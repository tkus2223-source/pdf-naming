#!/usr/bin/env python3
"""Batch rename PDF files to: '<year> <journal_abbrev> <title>.pdf'.

What’s improved here (vs basic versions):
- Even if PDF metadata is empty, attempts to extract TITLE and YEAR from page-1 content.
- If PyMuPDF (fitz) is available, uses font-size-aware title detection from page 1 (much better than plain text).
- YEAR detection from page 1 uses multiple heuristics (copyright/published/citation lines),
  then falls back to the most frequent plausible year on page 1.
- Priority: filename parsing > page-1 (pymupdf dict/text, fallback pypdf text) > Crossref (optional) > metadata (last)
- Never forces defaults like '0000 UNKNOWNJ Untitled'. Missing parts are omitted.
- Title never blank: at minimum original filename stem.
- Collision-safe names: base.pdf, base (2).pdf, base (3).pdf (no '(2) (2)' chaining)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import fitz  # pymupdf
except Exception:
    fitz = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


FORBIDDEN_CHARS_PATTERN = re.compile(r'[\\/:*?"<>|]')
MULTI_SPACE_PATTERN = re.compile(r"\s+")
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")
UPPER_TOKEN_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{1,10}$")
SUFFIX_PATTERN = re.compile(r"\s*\((\d+)\)$")

# Common non-title header/footer phrases (very lightweight)
TITLE_BAD_SUBSTRINGS = [
    "abstract",
    "keywords",
    "introduction",
    "doi",
    "http",
    "www.",
    "copyright",
    "all rights reserved",
    "published",
    "accepted",
    "received",
    "correspondence",
    "author",
    "authors",
    "affiliation",
    "department",
    "university",
    "hospital",
    "volume",
    "issue",
    "pages",
    "pp.",
    "©",
]


@dataclass
class FieldValue:
    value: Optional[str] = None
    source: Optional[str] = None


@dataclass
class PdfInfo:
    year: FieldValue = field(default_factory=FieldValue)
    journal: FieldValue = field(default_factory=FieldValue)
    title: FieldValue = field(default_factory=FieldValue)
    doi: FieldValue = field(default_factory=FieldValue)


@dataclass
class RenamePlan:
    src: Path
    dest: Path
    info: PdfInfo


def sanitize_text(text: str) -> str:
    text = (text or "").replace("_", " ")
    text = FORBIDDEN_CHARS_PATTERN.sub(" ", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip().strip(".")
    return text


def set_if_empty(field_obj: FieldValue, value: Optional[str], source: str) -> None:
    if field_obj.value:
        return
    if value and value.strip():
        field_obj.value = sanitize_text(value)
        field_obj.source = source


def detect_doi(text: str) -> Optional[str]:
    m = DOI_PATTERN.search(text or "")
    return m.group(0).rstrip(".,);]") if m else None


def detect_year(text: str) -> Optional[str]:
    m = YEAR_PATTERN.search(text or "")
    return m.group(1) if m else None


def load_journal_map(path: Path) -> Dict[str, str]:
    """journals.json: { "Full Journal Name": "ABBR", ... }.
    Keys stored lowercased; values sanitized.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        out: Dict[str, str] = {}
        for k, v in (data or {}).items():
            kk = str(k).strip().lower()
            vv = sanitize_text(str(v))
            if kk and vv:
                out[kk] = vv
        return out
    except Exception:
        return {}


def abbreviate_journal(journal: str) -> str:
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", journal) if t]
    initials = "".join(t[0].upper() for t in tokens if t)
    if 2 <= len(initials) <= 10:
        return initials
    short = sanitize_text(journal)
    return short[:24].rstrip() if len(short) > 24 else short


def parse_from_filename(path: Path, journal_map: Dict[str, str]) -> PdfInfo:
    """Best-effort parse from filename stem.
    Note: this is a conservative parser; page-1 extraction handles cases filename is messy.
    """
    info = PdfInfo()
    stem = sanitize_text(path.stem)

    # title fallback: always at least filename stem
    set_if_empty(info.title, stem, "filename")

    # year
    year = detect_year(stem)
    set_if_empty(info.year, year, "filename")

    tokens = stem.split()
    journal_candidate: Optional[str] = None

    # obvious abbrev token: NEJM, BJA, JAMA...
    for tok in tokens:
        if UPPER_TOKEN_PATTERN.match(tok) and not YEAR_PATTERN.fullmatch(tok):
            journal_candidate = tok
            break

    # substring match full journal name from map
    if not journal_candidate and journal_map:
        lowered = stem.lower()
        for full_name_lc, abbr in journal_map.items():
            if full_name_lc in lowered:
                journal_candidate = abbr
                break

    if journal_candidate:
        set_if_empty(info.journal, journal_candidate, "filename")

    # title candidate: remaining tokens minus year and journal token
    if info.year.value:
        tokens = [t for t in tokens if t != info.year.value]
    if info.journal.value:
        jv = info.journal.value
        tokens = [t for t in tokens if t != jv]

    candidate_title = sanitize_text(" ".join(tokens))
    if candidate_title and len(candidate_title) >= 6:
        info.title.value = candidate_title
        info.title.source = "filename"

    return info


def extract_text_pymupdf(pdf_path: Path) -> str:
    if fitz is None:
        return ""
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            return ""
        text = doc[0].get_text("text") or ""
        doc.close()
        return text
    except Exception:
        return ""


def extract_text_pypdf(pdf_path: Path) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(str(pdf_path))
        if not reader.pages:
            return ""
        return reader.pages[0].extract_text() or ""
    except Exception:
        return ""


def extract_page1_lines_with_fonts_pymupdf(pdf_path: Path) -> List[Tuple[str, float]]:
    """Return list of (line_text, approx_font_size) from page 1 using PyMuPDF.
    If anything fails, returns [].
    """
    if fitz is None:
        return []
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            return []
        page = doc[0]
        d = page.get_text("dict")
        doc.close()

        out: List[Tuple[str, float]] = []
        for block in d.get("blocks", []):
            if block.get("type") != 0:  # text blocks
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                texts = []
                sizes = []
                for sp in spans:
                    t = sp.get("text", "")
                    if t and t.strip():
                        texts.append(t.strip())
                        try:
                            sizes.append(float(sp.get("size", 0.0)))
                        except Exception:
                            pass
                if not texts:
                    continue
                line_text = sanitize_text(" ".join(texts))
                if not line_text:
                    continue
                font_size = sum(sizes) / len(sizes) if sizes else 0.0
                out.append((line_text, font_size))
        return out
    except Exception:
        return []


def looks_like_title_line(line: str) -> bool:
    """Heuristic for title-like lines (text-only fallback)."""
    line_s = sanitize_text(line)
    if len(line_s) < 25:
        return False
    if DOI_PATTERN.search(line_s) or YEAR_PATTERN.search(line_s):
        return False
    words = line_s.split()
    if len(words) < 5:
        return False
    low = line_s.lower()
    if any(bad in low for bad in TITLE_BAD_SUBSTRINGS):
        return False
    # Avoid lines that are almost all uppercase (often headers)
    letters = [c for c in line_s if c.isalpha()]
    if letters:
        upper_ratio = sum(c.isupper() for c in letters) / len(letters)
        if upper_ratio > 0.85:
            return False
    return True


def detect_title_from_page1(text: str) -> Optional[str]:
    """Text-only fallback title detection."""
    best: Optional[str] = None
    best_score = -1.0
    for raw in (text or "").splitlines():
        line = sanitize_text(raw)
        if not looks_like_title_line(line):
            continue
        # score: longer + more words, penalize very long (often includes affiliations)
        n = len(line)
        w = len(line.split())
        score = (w * 3.0) + (n * 0.05)
        if n > 140:
            score -= 10
        if score > best_score:
            best_score = score
            best = line
    return best


def detect_title_from_page1_pymupdf_fonts(pdf_path: Path) -> Optional[str]:
    """Font-aware title detection.
    Strategy:
      - rank by font size (desc), focus on top region lines by size
      - filter out obvious non-title lines
      - pick best candidate by (font_size, words, length)
    """
    lines = extract_page1_lines_with_fonts_pymupdf(pdf_path)
    if not lines:
        return None

    # Consider top font sizes first
    lines_sorted = sorted(lines, key=lambda x: x[1], reverse=True)

    # Determine a threshold: lines within 90% of max font size are likely title candidates
    max_size = lines_sorted[0][1] if lines_sorted else 0.0
    if max_size <= 0:
        return None
    threshold = max_size * 0.90

    candidates: List[Tuple[str, float, float]] = []  # (line, size, score)
    for line, size in lines_sorted[:80]:
        if size < threshold:
            # still allow a few next lines (titles sometimes split across 2 sizes)
            if size < max_size * 0.80:
                break
        if not looks_like_title_line(line):
            continue
        words = len(line.split())
        length = len(line)
        # score: font size dominates, then word count/length
        score = (size * 10.0) + (words * 2.0) + (length * 0.02)
        candidates.append((line, size, score))

    if not candidates:
        # fallback: try any title-like lines, not just top-size
        for line, size in lines_sorted[:120]:
            if not looks_like_title_line(line):
                continue
            words = len(line.split())
            length = len(line)
            score = (size * 6.0) + (words * 2.0) + (length * 0.02)
            candidates.append((line, size, score))

    if not candidates:
        return None

    # sometimes titles are split across two consecutive big lines; merge adjacent candidates if they look like continuation
    best_line, _, _ = max(candidates, key=lambda x: x[2])

    # Try to append the next line if it is also big and title-like but shorter (continuation)
    # This is conservative: only if combined looks better and remains reasonable length
    try_merge = []
    for line, size in lines_sorted:
        if line == best_line:
            continue
        if size < max_size * 0.85:
            continue
        if not looks_like_title_line(line):
            continue
        # merge candidate if it does not repeat too much and combined length reasonable
        if len(best_line) < 120 and len(line) < 80:
            combined = sanitize_text(best_line + " " + line)
            if looks_like_title_line(combined) and len(combined) <= 170:
                try_merge.append(combined)

    if try_merge:
        # pick the longest merged (often the true full title)
        merged_best = max(try_merge, key=lambda s: len(s))
        return merged_best

    return best_line


def detect_year_from_page1(text: str) -> Optional[str]:
    """Heuristic year detection from page-1 text.

    Steps:
      1) Prefer years on lines with patterns: ©, copyright, published, accepted, received, journal citation-like.
      2) Else choose the most frequent plausible year on page 1.
    """
    if not text:
        return None

    current_year = dt.datetime.now().year
    min_year = 1900

    def valid(y: str) -> bool:
        try:
            yi = int(y)
        except Exception:
            return False
        return min_year <= yi <= current_year

    lines = [sanitize_text(l) for l in text.splitlines() if sanitize_text(l)]
    if not lines:
        return None

    priority_hits: List[str] = []
    for line in lines[:80]:
        low = line.lower()
        if any(k in low for k in ["©", "copyright", "published", "accepted", "received", "online", "journal", "vol", "volume"]):
            for y in YEAR_PATTERN.findall(line):
                if valid(y):
                    priority_hits.append(y)

    if priority_hits:
        # if multiple, choose the most frequent among priority lines
        return max(set(priority_hits), key=priority_hits.count)

    # fallback: mode of all years on page 1
    years_all = [y for y in YEAR_PATTERN.findall(text) if valid(y)]
    if not years_all:
        return None
    return max(set(years_all), key=years_all.count)


def extract_metadata(pdf_path: Path) -> Dict[str, str]:
    if PdfReader is None:
        return {}
    try:
        reader = PdfReader(str(pdf_path))
        meta = reader.metadata or {}
        out: Dict[str, str] = {}
        for k, v in meta.items():
            if v is not None:
                out[str(k)] = str(v)
        return out
    except Exception:
        return {}


def crossref_lookup(doi: str, timeout: float = 8.0) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    req = urllib.request.Request(url, headers={"User-Agent": "pdf-renamer/2.2"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None, None, None

    msg = payload.get("message", {})
    title = ((msg.get("title") or [None])[0] or None)
    journal = ((msg.get("short-container-title") or [None])[0] or (msg.get("container-title") or [None])[0] or None)

    year = None
    parts = msg.get("issued", {}).get("date-parts", [])
    if parts and parts[0]:
        year = str(parts[0][0])

    return year, journal, title


def choose_journal_abbrev(journal: Optional[str], journal_map: Dict[str, str]) -> Optional[str]:
    if not journal:
        return None
    j_clean = sanitize_text(journal)
    if not j_clean:
        return None

    mapped = journal_map.get(j_clean.lower())
    if mapped:
        return mapped

    if UPPER_TOKEN_PATTERN.match(j_clean):
        return j_clean

    return abbreviate_journal(j_clean)


def build_name(info: PdfInfo, journal_map: Dict[str, str], max_len: int = 160) -> str:
    # Do not force defaults; omit missing parts
    title = sanitize_text(info.title.value or "")
    if not title:
        title = "document"

    year = sanitize_text(info.year.value) if info.year.value else ""
    journal_raw = sanitize_text(info.journal.value) if info.journal.value else ""
    journal = choose_journal_abbrev(journal_raw, journal_map) if journal_raw else ""

    parts = [p for p in [year, journal, title] if p]
    stem = sanitize_text(" ".join(parts))
    if not stem:
        stem = sanitize_text(info.title.value or "") or "document"

    if len(stem) + 4 > max_len:
        overflow = len(stem) + 4 - max_len
        title_cut = title[:-overflow].rstrip() if overflow < len(title) else title[:40].rstrip()
        parts = [p for p in [year, journal, title_cut] if p]
        stem = sanitize_text(" ".join(parts)) or sanitize_text(title_cut) or "document"

    return f"{stem}.pdf"


def strip_numeric_suffix(stem: str) -> str:
    s = stem
    while True:
        m = SUFFIX_PATTERN.search(s)
        if not m:
            break
        s = s[: m.start()].rstrip()
    return s


def unique_destination(src: Path, desired: Path, seen: set[Path]) -> Path:
    base = strip_numeric_suffix(desired.stem)
    first = desired.with_name(f"{base}{desired.suffix}")

    def is_available(p: Path) -> bool:
        if p in seen:
            return False
        if p == src:
            return True
        return not p.exists()

    if is_available(first):
        return first

    i = 2
    while True:
        candidate = desired.with_name(f"{base} ({i}){desired.suffix}")
        if is_available(candidate):
            return candidate
        i += 1


def enrich_info(pdf_path: Path, info: PdfInfo, use_crossref: bool) -> None:
    # Page-1 extraction (prefer PyMuPDF)
    text = extract_text_pymupdf(pdf_path)
    text_source = "pdftext:pymupdf"
    if not text:
        text = extract_text_pypdf(pdf_path)
        text_source = "pdftext:pypdf"

    if text:
        # DOI from page 1
        set_if_empty(info.doi, detect_doi(text), text_source)

        # Improved TITLE from page 1: font-aware if available
        title_from_fonts = detect_title_from_page1_pymupdf_fonts(pdf_path) if fitz is not None else None
        if title_from_fonts:
            set_if_empty(info.title, title_from_fonts, "pdftext:pymupdf:fonts")
        else:
            set_if_empty(info.title, detect_title_from_page1(text), text_source)

        # Improved YEAR from page 1 text (do this before metadata)
        set_if_empty(info.year, detect_year_from_page1(text), text_source)

    # Crossref enrichment (optional) if DOI found
    if use_crossref and info.doi.value:
        y, j, t = crossref_lookup(info.doi.value)
        set_if_empty(info.year, y, "crossref")
        set_if_empty(info.journal, j, "crossref")
        set_if_empty(info.title, t, "crossref")

    # Metadata last
    meta = extract_metadata(pdf_path)
    if meta:
        set_if_empty(info.year, detect_year(meta.get("/CreationDate", "")), "metadata")
        set_if_empty(info.journal, meta.get("/Journal") or meta.get("/Subject"), "metadata")
        set_if_empty(info.title, meta.get("/Title"), "metadata")
        if not info.doi.value:
            set_if_empty(info.doi, detect_doi("\n".join(meta.values())), "metadata")


def plan_renames(folder: Path, journal_map: Dict[str, str], use_crossref: bool) -> List[RenamePlan]:
    plans: List[RenamePlan] = []
    seen: set[Path] = set()

    for src in sorted(folder.glob("*.pdf")):
        info = parse_from_filename(src, journal_map)
        enrich_info(src, info, use_crossref)

        # Title must never be blank
        if not info.title.value:
            info.title.value = sanitize_text(src.stem)
            info.title.source = "filename"

        target_name = build_name(info, journal_map)
        dest = unique_destination(src, src.with_name(target_name), seen)
        seen.add(dest)
        plans.append(RenamePlan(src=src, dest=dest, info=info))

    return plans


def source_of(field: FieldValue) -> str:
    return field.source or "n/a"


def write_report(path: Path, plans: List[RenamePlan], dry_run: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "old_name",
                "new_name",
                "year",
                "year_source",
                "journal",
                "journal_source",
                "title",
                "title_source",
                "doi",
                "doi_source",
                "mode",
            ]
        )
        for p in plans:
            w.writerow(
                [
                    p.src.name,
                    p.dest.name,
                    p.info.year.value or "",
                    source_of(p.info.year),
                    p.info.journal.value or "",
                    source_of(p.info.journal),
                    p.info.title.value or "",
                    source_of(p.info.title),
                    p.info.doi.value or "",
                    source_of(p.info.doi),
                    "dry-run" if dry_run else "rename",
                ]
            )


def execute(plans: List[RenamePlan], dry_run: bool, log_dir: Path) -> Optional[Path]:
    if not plans:
        print("No PDF files found in the target folder.")
        return None

    now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"rename_log_{now}.csv"

    logs: List[Tuple[str, str, str, str]] = []  # ts, old_path, new_path, status

    for p in plans:
        print(
            f"{p.src.name} -> {p.dest.name} | "
            f"year={p.info.year.value or '-'}({source_of(p.info.year)}), "
            f"journal={p.info.journal.value or '-'}({source_of(p.info.journal)}), "
            f"title={p.info.title.value or '-'}({source_of(p.info.title)}), "
            f"doi={p.info.doi.value or '-'}({source_of(p.info.doi)})"
        )

        status = "dry-run"
        if not dry_run and p.src != p.dest:
            p.src.rename(p.dest)
            status = "renamed"
        elif not dry_run:
            status = "skipped-same-name"

        logs.append((dt.datetime.now().isoformat(), str(p.src), str(p.dest), status))

    if dry_run:
        print("\nDry run complete. No files were renamed.")
        return None

    with log_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "old_path", "new_path", "status"])
        w.writerows(logs)

    return log_path


def undo(log_path: Path, dry_run: bool) -> None:
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    with log_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in reversed(rows):
        status = row.get("status", "")
        if status != "renamed":
            continue

        old = Path(row["old_path"])
        new = Path(row["new_path"])

        print(f"UNDO {new.name} -> {old.name}")
        if not dry_run and new.exists():
            old.parent.mkdir(parents=True, exist_ok=True)
            new.rename(old)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rename PDFs to '<year> <journal_abbrev> <title>.pdf'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python pdf_renamer.py --dry-run
              python pdf_renamer.py --dry-run --use-crossref
              python pdf_renamer.py --report report.csv
              python pdf_renamer.py --undo logs/rename_log_20260101_120000.csv
              python pdf_renamer.py --undo logs/rename_log_20260101_120000.csv --dry-run
            """
        ).strip(),
    )
    parser.add_argument("--folder", default="./pdfs", help="Target folder containing PDF files (non-recursive).")
    parser.add_argument("--journals", default="./journals.json", help="Path to journal abbreviation mapping JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Preview rename results without changing files.")
    parser.add_argument("--undo", help="Undo renames using a CSV log file path.")
    parser.add_argument("--log-dir", default="./logs", help="Directory to save rename logs.")
    parser.add_argument("--use-crossref", action="store_true", help="Enable DOI Crossref enrichment.")
    parser.add_argument("--report", default="report.csv", help="Write rename analysis report CSV.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.undo:
        undo(Path(args.undo), dry_run=args.dry_run)
        return 0

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Target folder not found: {folder}")
        return 1

    journal_map = load_journal_map(Path(args.journals))
    plans = plan_renames(folder, journal_map, use_crossref=args.use_crossref)

    if args.report:
        write_report(Path(args.report), plans, args.dry_run)
        print(f"Report written: {args.report}")

    log_path = execute(plans, args.dry_run, Path(args.log_dir))
    if log_path:
        print(f"Rename log written: {log_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
