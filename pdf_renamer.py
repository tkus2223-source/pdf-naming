#!/usr/bin/env python3
"""Batch rename PDF files to: '<year> <journal_abbrev> <title>.pdf'."""

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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

FORBIDDEN_CHARS_PATTERN = re.compile(r'[\\/:*?"<>|]')
MULTI_SPACE_PATTERN = re.compile(r"\s+")
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2}|2100)\b")
STOPWORDS = {
    "of",
    "and",
    "the",
    "in",
    "for",
    "to",
    "on",
    "a",
    "an",
    "at",
    "by",
    "from",
    "with",
    "journal",
}


@dataclass
class PdfInfo:
    year: Optional[str] = None
    journal: Optional[str] = None
    title: Optional[str] = None
    doi: Optional[str] = None


@dataclass
class RenamePlan:
    src: Path
    dest: Path
    info: PdfInfo


def sanitize_filename_component(value: str) -> str:
    cleaned = FORBIDDEN_CHARS_PATTERN.sub(" ", value)
    cleaned = MULTI_SPACE_PATTERN.sub(" ", cleaned).strip()
    cleaned = cleaned.strip(".")
    return cleaned


def parse_year(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    match = YEAR_PATTERN.search(text)
    return match.group(1) if match else None


def detect_doi(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    match = DOI_PATTERN.search(text)
    return match.group(0).rstrip(".,);]") if match else None


def looks_like_title(line: str) -> bool:
    line = line.strip()
    if len(line) < 20:
        return False
    words = [w for w in re.split(r"\s+", line) if w]
    if len(words) < 4:
        return False
    if YEAR_PATTERN.search(line):
        return False
    if DOI_PATTERN.search(line):
        return False
    lowercase_ratio = sum(ch.islower() for ch in line) / max(len([c for c in line if c.isalpha()]), 1)
    return lowercase_ratio > 0.2


def extract_first_page_text(pdf_path: Path) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(str(pdf_path))
        if not reader.pages:
            return ""
        return reader.pages[0].extract_text() or ""
    except Exception:
        return ""


def extract_metadata(pdf_path: Path) -> Tuple[Dict[str, str], str]:
    metadata: Dict[str, str] = {}
    first_page_text = ""
    if PdfReader is None:
        return metadata, first_page_text

    try:
        reader = PdfReader(str(pdf_path))
        raw_meta = reader.metadata or {}
        for key, value in raw_meta.items():
            if value is None:
                continue
            metadata[str(key)] = str(value)
        if reader.pages:
            first_page_text = reader.pages[0].extract_text() or ""
    except Exception:
        pass
    return metadata, first_page_text


def parse_title_from_text(first_page_text: str) -> Optional[str]:
    for line in (ln.strip() for ln in first_page_text.splitlines()):
        if looks_like_title(line):
            return line
    return None


def parse_journal_from_text(first_page_text: str) -> Optional[str]:
    lines = [ln.strip() for ln in first_page_text.splitlines() if ln.strip()]
    for line in lines[:20]:
        if "journal" in line.lower() or line.isupper():
            if len(line) <= 80 and len(line.split()) <= 10:
                return line
    return None


def crossref_lookup(doi: str, timeout: float = 8.0) -> PdfInfo:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    req = urllib.request.Request(url, headers={"User-Agent": "pdf-renamer/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return PdfInfo(doi=doi)

    message = payload.get("message", {})
    title = (message.get("title") or [None])[0]
    journal = (message.get("short-container-title") or [None])[0] or (message.get("container-title") or [None])[0]

    year = None
    issued = message.get("issued", {}).get("date-parts", [])
    if issued and issued[0]:
        year = str(issued[0][0])

    return PdfInfo(year=year, journal=journal, title=title, doi=doi)


def load_journal_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(k).strip().lower(): str(v).strip() for k, v in raw.items() if str(k).strip() and str(v).strip()}
    except Exception:
        return {}


def auto_abbreviate_journal(journal: str) -> str:
    words = [w for w in re.split(r"[^A-Za-z0-9]+", journal) if w]
    filtered = [w for w in words if w.lower() not in STOPWORDS]
    base = filtered or words
    initials = "".join(w[0].upper() for w in base if w)

    if 2 <= len(initials) <= 10:
        return initials

    short = sanitize_filename_component(journal)
    if len(short) > 24:
        short = short[:24].rstrip()
    return short


def choose_journal_abbrev(journal: Optional[str], journal_map: Dict[str, str]) -> str:
    if not journal:
        return "UNKNOWNJ"
    mapped = journal_map.get(journal.strip().lower())
    if mapped:
        return sanitize_filename_component(mapped)
    return sanitize_filename_component(auto_abbreviate_journal(journal)) or "UNKNOWNJ"


def build_target_name(info: PdfInfo, journal_map: Dict[str, str], max_len: int = 160) -> str:
    year = sanitize_filename_component(info.year or "0000")
    journal_abbrev = choose_journal_abbrev(info.journal, journal_map)
    title = sanitize_filename_component(info.title or "Untitled")

    stem = f"{year} {journal_abbrev} {title}".strip()
    suffix = ".pdf"
    if len(stem) + len(suffix) > max_len:
        overflow = len(stem) + len(suffix) - max_len
        title = title[:-overflow].rstrip() if overflow < len(title) else "Untitled"
        stem = f"{year} {journal_abbrev} {title}".strip()

    stem = MULTI_SPACE_PATTERN.sub(" ", stem)
    return f"{stem}{suffix}"


def unique_destination(dest: Path) -> Path:
    if not dest.exists():
        return dest
    idx = 2
    while True:
        candidate = dest.with_name(f"{dest.stem} ({idx}){dest.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def extract_pdf_info(pdf_path: Path) -> PdfInfo:
    metadata, first_page = extract_metadata(pdf_path)
    meta_text = "\n".join(metadata.values())

    info = PdfInfo(
        year=parse_year(metadata.get("/CreationDate") or meta_text),
        journal=metadata.get("/Journal") or metadata.get("/Subject"),
        title=metadata.get("/Title"),
        doi=detect_doi(meta_text),
    )

    if not info.title:
        info.title = parse_title_from_text(first_page)
    if not info.journal:
        info.journal = parse_journal_from_text(first_page)
    if not info.year:
        info.year = parse_year(first_page)
    if not info.doi:
        info.doi = detect_doi(first_page)

    if info.doi:
        crossref = crossref_lookup(info.doi)
        info.year = info.year or crossref.year
        info.journal = info.journal or crossref.journal
        info.title = info.title or crossref.title

    return info


def plan_renames(folder: Path, journal_map: Dict[str, str]) -> List[RenamePlan]:
    plans: List[RenamePlan] = []
    seen_destinations: set[Path] = set()

    for pdf_path in sorted(folder.glob("*.pdf")):
        info = extract_pdf_info(pdf_path)
        target_name = build_target_name(info, journal_map)
        dest = pdf_path.with_name(target_name)
        dest = unique_destination(dest)
        while dest in seen_destinations:
            dest = unique_destination(dest.with_name(f"{dest.stem} (2){dest.suffix}"))

        seen_destinations.add(dest)
        plans.append(RenamePlan(src=pdf_path, dest=dest, info=info))

    return plans


def write_log(log_path: Path, rows: Iterable[Tuple[str, str, str, str]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "old_path", "new_path", "status"])
        for row in rows:
            writer.writerow(row)


def apply_renames(plans: List[RenamePlan], dry_run: bool, log_dir: Path) -> Optional[Path]:
    if not plans:
        print("No PDF files found in the target folder.")
        return None

    now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"rename_log_{now}.csv"
    log_rows = []

    for plan in plans:
        print(f"{plan.src.name}  ->  {plan.dest.name}")
        status = "dry-run"
        if not dry_run and plan.src != plan.dest:
            plan.src.rename(plan.dest)
            status = "renamed"
        elif not dry_run and plan.src == plan.dest:
            status = "skipped-same-name"
        log_rows.append((dt.datetime.now().isoformat(), str(plan.src), str(plan.dest), status))

    if dry_run:
        print("\nDry run complete. No files were renamed.")
        return None

    write_log(log_path, log_rows)
    print(f"\nRename complete. Log written to: {log_path}")
    return log_path


def undo_from_log(log_path: Path, dry_run: bool) -> None:
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    with log_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    # Reverse order to safely handle chains.
    for row in reversed(rows):
        old_path = Path(row["old_path"])
        new_path = Path(row["new_path"])
        status = row.get("status", "")
        if status not in {"renamed", "skipped-same-name"}:
            continue

        if status == "skipped-same-name":
            print(f"SKIP {new_path} (same name originally)")
            continue

        if not new_path.exists():
            print(f"MISSING {new_path} (cannot undo)")
            continue

        print(f"UNDO {new_path.name}  ->  {old_path.name}")
        if not dry_run:
            old_path.parent.mkdir(parents=True, exist_ok=True)
            new_path.rename(old_path)

    print("Undo complete." if not dry_run else "Undo dry run complete.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch rename PDFs to '<year> <journal_abbrev> <title>.pdf'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python pdf_renamer.py --dry-run
              python pdf_renamer.py
              python pdf_renamer.py --undo logs/rename_log_20260101_120000.csv
              python pdf_renamer.py --undo logs/rename_log_20260101_120000.csv --dry-run
            """
        ),
    )
    parser.add_argument("--folder", default="./pdfs", help="Target folder containing PDF files (non-recursive).")
    parser.add_argument("--journals", default="./journals.json", help="Path to journal abbreviation mapping JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Preview rename results without changing files.")
    parser.add_argument("--undo", help="Undo renames using a CSV log file path.")
    parser.add_argument("--log-dir", default="./logs", help="Directory to save rename logs.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    folder = Path(args.folder)
    journal_map = load_journal_map(Path(args.journals))

    if args.undo:
        undo_from_log(Path(args.undo), dry_run=args.dry_run)
        return 0

    if not folder.exists():
        print(f"Target folder not found: {folder}")
        return 1

    plans = plan_renames(folder=folder, journal_map=journal_map)
    apply_renames(plans, dry_run=args.dry_run, log_dir=Path(args.log_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
