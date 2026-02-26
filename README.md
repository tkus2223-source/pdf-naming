# PDF Naming Tool

`./pdfs` 폴더의 PDF를 일괄 리네이밍합니다.

목표 형식:
- `출판연도 저널약자 논문제목.pdf`

## 핵심 변경점

추출 우선순위(최신):
1. **파일명 파싱** (최우선)
2. **PDF 1페이지 텍스트(기본: pymupdf, fallback: pypdf)** 에서 DOI/제목 추출
3. **Crossref 조회**로 year/journal/title 보완 (`--use-crossref` 사용 시)

중요:
- 정보가 부족해도 `0000 UNKNOWNJ Untitled` 같은 강제 기본값으로 덮어쓰지 않습니다.
- 최소 title은 항상 기존 파일명(stem)을 사용합니다.

## 설치

```bash
pip install pymupdf pypdf
```

## 사용법

### Dry-run

```bash
python pdf_renamer.py --dry-run --report report.csv
```

출력:
- `원본 -> 새이름`
- `year/journal/title` 값과 각각의 source (`filename`, `pdftext:pymupdf`, `pdftext:pypdf`, `crossref`)

### Crossref 보완 사용

```bash
python pdf_renamer.py --dry-run --use-crossref --report report.csv
```

### 실제 리네이밍

```bash
python pdf_renamer.py --report report.csv
```

- 실제 변경 로그: `logs/rename_log.csv`
- 분석 리포트: `report.csv`

### Undo

```bash
python pdf_renamer.py --undo logs/rename_log.csv
```

## 파일명 파싱 규칙

- 4자리 연도(`19xx`, `20xx`) 탐지
- 저널 약자 후보:
  - 대문자 토큰 (`NEJM`, `BJA`, `JAMA` 등)
  - 또는 `journals.json`에 있는 저널명 매핑
- 남은 토큰은 제목으로 사용
- `_` 및 다중 공백 정리

## 충돌 처리

동일 이름이면:
- `base.pdf`
- `base (2).pdf`
- `base (3).pdf`

중첩 suffix (`(2) (2)`)는 생성하지 않습니다.
