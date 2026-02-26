# PDF Naming Tool

`./pdfs` 폴더의 PDF 파일명을 아래 형식으로 일괄 변경하는 스크립트입니다.

- 목표 형식: `출판연도 저널약자 논문제목.pdf`
- 예시: `2019 NEJM Dapagliflozin in Heart Failure.pdf`

## 파일

- `pdf_renamer.py`: 리네이밍/드라이런/원복 기능
- `journals.json`: 저널명 → 약자 매핑 파일
- `logs/*.csv`: 실제 변경 시 생성되는 로그 파일

## 동작 규칙

1. 대상 폴더: 기본 `./pdfs` (하위폴더 미포함)
2. 메타 추출 우선순위
   - (a) PDF 메타데이터 (`/Title`, `/Subject`, `/Journal`, `/CreationDate`)
   - (b) 첫 페이지 텍스트에서 title-like line 탐지
   - (c) DOI 검출 시 Crossref 조회로 year/journal/title 보완
3. 저널 약자 규칙
   - `journals.json` 매핑 우선
   - 미매핑 시 자동 약자 생성(단어 이니셜)
   - 이상한 경우 저널명 단축 fallback
4. 파일명 정리
   - 금지문자 제거: `\ / : * ? " < > |`
   - 연속 공백 축소
   - 총 파일명 길이 160자 초과 시 제목 잘라서 조정
5. 충돌 처리
   - 동일 파일명 존재 시 ` (2)`, ` (3)` suffix 부여
6. `--dry-run`
   - 변경될 결과만 출력, 실제 파일명 변경 없음
7. 실제 변경 모드
   - 리네이밍 수행 후 CSV 로그 생성
8. `--undo`
   - CSV 로그 기반으로 원복 지원

## 요구 패키지

```bash
pip install pypdf
```

> `pypdf`가 없으면 PDF 메타/본문 추출이 제한됩니다.

## 사용법

### 1) 드라이런 (미리보기)

```bash
python pdf_renamer.py --dry-run
```

### 2) 실제 리네이밍

```bash
python pdf_renamer.py
```

실행 후 예:

```text
old_file.pdf  ->  2019 NEJM Dapagliflozin in Heart Failure.pdf

Rename complete. Log written to: logs/rename_log_20260226_100000.csv
```

### 3) 원복

```bash
python pdf_renamer.py --undo logs/rename_log_20260226_100000.csv
```

원복 미리보기:

```bash
python pdf_renamer.py --undo logs/rename_log_20260226_100000.csv --dry-run
```

## journals.json 커스터마이징

```json
{
  "New England Journal of Medicine": "NEJM",
  "Circulation": "Circ",
  "European Heart Journal": "EHJ"
}
```

키 비교는 대소문자 무시로 처리됩니다.
