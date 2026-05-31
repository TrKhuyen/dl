import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_FINETUNE_PATH = Path("data/processed/legal_finetune_dataset_2024_2026.jsonl")
DEFAULT_CHUNKS_PATH = Path("data/processed/legal_chunks_2024_2026.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/legal_finetune_dataset_2024_2026_raw_context.jsonl")
DEFAULT_REPORT_PATH = Path("data/processed/raw_context_enrichment_report.csv")

DIEU = "\u0110i\u1ec1u"
KHOAN = "Kho\u1ea3n"

ARTICLE_RE = re.compile(rf"\b{DIEU}\s+(\d+[a-zA-Z]?)", re.IGNORECASE)
CLAUSE_RE = re.compile(rf"\b{KHOAN}\s+(\d+[a-zA-Z]?)", re.IGNORECASE)
CLAUSE_START_RE = re.compile(r"(?m)^\s*(\d+[a-zA-Z]?)\.\s+")
TOKEN_RE = re.compile(r"\w+", re.UNICODE)

STOPWORDS = {
    "theo",
    "cua",
    "cua",
    "va",
    "la",
    "duoc",
    "nhung",
    "cac",
    "mot",
    "trong",
    "quy",
    "dinh",
    "luat",
    "dieu",
    "khoan",
    "hay",
    "neu",
    "phan",
    "tich",
}


def configure_stdout():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def increase_csv_field_limit():
    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            return
        except OverflowError:
            field_limit = int(field_limit / 10)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_for_contains(text: str) -> str:
    return normalize_space(text).casefold()


def normalize_article_number(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def extract_document_number(context: str) -> str:
    matches = re.findall(r",\s*s\u1ed1\s+([^,]+)$", context, flags=re.IGNORECASE)
    if not matches:
        return ""
    return normalize_space(matches[-1])


def extract_article_numbers(text: str) -> list[str]:
    seen = set()
    numbers = []

    for match in ARTICLE_RE.finditer(text):
        number = normalize_article_number(match.group(1))
        if number not in seen:
            numbers.append(number)
            seen.add(number)

    return numbers


def extract_clause_numbers(text: str) -> list[str]:
    seen = set()
    numbers = []

    for match in CLAUSE_RE.finditer(text):
        number = normalize_article_number(match.group(1))
        if number not in seen:
            numbers.append(number)
            seen.add(number)

    return numbers


def strip_chunk_metadata(chunk_text: str, article_number: str) -> str:
    if not chunk_text:
        return ""

    article_pattern = re.compile(
        rf"(?m)^\s*{DIEU}\s+{re.escape(article_number)}(?:\.|\s)",
        re.IGNORECASE,
    )
    match = article_pattern.search(chunk_text)
    if match:
        return chunk_text[match.start() :].strip()

    generic_match = re.search(rf"(?m)^\s*{DIEU}\s+\d+[a-zA-Z]?(?:\.|\s)", chunk_text)
    if generic_match:
        return chunk_text[generic_match.start() :].strip()

    parts = re.split(r"\n\s*\n", chunk_text, maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()

    return chunk_text.strip()


def extract_selected_clauses(article_text: str, clause_numbers: list[str]) -> str:
    if not article_text or not clause_numbers:
        return article_text

    matches = list(CLAUSE_START_RE.finditer(article_text))
    if not matches:
        return article_text

    header = article_text[: matches[0].start()].strip()
    clause_blocks = {}

    for index, match in enumerate(matches):
        clause_number = normalize_article_number(match.group(1))
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(article_text)
        clause_blocks[clause_number] = article_text[start:end].strip()

    selected = [clause_blocks[number] for number in clause_numbers if number in clause_blocks]
    if not selected:
        return article_text

    if header:
        return header + "\n\n" + "\n\n".join(selected)
    return "\n\n".join(selected)


def remove_repeated_article_heading(raw_text: str, article_number: str) -> str:
    lines = raw_text.splitlines()
    first_content_index = None

    for index, line in enumerate(lines):
        if line.strip():
            first_content_index = index
            break

    if first_content_index is None:
        return ""

    heading_re = re.compile(
        rf"^\s*{DIEU}\s+{re.escape(article_number)}(?:\.|\s)",
        re.IGNORECASE,
    )
    if not heading_re.search(lines[first_content_index]):
        return raw_text.strip()

    return "\n".join(lines[first_content_index + 1 :]).strip()


def combine_article_parts(parts: list[dict]) -> dict:
    if not parts:
        return {}

    first_part = parts[0]
    article_number = first_part["article_number"]
    raw_parts = []

    for index, part in enumerate(parts):
        raw_text = part["raw_text"]
        if index > 0:
            raw_text = remove_repeated_article_heading(raw_text, article_number)
        if raw_text:
            raw_parts.append(raw_text)

    combined_raw_text = "\n\n".join(raw_parts)

    return {
        "chunk_id": "|".join(part["chunk_id"] for part in parts),
        "article_number": article_number,
        "article_title": first_part["article_title"],
        "raw_text": combined_raw_text,
        "norm_text": normalize_for_contains(combined_raw_text),
        "tokens": tokenize(combined_raw_text),
    }


def tokenize(text: str) -> set[str]:
    tokens = set()
    for token in TOKEN_RE.findall(text.casefold()):
        if len(token) <= 2 or token in STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def read_finetune_samples(path: Path):
    samples = []
    doc_numbers = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc

            source_context = str(sample.get("context", "")).strip()
            document_number = extract_document_number(source_context)
            if document_number:
                doc_numbers.add(document_number)

            samples.append(
                {
                    "line_number": line_number,
                    "sample": sample,
                    "source_context": source_context,
                    "document_number": document_number,
                }
            )

    return samples, doc_numbers


def build_chunk_index(chunks_path: Path, doc_numbers: set[str]):
    increase_csv_field_limit()

    article_parts = defaultdict(lambda: defaultdict(list))

    with chunks_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {
            "document_number",
            "chunk_id",
            "chunk_type",
            "article_number",
            "article_title",
            "chunk_text",
        }
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing chunk columns: {sorted(missing_columns)}")

        for row in reader:
            document_number = str(row.get("document_number", "")).strip()
            if document_number not in doc_numbers:
                continue

            if row.get("chunk_type") not in {"article", "article_part"}:
                continue

            article_number = normalize_article_number(row.get("article_number", ""))
            if not article_number:
                continue

            raw_text = strip_chunk_metadata(str(row.get("chunk_text", "")), article_number)
            chunk = {
                "chunk_id": str(row.get("chunk_id", "")),
                "article_number": article_number,
                "article_title": str(row.get("article_title", "")),
                "raw_text": raw_text,
                "tokens": tokenize(raw_text),
            }

            article_parts[document_number][article_number].append(chunk)

    article_index = defaultdict(dict)
    fallback_chunks = defaultdict(list)

    for document_number, articles in article_parts.items():
        for article_number, parts in articles.items():
            combined_chunk = combine_article_parts(parts)
            if not combined_chunk:
                continue
            article_index[document_number][article_number] = combined_chunk
            fallback_chunks[document_number].append(combined_chunk)

    return article_index, fallback_chunks


def extract_search_phrases(text: str) -> list[str]:
    phrases = []

    for phrase in re.split(r"[.!?\n]+", text):
        phrase = normalize_space(phrase)
        if 35 <= len(phrase) <= 260:
            phrases.append(phrase)

    return phrases[:8]


def best_fallback_chunk(fallback_chunks: list[dict], instruction: str, response: str):
    instruction_tokens = tokenize(instruction)
    response_tokens = tokenize(response)
    if not instruction_tokens and not response_tokens:
        return None, 0

    search_phrases = extract_search_phrases(response) + extract_search_phrases(instruction)
    best_chunk = None
    best_score = 0

    for chunk in fallback_chunks:
        score = len(instruction_tokens & chunk["tokens"])
        score += 2 * len(response_tokens & chunk["tokens"])

        chunk_norm_text = chunk.get("norm_text", "")
        for phrase in search_phrases:
            if normalize_for_contains(phrase) in chunk_norm_text:
                score += min(500, max(50, len(phrase)))

        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_chunk, best_score


def build_raw_context_for_sample(
    item,
    article_index,
    fallback_chunks,
    context_level: str,
):
    sample = item["sample"]
    document_number = item["document_number"]
    instruction = str(sample.get("instruction", ""))
    response = str(sample.get("response", ""))

    instruction_articles = extract_article_numbers(instruction)
    response_articles = extract_article_numbers(response)
    article_candidates = instruction_articles or response_articles
    clause_numbers = extract_clause_numbers(instruction + " " + response)

    matched_chunks = []
    for article_number in article_candidates:
        chunk = article_index.get(document_number, {}).get(article_number)
        if chunk:
            matched_chunks.append(chunk)

    match_method = "article"
    fallback_score = ""

    if not matched_chunks:
        fallback_chunk, score = best_fallback_chunk(
            fallback_chunks.get(document_number, []),
            instruction=instruction,
            response=response,
        )
        fallback_score = score
        if fallback_chunk:
            matched_chunks = [fallback_chunk]
            match_method = "fallback_similarity"

    if not matched_chunks:
        return {
            "raw_context": item["source_context"],
            "status": "missing",
            "match_method": "missing",
            "article_numbers": article_candidates,
            "clause_numbers": clause_numbers,
            "chunk_ids": [],
            "fallback_score": fallback_score,
        }

    raw_parts = []
    for chunk in matched_chunks:
        raw_text = chunk["raw_text"]
        if context_level == "clause" and clause_numbers:
            raw_text = extract_selected_clauses(raw_text, clause_numbers)
        raw_parts.append(raw_text)

    return {
        "raw_context": "\n\n---\n\n".join(part for part in raw_parts if part),
        "status": "matched",
        "match_method": match_method,
        "article_numbers": [chunk["article_number"] for chunk in matched_chunks],
        "clause_numbers": clause_numbers,
        "chunk_ids": [chunk["chunk_id"] for chunk in matched_chunks],
        "fallback_score": fallback_score,
    }


def write_jsonl(samples, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_report(report_rows, report_path: Path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "line_number",
        "status",
        "match_method",
        "document_number",
        "source_context",
        "article_numbers",
        "clause_numbers",
        "chunk_ids",
        "fallback_score",
        "raw_context_char_len",
    ]

    with report_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)


def enrich_finetune_contexts(
    finetune_path: Path,
    chunks_path: Path,
    output_path: Path,
    report_path: Path,
    context_level: str,
    in_place: bool,
):
    samples, doc_numbers = read_finetune_samples(finetune_path)
    article_index, fallback_chunks = build_chunk_index(chunks_path, doc_numbers)

    enriched_samples = []
    report_rows = []
    status_counts = defaultdict(int)
    method_counts = defaultdict(int)

    for item in samples:
        result = build_raw_context_for_sample(
            item=item,
            article_index=article_index,
            fallback_chunks=fallback_chunks,
            context_level=context_level,
        )

        sample = dict(item["sample"])
        sample["context"] = result["raw_context"]
        enriched_samples.append(sample)

        status_counts[result["status"]] += 1
        method_counts[result["match_method"]] += 1

        report_rows.append(
            {
                "line_number": item["line_number"],
                "status": result["status"],
                "match_method": result["match_method"],
                "document_number": item["document_number"],
                "source_context": item["source_context"],
                "article_numbers": "|".join(result["article_numbers"]),
                "clause_numbers": "|".join(result["clause_numbers"]),
                "chunk_ids": "|".join(result["chunk_ids"]),
                "fallback_score": result["fallback_score"],
                "raw_context_char_len": len(result["raw_context"]),
            }
        )

    final_output_path = output_path
    if in_place:
        backup_path = finetune_path.with_suffix(finetune_path.suffix + ".title_context_backup")
        if not backup_path.exists():
            with finetune_path.open("r", encoding="utf-8") as source, backup_path.open(
                "w", encoding="utf-8", newline="\n"
            ) as backup:
                for line in source:
                    backup.write(line)
        final_output_path = finetune_path

    temp_path = final_output_path.with_name(f"{final_output_path.name}.tmp")
    write_jsonl(enriched_samples, temp_path)
    os.replace(temp_path, final_output_path)
    write_report(report_rows, report_path)

    return {
        "samples": len(samples),
        "doc_numbers": len(doc_numbers),
        "status_counts": dict(status_counts),
        "method_counts": dict(method_counts),
        "output_path": final_output_path,
        "report_path": report_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replace finetune context source names with raw legal Article/Clause text."
    )
    parser.add_argument("--finetune", type=Path, default=DEFAULT_FINETUNE_PATH)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--context-level",
        choices=["article", "clause"],
        default="article",
        help="Use full Article text, or selected Clause text when clauses are detected.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite --finetune and create a .title_context_backup file first.",
    )
    return parser.parse_args()


def main():
    configure_stdout()
    args = parse_args()

    if not args.finetune.exists():
        raise FileNotFoundError(f"Finetune file not found: {args.finetune}")
    if not args.chunks.exists():
        raise FileNotFoundError(f"Chunks file not found: {args.chunks}")

    summary = enrich_finetune_contexts(
        finetune_path=args.finetune,
        chunks_path=args.chunks,
        output_path=args.output,
        report_path=args.report,
        context_level=args.context_level,
        in_place=args.in_place,
    )

    print(f"Samples processed: {summary['samples']}")
    print(f"Documents indexed: {summary['doc_numbers']}")
    print(f"Status counts: {summary['status_counts']}")
    print(f"Match method counts: {summary['method_counts']}")
    print(f"Output saved to: {summary['output_path']}")
    print(f"Report saved to: {summary['report_path']}")


if __name__ == "__main__":
    main()
