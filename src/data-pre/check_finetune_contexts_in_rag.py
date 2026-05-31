import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path


DEFAULT_FINETUNE_PATH = Path("data/processed/legal_finetune_dataset_2024_2026.jsonl")
DEFAULT_RAG_PATH = Path("data/processed/legal_content_2024_2026.csv")
DEFAULT_REPORT_PATH = Path("data/processed/finetune_contexts_in_rag_report.csv")
DEFAULT_HEADER_CHARS = 5000


def configure_stdout():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def increase_csv_field_limit():
    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            return
        except OverflowError:
            field_limit = int(field_limit / 10)


def read_context_counts(jsonl_path: Path) -> Counter:
    context_counts = Counter()

    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc

            context = normalize_space(sample.get("context", ""))
            if context:
                context_counts[context] += 1

    return context_counts


def extract_document_number(context: str) -> str:
    match = re.search(r",\s*số\s+(.+)$", context, flags=re.IGNORECASE)
    if not match:
        return ""
    return normalize_space(match.group(1))


def extract_title(context: str) -> str:
    title = re.sub(r",\s*số\s+.+$", "", context, flags=re.IGNORECASE)
    title = title.split(" của ", maxsplit=1)[0]
    return normalize_space(title)


def build_context_rows(context_counts: Counter):
    rows = []

    for context, sample_count in context_counts.most_common():
        title = extract_title(context)
        document_number = extract_document_number(context)
        rows.append(
            {
                "context": context,
                "sample_count": sample_count,
                "title": title,
                "document_number": document_number,
                "context_key": context.casefold(),
                "title_key": title.casefold(),
                "document_number_key": document_number.casefold(),
                "exact_context_found": False,
                "document_number_found": False,
                "document_number_in_header": False,
                "title_found": False,
                "title_in_header": False,
                "matched_rag_ids": [],
                "strong_matched_rag_ids": [],
            }
        )

    return rows


def update_matches(
    rows,
    rag_id: str,
    content: str,
    max_ids_per_context: int,
    header_chars: int,
):
    content_key = content.casefold()
    header_key = content_key[:header_chars]

    for row in rows:
        if row["context_key"] and row["context_key"] in content_key:
            row["exact_context_found"] = True
            append_matched_ids(row, [rag_id], max_ids_per_context)
            append_matched_ids(row, [rag_id], max_ids_per_context, key="strong_matched_rag_ids")

        document_number_key = row["document_number_key"]
        if document_number_key and document_number_key in content_key:
            row["document_number_found"] = True
            append_matched_ids(row, [rag_id], max_ids_per_context)
            if document_number_key in header_key:
                row["document_number_in_header"] = True
                append_matched_ids(row, [rag_id], max_ids_per_context, key="strong_matched_rag_ids")

        title_key = row["title_key"]
        if title_key and title_key in content_key:
            row["title_found"] = True
            append_matched_ids(row, [rag_id], max_ids_per_context)
            if title_key in header_key:
                row["title_in_header"] = True
                if not document_number_key:
                    append_matched_ids(
                        row,
                        [rag_id],
                        max_ids_per_context,
                        key="strong_matched_rag_ids",
                    )


def append_matched_ids(row, ids, max_ids_per_context: int, key: str = "matched_rag_ids"):
    for rag_id in ids:
        if rag_id in row[key]:
            continue
        if len(row[key]) >= max_ids_per_context:
            break
        row[key].append(rag_id)


def finalize_rows(rows):
    finalized = []

    for row in rows:
        found = (
            row["exact_context_found"]
            or row["document_number_found"]
            or row["title_found"]
        )
        strong_found = bool(row["strong_matched_rag_ids"])
        match_types = []
        if row["exact_context_found"]:
            match_types.append("exact_context")
        if row["document_number_found"]:
            match_types.append("document_number")
        if row["document_number_in_header"]:
            match_types.append("document_number_in_header")
        if row["title_found"]:
            match_types.append("title")
        if row["title_in_header"]:
            match_types.append("title_in_header")

        finalized.append(
            {
                "context": row["context"],
                "sample_count": row["sample_count"],
                "title": row["title"],
                "document_number": row["document_number"],
                "found_in_rag": found,
                "strong_found_in_rag": strong_found,
                "match_types": "|".join(match_types),
                "matched_rag_ids": "|".join(row["matched_rag_ids"]),
                "strong_matched_rag_ids": "|".join(row["strong_matched_rag_ids"]),
            }
        )

    return finalized


def check_contexts_in_rag(
    finetune_path: Path,
    rag_path: Path,
    report_path: Path,
    max_ids_per_context: int,
    progress_every: int,
    header_chars: int,
):
    increase_csv_field_limit()

    context_counts = read_context_counts(finetune_path)
    rows = build_context_rows(context_counts)

    if not rows:
        raise ValueError(f"No contexts found in: {finetune_path}")

    total_rag_rows = 0
    with rag_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"id", "content"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing required RAG columns: {sorted(missing_columns)}")

        for rag_row in reader:
            total_rag_rows += 1
            update_matches(
                rows,
                rag_id=str(rag_row.get("id", "")),
                content=str(rag_row.get("content", "")),
                max_ids_per_context=max_ids_per_context,
                header_chars=header_chars,
            )

            if progress_every > 0 and total_rag_rows % progress_every == 0:
                print(f"Scanned RAG rows: {total_rag_rows}", file=sys.stderr)

    report_rows = finalize_rows(rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)

    found_count = sum(row["found_in_rag"] for row in report_rows)
    strong_found_count = sum(row["strong_found_in_rag"] for row in report_rows)
    missing_count = len(report_rows) - found_count
    strong_missing_count = len(report_rows) - strong_found_count

    return {
        "total_finetune_samples": sum(context_counts.values()),
        "unique_contexts": len(context_counts),
        "total_rag_rows": total_rag_rows,
        "found_contexts": found_count,
        "strong_found_contexts": strong_found_count,
        "missing_contexts": missing_count,
        "strong_missing_contexts": strong_missing_count,
        "report_path": report_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check whether finetune JSONL contexts appear in the RAG CSV content."
    )
    parser.add_argument("--finetune", type=Path, default=DEFAULT_FINETUNE_PATH)
    parser.add_argument("--rag", type=Path, default=DEFAULT_RAG_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--max-ids-per-context", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--header-chars",
        type=int,
        default=DEFAULT_HEADER_CHARS,
        help="Number of leading content characters used for strong source-document matches.",
    )
    return parser.parse_args()


def main():
    configure_stdout()
    args = parse_args()

    if not args.finetune.exists():
        raise FileNotFoundError(f"Finetune file not found: {args.finetune}")
    if not args.rag.exists():
        raise FileNotFoundError(f"RAG file not found: {args.rag}")

    summary = check_contexts_in_rag(
        finetune_path=args.finetune,
        rag_path=args.rag,
        report_path=args.report,
        max_ids_per_context=args.max_ids_per_context,
        progress_every=args.progress_every,
        header_chars=args.header_chars,
    )

    print(f"Finetune samples: {summary['total_finetune_samples']}")
    print(f"Unique contexts: {summary['unique_contexts']}")
    print(f"RAG rows scanned: {summary['total_rag_rows']}")
    print(f"Found contexts: {summary['found_contexts']}")
    print(f"Missing contexts: {summary['missing_contexts']}")
    print(f"Strong found contexts: {summary['strong_found_contexts']}")
    print(f"Strong missing contexts: {summary['strong_missing_contexts']}")
    print(f"Report saved to: {summary['report_path']}")


if __name__ == "__main__":
    main()
