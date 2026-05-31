import argparse
import csv
import json
import os
import sys
from pathlib import Path


DEFAULT_FINETUNE_PATH = Path("data/processed/legal_finetune_dataset_2024_2026.jsonl")
DEFAULT_REPORT_PATH = Path("data/processed/finetune_contexts_in_rag_report.csv")


def configure_stdout():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def read_missing_contexts(report_path: Path, use_strong_match: bool) -> set[str]:
    missing_contexts = set()

    with report_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"Report is empty: {report_path}")

        match_column = "strong_found_in_rag" if use_strong_match else "found_in_rag"
        if match_column not in reader.fieldnames:
            raise ValueError(f"Missing report column: {match_column}")

        for row in reader:
            if row.get(match_column, "").strip().casefold() == "false":
                context = row.get("context", "").strip()
                if context:
                    missing_contexts.add(context)

    return missing_contexts


def remove_contexts_from_jsonl(input_path: Path, missing_contexts: set[str]) -> tuple[int, int]:
    temp_path = input_path.with_name(f"{input_path.name}.tmp")
    total_count = 0
    removed_count = 0

    with input_path.open("r", encoding="utf-8") as reader, temp_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as writer:
        for line_number, line in enumerate(reader, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            total_count += 1
            try:
                sample = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc

            if str(sample.get("context", "")).strip() in missing_contexts:
                removed_count += 1
                continue

            writer.write(json.dumps(sample, ensure_ascii=False) + "\n")

    os.replace(temp_path, input_path)
    return total_count, removed_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove finetune JSONL samples whose contexts are missing from the RAG report."
    )
    parser.add_argument("--finetune", type=Path, default=DEFAULT_FINETUNE_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--loose-match",
        action="store_true",
        help="Use found_in_rag instead of strong_found_in_rag when reading the report.",
    )
    return parser.parse_args()


def main():
    configure_stdout()
    args = parse_args()

    if not args.finetune.exists():
        raise FileNotFoundError(f"Finetune file not found: {args.finetune}")
    if not args.report.exists():
        raise FileNotFoundError(f"Report file not found: {args.report}")

    missing_contexts = read_missing_contexts(
        report_path=args.report,
        use_strong_match=not args.loose_match,
    )
    if not missing_contexts:
        print("No missing contexts found in report.")
        return

    total_count, removed_count = remove_contexts_from_jsonl(args.finetune, missing_contexts)
    kept_count = total_count - removed_count

    print(f"Missing contexts removed: {len(missing_contexts)}")
    for context in sorted(missing_contexts):
        print(f"- {context}")
    print(f"Samples before: {total_count}")
    print(f"Samples removed: {removed_count}")
    print(f"Samples kept: {kept_count}")
    print(f"Updated file: {args.finetune}")


if __name__ == "__main__":
    main()
