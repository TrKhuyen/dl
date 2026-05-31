import argparse
import json
import re
import sys
from pathlib import Path


DEFAULT_INPUT = Path("data/processed/legal_finetune_dataset.jsonl")
DEFAULT_OUTPUT = Path("data/processed/legal_finetune_dataset_2024_2026.jsonl")


def build_year_pattern(start_year: int, end_year: int) -> re.Pattern:
    years = "|".join(str(year) for year in range(start_year, end_year + 1))
    return re.compile(rf"(?<!\d)({years})(?!\d)")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc


def filter_samples_by_context_year(input_path: Path, start_year: int, end_year: int):
    year_pattern = build_year_pattern(start_year, end_year)

    for line_number, sample in iter_jsonl(input_path):
        context = sample.get("context", "")
        if not isinstance(context, str):
            context = str(context)

        if year_pattern.search(context):
            yield line_number, sample


def write_jsonl(samples, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for _, sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1

    return count


def print_jsonl(samples) -> int:
    count = 0
    for _, sample in samples:
        print(json.dumps(sample, ensure_ascii=False))
        count += 1
    return count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter legal finetune JSONL samples whose context contains years in a range."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSONL path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument("--start-year", type=int, default=2024, help="Start year, inclusive.")
    parser.add_argument("--end-year", type=int, default=2026, help="End year, inclusive.")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print matched samples to stdout instead of writing to --output.",
    )
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()

    if args.start_year > args.end_year:
        raise ValueError("--start-year must be less than or equal to --end-year")

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    samples = filter_samples_by_context_year(args.input, args.start_year, args.end_year)

    if args.stdout:
        count = print_jsonl(samples)
        print(f"\nMatched samples: {count}", file=sys.stderr)
    else:
        count = write_jsonl(samples, args.output)
        print(f"Matched samples: {count}")
        print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
