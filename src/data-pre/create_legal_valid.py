import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


DEFAULT_INPUT = Path("data/processed/legal_finetune_dataset_2024_2026.jsonl")
DEFAULT_OUTPUT = Path("data/processed/legal_valid.json")
DEFAULT_FIELDS = ("instruction", "context", "response")


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def iter_jsonl_lines(path: Path, start_line: int, end_line: int) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line_number < start_line:
                continue
            if line_number > end_line:
                break

            stripped_line = line.strip()
            if not stripped_line:
                continue

            try:
                sample = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc

            if not isinstance(sample, dict):
                raise ValueError(f"Expected JSON object at line {line_number}, got {type(sample).__name__}")

            yield line_number, sample


def keep_validation_fields(sample: Dict[str, Any]) -> Dict[str, str]:
    return {field: str(sample.get(field, "")) for field in DEFAULT_FIELDS}


def write_json(samples: Iterable[Dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [keep_validation_fields(sample) for sample in samples]

    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a JSON validation file from a line range in the legal finetune JSONL dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSONL path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path.")
    parser.add_argument("--start-line", type=int, default=1, help="First line to include, 1-indexed.")
    parser.add_argument("--end-line", type=int, default=100, help="Last line to include, inclusive.")
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()

    if args.start_line < 1:
        raise ValueError("--start-line must be >= 1")
    if args.end_line < args.start_line:
        raise ValueError("--end-line must be >= --start-line")
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    samples = (sample for _, sample in iter_jsonl_lines(args.input, args.start_line, args.end_line))
    count = write_json(samples, args.output)

    print(f"Saved {count} samples to: {args.output}")


if __name__ == "__main__":
    main()
