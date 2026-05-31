"""Run Legal RAG validation on legal_valid.json with two embedding caches."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag.llm import generate_llm_answer
from src.rag.vector_db import DEFAULT_CONFIG_PATH, build_rag_system, hybrid_search


DEFAULT_INPUT = Path("data/processed/legal_valid.json")
BASE_FIELDS = ("instruction", "context", "response")
SBERT_FIELD = "qwen + RAG with vietnamese-sbert"
HALONG_FIELD = "qwen + RAG with halong_embedding"


RAG_RUNS = (
    {
        "field": SBERT_FIELD,
        "cache_dir_key": "cache_dir",
        "encoder_key": "encoder",
    },
    {
        "field": HALONG_FIELD,
        "cache_dir_key": "cache_dir_halong",
        "encoder_key": "encoder_halong",
    },
)


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in: {path}")
        rows = data
    else:
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                rows.append(json.loads(stripped_line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object at item {index}, got {type(row).__name__}")
    return rows


def normalize_records(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for row in rows:
        record = {field: str(row.get(field, "")) for field in BASE_FIELDS}
        record[SBERT_FIELD] = str(row.get(SBERT_FIELD, ""))
        record[HALONG_FIELD] = str(row.get(HALONG_FIELD, ""))
        records.append(record)
    return records


def write_json_atomic(records: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.name}.tmp")

    with temp_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
        file.write("\n")

    os.replace(temp_path, output_path)


def answer_with_rag(
    state: Dict[str, Any],
    query: str,
    args: argparse.Namespace,
) -> str:
    results = hybrid_search(
        state,
        query=query,
        top_k=args.top_k,
        vector_top_k=args.vector_top_k,
        bm25_top_k=args.bm25_top_k,
        rerank_top_n=args.rerank_top_n,
        verbose=args.verbose,
    )
    return generate_llm_answer(
        state,
        query=query,
        results=results,
        model=args.llm_model,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
    )


def run_validation(records: List[Dict[str, str]], args: argparse.Namespace) -> None:
    total = len(records)
    if total == 0:
        write_json_atomic(records, args.output)
        print(f"No samples found. Saved empty JSON to: {args.output}")
        return

    for rag_run in RAG_RUNS:
        field = rag_run["field"]
        print(f"\nLoading RAG cache for: {field}")
        state = build_rag_system(
            config_path=args.config,
            csv_path=args.csv,
            cache_dir=None,
            cache_dir_key=rag_run["cache_dir_key"],
            encoder_key=rag_run["encoder_key"],
            llm_model=args.llm_model,
            index_type=args.index_type,
            rebuild_cache=args.rebuild_cache,
            vector_top_k=args.vector_top_k,
            bm25_top_k=args.bm25_top_k,
            rerank_top_n=args.rerank_top_n,
            final_top_k=args.top_k,
            hnsw_ef_search=args.hnsw_ef_search,
            ivf_nprobe=args.ivf_nprobe,
        )

        try:
            for index, record in enumerate(records, start=1):
                query = record["instruction"].strip()
                if record.get(field, "").strip() and not args.overwrite:
                    print(f"[{field}] {index}/{total} skipped")
                    continue

                print(f"[{field}] {index}/{total}")
                item_started_at = time.perf_counter()
                if not query:
                    record[field] = ""
                    elapsed = time.perf_counter() - item_started_at
                    print(f"[{field}] {index}/{total} done in {elapsed:.1f}s | empty instruction")
                    continue

                last_error = None
                for attempt in range(1, args.max_retries + 2):
                    try:
                        record[field] = answer_with_rag(state, query, args)
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt > args.max_retries:
                            break
                        print(f"Error at item {index}, retry {attempt}/{args.max_retries}: {exc}")
                        time.sleep(args.retry_sleep)

                if last_error is not None:
                    write_json_atomic(records, args.output)
                    raise RuntimeError(
                        f"Failed at item {index}/{total} for field '{field}'. "
                        f"Progress was saved to: {args.output}"
                    ) from last_error

                if args.checkpoint_every > 0 and index % args.checkpoint_every == 0:
                    write_json_atomic(records, args.output)
                    saved_suffix = f" | saved to {args.output}"
                else:
                    saved_suffix = ""

                elapsed = time.perf_counter() - item_started_at
                answer_chars = len(record.get(field, ""))
                print(f"[{field}] {index}/{total} done in {elapsed:.1f}s | answer_chars={answer_chars}{saved_suffix}")

            write_json_atomic(records, args.output)
        finally:
            del state
            gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load legal_valid.json, answer each instruction with Qwen + RAG using two embedding caches, and write JSON."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input legal_valid JSON path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to --input, so legal_valid.json is updated in place.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Vector DB YAML config.")
    parser.add_argument("--csv", default=None, help="Override paths.csv in YAML.")
    parser.add_argument("--index-type", default=None, choices=["hnsw", "ivf", "flat"], help="Override index.type in YAML.")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval.final_top_k in YAML.")
    parser.add_argument("--vector-top-k", type=int, default=None, help="Override retrieval.vector_top_k in YAML.")
    parser.add_argument("--bm25-top-k", type=int, default=None, help="Override retrieval.bm25_top_k in YAML.")
    parser.add_argument("--rerank-top-n", type=int, default=None, help="Override retrieval.rerank_top_n in YAML.")
    parser.add_argument("--llm-model", default=None, help="Override models.llm in YAML.")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature.")
    parser.add_argument("--num-ctx", type=int, default=8192, help="Ollama context window.")
    parser.add_argument("--hnsw-ef-search", type=int, default=None, help="Override index.hnsw_ef_search in YAML.")
    parser.add_argument("--ivf-nprobe", type=int, default=None, help="Override index.ivf_nprobe in YAML.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Write progress every N samples. Use 0 to disable.")
    parser.add_argument("--max-retries", type=int, default=1, help="Retry each failed LLM call this many times before stopping.")
    parser.add_argument("--retry-sleep", type=float, default=5.0, help="Seconds to wait between retries.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute fields that already have answers.")
    parser.add_argument("--rebuild-cache", action="store_true", default=None, help="Rebuild embedding, BM25, and FAISS cache.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed retrieval logs.")
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    if args.output is None:
        args.output = args.input

    rows = load_json_or_jsonl(args.input)
    records = normalize_records(rows)
    run_validation(records, args)
    write_json_atomic(records, args.output)

    print(f"\nSaved {len(records)} validation samples to: {args.output}")
    print(f"Fields: {', '.join([*BASE_FIELDS, SBERT_FIELD, HALONG_FIELD])}")


if __name__ == "__main__":
    main()
