"""STS evaluation for RAG prompt outputs.

Compares the reference `response` with two model outputs:
- qwen + RAG with vietnamese-sbert
- qwen + RAG with halong_embedding
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

SBERT_FIELD = "qwen + RAG with vietnamese-sbert"
HALONG_FIELD = "qwen + RAG with halong_embedding"


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
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object at item {index}, got {type(row).__name__}")
    return rows


def to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def compute_metrics(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "avg": 0.0,
            "median": 0.0,
            "std": 0.0,
            "pct_ge_0.85": 0.0,
            "pct_ge_0.70": 0.0,
            "pct_lt_0.60": 0.0,
        }

    return {
        "avg": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "pct_ge_0.85": float((values >= 0.85).mean()),
        "pct_ge_0.70": float((values >= 0.70).mean()),
        "pct_lt_0.60": float((values < 0.60).mean()),
    }


def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError(f"Embedding shapes do not match: {a.shape} vs {b.shape}")
    # Embeddings are normalized, so dot product equals cosine similarity.
    return np.sum(a * b, axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STS eval for RAG outputs vs reference response.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/legal_valid.json"),
        help="Input JSON or JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/legal_valid_sts_eval.json"),
        help="Output JSON with metrics and per-item scores.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional CSV output path for quick inspection.",
    )
    parser.add_argument(
        "--model",
        default="keepitreal/vietnamese-sbert",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=200,
        help="Maximum number of items to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_json_or_jsonl(args.input)
    if args.max_items is not None:
        records = records[: max(0, args.max_items)]

    if not records:
        raise ValueError("No records found in input file.")

    responses = [to_text(r.get("response")) for r in records]
    sbert_answers = [to_text(r.get(SBERT_FIELD)) for r in records]
    halong_answers = [to_text(r.get(HALONG_FIELD)) for r in records]

    model = SentenceTransformer(args.model)
    emb_ref = encode_texts(model, responses, args.batch_size)
    emb_sbert = encode_texts(model, sbert_answers, args.batch_size)
    emb_halong = encode_texts(model, halong_answers, args.batch_size)

    sims_sbert = cosine_sim(emb_ref, emb_sbert)
    sims_halong = cosine_sim(emb_ref, emb_halong)

    items_out = []
    for idx, (rec, sim_sbert, sim_halong) in enumerate(
        zip(records, sims_sbert, sims_halong), start=1
    ):
        items_out.append(
            {
                "index": idx,
                "instruction": rec.get("instruction", ""),
                "response": rec.get("response", ""),
                SBERT_FIELD: rec.get(SBERT_FIELD, ""),
                HALONG_FIELD: rec.get(HALONG_FIELD, ""),
                "sts_vs_response": {
                    SBERT_FIELD: float(sim_sbert),
                    HALONG_FIELD: float(sim_halong),
                },
            }
        )

    result = {
        "input_path": str(args.input),
        "model": args.model,
        "count": len(records),
        "metrics": {
            SBERT_FIELD: compute_metrics(sims_sbert),
            HALONG_FIELD: compute_metrics(sims_halong),
        },
        "items": items_out,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", encoding="utf-8", newline="") as f:
            f.write("index,sts_sbert,sts_halong\n")
            for idx, sim_sbert, sim_halong in zip(range(1, len(records) + 1), sims_sbert, sims_halong):
                f.write(f"{idx},{float(sim_sbert):.6f},{float(sim_halong):.6f}\n")

    print("STS eval done.")
    print(f"Items: {len(records)}")
    print(f"Output: {args.output}")
    if args.csv_out is not None:
        print(f"CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
