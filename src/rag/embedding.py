"""Chunk loading and embedding cache utilities for Legal RAG."""

from __future__ import annotations

import json
import os
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = str(ROOT_DIR / "configs" / "rag" / "vector_db.yaml")


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def compact_text(text: Any) -> str:
    text = safe_str(text)
    text = text.replace("\u00a0", " ").replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def embeddings_path(config: Dict[str, Any]) -> str:
    return os.path.join(config["cache_dir"], "embedding_cache.npy")


def embedding_meta_path(config: Dict[str, Any]) -> str:
    return os.path.join(config["cache_dir"], "embedding_cache_meta.json")


def file_signature(path: str) -> Dict[str, Any]:
    stat = os.stat(path)
    return {
        "csv_path": os.path.abspath(path),
        "csv_size": stat.st_size,
        "csv_mtime_ns": stat.st_mtime_ns,
    }


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def cache_meta_matches(path: str, expected: Dict[str, Any]) -> bool:
    if not os.path.exists(path):
        return False
    try:
        return read_json(path) == expected
    except Exception:
        return False


def get_raw_context_column(df: pd.DataFrame) -> str:
    if "chunk_text" in df.columns:
        return "chunk_text"
    if "context" in df.columns:
        return "context"
    raise ValueError("CSV phải có cột `chunk_text` từ chunking.py hoặc cột legacy `context`.")


def get_search_context_column(df: pd.DataFrame, raw_context_column: str) -> str:
    if "context_search" in df.columns:
        return "context_search"
    return raw_context_column


def ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df


def fill_missing_chunk_ids(df: pd.DataFrame) -> pd.DataFrame:
    missing = df["chunk_id"].fillna("").astype(str).str.strip().eq("")
    if not missing.any():
        return df

    ids = []
    for idx, row in df.loc[missing].iterrows():
        doc_id = compact_text(row.get("doc_id", "")) or compact_text(row.get("id", ""))
        if doc_id:
            ids.append(f"{doc_id}__{int(idx):04d}")
        else:
            ids.append(f"chunk_{int(idx):08d}")
    df.loc[missing, "chunk_id"] = ids
    return df


def load_chunk_data(
    config: Dict[str, Any],
    chunking_columns: Sequence[str],
    legacy_columns: Sequence[str],
) -> Dict[str, Any]:
    print("Đang đọc chunk CSV...")

    csv_path = config["csv_path"]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    raw_context_column = get_raw_context_column(df)
    search_context_column = get_search_context_column(df, raw_context_column)

    df = ensure_columns(df, list(chunking_columns) + list(legacy_columns))
    df["_rag_context"] = df[raw_context_column].fillna("").astype(str).map(compact_text)
    df["_rag_search"] = df[search_context_column].fillna("").astype(str).map(compact_text)
    df = df[df["_rag_context"].str.len() >= config["min_context_chars"]].reset_index(drop=True)

    if df.empty:
        raise ValueError("Không còn chunk hợp lệ sau khi lọc min_context_chars.")

    df = fill_missing_chunk_ids(df)

    docs = df["_rag_search"].astype(str).tolist()
    raw_docs = df["_rag_context"].astype(str).tolist()

    print(f"Đã nạp {len(df):,} chunks.")
    print(f"Context column: {raw_context_column} | Search column: {search_context_column}")

    return {
        "df": df,
        "docs": docs,
        "raw_docs": raw_docs,
        "raw_context_column": raw_context_column,
        "search_context_column": search_context_column,
        "data_signature": file_signature(csv_path),
    }


def data_meta(state: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(state["data_signature"])
    meta.update(
        {
            "num_docs": len(state["docs"]),
            "raw_context_column": state["raw_context_column"],
            "search_context_column": state["search_context_column"],
        }
    )
    return meta


def embedding_meta(state: Dict[str, Any]) -> Dict[str, Any]:
    meta = data_meta(state)
    meta.update(
        {
            "encoder_name": state["config"]["encoder_name"],
            "dim": state["dim"],
            "normalize_embeddings": True,
        }
    )
    return meta


def load_or_build_embeddings(state: Dict[str, Any]) -> np.ndarray:
    config = state["config"]
    expected_meta = embedding_meta(state)
    path = embeddings_path(config)
    meta_path = embedding_meta_path(config)

    can_load = (
        not config["rebuild_cache"]
        and os.path.exists(path)
        and cache_meta_matches(meta_path, expected_meta)
    )

    if can_load:
        print("Đang tải embedding cache hợp lệ...")
        embeddings = np.load(path)
        if embeddings.shape == (len(state["docs"]), state["dim"]):
            return embeddings.astype("float32", copy=False)
        print("Shape embedding cache không khớp. Build lại embedding...")

    print("Đang encode embeddings mới với normalize_embeddings=True...")
    embeddings = state["encoder"].encode(
        state["docs"],
        batch_size=config["batch_size"],
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    np.save(path, embeddings)
    write_json(meta_path, expected_meta)
    print(f"Đã lưu embeddings: {path}")
    return embeddings


def build_embedding_state(
    config: Dict[str, Any],
    chunking_columns: Sequence[str],
    legacy_columns: Sequence[str],
) -> Dict[str, Any]:
    os.makedirs(config["cache_dir"], exist_ok=True)
    chunk_data = load_chunk_data(config, chunking_columns, legacy_columns)

    print(f"Encoder:  {config['encoder_name']}")
    encoder = SentenceTransformer(config["encoder_name"])

    state = {
        "config": config,
        "encoder": encoder,
        "dim": int(encoder.get_sentence_embedding_dimension()),
        **chunk_data,
    }
    state["embeddings"] = load_or_build_embeddings(state)
    return state


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Build embedding cache từ file chunks.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config cho vector DB/RAG.")
    parser.add_argument("--csv", default=None, help="Override paths.csv trong YAML.")
    parser.add_argument("--cache-dir", default=None, help="Override paths.cache_dir trong YAML.")
    parser.add_argument("--encoder-name", default=None, help="Override models.encoder trong YAML.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override index.batch_size trong YAML.")
    parser.add_argument("--rebuild-cache", action="store_true", default=None, help="Build lại embedding cache.")
    return parser.parse_args()


def main() -> None:
    from src.rag.vector_db import build_runtime_config, get_vector_settings, read_yaml_config

    args = parse_args()
    settings = read_yaml_config(args.config)
    chunking_columns, legacy_columns, *_ = get_vector_settings(settings)
    config = build_runtime_config(
        settings,
        csv_path=args.csv,
        cache_dir=args.cache_dir,
        encoder_name=args.encoder_name,
        batch_size=args.batch_size,
        rebuild_cache=args.rebuild_cache,
    )

    state = build_embedding_state(config, chunking_columns, legacy_columns)
    print("\nEmbedding build xong.")
    print(f"Chunks: {len(state['docs']):,}")
    print(f"Dim: {state['dim']}")
    print(f"Cache: {embeddings_path(config)}")


if __name__ == "__main__":
    main()
