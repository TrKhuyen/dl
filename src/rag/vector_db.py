"""Hybrid vector database and retrieval layer for Legal RAG.

This module owns BM25, FAISS, reranking, and retrieval orchestration.
LLM prompt formatting and answer generation live in llm.py.
"""

from __future__ import annotations

import os
import pickle
import re
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.rag.embedding import (
    build_embedding_state,
    cache_meta_matches,
    compact_text,
    data_meta,
    embedding_meta,
    safe_str,
    write_json,
)

try:
    import yaml
except ImportError:
    yaml = None


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = str(ROOT_DIR / "configs" / "rag" / "vector_db.yaml")


def read_yaml_config(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("Chưa cài PyYAML. Hãy chạy: pip install PyYAML")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config YAML phải là object: {path}")
    return data


def build_runtime_config(
    settings: Dict[str, Any],
    csv_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    encoder_name: Optional[str] = None,
    reranker_name: Optional[str] = None,
    llm_model: Optional[str] = None,
    index_type: Optional[str] = None,
    rebuild_cache: Optional[bool] = None,
    vector_top_k: Optional[int] = None,
    bm25_top_k: Optional[int] = None,
    rerank_top_n: Optional[int] = None,
    final_top_k: Optional[int] = None,
    batch_size: Optional[int] = None,
    hnsw_ef_search: Optional[int] = None,
    ivf_nprobe: Optional[int] = None,
) -> Dict[str, Any]:
    paths = settings.get("paths", {})
    models = settings.get("models", {})
    index = settings.get("index", {})
    retrieval = settings.get("retrieval", {})
    cache = settings.get("cache", {})

    config = {
        "csv_path": csv_path or paths.get("csv", ""),
        "cache_dir": cache_dir or paths.get("cache_dir", ""),
        "encoder_name": encoder_name or models.get("encoder", ""),
        "reranker_name": reranker_name or models.get("reranker", ""),
        "llm_model": llm_model or models.get("llm", ""),
        "index_type": (index_type or index.get("type", "hnsw")).lower(),
        "batch_size": batch_size if batch_size is not None else int(index.get("batch_size", 32)),
        "hnsw_m": int(index.get("hnsw_m", 32)),
        "hnsw_ef_construction": int(index.get("hnsw_ef_construction", 100)),
        "hnsw_ef_search": (
            hnsw_ef_search if hnsw_ef_search is not None else int(index.get("hnsw_ef_search", 128))
        ),
        "ivf_nlist": index.get("ivf_nlist"),
        "ivf_nprobe": ivf_nprobe if ivf_nprobe is not None else int(index.get("ivf_nprobe", 32)),
        "vector_top_k": vector_top_k if vector_top_k is not None else int(retrieval.get("vector_top_k", 50)),
        "bm25_top_k": bm25_top_k if bm25_top_k is not None else int(retrieval.get("bm25_top_k", 50)),
        "rerank_top_n": rerank_top_n if rerank_top_n is not None else int(retrieval.get("rerank_top_n", 50)),
        "final_top_k": final_top_k if final_top_k is not None else int(retrieval.get("final_top_k", 5)),
        "min_context_chars": int(retrieval.get("min_context_chars", 30)),
        "weak_score_threshold": float(retrieval.get("weak_score_threshold", -2.0)),
        "rebuild_cache": rebuild_cache if rebuild_cache is not None else bool(cache.get("rebuild", False)),
    }
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    if config["index_type"] not in {"hnsw", "ivf", "flat"}:
        raise ValueError("index_type phải là 'hnsw', 'ivf' hoặc 'flat'.")
    if config["batch_size"] <= 0:
        raise ValueError("batch_size phải lớn hơn 0.")
    if config["vector_top_k"] <= 0 or config["bm25_top_k"] <= 0:
        raise ValueError("vector_top_k và bm25_top_k phải lớn hơn 0.")
    if config["rerank_top_n"] <= 0 or config["final_top_k"] <= 0:
        raise ValueError("rerank_top_n và final_top_k phải lớn hơn 0.")
    if config["min_context_chars"] < 0:
        raise ValueError("min_context_chars không được âm.")


def get_vector_settings(settings: Dict[str, Any]) -> Tuple[Sequence[str], Sequence[str], Dict[str, str], str, str]:
    schema = settings.get("schema", {})
    chunking_columns = list(schema.get("chunking_columns", []))
    legacy_columns = list(schema.get("legacy_columns", []))
    legal_synonyms = dict(settings.get("legal_synonyms", {}))
    llm_system_prompt = settings.get("llm", {}).get("system_prompt", "")
    bm25_tokenizer_version = settings.get("cache", {}).get("bm25_tokenizer_version", "")
    return chunking_columns, legacy_columns, legal_synonyms, llm_system_prompt, bm25_tokenizer_version


def tokenize_vi(text: Any) -> List[str]:
    text = safe_str(text).lower()
    text = re.sub(r"[^\w\s/\.\-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return text.split()


def bm25_path(config: Dict[str, Any]) -> str:
    return os.path.join(config["cache_dir"], "bm25.pkl")


def bm25_meta_path(config: Dict[str, Any]) -> str:
    return bm25_path(config) + ".meta.json"


def faiss_index_path(config: Dict[str, Any]) -> str:
    return os.path.join(config["cache_dir"], f"faiss_{config['index_type']}.index")


def faiss_meta_path(config: Dict[str, Any]) -> str:
    return faiss_index_path(config) + ".meta.json"


def bm25_meta(state: Dict[str, Any]) -> Dict[str, Any]:
    meta = data_meta(state)
    meta.update(
        {
            "bm25_tokenizer_version": state["bm25_tokenizer_version"],
            "keep_vietnamese_accents": True,
        }
    )
    return meta


def resolve_ivf_nlist(config: Dict[str, Any], n_docs: int) -> int:
    if config["ivf_nlist"] is not None:
        return max(1, min(int(config["ivf_nlist"]), max(1, n_docs)))

    if n_docs < 50_000:
        return min(256, max(1, int(np.sqrt(n_docs))))
    if n_docs < 300_000:
        return 1024
    if n_docs < 700_000:
        return 2048
    return 4096


def faiss_meta(state: Dict[str, Any]) -> Dict[str, Any]:
    config = state["config"]
    meta = embedding_meta(state)
    meta.update(
        {
            "index_type": config["index_type"],
            "metric": "inner_product_cosine_normalized",
            "hnsw_m": config["hnsw_m"],
            "hnsw_ef_construction": config["hnsw_ef_construction"],
            "ivf_nlist": resolve_ivf_nlist(config, len(state["docs"])),
        }
    )
    return meta


def load_or_build_bm25(state: Dict[str, Any]) -> BM25Okapi:
    config = state["config"]
    expected_meta = bm25_meta(state)
    path = bm25_path(config)
    meta_path = bm25_meta_path(config)

    can_load = (
        not config["rebuild_cache"]
        and os.path.exists(path)
        and cache_meta_matches(meta_path, expected_meta)
    )

    if can_load:
        print("Đang tải BM25 cache hợp lệ...")
        with open(path, "rb") as f:
            return pickle.load(f)

    print("Đang xây dựng BM25 index...")
    tokenized_corpus = [tokenize_vi(doc) for doc in state["docs"]]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(path, "wb") as f:
        pickle.dump(bm25, f)
    write_json(meta_path, expected_meta)

    print(f"Đã lưu BM25 cache: {path}")
    return bm25


def apply_runtime_index_params(index: faiss.Index, config: Dict[str, Any], n_docs: int) -> None:
    if config["index_type"] == "hnsw":
        index.hnsw.efSearch = config["hnsw_ef_search"]
    elif config["index_type"] == "ivf":
        index.nprobe = min(config["ivf_nprobe"], resolve_ivf_nlist(config, n_docs))


def build_faiss_index(state: Dict[str, Any]) -> faiss.Index:
    config = state["config"]
    dim = state["dim"]
    embeddings = state["embeddings"]

    if config["index_type"] == "hnsw":
        index = faiss.IndexHNSWFlat(dim, config["hnsw_m"], faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = config["hnsw_ef_construction"]
        index.add(embeddings)
    elif config["index_type"] == "ivf":
        nlist = resolve_ivf_nlist(config, len(state["docs"]))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

    apply_runtime_index_params(index, config, len(state["docs"]))
    return index


def load_or_build_faiss_index(state: Dict[str, Any]) -> faiss.Index:
    config = state["config"]
    expected_meta = faiss_meta(state)
    path = faiss_index_path(config)
    meta_path = faiss_meta_path(config)

    can_load = (
        not config["rebuild_cache"]
        and os.path.exists(path)
        and cache_meta_matches(meta_path, expected_meta)
    )

    if can_load:
        print(f"Đang tải FAISS {config['index_type'].upper()} index hợp lệ...")
        index = faiss.read_index(path)
        apply_runtime_index_params(index, config, len(state["docs"]))
        return index

    print(f"Đang xây dựng FAISS {config['index_type'].upper()} index...")
    index = build_faiss_index(state)
    faiss.write_index(index, path)
    write_json(meta_path, expected_meta)

    print(f"Đã lưu FAISS index: {path}")
    return index


def build_rag_system(
    config_path: str = DEFAULT_CONFIG_PATH,
    csv_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    encoder_name: Optional[str] = None,
    reranker_name: Optional[str] = None,
    llm_model: Optional[str] = None,
    index_type: Optional[str] = None,
    rebuild_cache: Optional[bool] = None,
    vector_top_k: Optional[int] = None,
    bm25_top_k: Optional[int] = None,
    rerank_top_n: Optional[int] = None,
    final_top_k: Optional[int] = None,
    batch_size: Optional[int] = None,
    hnsw_ef_search: Optional[int] = None,
    ivf_nprobe: Optional[int] = None,
    load_reranker: bool = True,
) -> Dict[str, Any]:
    print("Đang nạp hệ thống Hybrid Legal RAG...")
    settings = read_yaml_config(config_path)
    chunking_columns, legacy_columns, legal_synonyms, llm_system_prompt, bm25_tokenizer_version = (
        get_vector_settings(settings)
    )
    config = build_runtime_config(
        settings,
        csv_path=csv_path,
        cache_dir=cache_dir,
        encoder_name=encoder_name,
        reranker_name=reranker_name,
        llm_model=llm_model,
        index_type=index_type,
        rebuild_cache=rebuild_cache,
        vector_top_k=vector_top_k,
        bm25_top_k=bm25_top_k,
        rerank_top_n=rerank_top_n,
        final_top_k=final_top_k,
        batch_size=batch_size,
        hnsw_ef_search=hnsw_ef_search,
        ivf_nprobe=ivf_nprobe,
    )

    state = build_embedding_state(config, chunking_columns, legacy_columns)

    state.update(
        {
            "legal_synonyms": legal_synonyms,
            "llm_system_prompt": llm_system_prompt,
            "bm25_tokenizer_version": bm25_tokenizer_version,
        }
    )

    if load_reranker:
        print(f"Reranker: {config['reranker_name']}")
        state["reranker"] = CrossEncoder(config["reranker_name"])
    else:
        print("Bỏ qua reranker, chỉ build embedding + BM25 + FAISS.")

    state["bm25"] = load_or_build_bm25(state)
    state["index"] = load_or_build_faiss_index(state)
    return state


def normalize_for_exact_dedupe(text: Any) -> str:
    return compact_text(text).lower()


def expand_legal_query(query: str, legal_synonyms: Dict[str, str]) -> str:
    query_clean = compact_text(query)
    query_l = query_clean.lower()
    expanded_parts = [query_clean]

    for short_form, formal_form in legal_synonyms.items():
        if short_form.lower() in query_l and formal_form.lower() not in query_l:
            expanded_parts.append(formal_form)

    seen = set()
    out = []
    for part in expanded_parts:
        key = part.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(part)
    return " ".join(out)


def vector_search(state: Dict[str, Any], query: str, k: int) -> Tuple[List[int], Dict[int, float]]:
    q_emb = state["encoder"].encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = state["index"].search(q_emb, k=min(k, len(state["docs"])))

    out_indices: List[int] = []
    score_map: Dict[int, float] = {}
    for score, idx in zip(scores[0], indices[0]):
        idx = int(idx)
        if idx < 0:
            continue
        out_indices.append(idx)
        score_map[idx] = float(score)
    return out_indices, score_map


def bm25_search(state: Dict[str, Any], query: str, k: int) -> Tuple[List[int], Dict[int, float]]:
    tokens = tokenize_vi(query)
    if not tokens:
        return [], {}

    scores = state["bm25"].get_scores(tokens)
    k = min(k, len(scores))
    if k <= 0:
        return [], {}

    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    indices = [int(i) for i in top_idx]
    score_map = {int(i): float(scores[i]) for i in top_idx}
    return indices, score_map


def merge_interleave(a: Sequence[int], b: Sequence[int]) -> List[int]:
    merged: List[int] = []
    max_len = max(len(a), len(b))

    for i in range(max_len):
        if i < len(a):
            merged.append(a[i])
        if i < len(b):
            merged.append(b[i])

    seen = set()
    out: List[int] = []
    for idx in merged:
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def candidate_dedupe_key(state: Dict[str, Any], doc_idx: int) -> Tuple[str, str, str, str]:
    row = state["df"].iloc[doc_idx]
    document_key = (
        row.get("document_number", "")
        or row.get("law_id", "")
        or row.get("doc_id", "")
        or row.get("id", "")
    )
    article_key = row.get("article_number", "") or row.get("article_id", "")
    chunk_type = row.get("chunk_type", "")
    return (
        normalize_for_exact_dedupe(document_key),
        normalize_for_exact_dedupe(article_key),
        normalize_for_exact_dedupe(chunk_type),
        normalize_for_exact_dedupe(row.get("_rag_context", ""))[:800],
    )


def dedupe_candidate_indices_by_content(state: Dict[str, Any], indices: Sequence[int]) -> List[int]:
    seen = set()
    out: List[int] = []

    for idx in indices:
        key = candidate_dedupe_key(state, idx)
        if key in seen:
            continue
        seen.add(key)
        out.append(idx)
    return out


def row_to_result(
    state: Dict[str, Any],
    doc_idx: int,
    rerank_score: Optional[float] = None,
    vector_score: Optional[float] = None,
    bm25_score: Optional[float] = None,
    rank: Optional[int] = None,
) -> Dict[str, Any]:
    row = state["df"].iloc[doc_idx]

    document_number = safe_str(row.get("document_number", "")) or safe_str(row.get("law_id", ""))
    article_number = safe_str(row.get("article_number", "")) or safe_str(row.get("article_id", ""))

    result = {
        "rank": rank,
        "doc_index": int(doc_idx),
        "chunk_id": safe_str(row.get("chunk_id", "")),
        "doc_id": safe_str(row.get("doc_id", "")) or safe_str(row.get("id", "")),
        "document_number": document_number,
        "title": safe_str(row.get("title", "")),
        "url": safe_str(row.get("url", "")),
        "legal_type": safe_str(row.get("legal_type", "")),
        "legal_sectors": safe_str(row.get("legal_sectors", "")),
        "issuing_authority": safe_str(row.get("issuing_authority", "")),
        "issuance_date": safe_str(row.get("issuance_date", "")),
        "signers": safe_str(row.get("signers", "")),
        "chunk_index": safe_str(row.get("chunk_index", "")),
        "chunk_type": safe_str(row.get("chunk_type", "")),
        "article_number": article_number,
        "article_title": safe_str(row.get("article_title", "")),
        "chunk_char_len": safe_str(row.get("chunk_char_len", "")) or safe_str(row.get("char_len", "")),
        "context": safe_str(row.get("_rag_context", "")),
        "law_id": document_number,
        "article_id": article_number,
    }

    if rerank_score is not None:
        result["rerank_score"] = float(rerank_score)
    if vector_score is not None:
        result["vector_score"] = float(vector_score)
    if bm25_score is not None:
        result["bm25_score"] = float(bm25_score)
    return result


def hybrid_search(
    state: Dict[str, Any],
    query: str,
    top_k: Optional[int] = None,
    vector_top_k: Optional[int] = None,
    bm25_top_k: Optional[int] = None,
    rerank_top_n: Optional[int] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    config = state["config"]
    final_top_k = top_k or config["final_top_k"]
    vector_k = vector_top_k or config["vector_top_k"]
    bm25_k = bm25_top_k or config["bm25_top_k"]
    rerank_n = rerank_top_n or config["rerank_top_n"]

    expanded_query = expand_legal_query(query, state["legal_synonyms"])
    if verbose:
        print(f"\n[TRUY VẤN]: {query}")
        if expanded_query != query:
            print(f"Query expansion: {expanded_query}")

    t0 = time.perf_counter()
    vector_indices, vector_score_map = vector_search(state, expanded_query, vector_k)
    t_vec = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    bm25_indices, bm25_score_map = bm25_search(state, expanded_query, bm25_k)
    t_bm25 = (time.perf_counter() - t1) * 1000

    candidate_indices = merge_interleave(vector_indices, bm25_indices)
    candidate_indices = dedupe_candidate_indices_by_content(state, candidate_indices)
    candidate_indices = candidate_indices[: max(rerank_n, final_top_k)]

    if verbose:
        print(f"Latency: Vector {t_vec:.1f}ms | BM25 {t_bm25:.1f}ms")
        print(f"Candidates sau khi merge + dedupe: {len(candidate_indices)}")

    if not candidate_indices:
        return []

    if "reranker" not in state:
        raise ValueError("State chưa load reranker. Hãy build_rag_system(..., load_reranker=True) để search.")

    candidate_docs = [state["raw_docs"][i] for i in candidate_indices]
    pairs = [[query, doc] for doc in candidate_docs]

    t2 = time.perf_counter()
    rerank_scores = np.asarray(state["reranker"].predict(pairs, batch_size=16), dtype="float32")
    t_rerank = (time.perf_counter() - t2) * 1000

    ranked_local_indices = np.argsort(rerank_scores)[::-1][:final_top_k]
    results: List[Dict[str, Any]] = []
    for rank, local_i in enumerate(ranked_local_indices, start=1):
        doc_idx = candidate_indices[int(local_i)]
        results.append(
            row_to_result(
                state=state,
                doc_idx=doc_idx,
                rerank_score=float(rerank_scores[int(local_i)]),
                vector_score=vector_score_map.get(doc_idx),
                bm25_score=bm25_score_map.get(doc_idx),
                rank=rank,
            )
        )

    if verbose:
        print(f"Rerank latency: {t_rerank:.1f}ms")
        for item in results:
            print(
                f"  [{item['rank']}] "
                f"rerank={item.get('rerank_score', 0):.4f} "
                f"chunk={item.get('chunk_id', '')} "
                f"doc={item.get('document_number', '')} "
                f"article={item.get('article_number', '')}"
            )

        top_score = float(results[0].get("rerank_score", 0.0)) if results else 0.0
        if results and top_score < config["weak_score_threshold"]:
            print("Cảnh báo: reranker score top-1 thấp, retrieval có thể chưa đủ căn cứ.")

    return results


def result_to_context_text(result: Any) -> str:
    if isinstance(result, str):
        return result

    lines = []
    if result.get("document_number"):
        lines.append(f"Số hiệu: {result['document_number']}")
    if result.get("title"):
        lines.append(f"Tên văn bản: {result['title']}")
    if result.get("article_number"):
        article = f"Điều {result['article_number']}"
        if result.get("article_title"):
            article = f"{article}. {result['article_title']}"
        lines.append(article)
    if result.get("url"):
        lines.append(f"URL: {result['url']}")

    lines.append("Nội dung:")
    lines.append(safe_str(result.get("context", "")))
    return "\n".join(line for line in lines if line)


def results_to_context_texts(results: Sequence[Any]) -> List[str]:
    return [result_to_context_text(result) for result in results]


def search_contexts(
    state: Dict[str, Any],
    query: str,
    top_k: int = 3,
    verbose: bool = False,
) -> List[str]:
    results = hybrid_search(state, query=query, top_k=top_k, verbose=verbose)
    return results_to_context_texts(results)


def HybridRAGSystem(
    config_path: str = DEFAULT_CONFIG_PATH,
    csv_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    cache_file: Optional[str] = None,
    index_type: Optional[str] = None,
    rebuild_cache: Optional[bool] = None,
    vector_top_k: Optional[int] = None,
    bm25_top_k: Optional[int] = None,
    rerank_top_n: Optional[int] = None,
    final_top_k: Optional[int] = None,
    llm_model: Optional[str] = None,
    hnsw_ef_search: Optional[int] = None,
    ivf_nprobe: Optional[int] = None,
) -> SimpleNamespace:
    """Compatibility factory for older code that imported HybridRAGSystem."""
    if cache_file and cache_dir is None:
        cache_dir_from_file = os.path.dirname(cache_file)
        if cache_dir_from_file:
            cache_dir = cache_dir_from_file

    state = build_rag_system(
        config_path=config_path,
        csv_path=csv_path,
        cache_dir=cache_dir,
        llm_model=llm_model,
        index_type=index_type,
        rebuild_cache=rebuild_cache,
        vector_top_k=vector_top_k,
        bm25_top_k=bm25_top_k,
        rerank_top_n=rerank_top_n,
        final_top_k=final_top_k,
        hnsw_ef_search=hnsw_ef_search,
        ivf_nprobe=ivf_nprobe,
    )

    def bound_hybrid_search(
        query: str,
        top_k: Optional[int] = None,
        vector_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
        verbose: bool = True,
        return_texts: bool = True,
    ) -> List[Any]:
        results = hybrid_search(
            state,
            query=query,
            top_k=top_k,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            rerank_top_n=rerank_top_n,
            verbose=verbose,
        )
        if return_texts:
            return results_to_context_texts(results)
        return results

    return SimpleNamespace(
        state=state,
        config=state["config"],
        df=state["df"],
        hybrid_search=bound_hybrid_search,
        search_results=lambda query, top_k=None, verbose=True: bound_hybrid_search(
            query=query,
            top_k=top_k,
            verbose=verbose,
            return_texts=False,
        ),
        search_contexts=lambda query, top_k=3, verbose=False: search_contexts(
            state, query=query, top_k=top_k, verbose=verbose
        ),
    )


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Build vector DB hoặc test retrieval cho Legal RAG.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config cho vector DB.")
    parser.add_argument("--csv", default=None, help="Override paths.csv trong YAML.")
    parser.add_argument("--cache-dir", default=None, help="Override paths.cache_dir trong YAML.")
    parser.add_argument("--index-type", default=None, choices=["hnsw", "ivf", "flat"], help="Override index.type trong YAML.")
    parser.add_argument("--encoder-name", default=None, help="Override models.encoder trong YAML.")
    parser.add_argument("--reranker-name", default=None, help="Override models.reranker trong YAML.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override index.batch_size trong YAML.")
    parser.add_argument("--vector-top-k", type=int, default=None, help="Override retrieval.vector_top_k trong YAML.")
    parser.add_argument("--bm25-top-k", type=int, default=None, help="Override retrieval.bm25_top_k trong YAML.")
    parser.add_argument("--rerank-top-n", type=int, default=None, help="Override retrieval.rerank_top_n trong YAML.")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval.final_top_k trong YAML.")
    parser.add_argument("--hnsw-ef-search", type=int, default=None, help="Override index.hnsw_ef_search trong YAML.")
    parser.add_argument("--ivf-nprobe", type=int, default=None, help="Override index.ivf_nprobe trong YAML.")
    parser.add_argument("--query", default=None, help="Query để test retrieval sau khi build/load vector DB.")
    parser.add_argument("--no-reranker", action="store_true", help="Chỉ build embedding, BM25, FAISS; không load reranker.")
    parser.add_argument("--rebuild-cache", action="store_true", default=None, help="Build lại embedding, BM25 và FAISS index.")
    return parser.parse_args()


def print_result_preview(results: Sequence[Dict[str, Any]]) -> None:
    for item in results:
        print(
            f"[{item.get('rank')}] "
            f"doc={item.get('document_number', '')} "
            f"article={item.get('article_number', '')} "
            f"rerank={item.get('rerank_score', 0):.4f}"
        )
        print(compact_text(item.get("context", ""))[:500])
        print("-" * 80)


def main() -> None:
    args = parse_args()
    state = build_rag_system(
        config_path=args.config,
        csv_path=args.csv,
        cache_dir=args.cache_dir,
        encoder_name=args.encoder_name,
        reranker_name=args.reranker_name,
        index_type=args.index_type,
        rebuild_cache=args.rebuild_cache,
        vector_top_k=args.vector_top_k,
        bm25_top_k=args.bm25_top_k,
        rerank_top_n=args.rerank_top_n,
        final_top_k=args.top_k,
        batch_size=args.batch_size,
        hnsw_ef_search=args.hnsw_ef_search,
        ivf_nprobe=args.ivf_nprobe,
        load_reranker=not args.no_reranker,
    )

    print("\nVector DB build/load xong.")
    print(f"Chunks: {len(state['docs']):,}")
    print(f"Index type: {state['config']['index_type']}")
    print(f"Cache dir: {state['config']['cache_dir']}")

    if not args.query:
        return
    if args.no_reranker:
        raise ValueError("Muốn search thì bỏ `--no-reranker` để load reranker.")

    results = hybrid_search(
        state,
        query=args.query,
        top_k=args.top_k,
        vector_top_k=args.vector_top_k,
        bm25_top_k=args.bm25_top_k,
        rerank_top_n=args.rerank_top_n,
        verbose=True,
    )
    print_result_preview(results)


if __name__ == "__main__":
    main()
