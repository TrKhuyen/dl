"""Manual test runner for Legal RAG.

Examples:
python src/rag/test_rag.py --no-llm
python src/rag/test_rag.py --query "vượt đèn đỏ phạt bao nhiêu tiền"
"""

from __future__ import annotations

import argparse

from src.rag.llm import build_user_message, generate_llm_answer
from src.rag.vector_db import DEFAULT_CONFIG_PATH, build_rag_system, hybrid_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Legal RAG retrieval and optional Ollama LLM.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config cho vector DB.")
    parser.add_argument("--csv", default=None, help="Override paths.csv trong YAML.")
    parser.add_argument("--cache-dir", default=None, help="Override paths.cache_dir trong YAML.")
    parser.add_argument("--index-type", default=None, choices=["hnsw", "ivf", "flat"], help="Override index.type trong YAML.")
    parser.add_argument("--query", default=None, help="Câu hỏi cần test. Nếu bỏ trống, chạy vài query mẫu.")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval.final_top_k trong YAML.")
    parser.add_argument("--vector-top-k", type=int, default=None, help="Override retrieval.vector_top_k trong YAML.")
    parser.add_argument("--bm25-top-k", type=int, default=None, help="Override retrieval.bm25_top_k trong YAML.")
    parser.add_argument("--rerank-top-n", type=int, default=None, help="Override retrieval.rerank_top_n trong YAML.")
    parser.add_argument("--llm-model", default=None, help="Override models.llm trong YAML.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature cho LLM.")
    parser.add_argument("--num-ctx", type=int, default=8192, help="Context window cho Ollama.")
    parser.add_argument("--hnsw-ef-search", type=int, default=None, help="Override index.hnsw_ef_search trong YAML.")
    parser.add_argument("--ivf-nprobe", type=int, default=None, help="Override index.ivf_nprobe trong YAML.")
    parser.add_argument("--no-llm", action="store_true", help="Chỉ retrieve và in prompt, không gọi Ollama.")
    parser.add_argument("--rebuild-cache", action="store_true", default=None, help="Build lại embedding, BM25 và FAISS index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = build_rag_system(
        config_path=args.config,
        csv_path=args.csv,
        cache_dir=args.cache_dir,
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

    queries = [args.query] if args.query else [
        "vượt đèn đỏ phạt bao nhiêu tiền",
        "lái xe khi say rượu bị xử lý thế nào",
        "xây dựng trái phép trên đất nông nghiệp bị xử phạt ra sao",
    ]

    for query in queries:
        print("\n" + "=" * 100)
        print(f"QUESTION: {query}")
        print("=" * 100)

        results = hybrid_search(
            state,
            query=query,
            top_k=args.top_k,
            vector_top_k=args.vector_top_k,
            bm25_top_k=args.bm25_top_k,
            rerank_top_n=args.rerank_top_n,
            verbose=True,
        )

        if args.no_llm:
            print("\nSYSTEM MESSAGE:")
            print(state["llm_system_prompt"])
            print("\nUSER MESSAGE:")
            print(build_user_message(query, results))
            continue

        print("\nĐang gọi Ollama LLM để tổng hợp câu trả lời...")
        try:
            answer = generate_llm_answer(
                state,
                query=query,
                results=results,
                model=args.llm_model,
                temperature=args.temperature,
                num_ctx=args.num_ctx,
            )
        except Exception as exc:
            print("\nLỗi khi gọi Ollama LLM:")
            print(exc)
            print("\nUSER MESSAGE để debug:")
            print(build_user_message(query, results))
            continue

        print("\n" + "-" * 100)
        print("ANSWER:")
        print("-" * 100)
        print(answer)


if __name__ == "__main__":
    main()
