"""LLM prompt formatting and Ollama answer generation for Legal RAG."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def format_context_for_llm(results: Sequence[Any]) -> str:
    if not results:
        return "Không tìm thấy context phù hợp từ hệ thống truy xuất."

    blocks: List[str] = []
    for rank, item in enumerate(results, start=1):
        if isinstance(item, str):
            blocks.append("\n".join([f"[Context {rank}]", "content:", item]))
            continue

        lines = [
            f"[Context {item.get('rank', '?')}]",
            f"chunk_id: {item.get('chunk_id', '')}",
            f"document_number: {item.get('document_number', '')}",
            f"article_number: {item.get('article_number', '')}",
            f"article_title: {item.get('article_title', '')}",
            f"title: {item.get('title', '')}",
            f"url: {item.get('url', '')}",
            f"chunk_type: {item.get('chunk_type', '')}",
            "content:",
            item.get("context", ""),
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_user_message(query: str, results: Sequence[Dict[str, Any]]) -> str:
    context = format_context_for_llm(results)
    return f"""CONTEXT:
{context}

CÂU HỎI:
{query}

Hãy trả lời theo đúng quy tắc và format trong system message.
"""


def extract_ollama_content(response: Any) -> str:
    if isinstance(response, dict):
        message = response.get("message", {})
        if isinstance(message, dict):
            return str(message.get("content", ""))
        return str(getattr(message, "content", ""))

    message = getattr(response, "message", None)
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def generate_llm_answer(
    state: Dict[str, Any],
    query: str,
    results: Sequence[Dict[str, Any]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    num_ctx: int = 8192,
) -> str:
    model_name = model or state["config"]["llm_model"]
    try:
        import ollama
    except ImportError as exc:
        raise ImportError(
            "Chưa cài thư viện `ollama`. Hãy chạy:\n"
            "pip install ollama\n\n"
            f"Sau đó pull model:\nollama pull {model_name}"
        ) from exc

    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": state["llm_system_prompt"]},
            {"role": "user", "content": build_user_message(query, results)},
        ],
        options={
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
        stream=False,
    )
    return extract_ollama_content(response).strip()
