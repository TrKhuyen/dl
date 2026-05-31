"""LLM prompt formatting and Ollama answer generation for Legal RAG."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence


DIRECT_SYSTEM_PROMPT = """Bạn là Trợ lý Pháp Luật AI chuyên về pháp luật Việt Nam.

Nhiệm vụ:
- Trả lời trực tiếp câu hỏi của người dùng bằng tiếng Việt, rõ ràng và ngắn gọn.
- Có thể dùng kiến thức chung của bạn, nhưng không được bịa căn cứ pháp lý khi không chắc chắn.
- Nếu câu hỏi cần tư vấn pháp lý cụ thể hoặc dữ liệu mới, hãy nêu rõ giới hạn và khuyến nghị kiểm tra văn bản hiện hành hoặc hỏi luật sư.

Format trả lời:
- Kết luận ngắn:
- Giải thích:
- Lưu ý nếu có:
"""


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


def build_user_message(query: str, results: Sequence[Any]) -> str:
    context = format_context_for_llm(results)
    return f"""CONTEXT:
{context}

CÂU HỎI:
{query}

Hãy trả lời theo đúng quy tắc và format trong system message.
"""


def build_direct_user_message(query: str) -> str:
    return f"""CÂU HỎI:
{query}

Hãy trả lời trực tiếp theo đúng quy tắc và format trong system message.
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


def get_system_prompt(
    state: Dict[str, Any],
    use_rag: bool = True,
    system_prompt: Optional[str] = None,
) -> str:
    if system_prompt:
        return system_prompt
    if use_rag:
        return state.get("llm_system_prompt") or DIRECT_SYSTEM_PROMPT
    return state.get("direct_system_prompt") or DIRECT_SYSTEM_PROMPT


def build_ollama_messages(
    state: Dict[str, Any],
    query: str,
    results: Optional[Sequence[Any]] = None,
    use_rag: bool = True,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    user_message = build_user_message(query, results or []) if use_rag else build_direct_user_message(query)
    return [
        {"role": "system", "content": get_system_prompt(state, use_rag=use_rag, system_prompt=system_prompt)},
        {"role": "user", "content": user_message},
    ]


def stream_llm_answer(
    state: Dict[str, Any],
    query: str,
    results: Optional[Sequence[Any]] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    num_ctx: int = 8192,
    use_rag: bool = True,
    system_prompt: Optional[str] = None,
) -> Iterator[str]:
    model_name = model or state["config"]["llm_model"]
    try:
        import ollama
    except ImportError as exc:
        raise ImportError(
            "Chưa cài thư viện `ollama`. Hãy chạy:\n"
            "pip install ollama\n\n"
            f"Sau đó pull model:\nollama pull {model_name}"
        ) from exc

    stream = ollama.chat(
        model=model_name,
        messages=build_ollama_messages(
            state,
            query=query,
            results=results,
            use_rag=use_rag,
            system_prompt=system_prompt,
        ),
        options={
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
        stream=True,
    )
    for chunk in stream:
        content = extract_ollama_content(chunk)
        if content:
            yield content


def generate_llm_answer(
    state: Dict[str, Any],
    query: str,
    results: Optional[Sequence[Any]] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    num_ctx: int = 8192,
    use_rag: bool = True,
    system_prompt: Optional[str] = None,
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
        messages=build_ollama_messages(
            state,
            query=query,
            results=results,
            use_rag=use_rag,
            system_prompt=system_prompt,
        ),
        options={
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
        stream=False,
    )
    return extract_ollama_content(response).strip()
