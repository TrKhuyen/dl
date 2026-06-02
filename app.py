"""
Trợ lý Pháp luật AI - giao diện Streamlit so sánh 3 chế độ trả lời.

Chạy:
    conda activate dl
    streamlit run app.py
"""

from __future__ import annotations

import gc
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

RAG_CONFIG = ROOT / "configs" / "rag" / "vector_db.yaml"
DEFAULT_DATA_CSV = ROOT / "data" / "processed" / "legal_chunks_2024_2026.csv"
DATA_CSV = DEFAULT_DATA_CSV
DEFAULT_ENCODER = "keepitreal/vietnamese-sbert"
DEFAULT_HALONG_ENCODER = "hiieu/halong_embedding"
RAG_TOP_K = 3
RAG_LOAD_ERRORS_KEY = "rag_load_errors"
WHISPER_MODEL_SIZE = "small"
REQUIRED_CACHE_FILES = (
    "embedding_cache.npy",
    "embedding_cache_meta.json",
    "bm25.pkl",
    "bm25.pkl.meta.json",
)


st.set_page_config(
    page_title="Trợ lý Pháp luật AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .gradient-title {
        background: linear-gradient(135deg, #6C63FF 0%, #48CAE4 50%, #06D6A0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        color: #8B8FA8;
        font-size: 0.92rem;
        margin-top: 4px;
        margin-bottom: 1.25rem;
    }

    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 0.65rem;
        padding: 0.5rem 1rem;
    }

    [data-testid="stChatMessage"][data-testid*="assistant"] {
        border-left: 3px solid #6C63FF;
    }

    .source-badge {
        display: inline-block;
        background: rgba(108, 99, 255, 0.15);
        color: #9D97FF;
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 8px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin: 2px 3px;
        font-weight: 500;
    }

    .model-card {
        border: 1px solid rgba(108, 99, 255, 0.22);
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.65rem;
        background: rgba(108, 99, 255, 0.06);
    }

    .model-name {
        font-weight: 700;
        color: #E8E7FF;
        font-size: 0.92rem;
    }

    .model-meta {
        color: #A7ABC4;
        font-size: 0.78rem;
        margin-top: 0.2rem;
    }

    .status-ok {
        color: #06D6A0;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .status-warn {
        color: #FFD60A;
        font-size: 0.78rem;
        font-weight: 600;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1C2E 0%, #0E1117 100%);
        border-right: 1px solid rgba(108, 99, 255, 0.2);
    }

    hr {
        border-color: rgba(108, 99, 255, 0.2) !important;
    }

    .welcome-card {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.08) 0%, rgba(72, 202, 228, 0.08) 100%);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0 1.5rem;
    }

    .welcome-card h3 {
        color: #9D97FF;
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


EXAMPLE_QUESTIONS = [
    "Vượt đèn đỏ bị phạt bao nhiêu tiền?",
    "Lái xe khi say rượu bị xử lý thế nào?",
    "Không đội mũ bảo hiểm phạt như thế nào?",
    "Xe máy không có gương chiếu hậu có bị phạt không?",
    "Xây nhà thì cần điều kiện gì?",
    "Ly hôn đơn phương có được không? Điều kiện ra sao?",
]


def project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def session_get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def session_set(key: str, value: Any) -> None:
    st.session_state[key] = value


def get_rag_load_error(cache_dir_key: str) -> Optional[str]:
    errors = session_get(RAG_LOAD_ERRORS_KEY, {})
    if not isinstance(errors, dict):
        return None
    error = errors.get(cache_dir_key)
    return str(error) if error else None


def set_rag_load_error(cache_dir_key: str, error: Optional[str]) -> None:
    errors = session_get(RAG_LOAD_ERRORS_KEY, {})
    errors = dict(errors) if isinstance(errors, dict) else {}
    if error:
        errors[cache_dir_key] = error
    else:
        errors.pop(cache_dir_key, None)
    session_set(RAG_LOAD_ERRORS_KEY, errors)


def ensure_messages() -> List[Dict[str, Any]]:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    return st.session_state["messages"]


@st.cache_data(show_spinner=False)
def load_rag_settings() -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Chưa cài PyYAML. Hãy chạy: pip install PyYAML") from exc

    with open(RAG_CONFIG, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config YAML phải là object: {RAG_CONFIG}")
    return data


def resolve_csv_path(settings: Dict[str, Any]) -> Path:
    if DATA_CSV != DEFAULT_DATA_CSV:
        return DATA_CSV

    configured = settings.get("paths", {}).get("csv")
    return project_path(configured) if configured else DATA_CSV


@st.cache_resource(show_spinner="Đang đọc cấu hình LLM...")
def load_llm_state() -> Dict[str, Any]:
    from src.rag.llm import DIRECT_SYSTEM_PROMPT

    settings = load_rag_settings()
    model = settings.get("models", {}).get("llm", "qwen2.5:7b")
    rag_prompt = settings.get("llm", {}).get("system_prompt", "")
    return {
        "config": {"llm_model": model},
        "llm_system_prompt": rag_prompt,
        "direct_system_prompt": DIRECT_SYSTEM_PROMPT,
    }


@st.cache_resource(show_spinner="Đang kiểm tra Ollama...")
def load_ollama_client() -> Optional[Any]:
    try:
        import ollama

        return ollama
    except ImportError:
        st.warning("Chưa cài `ollama`. Hãy chạy `pip install ollama` và pull model trong YAML.")
        return None


@st.cache_resource(show_spinner="Đang tải Whisper...")
def load_voice_assistant(model_size: str = WHISPER_MODEL_SIZE) -> Optional[Any]:
    try:
        from src.whisper.audio_processor import LegalVoiceAssistant

        return LegalVoiceAssistant(model_size=model_size)
    except ImportError as exc:
        st.warning(f"Chưa cài đủ thư viện Whisper: {exc}")
        return None
    except Exception as exc:
        st.warning(f"Không thể tải Whisper `{model_size}`: {exc}")
        return None


@st.cache_resource(show_spinner="Đang tải bộ chuẩn hóa câu hỏi...")
def load_voice_query_rewriter() -> Optional[Any]:
    try:
        from src.whisper.llm_rewriter import LocalQueryRewriter

        return LocalQueryRewriter()
    except Exception:
        return None


def build_rag_resource(
    config_path: str,
    csv_path: str,
    cache_dir: Optional[str],
    cache_dir_key: str,
    encoder_key: str,
    encoder_name: Optional[str],
) -> Any:
    from src.rag.vector_db import HybridRAGSystem

    return HybridRAGSystem(
        config_path=config_path,
        csv_path=csv_path,
        cache_dir=cache_dir,
        cache_dir_key=cache_dir_key,
        encoder_key=encoder_key,
        encoder_name=encoder_name,
    )


def load_rag(
    cache_dir_key: str = "cache_dir",
    encoder_key: str = "encoder",
    encoder_name: Optional[str] = None,
) -> Optional[Any]:
    try:
        settings = load_rag_settings()
        csv_path = resolve_csv_path(settings)
        if not csv_path.exists():
            set_rag_load_error(cache_dir_key, f"Không tìm thấy CSV dữ liệu: {csv_path}")
            return None
        cache_dir_value = settings.get("paths", {}).get(cache_dir_key)
        cache_dir = str(project_path(cache_dir_value)) if cache_dir_value else None

        rag = build_rag_resource(
            config_path=str(RAG_CONFIG),
            csv_path=str(csv_path),
            cache_dir=cache_dir,
            cache_dir_key=cache_dir_key,
            encoder_key=encoder_key,
            encoder_name=encoder_name,
        )
        set_rag_load_error(cache_dir_key, None)
        return rag
    except Exception as exc:
        error = str(exc)
        set_rag_load_error(cache_dir_key, error)
        st.warning(f"Không thể load RAG `{cache_dir_key}`: {error}")
        return None


def cache_dir_path(cache_dir_key: str) -> Optional[Path]:
    try:
        settings = load_rag_settings()
    except Exception:
        return None
    cache_dir_value = settings.get("paths", {}).get(cache_dir_key)
    return project_path(cache_dir_value) if cache_dir_value else None


def cache_artifacts_ready(cache_dir_key: str) -> bool:
    try:
        settings = load_rag_settings()
    except Exception:
        return False

    cache_dir = cache_dir_path(cache_dir_key)
    if cache_dir is None or not cache_dir.exists():
        return False

    index_type = str(settings.get("index", {}).get("type", "hnsw")).lower()
    required_files = (
        *REQUIRED_CACHE_FILES,
        f"faiss_{index_type}.index",
        f"faiss_{index_type}.index.meta.json",
    )
    return all((cache_dir / filename).exists() for filename in required_files)


def get_encoder_name(encoder_key: str = "encoder", default: str = DEFAULT_ENCODER) -> str:
    try:
        settings = load_rag_settings()
    except Exception:
        return default
    return settings.get("models", {}).get(encoder_key, default)


def get_halong_encoder_name() -> str:
    return get_encoder_name("encoder_halong", DEFAULT_HALONG_ENCODER)


def format_elapsed(seconds: Optional[float]) -> str:
    if seconds is None:
        return "N/A"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


def render_timing(
    *,
    llm_elapsed: Optional[float] = None,
    cache_elapsed: Optional[float] = None,
    cache_label: Optional[str] = None,
) -> None:
    return


def init_session() -> None:
    ensure_messages()

    if "llm_state" not in st.session_state:
        session_set("llm_state", load_llm_state())

    if "ollama_client" not in st.session_state:
        session_set("ollama_client", load_ollama_client())

def retrieve_contexts(rag: Optional[Any], query: str, label: str) -> List[str]:
    if rag is None:
        return []

    try:
        return rag.hybrid_search(query, top_k=RAG_TOP_K, verbose=False)
    except TypeError:
        try:
            return rag.hybrid_search(query, top_k=RAG_TOP_K)
        except Exception as exc:
            error = str(exc)
            set_rag_load_error(label, error)
            st.warning(f"Không thể truy xuất context từ {label}: {error}")
            return []
    except Exception as exc:
        error = str(exc)
        set_rag_load_error(label, error)
        st.warning(f"Không thể truy xuất context từ {label}: {error}")
        return []


def release_rag_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def retrieve_contexts_from_cache(
    *,
    query: str,
    cache_dir_key: str,
    encoder_key: str,
    label: str,
    encoder_name: Optional[str] = None,
) -> tuple[List[str], bool, Optional[str]]:
    rag = None
    try:
        rag = load_rag(
            cache_dir_key=cache_dir_key,
            encoder_key=encoder_key,
            encoder_name=encoder_name,
        )
        if rag is None:
            return [], False, get_rag_load_error(cache_dir_key)

        contexts = retrieve_contexts(rag, query, label)
        if not contexts:
            error = get_rag_load_error(cache_dir_key) or f"Không tìm thấy context phù hợp từ `{cache_dir_key}`."
            return [], False, error
        return contexts, True, None
    finally:
        del rag
        release_rag_memory()


def render_sources(title: str, sources: List[str]) -> None:
    if not sources:
        return

    with st.expander(title, expanded=False):
        for i, ctx in enumerate(sources, 1):
            st.markdown(f'<span class="source-badge">Nguồn {i}</span>', unsafe_allow_html=True)
            st.caption(ctx[:500] + "..." if len(ctx) > 500 else ctx)


def transcribe_uploaded_audio(uploaded_audio: Any) -> tuple[str, str, Optional[str]]:
    assistant = load_voice_assistant()
    if assistant is None:
        return "", "", "Whisper chưa sẵn sàng."

    rewriter = load_voice_query_rewriter()
    suffix = Path(uploaded_audio.name or "").suffix.lower() or ".mp3"
    with tempfile.TemporaryDirectory(prefix="voice_question_") as temp_dir:
        audio_path = Path(temp_dir) / f"uploaded{suffix}"
        audio_path.write_bytes(uploaded_audio.getvalue())
        try:
            return assistant.transcribe_for_rag(str(audio_path), rewriter=rewriter)
        except Exception as exc:
            return "", "", f"Whisper không xử lý được file MP3: {exc}"


def render_streamed_answer(
    *,
    label: str,
    avatar: str,
    query: str,
    contexts: Optional[List[str]],
    use_rag: bool,
    llm_state: Dict[str, Any],
    cache_elapsed: Optional[float] = None,
    cache_label: Optional[str] = None,
) -> tuple[str, float]:
    from src.rag.llm import stream_llm_answer

    response = ""
    llm_started = perf_counter()
    st.caption(label)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            for chunk in stream_llm_answer(
                llm_state,
                query=query,
                results=contexts or [],
                model=llm_state["config"]["llm_model"],
                use_rag=use_rag,
            ):
                response += chunk
                placeholder.markdown(response + "▌")
            placeholder.markdown(response or "_Không có nội dung trả lời._")
        except Exception as exc:
            response = f"Lỗi xử lý: {exc}"
            placeholder.markdown(response)
        llm_elapsed = perf_counter() - llm_started
        render_timing(llm_elapsed=llm_elapsed, cache_elapsed=cache_elapsed, cache_label=cache_label)
    return response, llm_elapsed


def render_unavailable_answer(
    label: str,
    avatar: str,
    message: str,
    detail: Optional[str] = None,
    cache_elapsed: Optional[float] = None,
    cache_label: Optional[str] = None,
) -> str:
    st.caption(label)
    warning_text = f"{message}\n\nChi tiết: {detail}" if detail else message
    with st.chat_message("assistant"):
        st.warning(warning_text)
        render_timing(cache_elapsed=cache_elapsed, cache_label=cache_label)
    return f"Lỗi: {warning_text}"


def process_query(query: str) -> None:
    if not query.strip():
        return

    messages = ensure_messages()
    messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    llm_state = session_get("llm_state") or load_llm_state()
    answers: List[Dict[str, Any]] = []

    no_rag_answer, no_rag_llm_elapsed = render_streamed_answer(
        label="1. Qwen 2.5 - Không RAG",
        avatar="",
        query=query,
        contexts=[],
        use_rag=False,
        llm_state=llm_state,
    )
    answers.append(
        {
            "key": "no_rag",
            "label": "1. Qwen 2.5 - Không RAG",
            "avatar": "",
            "content": no_rag_answer,
            "sources": [],
            "timing": {"llm": no_rag_llm_elapsed},
        }
    )

    default_cache_started = perf_counter()
    default_contexts, default_ready, default_error = retrieve_contexts_from_cache(
        query=query,
        cache_dir_key="cache_dir",
        encoder_key="encoder",
        label="cache_dir",
    )
    default_cache_elapsed = perf_counter() - default_cache_started
    default_llm_elapsed = None
    if not default_ready:
        default_answer = render_unavailable_answer(
            "2. Qwen 2.5 + RAG with vietnamese-sbert",
            "",
            "RAG `cache_dir` chưa sẵn sàng hoặc chưa load được dữ liệu.",
            detail=default_error,
            cache_elapsed=default_cache_elapsed,
            cache_label="cache_dir",
        )
    else:
        default_answer, default_llm_elapsed = render_streamed_answer(
            label="2. Qwen 2.5 + RAG with vietnamese-sbert",
            avatar="",
            query=query,
            contexts=default_contexts,
            use_rag=True,
            llm_state=llm_state,
            cache_elapsed=default_cache_elapsed,
            cache_label="cache_dir",
        )
        render_sources("Nguồn tham chiếu - cache_dir", default_contexts)
    answers.append(
        {
            "key": "rag_default",
            "label": "2. Qwen 2.5 + RAG with vietnamese-sbert",
            "avatar": "",
            "content": default_answer,
            "sources": default_contexts,
            "timing": {"llm": default_llm_elapsed, "cache": default_cache_elapsed, "cache_label": "cache_dir"},
        }
    )

    halong_cache_started = perf_counter()
    halong_contexts, halong_ready, halong_error = retrieve_contexts_from_cache(
        query=query,
        cache_dir_key="cache_dir_halong",
        encoder_key="encoder_halong",
        encoder_name=get_halong_encoder_name(),
        label="cache_dir_halong",
    )
    halong_cache_elapsed = perf_counter() - halong_cache_started
    halong_llm_elapsed = None
    if not halong_ready:
        halong_answer = render_unavailable_answer(
            "3. Qwen 2.5 + RAG with halong_embedding",
            "",
            "RAG `cache_dir_halong` chưa sẵn sàng hoặc chưa load được dữ liệu.",
            detail=halong_error,
            cache_elapsed=halong_cache_elapsed,
            cache_label="cache_dir_halong",
        )
    else:
        halong_answer, halong_llm_elapsed = render_streamed_answer(
            label="3. Qwen 2.5 + RAG with halong_embedding",
            avatar="",
            query=query,
            contexts=halong_contexts,
            use_rag=True,
            llm_state=llm_state,
            cache_elapsed=halong_cache_elapsed,
            cache_label="cache_dir_halong",
        )
        render_sources("Nguồn tham chiếu - cache_dir_halong", halong_contexts)
    answers.append(
        {
            "key": "rag_halong",
            "label": "3. Qwen 2.5 + RAG with halong_embedding",
            "avatar": "",
            "content": halong_answer,
            "sources": halong_contexts,
            "timing": {"llm": halong_llm_elapsed, "cache": halong_cache_elapsed, "cache_label": "cache_dir_halong"},
        }
    )

    messages.append({"role": "triple_assistant", "answers": answers})


def render_model_card(index: int, name: str, meta: str, ready: bool) -> None:
    status_class = "status-ok" if ready else "status-warn"
    status_text = "Sẵn sàng" if ready else "Chưa sẵn sàng"
    st.markdown(
        f"""
        <div class="model-card">
            <div class="model-name">{index}. {name}</div>
            <div class="model-meta">{meta}</div>
            <div class="{status_class}">{status_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem 0 0.5rem;">
                <div style="font-size: 2rem; font-weight: 700; color: #9D97FF;">AI</div>
                <div style="font-weight: 700; font-size: 1.1rem; color: #9D97FF;">Trợ lý Pháp luật AI</div>
                <div style="color: #A7ABC4; font-size: 0.78rem;">So sánh 2 embedding RAG</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        llm_state = session_get("llm_state") or {"config": {"llm_model": "qwen2.5:7b"}}
        model_name = llm_state["config"]["llm_model"]
        ollama_ready = session_get("ollama_client") is not None
        rag_default_ready = cache_artifacts_ready("cache_dir")
        rag_halong_ready = cache_artifacts_ready("cache_dir_halong")

        st.caption(f"Model LLM đang dùng: `{model_name}`")
        render_model_card(1, "Không RAG", "Trả lời trực tiếp câu hỏi", ollama_ready)
        render_model_card(2, "RAG with vietnamese-sbert", f"Cache embedding: {get_encoder_name()}", ollama_ready and rag_default_ready)
        render_model_card(3, "RAG with halong_embedding", f"Cache embedding: {get_halong_encoder_name()}", ollama_ready and rag_halong_ready)

        st.divider()

        msg_count = len([m for m in session_get("messages", []) if m.get("role") == "user"])
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Câu hỏi", msg_count)
        with col2:
            ready_count = sum(
                [
                    ollama_ready,
                    ollama_ready and rag_default_ready,
                    ollama_ready and rag_halong_ready,
                ]
            )
            st.metric("Sẵn sàng", f"{ready_count}/3")

        st.divider()

        if st.button("Xóa lịch sử", use_container_width=True, type="secondary"):
            session_set("messages", [])
            st.rerun()

        st.divider()
        st.caption("PTIT - NLP - Deep Learning Project")
        st.caption("Powered by Ollama + Hybrid RAG")


def render_saved_answer(answer: Dict[str, Any]) -> None:
    st.caption(answer.get("label", "Assistant"))
    with st.chat_message("assistant"):
        st.markdown(answer.get("content", ""))
        timing = answer.get("timing", {})
        render_timing(
            llm_elapsed=timing.get("llm"),
            cache_elapsed=timing.get("cache"),
            cache_label=timing.get("cache_label"),
        )
    render_sources(f"Nguồn tham chiếu - {answer.get('label', '')}", answer.get("sources", []))


def render_chat() -> None:
    st.markdown('<p class="gradient-title">Trợ lý Pháp luật AI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Một LLM, ba chế độ trả lời: không RAG, RAG with vietnamese-sbert, RAG with halong_embedding.</p>',
        unsafe_allow_html=True,
    )

    messages = session_get("messages", [])
    if not messages:
        st.markdown(
            """
            <div class="welcome-card">
                <h3>Xin chào! Hãy đặt một câu hỏi pháp luật Việt Nam.</h3>
                <p style="color: #A7ABC4; font-size: 0.92rem;">
                    Hệ thống sẽ lần lượt trả lời bằng cùng một model LLM:
                    bản không RAG, bản RAG với cache mặc định, và bản RAG với cache Halong.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Câu hỏi gợi ý:**")
        cols = st.columns(2)
        for i, question in enumerate(EXAMPLE_QUESTIONS):
            with cols[i % 2]:
                if st.button(question, key=f"example_{i}", use_container_width=True):
                    process_query(question)
                    st.rerun()

    for msg in messages:
        role = msg.get("role")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(msg.get("content", ""))
        elif role == "triple_assistant":
            for answer in msg.get("answers", []):
                render_saved_answer(answer)
        elif role == "dual_assistant":
            legacy_answers = [
                {
                    "label": "Gemini AI",
                    "avatar": "",
                    "content": msg.get("content_gemini", ""),
                    "sources": msg.get("sources", []),
                },
                {
                    "label": "Qwen 2.5",
                    "avatar": "",
                    "content": msg.get("content_qwen", ""),
                    "sources": msg.get("sources", []),
                },
            ]
            for answer in legacy_answers:
                render_saved_answer(answer)


def render_input() -> None:
    uploaded_audio = st.file_uploader(
        "MP3 câu hỏi",
        type=["mp3"],
        accept_multiple_files=False,
        key="voice_mp3_upload",
    )
    if uploaded_audio is not None:
        st.audio(uploaded_audio, format="audio/mp3")
        if st.button("Gửi MP3", type="primary", use_container_width=True):
            with st.spinner("Đang chuyển MP3 thành câu hỏi..."):
                raw_query, voice_query, warning = transcribe_uploaded_audio(uploaded_audio)

            if raw_query and voice_query:
                if warning:
                    st.warning(warning)
                if raw_query != voice_query:
                    st.info(f"Whisper: {raw_query}\n\nCâu hỏi RAG: {voice_query}")
                else:
                    st.info(f"Câu hỏi RAG: {voice_query}")
                process_query(voice_query)
                st.rerun()
            else:
                st.error(warning or "Không thể tạo câu hỏi từ file MP3.")

    prompt = st.chat_input("Hỏi về pháp luật... (VD: Vượt đèn đỏ phạt bao nhiêu?)")
    if prompt:
        process_query(prompt)
        st.rerun()


def main() -> None:
    init_session()
    render_sidebar()
    render_chat()
    render_input()

    if session_get("ollama_client") is None:
        st.error("Không thể import `ollama`. Cài bằng `pip install ollama`, sau đó chạy `ollama pull` model trong YAML.")


if __name__ == "__main__":
    main()
