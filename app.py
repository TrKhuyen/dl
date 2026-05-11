"""
🚦 Trợ lý Pháp lý AI — Giao diện Streamlit
=========================================
Giao diện chat kiểu ChatGPT/Gemini cho hệ thống tra cứu luật giao thông Việt Nam.

Chạy:
    conda activate dl
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Page Config (phải gọi trước mọi lệnh st khác) ───────────────────────────
st.set_page_config(
    page_title="Trợ lý Pháp lý AI 🚦",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Gradient header */
    .gradient-title {
        background: linear-gradient(135deg, #6C63FF 0%, #48CAE4 50%, #06D6A0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0;
    }

    .subtitle {
        color: #8B8FA8;
        font-size: 0.9rem;
        margin-top: 4px;
        margin-bottom: 1.5rem;
    }

    /* Chat messages styling */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        margin-bottom: 0.5rem;
        padding: 0.5rem 1rem;
    }

    /* User message */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: linear-gradient(135deg, #2D2B55 0%, #1A1C2E 100%);
    }

    /* Assistant message */
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        background: linear-gradient(135deg, #0D2137 0%, #0E1117 100%);
        border-left: 3px solid #6C63FF;
    }

    /* Source badge */
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

    /* Status badge */
    .status-rag-on {
        background: rgba(6, 214, 160, 0.15);
        color: #06D6A0;
        border: 1px solid rgba(6, 214, 160, 0.3);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .status-rag-off {
        background: rgba(255, 214, 10, 0.15);
        color: #FFD60A;
        border: 1px solid rgba(255, 214, 10, 0.3);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Sidebar style */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1C2E 0%, #0E1117 100%);
        border-right: 1px solid rgba(108, 99, 255, 0.2);
    }

    /* Divider */
    hr {
        border-color: rgba(108, 99, 255, 0.2) !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #6C63FF !important;
    }


    /* Welcome card */
    .welcome-card {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.08) 0%, rgba(72, 202, 228, 0.08) 100%);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1rem 0 2rem;
    }

    .welcome-card h3 {
        color: #9D97FF;
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
    }


    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ────────────────────────────────────────────────────────────────
DATA_CSV = ROOT / "data" / "processed" / "optimized_corpus.csv"
EXAMPLE_QUESTIONS = [
    "Vượt đèn đỏ bị phạt bao nhiêu tiền?",
    "Lái xe khi say rượu bị xử lý thế nào?",
    "Không đội mũ bảo hiểm phạt như thế nào?",
    "Xe máy không có gương chiếu hậu có bị phạt không?",
]


# ── Cached resource loaders ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="🤖 Đang khởi tạo Gemini AI...")
def load_gemini():
    """Load Gemini client (cached, chỉ init 1 lần)."""
    from src.ui.gemini_client import GeminiLegalAssistant
    return GeminiLegalAssistant()


@st.cache_resource(show_spinner="🔍 Đang nạp Cơ sở dữ liệu Luật (RAG)...")
def load_rag():
    """Load RAG system nếu data tồn tại (cached)."""
    if not DATA_CSV.exists():
        return None
    try:
        from src.rag.vector_db import HybridRAGSystem
        return HybridRAGSystem(csv_path=str(DATA_CSV))
    except Exception as e:
        st.warning(f"⚠️ Không thể load RAG: {e}")
        return None


# ── Session State Init ────────────────────────────────────────────────────────

def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "gemini" not in st.session_state:
        try:
            st.session_state.gemini = load_gemini()
            st.session_state.gemini.new_chat()
        except Exception as e:
            st.session_state.gemini = None
            st.session_state.gemini_error = str(e)
    if "rag" not in st.session_state:
        st.session_state.rag = load_rag()


# ── Helper: process query ─────────────────────────────────────────────────────

def process_query(query: str):
    """Gửi query → RAG → Gemini, lưu vào session state."""
    if not query.strip():
        return

    gemini = st.session_state.get("gemini")
    if gemini is None:
        st.error(f"❌ Gemini chưa được khởi tạo. Lỗi: {st.session_state.get('gemini_error', 'Unknown')}")
        return

    # Thêm user message
    st.session_state.messages.append({
        "role": "user",
        "content": query,
    })

    # Lấy RAG context nếu có
    contexts = []
    rag = st.session_state.get("rag")
    if rag is not None:
        try:
            contexts = rag.hybrid_search(query, top_k=3)
        except Exception:
            contexts = []

    # Stream câu trả lời từ Gemini
    with st.chat_message("assistant", avatar="⚖️"):
        placeholder = st.empty()
        full_response = ""
        try:
            for chunk in gemini.answer_stream(query, contexts=contexts if contexts else None):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"⚠️ Lỗi: {str(e)}"
            placeholder.markdown(full_response)

        # Hiển thị nguồn tham chiếu nếu có
        if contexts:
            with st.expander("📚 Nguồn tham chiếu từ CSDL Luật", expanded=False):
                for i, ctx in enumerate(contexts, 1):
                    st.markdown(
                        f'<span class="source-badge">Nguồn {i}</span>', 
                        unsafe_allow_html=True
                    )
                    st.caption(ctx[:300] + "..." if len(ctx) > 300 else ctx)

    # Lưu assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": contexts,
    })


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        # Logo & Title
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 0.5rem;">
            <div style="font-size: 3rem;">⚖️</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: #9D97FF;">Trợ lý Pháp lý AI</div>
            <div style="color: #5A5D7A; font-size: 0.78rem;">Luật Giao thông Việt Nam</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # RAG Status
        rag_active = st.session_state.get("rag") is not None
        if rag_active:
            st.markdown('<div style="text-align:center"><span class="status-rag-on">🟢 RAG: Đang hoạt động</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:center"><span class="status-rag-off">🟡 RAG: Chưa có dữ liệu</span></div>', unsafe_allow_html=True)
            st.caption("Chạy data pipeline để kích hoạt RAG.")

        st.divider()

        # Stats
        msg_count = len([m for m in st.session_state.get("messages", []) if m["role"] == "user"])
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💬 Câu hỏi", msg_count)
        with col2:
            st.metric("🔍 Nguồn RAG", "✓" if rag_active else "✗")

        st.divider()

        # Clear button
        if st.button("🗑️ Xóa lịch sử", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            if st.session_state.get("gemini"):
                st.session_state.gemini.new_chat()
            st.rerun()

        st.divider()

        st.caption("🏫 PTIT · NLP · Deep Learning Project")
        st.caption("Powered by Gemini 2.5 Flash + Hybrid RAG")


# ── Main Chat Area ────────────────────────────────────────────────────────────

def render_chat():
    # Header
    st.markdown('<p class="gradient-title">🚦 Trợ lý Pháp lý AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Hỏi đáp Luật Giao thông Đường bộ Việt Nam · Powered by Gemini AI + Hybrid RAG</p>', unsafe_allow_html=True)

    # Welcome screen khi chưa có message
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-card">
            <h3>👋 Xin chào! Tôi có thể giúp gì cho bạn?</h3>
            <p style="color: #6B7280; font-size: 0.9rem;">
                Hãy đặt câu hỏi về luật giao thông đường bộ Việt Nam. 
                Tôi sẽ tra cứu từ cơ sở dữ liệu pháp luật và trả lời chính xác.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**💡 Câu hỏi gợi ý:**")
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            with cols[i % 2]:
                if st.button(q, key=f"example_{i}", use_container_width=True):
                    process_query(q)
                    st.rerun()

    # Render chat history
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        avatar = "🧑‍💻" if role == "user" else "⚖️"

        with st.chat_message(role, avatar=avatar):
            st.markdown(content)
            # Nguồn tham chiếu cho assistant message
            if role == "assistant" and msg.get("sources"):
                with st.expander("📚 Nguồn tham chiếu", expanded=False):
                    for i, ctx in enumerate(msg["sources"], 1):
                        st.markdown(f'<span class="source-badge">Nguồn {i}</span>', unsafe_allow_html=True)
                        st.caption(ctx[:300] + "..." if len(ctx) > 300 else ctx)


def render_input():
    """Render text chat input."""
    if prompt := st.chat_input("Hỏi về luật giao thông... (VD: Vượt đèn đỏ phạt bao nhiêu?)"):
        process_query(prompt)
        st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    init_session()
    render_sidebar()
    render_chat()
    render_input()

    # Gemini error banner
    if st.session_state.get("gemini") is None:
        st.error(
            f"❌ **Lỗi khởi tạo Gemini API**\n\n"
            f"{st.session_state.get('gemini_error', 'Unknown error')}\n\n"
            f"Kiểm tra `api_key` trong file `.env`."
        )


if __name__ == "__main__":
    main()
