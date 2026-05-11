"""
Tests cho GeminiLegalAssistant (src/ui/gemini_client.py)

Chiến lược:
- Mock toàn bộ google.genai để không cần API key thật khi chạy test
- Kiểm tra: init, key loading, prompt building, answer(), answer_stream(), new_chat()
- Kiểm tra: edge cases, error handling, history rollback

Chạy:
    conda activate dl
    pytest tests/test_gemini_client.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Đảm bảo root trong path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_genai(tmp_path):
    """Patch toàn bộ google.genai để không cần network."""
    with patch("src.ui.gemini_client.genai") as mock_g, \
         patch("src.ui.gemini_client.types") as mock_types:

        # Client mock
        mock_client = MagicMock()
        mock_g.Client.return_value = mock_client

        # types.Content / types.Part — trả về object có thể so sánh
        mock_types.Content.side_effect = lambda role, parts: {"role": role, "parts": parts}
        mock_types.Part.side_effect = lambda text: {"text": text}
        mock_types.GenerateContentConfig.return_value = MagicMock()

        yield mock_g, mock_types, mock_client


@pytest.fixture
def assistant(mock_genai, tmp_path):
    """GeminiLegalAssistant với API key giả và genai đã mock."""
    from src.ui.gemini_client import GeminiLegalAssistant
    asst = GeminiLegalAssistant(api_key="fake-test-key-1234")
    asst.new_chat()
    return asst


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    """Tạo file .env tạm với api_key."""
    env = tmp_path / ".env"
    env.write_text("api_key=env-file-key-5678\n", encoding="utf-8")
    # Trỏ ROOT của module về tmp_path
    monkeypatch.setattr("src.ui.gemini_client.ROOT", tmp_path)
    return env


# ── Tests: Initialization ────────────────────────────────────────────────────

class TestInit:

    def test_init_with_explicit_key(self, mock_genai):
        """Khởi tạo thành công khi truyền api_key trực tiếp."""
        mock_g, _, _ = mock_genai
        from src.ui.gemini_client import GeminiLegalAssistant
        asst = GeminiLegalAssistant(api_key="direct-key")
        mock_g.Client.assert_called_once_with(api_key="direct-key")
        assert asst._history == []

    def test_init_reads_key_from_env_var(self, mock_genai, monkeypatch):
        """Đọc api_key từ biến môi trường khi không truyền tham số."""
        mock_g, _, _ = mock_genai
        monkeypatch.setenv("api_key", "env-var-key")
        # Patch ROOT để _load_from_env_file không tìm thấy file
        with patch("src.ui.gemini_client.ROOT", Path("/nonexistent")):
            from src.ui.gemini_client import GeminiLegalAssistant
            asst = GeminiLegalAssistant()
        mock_g.Client.assert_called_once_with(api_key="env-var-key")

    def test_init_reads_key_from_env_file(self, mock_genai, env_file, monkeypatch):
        """Đọc api_key từ file .env khi không có env var."""
        mock_g, _, _ = mock_genai
        monkeypatch.delenv("api_key", raising=False)
        from src.ui.gemini_client import GeminiLegalAssistant
        asst = GeminiLegalAssistant()
        mock_g.Client.assert_called_once_with(api_key="env-file-key-5678")

    def test_init_raises_when_no_key(self, mock_genai, monkeypatch, tmp_path):
        """ValueError nếu không có API key từ bất kỳ nguồn nào."""
        monkeypatch.delenv("api_key", raising=False)
        with patch("src.ui.gemini_client.ROOT", tmp_path):  # tmp_path không có .env
            from src.ui.gemini_client import GeminiLegalAssistant
            with pytest.raises(ValueError, match="api_key"):
                GeminiLegalAssistant()

    def test_init_ignores_empty_env_var(self, mock_genai, env_file, monkeypatch):
        """Env var rỗng → fallback sang .env file."""
        mock_g, _, _ = mock_genai
        monkeypatch.setenv("api_key", "")
        from src.ui.gemini_client import GeminiLegalAssistant
        asst = GeminiLegalAssistant()
        # Should use env-file key (env var is falsy)
        mock_g.Client.assert_called_once_with(api_key="env-file-key-5678")


# ── Tests: new_chat ──────────────────────────────────────────────────────────

class TestNewChat:

    def test_new_chat_clears_history(self, assistant):
        """new_chat() xóa sạch _history."""
        assistant._history = [{"role": "user", "parts": [{"text": "test"}]}]
        assistant.new_chat()
        assert assistant._history == []

    def test_new_chat_idempotent(self, assistant):
        """Gọi new_chat() nhiều lần vẫn an toàn."""
        assistant.new_chat()
        assistant.new_chat()
        assert assistant._history == []


# ── Tests: _build_prompt ─────────────────────────────────────────────────────

class TestBuildPrompt:

    def test_no_context_returns_raw_query(self, assistant):
        """Không có context → trả về query gốc."""
        result = assistant._build_prompt("vượt đèn đỏ?", None)
        assert result == "vượt đèn đỏ?"

    def test_empty_context_list_returns_raw_query(self, assistant):
        """Context list rỗng → trả về query gốc."""
        result = assistant._build_prompt("câu hỏi", [])
        assert result == "câu hỏi"

    def test_with_single_context(self, assistant):
        """Một context → prompt chứa [Nguồn 1] và câu hỏi."""
        result = assistant._build_prompt("câu hỏi?", ["Điều 6: phạt 4 triệu"])
        assert "[Nguồn 1]" in result
        assert "Điều 6: phạt 4 triệu" in result
        assert "câu hỏi?" in result

    def test_with_multiple_contexts(self, assistant):
        """Nhiều context → có [Nguồn 1], [Nguồn 2], ..."""
        contexts = ["ctx1", "ctx2", "ctx3"]
        result = assistant._build_prompt("q", contexts)
        assert "[Nguồn 1]" in result
        assert "[Nguồn 2]" in result
        assert "[Nguồn 3]" in result

    def test_context_appears_before_question(self, assistant):
        """Context block xuất hiện trước câu hỏi trong prompt."""
        result = assistant._build_prompt("câu hỏi", ["ctx_data"])
        ctx_pos = result.index("ctx_data")
        q_pos = result.index("câu hỏi")
        assert ctx_pos < q_pos


# ── Tests: answer() ─────────────────────────────────────────────────────────

class TestAnswer:

    def _make_response(self, text):
        """Tạo mock response object."""
        resp = MagicMock()
        resp.text = text
        return resp

    def test_answer_happy_path(self, mock_genai, assistant):
        """answer() trả về text từ Gemini và lưu vào history."""
        _, _, mock_client = mock_genai
        mock_client.models.generate_content.return_value = self._make_response("Phạt 800k-1tr.")

        result = assistant.answer("xe máy vượt đèn đỏ?")

        assert result == "Phạt 800k-1tr."
        # History phải có user message và model response
        assert len(assistant._history) == 2
        assert assistant._history[0]["role"] == "user"
        assert assistant._history[1]["role"] == "model"

    def test_answer_with_rag_context(self, mock_genai, assistant):
        """answer() với RAG context → prompt chứa context."""
        _, _, mock_client = mock_genai
        mock_client.models.generate_content.return_value = self._make_response("Theo Điều 6...")
        contexts = ["Điều 6: phạt ô tô 4-6 triệu"]

        result = assistant.answer("ô tô vượt đèn đỏ?", contexts=contexts)

        assert result == "Theo Điều 6..."
        # Kiểm tra prompt đã được truyền vào generate_content
        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs.get("contents") or call_args[1].get("contents") or call_args[0][1]
        # Phần user message phải chứa context
        user_part_text = contents[0]["parts"][0]["text"]
        assert "Điều 6" in user_part_text

    def test_answer_accumulates_history(self, mock_genai, assistant):
        """Nhiều lần gọi answer() → history tích lũy đúng."""
        _, _, mock_client = mock_genai
        mock_client.models.generate_content.side_effect = [
            self._make_response("Trả lời 1"),
            self._make_response("Trả lời 2"),
        ]

        assistant.answer("Câu hỏi 1")
        assistant.answer("Câu hỏi 2")

        # 2 turns × (user + model) = 4 entries
        assert len(assistant._history) == 4

    def test_answer_on_api_error_returns_error_string(self, mock_genai, assistant):
        """API lỗi → trả về chuỗi lỗi, KHÔNG raise exception."""
        _, _, mock_client = mock_genai
        mock_client.models.generate_content.side_effect = Exception("Connection timeout")

        result = assistant.answer("câu hỏi")

        assert "⚠️" in result or "Lỗi" in result
        assert "Connection timeout" in result

    def test_answer_rollback_history_on_error(self, mock_genai, assistant):
        """Khi API lỗi → history rollback về trạng thái trước."""
        _, _, mock_client = mock_genai
        mock_client.models.generate_content.side_effect = Exception("API fail")

        history_before = len(assistant._history)
        assistant.answer("câu hỏi lỗi")
        # Sau rollback: history phải bằng state trước khi gọi
        assert len(assistant._history) == history_before

    def test_answer_empty_response_text(self, mock_genai, assistant):
        """Response.text = None → trả về chuỗi rỗng, không crash."""
        _, _, mock_client = mock_genai
        resp = MagicMock()
        resp.text = None
        mock_client.models.generate_content.return_value = resp

        result = assistant.answer("câu hỏi")
        assert result == ""


# ── Tests: answer_stream() ───────────────────────────────────────────────────

class TestAnswerStream:

    def _make_chunks(self, texts):
        """Tạo danh sách mock chunks để simulate streaming."""
        chunks = []
        for t in texts:
            c = MagicMock()
            c.text = t
            chunks.append(c)
        return chunks

    def test_stream_yields_all_chunks(self, mock_genai, assistant):
        """answer_stream() yield đủ các chunks từ API."""
        _, _, mock_client = mock_genai
        mock_client.models.generate_content_stream.return_value = \
            self._make_chunks(["Theo ", "Nghị định ", "100/2019..."])

        result = list(assistant.answer_stream("câu hỏi"))
        assert result == ["Theo ", "Nghị định ", "100/2019..."]

    def test_stream_saves_full_response_to_history(self, mock_genai, assistant):
        """answer_stream() lưu toàn bộ text (nối chunks) vào history sau khi stream xong."""
        _, _, mock_client = mock_genai
        mock_client.models.generate_content_stream.return_value = \
            self._make_chunks(["phần 1", " phần 2"])

        # Consume the generator
        list(assistant.answer_stream("câu hỏi"))

        # History phải có user + model
        assert len(assistant._history) == 2
        model_text = assistant._history[1]["parts"][0]["text"]
        assert model_text == "phần 1 phần 2"

    def test_stream_skips_none_chunks(self, mock_genai, assistant):
        """Chunk với text=None bị bỏ qua, không yield."""
        _, _, mock_client = mock_genai
        chunks = self._make_chunks(["OK"])
        null_chunk = MagicMock()
        null_chunk.text = None
        chunks.insert(1, null_chunk)
        mock_client.models.generate_content_stream.return_value = chunks

        result = list(assistant.answer_stream("q"))
        assert result == ["OK"]
        assert None not in result

    def test_stream_on_api_error_yields_error_string(self, mock_genai, assistant):
        """API lỗi khi stream → yield error string, không raise."""
        _, _, mock_client = mock_genai
        mock_client.models.generate_content_stream.side_effect = Exception("Stream fail")

        result = list(assistant.answer_stream("câu hỏi"))
        assert len(result) == 1
        assert "Lỗi" in result[0] or "⚠️" in result[0]

    def test_stream_rollback_history_on_error(self, mock_genai, assistant):
        """Stream lỗi → history rollback, không rò rỉ user message."""
        _, _, mock_client = mock_genai
        mock_client.models.generate_content_stream.side_effect = Exception("fail")

        history_before = len(assistant._history)
        list(assistant.answer_stream("câu hỏi"))
        assert len(assistant._history) == history_before

    def test_stream_then_answer_shares_history(self, mock_genai, assistant):
        """answer_stream() và answer() dùng chung _history → multi-turn hoạt động."""
        _, _, mock_client = mock_genai
        mock_client.models.generate_content_stream.return_value = \
            self._make_chunks(["Trả lời stream"])
        mock_client.models.generate_content.return_value = MagicMock(text="Trả lời answer")

        list(assistant.answer_stream("Câu 1"))
        assistant.answer("Câu 2")

        # 2 turns × 2 = 4 entries
        assert len(assistant._history) == 4


# ── Tests: new_chat resets history mid-conversation ──────────────────────────

class TestNewChatMidConversation:

    def test_new_chat_after_conversation_resets(self, mock_genai, assistant):
        """new_chat() giữa cuộc trò chuyện → history rỗng, model không thấy lịch sử cũ."""
        _, _, mock_client = mock_genai
        mock_client.models.generate_content.return_value = MagicMock(text="R1")
        assistant.answer("Câu 1")

        assistant.new_chat()
        assert assistant._history == []

        mock_client.models.generate_content.return_value = MagicMock(text="R2")
        assistant.answer("Câu 2")
        # History chỉ có turn mới nhất
        assert len(assistant._history) == 2


# ── Tests: load_rag in app.py ────────────────────────────────────────────────

class TestLoadRag:
    """Kiểm tra logic lazy-load RAG trong app.py."""

    def test_load_rag_returns_none_when_no_csv(self, tmp_path, monkeypatch):
        """load_rag() trả None khi DATA_CSV không tồn tại."""
        import importlib
        # Patch DATA_CSV sang path không tồn tại
        import app
        monkeypatch.setattr(app, "DATA_CSV", tmp_path / "nonexistent.csv")
        # Reset cache
        app.load_rag.clear()
        result = app.load_rag()
        assert result is None

    def test_load_rag_returns_none_on_import_error(self, tmp_path, monkeypatch):
        """load_rag() trả None gracefully khi HybridRAGSystem raise exception."""
        import app
        csv = tmp_path / "optimized_corpus.csv"
        csv.write_text("context\ntest data", encoding="utf-8")
        monkeypatch.setattr(app, "DATA_CSV", csv)
        app.load_rag.clear()

        # HybridRAGSystem được import cục bộ bên trong load_rag()
        # → patch đúng path import: src.rag.vector_db.HybridRAGSystem
        with patch("src.rag.vector_db.HybridRAGSystem", side_effect=Exception("model fail")):
            # Streamlit warning() cũng cần mock để tránh lỗi context
            with patch("app.st") as mock_st:
                mock_st.warning = MagicMock()
                result = app.load_rag()
        assert result is None


# ── Tests: process_query logic ───────────────────────────────────────────────

class TestProcessQuery:
    """Unit tests cho process_query() — mock Streamlit và session_state."""

    def test_empty_query_returns_early(self):
        """Query rỗng hoặc chỉ khoảng trắng → không gọi Gemini."""
        import app
        with patch("app.st") as mock_st:
            mock_st.session_state = MagicMock()
            mock_st.session_state.get.return_value = MagicMock()
            # Gọi với query rỗng
            app.process_query("   ")
            # session_state.messages.append không được gọi
            mock_st.session_state.messages.append.assert_not_called()

    def test_none_gemini_shows_error(self):
        """Khi gemini = None → st.error() được gọi."""
        import app
        with patch("app.st") as mock_st:
            mock_st.session_state = {"gemini": None, "gemini_error": "key missing", "rag": None}
            app.process_query("câu hỏi")
            mock_st.error.assert_called_once()
            error_msg = mock_st.error.call_args[0][0]
            assert "Gemini" in error_msg
