"""
Gemini API Wrapper cho hệ thống Trợ lý Pháp lý AI.
Sử dụng google-genai (SDK mới) để gọi Gemini.
"""

import os
import sys
from pathlib import Path

from google import genai
from google.genai import types

# Thêm root vào path để import được từ src/
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


SYSTEM_PROMPT = """Bạn là **Trợ lý Pháp lý AI** chuyên về Luật Giao thông Đường bộ Việt Nam.

NHIỆM VỤ:
- Trả lời các câu hỏi về luật giao thông đường bộ Việt Nam một cách chính xác, rõ ràng.
- Trích dẫn cụ thể Nghị định, Điều khoản liên quan khi có thể.
- Nếu được cung cấp ngữ cảnh từ cơ sở dữ liệu luật, ưu tiên sử dụng thông tin đó.

QUY TẮC:
1. Trả lời bằng tiếng Việt, ngắn gọn và dễ hiểu.
2. Nếu không có đủ thông tin, hãy nói rõ và khuyến khích tham khảo luật sư.
3. KHÔNG bịa đặt thông tin pháp lý không chắc chắn.
4. Định dạng câu trả lời rõ ràng, dùng danh sách khi phù hợp.

Bắt đầu mỗi câu trả lời bằng một câu tóm tắt ngắn, sau đó giải thích chi tiết."""

MODEL_NAME = "gemini-2.5-flash"
MAX_HISTORY_TURNS = 20  # Giữ tối đa N turns để tránh context quá dài


class GeminiLegalAssistant:
    """Wrapper cho Gemini API với System Prompt pháp lý (google-genai SDK)."""

    def __init__(self, api_key: str | None = None):
        """
        Khởi tạo Gemini client.

        Args:
            api_key: Gemini API key. Nếu None, đọc từ biến môi trường hoặc .env.
        """
        key = api_key or os.environ.get("api_key") or self._load_from_env_file()

        if not key:
            raise ValueError(
                "Không tìm thấy Gemini API key!\n"
                "Kiểm tra file .env hoặc biến môi trường 'api_key'."
            )

        self.client = genai.Client(api_key=key)
        # Lịch sử hội thoại: list of {"role": str, "parts": [str]}
        self._history: list[types.Content] = []

    def _load_from_env_file(self) -> str | None:
        """Đọc api_key từ file .env trong thư mục gốc project."""
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("api_key="):
                    return line.split("=", 1)[1].strip()
        return None

    def new_chat(self):
        """Bắt đầu phiên chat mới (xóa lịch sử hội thoại)."""
        self._history = []

    def _build_prompt(self, query: str, contexts: list[str] | None) -> str:
        """Xây dựng prompt với RAG context nếu có."""
        if contexts:
            context_block = "\n\n".join(
                f"[Nguồn {i+1}]: {ctx}" for i, ctx in enumerate(contexts)
            )
            return (
                f"📚 Thông tin từ cơ sở dữ liệu luật:\n{context_block}\n\n"
                f"Câu hỏi của người dùng: {query}\n\n"
                f"Hãy trả lời dựa trên thông tin trên."
            )
        return query

    def answer(self, query: str, contexts: list[str] | None = None) -> str:
        """
        Trả lời câu hỏi (non-streaming), có thể kèm context từ RAG.
        Lưu ý: app.py dùng answer_stream() cho streaming UX; method này dành cho testing/scripting.

        Args:
            query: Câu hỏi của người dùng.
            contexts: Danh sách đoạn văn bản luật từ RAG system (optional).

        Returns:
            Câu trả lời dạng string.
        """
        full_prompt = self._build_prompt(query, contexts)

        # Thêm vào history
        self._history.append(
            types.Content(role="user", parts=[types.Part(text=full_prompt)])
        )

        # Trim history nếu quá dài (sliding window)
        if len(self._history) > MAX_HISTORY_TURNS * 2:
            self._history = self._history[-(MAX_HISTORY_TURNS * 2):]

        try:
            response = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=self._history,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                ),
            )
            answer_text = response.text or ""
            # Thêm assistant response vào history
            self._history.append(
                types.Content(role="model", parts=[types.Part(text=answer_text)])
            )
            return answer_text
        except Exception as e:
            self._history.pop()  # Rollback nếu lỗi
            return f"⚠️ Lỗi khi gọi Gemini API: {str(e)}"

    def answer_stream(self, query: str, contexts: list[str] | None = None):
        """
        Streaming version — yield từng chunk text (typewriter effect).

        Yields:
            str: Từng đoạn text nhỏ từ Gemini.
        """
        full_prompt = self._build_prompt(query, contexts)

        self._history.append(
            types.Content(role="user", parts=[types.Part(text=full_prompt)])
        )

        # Trim history nếu quá dài (sliding window)
        if len(self._history) > MAX_HISTORY_TURNS * 2:
            self._history = self._history[-(MAX_HISTORY_TURNS * 2):]

        full_response_text = ""
        try:
            for chunk in self.client.models.generate_content_stream(
                model=MODEL_NAME,
                contents=self._history,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                ),
            ):
                if chunk.text:
                    full_response_text += chunk.text
                    yield chunk.text

            # Lưu response vào history sau khi stream xong
            self._history.append(
                types.Content(role="model", parts=[types.Part(text=full_response_text)])
            )
        except Exception as e:
            self._history.pop()  # Rollback
            yield f"⚠️ Lỗi khi gọi Gemini API: {str(e)}"
