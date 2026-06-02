import whisper
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

# Tắt các cảnh báo rác của thư viện FP16 để terminal sạch sẽ
warnings.filterwarnings("ignore", category=UserWarning)

FFMPEG_INSTALL_HINT = (
    "❌ Lỗi: Không tìm thấy ffmpeg trong PATH. "
    "Whisper cần ffmpeg để đọc file MP3. "
    "Hãy cài bằng: conda install -n dl -c conda-forge ffmpeg"
)

REWRITER_ERROR_PREFIX = "❌"


def ensure_ffmpeg_available():
    if shutil.which("ffmpeg") is not None:
        return True

    ffmpeg_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    conda_bin = Path(sys.prefix) / "Library" / "bin"
    conda_ffmpeg = conda_bin / ffmpeg_name
    if conda_ffmpeg.exists():
        path_parts = [str(conda_bin)]
        mingw_bin = Path(sys.prefix) / "Library" / "mingw-w64" / "bin"
        if mingw_bin.exists():
            path_parts.append(str(mingw_bin))
        path_parts.append(os.environ.get("PATH", ""))
        os.environ["PATH"] = os.pathsep.join(path_parts)
        return True

    return False


def clean_query_text(query: str) -> str:
    cleaned = query.strip()
    for prefix in ("Output:", "Câu đã sửa:", "Câu hỏi:", "Query:"):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned.strip("\"' ")


def rewrite_query_for_rag(raw_query: str, rewriter: Optional[object] = None) -> tuple[str, Optional[str]]:
    if rewriter is None:
        try:
            from src.whisper.llm_rewriter import LocalQueryRewriter

            rewriter = LocalQueryRewriter()
        except Exception as exc:
            return "", f"Không thể tải llm_rewriter nên chưa thể đưa câu hỏi vào RAG: {exc}"

    try:
        rewritten = clean_query_text(str(rewriter.correct_query(raw_query)))
    except Exception as exc:
        return "", f"Không thể chuẩn hóa câu hỏi bằng llm_rewriter nên chưa thể đưa vào RAG: {exc}"

    if not rewritten:
        return "", "llm_rewriter không trả về câu hỏi nên chưa thể đưa vào RAG."
    if rewritten.startswith(REWRITER_ERROR_PREFIX) or "Lỗi kết nối" in rewritten:
        return "", rewritten
    return rewritten, None


class LegalVoiceAssistant:
    def __init__(self, model_size="small"):
        """
        Khởi tạo mô hình Whisper.
        Vì máy bạn có GPU NVIDIA mạnh, dùng bản 'small' hoặc 'medium' sẽ cho tốc độ cực nhanh
        và độ chính xác tiếng Việt rất cao.
        """
        print(f"Đang tải mô hình AI Nhận diện giọng nói Whisper ({model_size})...")
        # Hệ thống sẽ tự động tìm và sử dụng GPU (CUDA) nếu có
        self.model = whisper.load_model(model_size)
        print("✅ Whisper đã sẵn sàng nhận lệnh!")

    def transcribe(self, audio_path):
        if not os.path.exists(audio_path):
            return "❌ Lỗi: Không tìm thấy file âm thanh."
        if not ensure_ffmpeg_available():
            return FFMPEG_INSTALL_HINT

        print(f"\n🎙️ Đang lắng nghe và giải mã file: {audio_path}...")
        
        # task='transcribe': Dịch tiếng nói
        # language='vi': Khóa ngôn ngữ là tiếng Việt để tăng độ chuẩn xác
        result = self.model.transcribe(audio_path, language='vi', task='transcribe')
        
        text = result['text'].strip()
        return text

    def transcribe_for_rag(self, audio_path, rewriter: Optional[object] = None) -> tuple[str, str, Optional[str]]:
        raw_query = clean_query_text(self.transcribe(audio_path))
        if not raw_query:
            return "", "", "Whisper không nhận diện được câu hỏi."
        if raw_query.startswith("❌"):
            return raw_query, "", raw_query

        rag_query, warning = rewrite_query_for_rag(raw_query, rewriter=rewriter)
        return raw_query, rag_query, warning

# --- KHỐI TEST CHẠY THỬ ---
if __name__ == "__main__":
    assistant = LegalVoiceAssistant(model_size="small")
    
    # Bạn hãy lấy điện thoại tự ghi âm 1 câu hỏi luật (khoảng 5-10 giây)
    # Ví dụ: "Vượt đèn đỏ bị phạt bao nhiêu tiền?"
    # Lưu thành file .mp3 hoặc .wav và thả vào đường dẫn này:
    test_audio = "data/audio/test2.mp3" 
    
    # Tự động tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(test_audio), exist_ok=True)
    
    if os.path.exists(test_audio):
        raw_query, rag_query, warning = assistant.transcribe_for_rag(test_audio)
        print(f"\n📝 Câu hỏi Whisper nghe được: '{raw_query}'")
        if warning:
            print(f"⚠️ Lưu ý: {warning}")
        if rag_query:
            print(f"✅ Câu hỏi sau llm_rewriter để đưa vào RAG: '{rag_query}'")
        print("-" * 50)
        if rag_query:
            print("Tiếp theo, câu hỏi đã sửa sẽ được tự động đưa vào RAG để tìm luật...")
        else:
            print("Chưa đưa vào RAG vì câu hỏi chưa qua được llm_rewriter.")
    else:
        print(f"\n⚠️ Hệ thống đang đợi file âm thanh.")
        print(f"Hãy copy 1 file ghi âm của bạn vào đường dẫn: {test_audio} và chạy lại nhé!")
