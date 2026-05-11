import whisper
import os
import warnings

# Tắt các cảnh báo rác của thư viện FP16 để terminal sạch sẽ
warnings.filterwarnings("ignore", category=UserWarning)

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

        print(f"\n🎙️ Đang lắng nghe và giải mã file: {audio_path}...")
        
        # task='transcribe': Dịch tiếng nói
        # language='vi': Khóa ngôn ngữ là tiếng Việt để tăng độ chuẩn xác
        result = self.model.transcribe(audio_path, language='vi', task='transcribe')
        
        text = result['text'].strip()
        return text

# --- KHỐI TEST CHẠY THỬ ---
if __name__ == "__main__":
    assistant = LegalVoiceAssistant(model_size="small")
    
    # Bạn hãy lấy điện thoại tự ghi âm 1 câu hỏi luật (khoảng 5-10 giây)
    # Ví dụ: "Vượt đèn đỏ bị phạt bao nhiêu tiền?"
    # Lưu thành file .mp3 hoặc .wav và thả vào đường dẫn này:
    test_audio = "data/audio/test3.mp3" 
    
    # Tự động tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(test_audio), exist_ok=True)
    
    if os.path.exists(test_audio):
        query_text = assistant.transcribe(test_audio)
        print(f"\n📝 Câu hỏi AI nghe được: '{query_text}'")
        print("-" * 50)
        print("Tiếp theo, câu hỏi này sẽ được tự động đưa vào ChromaDB để tìm luật...")
    else:
        print(f"\n⚠️ Hệ thống đang đợi file âm thanh.")
        print(f"Hãy copy 1 file ghi âm của bạn vào đường dẫn: {test_audio} và chạy lại nhé!")