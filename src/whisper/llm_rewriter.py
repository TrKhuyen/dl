import os

from openai import OpenAI

class LocalQueryRewriter:
    def __init__(self, model=None, base_url=None):
        """
        Khởi tạo kết nối đến máy chủ Ollama đang chạy ngầm trên máy tính.
        Không tốn tiền, không cần mạng!
        """
        self.model = model or os.getenv("OLLAMA_REWRITER_MODEL", "qwen2.5:7b")
        # Trỏ thẳng vào cổng 11434 mặc định của hệ thống Ollama trên máy bạn
        self.client = OpenAI(
            base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama" # Bắt buộc phải điền nhưng điền chữ gì cũng được (API Key ảo)
        )

    def correct_query(self, noisy_query):
        print(f"🧠 Đang nhờ Qwen-Local phân tích: '{noisy_query}'...")
        
        # Prompt Engineering: Đóng vai chuyên gia pháp lý
        # Kỹ thuật Few-Shot Prompting: Dạy AI bằng ví dụ
        system_prompt = """
        Bạn là một hệ thống AI tiền xử lý ngôn ngữ (Text-Normalizer). 
        Nhiệm vụ DUY NHẤT của bạn là SỬA LỖI CHÍNH TẢ và chuẩn hóa câu nói của người dùng.
        
        QUY TẮC THÉP (Bắt buộc tuân thủ):
        1. CHỈ in ra câu đã được sửa lỗi chính tả. KHÔNG trả lời câu hỏi. KHÔNG giải thích. KHÔNG bịa thêm thông tin.
        2. Nhận diện các lỗi do nhận diện giọng nói (STT errors) hoặc ngọng vùng miền:
           - "vật đen đỏ" -> "vượt đèn đỏ"
           - Lỗi "n/l": "nái xe" -> "lái xe", "nỗi" -> "lỗi"
           - Lỗi "r/d/gi": "rịu" -> "rượu", "dấy phép" -> "giấy phép"
        3. Nếu câu hỏi KHÔNG liên quan đến luật giao thông (ví dụ: điện thoại), chỉ sửa chính tả bình thường, tuyệt đối không cố bóp méo nó thành từ khóa giao thông.
        4. Nếu thông tin là số liệu thì chỉ sửa lỗi chính tả mà KHÔNG được thay đổi ý nghĩa của con số đó. Lưu ý rằng "phẩy" là "," (dấu phẩy trong tiếng Việt) (ví dụ: "nồng đọ cồn không phẩy năm" -> "nồng độ cồn không phẩy năm"), giữ nguyên ý nghĩa của con số và chữ "phẩy".
                
        VÍ DỤ CHUẨN:
        Input: vật đen đỏ phát bao nhiêu tiền
        Output: vượt đèn đỏ phạt bao nhiêu tiền
        
        Input: nái xe khi say rịu bị phạt thế nào
        Output: lái xe khi say rượu bị phạt thế nào
        
        Input: địn hoại bị rơi xún nước có xửa được không
        Output: điện thoại bị rơi xuống nước có sửa được không
        
        Input: luật mua bán xe cũ có quy địn gì thêm không
        Output: luật mua bán xe cũ có quy định gì thêm không
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Câu bị lỗi: {noisy_query}"}
                ],
                temperature=0.0 # Nhiệt độ bằng 0 để đảm bảo tính logic tuyệt đối, không sáng tạo vớ vẩn
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ Lỗi kết nối Ollama hoặc chưa pull model `{self.model}`: {e}"

# --- KHỐI TEST ---
if __name__ == "__main__":
    rewriter = LocalQueryRewriter() 
    
    test_cases = [
        "vật đen đỏ phát bao nhiêu tiền Việt Nam",
        "không có dấy phép nái xe bị phạt thế nào",
        "nồng đọ cồn một phẩy chín bị phạt bao nhiêu tiền",
        "địn hoại bị rơi xún nước có xửa được không",
        "nái xe khi say rịu bị phạt bao nhiêu tiền",
        "luật mua bán xe cũ có quy địn gì thêm không"
    ]
    
    print("\n--- 🧠 TEST LOCAL LLM REWRITER ---")
    for error_query in test_cases:
        corrected = rewriter.correct_query(error_query)
        print(f"❌ Whisper nghe sai: {error_query}")
        print(f"✅ Qwen-Local sửa lại: {corrected}\n")
