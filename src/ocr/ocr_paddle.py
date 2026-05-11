# src/ocr_paddle.py
import os
import logging
from paddleocr import PaddleOCR

# Tắt các log cảnh báo rác của Paddle để terminal sạch sẽ hơn
logging.getLogger("ppocr").setLevel(logging.WARNING)

class AdvancedLegalReader:
    def __init__(self, use_gpu=False):
        """
        Khởi tạo hệ thống OCR bằng PaddleOCR.
        - lang='vi': Hỗ trợ tiếng Việt.
        - use_angle_cls=True: Tự động lật thẳng ảnh nếu người dùng chụp ngược.
        """
        print("Đang tải khối mô hình Đa phương thức (Detection + Recognition)...")
        self.ocr = PaddleOCR(use_textline_orientation=True, lang='vi', enable_mkldnn=False)
        print("✅ Tải hệ thống thành công!")

    def process_document(self, image_path, confidence_threshold=0.85):
        """
        Quét toàn bộ trang tài liệu và ứng xử thông minh với chữ viết tay mờ.
        """
        if not os.path.exists(image_path):
            return "Lỗi: Không tìm thấy file ảnh.", False

        print(f"\nĐang quét và phân rã các dòng chữ trong ảnh: {image_path}...")
        
        result = self.ocr.predict(image_path)
        
        # Đảm bảo result là list
        if not isinstance(result, list):
            result = list(result)
            
        if len(result) == 0:
            return "Không phát hiện thấy văn bản nào trong hình ảnh.", False

        res_0 = result[0]
        
        # Lấy danh sách text và danh sách điểm số từ PaddleX Object/Dict
        if isinstance(res_0, dict):
            texts = res_0.get('rec_texts', [])
            scores = res_0.get('rec_scores', [])
        else:
            texts = getattr(res_0, 'rec_texts', [])
            scores = getattr(res_0, 'rec_scores', [])

        extracted_lines = []
        has_bad_handwriting = False

        # Dùng zip() để ghép từng cặp (chữ, điểm) lại với nhau
        for text, score in zip(texts, scores):
            if score >= confidence_threshold:
                extracted_lines.append(text)
            else:
                has_bad_handwriting = True
                extracted_lines.append(f"[??? Chữ mờ/Viết tay: {text} ???]")

        # Nối các dòng lại thành một văn bản hoàn chỉnh
        full_text = "\n".join(extracted_lines)
        
        return full_text, has_bad_handwriting

# --- KHỐI TEST THỬ NGHIỆM TỜ BIÊN BẢN CỦA BẠN ---
if __name__ == "__main__":
    reader = AdvancedLegalReader()
    
    # Trỏ đường dẫn tới file ảnh biên bản bạn vừa gửi
    test_img_path = "data/img/test-bien-ban.jpg" 
    
    final_text, is_blurry = reader.process_document(test_img_path)
    
    print("\n--- 📄 KẾT QUẢ TRÍCH XUẤT TỪ ẢNH ---")
    print(final_text)
    print("-" * 38)
    
    # Giả lập luồng hoạt động của Chatbot
    if is_blurry:
        print("\n🤖 CHATBOT PHẢN HỒI:")
        print("Xin lỗi, hình ảnh biên bản của bạn có một số đoạn chữ viết tay hơi mờ (được đánh dấu [???]).")
        print("Để tôi có thể tìm chính xác điều luật xử phạt, bạn có thể vui lòng gõ lại lỗi vi phạm cụ thể không?")
    else:
        print("\n🤖 CHATBOT PHẢN HỒI:")
        print("Đã đọc rõ biên bản. Đang tiến hành tra cứu hệ thống RAG...")
        # Ở đây bạn sẽ lấy 'final_text' đẩy sang Giai đoạn 3 (ChromaDB + SBERT)