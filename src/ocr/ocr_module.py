from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import os

class LegalDocumentReader:
    def __init__(self, use_gpu=False):
        """
        Khởi tạo hệ thống đọc ảnh.
        Mô hình 'vgg_transformer' cho độ chính xác cao nhất với tiếng Việt có dấu.
        """
        print("Đang tải trọng số mô hình VietOCR (VGG-Transformer)...")
        self.config = Cfg.load_config_from_name('vgg_transformer')
        
        # Cấu hình phần cứng
        # Nếu máy có GPU và đã cài CUDA, đổi thành 'cuda:0' để tăng tốc
        self.config['device'] = 'cuda:0' if use_gpu else 'cpu' 
        
        self.model = Predictor(self.config)
        print("✅ Tải mô hình OCR thành công!")

    def read_cropped_line(self, image_path):
        """
        Hàm này dùng để đọc TỪNG DÒNG CHỮ (dành cho VietOCR thuần).
        """
        try:
            img = Image.open(image_path)
            # Gọi model VietOCR để dịch ảnh thành text
            text = self.model.predict(img)
            return text
        except Exception as e:
            return f"[Lỗi đọc ảnh]: {str(e)}"

    def simulate_full_document_read(self, image_path):
        """
        Hàm giả lập đọc toàn bộ văn bản. 
        Trong thực tế đồ án, bạn nên tích hợp thêm bộ YOLO hoặc DBNet ở đây để cắt dòng.
        """
        # Tạm thời gọi trực tiếp để test luồng
        return self.read_cropped_line(image_path)

# --- KHỐI TEST THỬ NGHIỆM ---
if __name__ == "__main__":
    # Khởi tạo class (chỉnh use_gpu=True nếu máy bạn đã cài driver NVIDIA chuẩn trên Ubuntu)
    reader = LegalDocumentReader(use_gpu=False)
    
    # Tạo một thư mục test và bỏ một tấm ảnh mẫu vào đó
    test_img_path = "data/img/test-bien-ban.jpg"
    
    if os.path.exists(test_img_path):
        print(f"\nĐang quét ảnh: {test_img_path}")
        extracted_text = reader.simulate_full_document_read(test_img_path)
        print("\n--- KẾT QUẢ TRÍCH XUẤT ---")
        print(extracted_text)
        print("--------------------------")
    else:
        print(f"\nKhông tìm thấy ảnh test tại {test_img_path}.")