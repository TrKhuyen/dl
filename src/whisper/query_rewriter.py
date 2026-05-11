import os
import pandas as pd
from symspellpy import SymSpell
from underthesea import word_tokenize
import re

class LegalSpellChecker:
    def __init__(self, dict_path="data/processed/legal_dict.txt"):
        """
        Khởi tạo bộ sửa lỗi SymSpell.
        Thuật toán này cần một file từ điển chứa các từ khóa và tần suất xuất hiện.
        """
        self.dict_path = dict_path
        # max_dictionary_edit_distance=2: Cho phép sai tối đa 2 ký tự (VD: "vật" -> "vượt")
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self.load_or_build_dictionary()

    def load_or_build_dictionary(self):
        """
        Kiểm tra xem file từ điển đã có chưa. Nếu chưa có, tự động quét corpus để xây dựng.
        """
        if os.path.exists(self.dict_path):
            print("Đang tải từ điển pháp lý vào RAM...")
            self.sym_spell.load_dictionary(self.dict_path, term_index=0, count_index=1, encoding="utf-8")
            print(f"✅ Đã tải xong từ điển từ {self.dict_path}")
        else:
            print("⚠️ Chưa có từ điển. Đang tự động trích xuất từ vựng từ Corpus Luật...")
            self._build_dictionary_from_corpus()

    def _build_dictionary_from_corpus(self, corpus_path="data/processed/legal_corpus_chunks.csv"):
        """
        Hàm Data Engineering: Quét hàng ngàn điều luật để đếm tần suất từ vựng.
        """
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Không tìm thấy {corpus_path}. Hãy chạy data_loader.py trước!")

        df = pd.read_csv(corpus_path)
        word_freq = {}

        print("Đang phân tích NLP tách từ (Có thể mất 1-2 phút cho dữ liệu lớn)...")
        for text in df['context'].dropna():
            # Xóa dấu câu, chuyển chữ thường
            clean_text = re.sub(r'[^\w\s]', '', str(text).lower())
            
            # Dùng Underthesea để tách cụm từ ghép (VD: "vượt_đèn_đỏ", "giấy_phép_lái_xe")
            tokens = word_tokenize(clean_text, format="text").split()
            
            for token in tokens:
                # Đổi dấu "_" thành khoảng trắng để lưu vào từ điển
                word = token.replace('_', ' ')
                # Chỉ lấy các từ dài hơn 1 ký tự và không phải số
                if len(word) > 1 and not word.isnumeric():
                    word_freq[word] = word_freq.get(word, 0) + 1

        # Lưu ra file txt theo format chuẩn của SymSpell: "từ_khóa tần_suất"
        os.makedirs(os.path.dirname(self.dict_path), exist_ok=True)
        with open(self.dict_path, "w", encoding="utf-8") as f:
            for word, freq in word_freq.items():
                f.write(f"{word} {freq}\n")
        
        print("✅ Đã xây dựng và lưu từ điển thành công!")
        # Tải thẳng vào RAM sau khi tạo
        self.sym_spell.load_dictionary(self.dict_path, term_index=0, count_index=1, encoding="utf-8")

    def correct_query(self, noisy_query):
        """
        Sửa lỗi chính tả một câu hoàn chỉnh dựa trên ngữ cảnh từ điển.
        """
        # Chuyển về chữ thường để chuẩn hóa
        noisy_query = noisy_query.lower()
        
        # Hàm lookup_compound của SymSpell cực mạnh: nó tự cắt câu, 
        # tìm lỗi sai và nắn lại dựa trên xác suất tần suất từ vựng ghép.
        suggestions = self.sym_spell.lookup_compound(noisy_query, max_edit_distance=2, ignore_non_words=True)
        
        if suggestions:
            # Lấy kết quả có độ tin cậy cao nhất
            return suggestions[0].term
        return noisy_query

# --- KHỐI TEST ---
if __name__ == "__main__":
    checker = LegalSpellChecker()
    
    # Giả lập các câu bị Whisper nghe nhầm
    test_cases = [
        "vật đen đỏ phát bao nhiêu tiền",
        "không có dấy phép nái xe bị phạt thế nào",
        "nồng đọ cồn không phẩy năm"
    ]
    
    print("\n--- 🛠️ TEST BỘ NẮN TRUY VẤN (QUERY REWRITER) ---")
    for error_query in test_cases:
        corrected = checker.correct_query(error_query)
        print(f"❌ Whisper nghe sai : {error_query}")
        print(f"✅ SymSpell sửa lại: {corrected}\n")