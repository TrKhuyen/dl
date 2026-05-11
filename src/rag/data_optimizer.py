import pandas as pd
import re
import os

def optimize_legal_chunks(input_csv="data/processed/legal_corpus_chunks.csv", output_csv="data/processed/optimized_corpus.csv"):
    print("⏳ Đang tối ưu hóa (Chunking) dữ liệu Luật thành các đoạn nhỏ...")
    
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file {input_csv}. Vui lòng kiểm tra lại đường dẫn!")
        return

    optimized_data = []

    for _, row in df.iterrows():
        context = str(row['context'])
        # Giả sử tiêu đề điều luật nằm ở đầu (VD: "Điều 6. Xử phạt...")
        match = re.match(r"(Điều \d+\.[^\n]+)(.*)", context, re.DOTALL)
        
        if match:
            title = match.group(1).strip()
            body = match.group(2).strip()
            
            # Cắt nhỏ dựa trên các đầu mục: "1.", "2.", "a)", "b)"...
            # Sử dụng Regex để tìm các đoạn bắt đầu bằng số hoặc chữ cái kèm dấu chấm/ngoặc
            chunks = re.split(r'\n(?=\d+\.|[a-z]\))', body)
            
            for chunk in chunks:
                chunk = chunk.strip()
                if chunk:
                    # Gắn tên Điều luật vào đầu mỗi khúc nhỏ để giữ ngữ cảnh
                    optimized_data.append({"context": f"{title} - {chunk}"})
        else:
            # Nếu không tìm thấy cấu trúc "Điều X", giữ nguyên
            optimized_data.append({"context": context})

    optimized_df = pd.DataFrame(optimized_data)
    
    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    optimized_df.to_csv(output_csv, index=False)
    
    print(f"✅ Đã băm nhỏ thành công: Từ {len(df)} Điều luật -> {len(optimized_df)} đoạn nhỏ (Khoản/Điểm)!")
    print(f"💾 Đã lưu tại: {output_csv}")

if __name__ == "__main__":
    optimize_legal_chunks()