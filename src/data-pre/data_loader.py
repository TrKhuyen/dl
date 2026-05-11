# src/data_loader.py
import json
import pandas as pd
import os
import re

def clean_text(text):
    # Chuẩn hóa khoảng trắng và xóa ký tự rỗng
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_rag_corpus(input_path, output_path):
    print(f"Đọc dữ liệu từ {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    chunks = []
    
    # Duyệt qua từng đạo luật
    for law in corpus:
        law_id = law.get('law_id', 'Unknown')
        articles = law.get('articles', [])
        
        # Duyệt qua từng Điều luật bên trong đạo luật đó
        for article in articles:
            article_id = article.get('article_id', '')
            title = clean_text(article.get('title', ''))
            text = clean_text(article.get('text', ''))
            
            # Bỏ qua các Điều bị trống hoặc đã bị bãi bỏ (chỉ có title mà không có text)
            if len(text) < 10: 
                continue
                
            # Gộp Tiêu đề và Nội dung thành một khối Context hoàn chỉnh để RAG đọc
            full_context = f"{title}\n{text}"
            
            chunks.append({
                'chunk_id': f"{law_id}_{article_id}",  # Tạo ID duy nhất (VD: 47/2011/tt-bca_7)
                'law_id': law_id,
                'article_id': article_id,
                'context': full_context
            })

    # Lưu thành file CSV để làm nguồn cho Vector Database
    df = pd.DataFrame(chunks)
    
    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✅ Xử lý hoàn tất!")
    print(f"✅ Tổng số Điều luật (chunks) thu được: {len(df)}")
    print(f"✅ Đã lưu file tại: {output_path}\n")
    
    return df

if __name__ == "__main__":
    input_file = "data/raw/legal_text_retrieval/legal_corpus.json"
    output_file = "data/processed/legal_corpus_chunks.csv"
    
    df_chunks = prepare_rag_corpus(input_file, output_file)
    
    print("--- XEM TRƯỚC 2 DÒNG DỮ LIỆU ĐẦU TIÊN ---")
    print(df_chunks.head(2).to_string())