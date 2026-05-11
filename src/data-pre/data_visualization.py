import json

# 1. Khám phá kho luật
with open('data/raw/legal_text_retrieval/legal_corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

print("--- CẤU TRÚC FILE LUẬT ---")
print(f"Tổng số văn bản trong corpus: {len(corpus)}")
if len(corpus) > 0:
    first_law = corpus[0] # Lấy phần tử đầu tiên trong danh sách
    print(f"Các trường dữ liệu của 1 văn bản: {list(first_law.keys())}")
    # In thử nội dung text của trường đầu tiên (giới hạn 300 ký tự cho đỡ dài)
    print(f"Nội dung xem trước: {str(first_law)[:300]}...\n")

# 2. Khám phá file câu hỏi
with open('data/raw/legal_text_retrieval/train_question_answer.json', 'r', encoding='utf-8') as f:
    qa_data = json.load(f)

print("--- CẤU TRÚC FILE Q&A ---")
# Check an toàn xem file Q&A cấu trúc như thế nào (List hay Dict)
if isinstance(qa_data, dict) and 'items' in qa_data:
    items = qa_data['items']
elif isinstance(qa_data, list):
    items = qa_data
else:
    items = []

print(f"Tổng số câu hỏi: {len(items)}")
if len(items) > 0:
    first_qa = items[0]
    print(f"Các trường dữ liệu trong 1 câu hỏi: {list(first_qa.keys())}")
    print(f"Nội dung câu hỏi mẫu: {str(first_qa)[:300]}...")