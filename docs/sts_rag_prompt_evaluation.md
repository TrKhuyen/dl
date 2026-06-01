# STS for RAG Prompt Evaluation — Detailed Report

## 1. Mục tiêu

Báo cáo này mô tả cách sử dụng **Semantic Textual Similarity (STS)** để đánh giá chất lượng câu trả lời của LLM trong quy trình RAG (Retrieval-Augmented Generation). Với dữ liệu đầu vào dạng `jsonl` có các trường:
- `instruction`
- `context`
- `response`

mục tiêu là so sánh **response của LLM** với **response tham chiếu** (reference response) về mặt ngữ nghĩa thay vì so sánh từng chữ.

> Nếu bạn đang dùng RAG và muốn đánh giá prompt chain, STS giúp trả lời câu hỏi: “LLM có hiểu đúng ý chính và trả lời tương đương về mặt nghĩa không?”

---

## 2. Vì sao STS phù hợp cho đánh giá RAG prompt

### 2.1 Lợi ích chính
- So sánh ý nghĩa hơn là so sánh từ ngữ: đánh giá chính xác hơn khi câu trả lời dùng từ khác nhưng vẫn giữ cùng nội dung.
- Hữu ích với câu hỏi mở, câu trả lời dài, hoặc khi LLM đưa thêm chi tiết mà vẫn không thay đổi ý chính.
- Giúp phát hiện prompt drift: nếu LLM trả lời lan man, semantic similarity thấp cho thấy cần điều chỉnh prompt hoặc retrieval context.

### 2.2 Đặc điểm của bài toán RAG
- `instruction`: yêu cầu người dùng gửi vào
- `context`: tài liệu tìm kiếm được bởi retrieval
- `response`: output của LLM dựa trên prompt và context

Khi so sánh `response` với reference, STS cho biết mức độ:
- đúng chủ đề
- đúng nghĩa
- không bị lệch ý

---

## 3. Quy trình đánh giá bằng STS

### 3.1 Chuẩn bị dữ liệu
Chuẩn bị file `jsonl` có định dạng mỗi dòng:

```json
{
  "instruction": "Hãy giải thích tại sao ...",
  "context": "... tài liệu dẫn nguồn ...",
  "response": "... câu trả lời gốc ...",
}
```

Lưu ý:
- Dùng STS để đo độ ổn định giữa prompt của llm với response.

### 3.2 Mô hình STS đề xuất
Dùng `sentence-transformers` để tạo embedding:
- Bi-encoder model tiêu chuẩn: keepitreal/vietnamese-sbert

### 3.3 Các bước thực thi
1. Load JSONL và tách cặp text:
   - `(response, reference_response)`
2. Encode cả hai câu bằng SentenceTransformers.
3. Tính cosine similarity giữa các cặp embedding.
4. Lưu kết quả vào file mới hoặc tính stats:
   - mean similarity
   - median similarity
   - tỷ lệ cặp >= threshold
   - phân bố similarity theo bucket

---

## 4. Công thức đánh giá

### 4.1 Cosine similarity

Nếu `u` và `v` là vector embedding của hai câu:

```math
cosine(u, v) = \frac{u \cdot v}{\|u\| \|v\|}
```

Giá trị nằm trong khoảng `[-1, 1]`, với câu giống nghĩa càng cao thì càng gần `1`.

### 4.2 Ngưỡng đánh giá
Các ngưỡng tham khảo:
- `>= 0.85`: câu trả lời rất tương đồng về nghĩa
- `0.70 – 0.85`: tương đồng tốt, có thể có khác biệt nhỏ
- `< 0.70`: khác biệt đáng kể, cần kiểm tra lại prompt/retrieval

Ngưỡng có thể điều chỉnh theo task và dataset.

---

## 5. Đề xuất cấu trúc báo cáo STS

### 5.1 Metrics cơ bản
- `avg_similarity`: trung bình similarity trên toàn bộ dataset
- `median_similarity`
- `std_similarity`
- `percent_above_0.85`
- `percent_above_0.70`
- `percent_below_0.60`

### 5.2 Phân tích theo nhóm
- Nhóm theo nguồn `context` hoặc `instruction` tương tự
- Nhóm theo length của response
- Nhóm theo loại câu hỏi: factual / explanation / summary

### 5.3 Phát hiện prompt issues
Dùng STS để gắn nhãn các tình huống:
- `high_similarity` nhưng câu trả lời vẫn sai factual: lỗi retrieval hoặc reference không phù hợp
- `low_similarity` và prompt chưa rõ ràng: cần cải thiện prompt instruction
- `low_similarity` nhưng prompt rõ ràng: có thể do toxicity/hallucination hoặc context không đủ

---

## 6. Minh họa code mẫu

### 6.1 Python script đánh giá STS

```python
from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

records = []
with open("data/rag_prompts.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        records.append(item)

responses = [r["response"] for r in records]
refs = [r["reference_response"] for r in records]

emb_resp = model.encode(responses, convert_to_numpy=True, normalize_embeddings=True)
emb_refs = model.encode(refs, convert_to_numpy=True, normalize_embeddings=True)

sims = (emb_resp * emb_refs).sum(axis=1)

for item, sim in zip(records, sims):
    item["sts_similarity"] = float(sim)

with open("data/rag_prompts_with_sts.jsonl", "w", encoding="utf-8") as f:
    for item in records:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("avg", float(np.mean(sims)))
print("pct>=0.85", float((sims >= 0.85).mean()))
print("pct<0.70", float((sims < 0.70).mean()))
```

### 6.2 Gợi ý workflow
- Gán `sts_similarity` cho mỗi dòng
- Phân loại response thành `good / acceptable / bad` theo threshold
- Đánh giá prompt version: nếu prompt mới tăng `avg_similarity` và giảm `pct<0.70`, prompt đang tốt hơn

---

## 7. Ứng dụng trong đánh giá prompt RAG

### 7.1 So sánh prompt variant
- Với cùng `instruction` và `context`, sinh response bằng nhiều prompt khác nhau.
- Dùng STS để so sánh response mới với reference.
- Một prompt tốt thường tạo ra response có similarity cao hơn và ít trường hợp similarity thấp.

### 7.2 Kiểm tra retrieval quality
- Nếu response sai/ngớ ngẩn và similarity thấp, có thể là do `context` không phù hợp.
- Kết hợp với retrieval score và document relevance để tìm nguyên nhân: prompt tốt nhưng context xấu.

### 7.3 Phát hiện drift/prompt degradation
- So sánh model output theo thời gian: nếu similarity giảm với cùng reference, có thể do model cập nhật prompt không phù hợp hoặc retrieval dataset drift.

---

## 8. Kết luận và khuyến nghị

### 8.1 Kết luận
Dùng STS để đánh giá câu trả lời LLM trong quy trình RAG giúp đo lường chất lượng semantic, đặc biệt hữu ích với câu trả lời mở và multi-turn. Đây là phép đo thứ cấp quan trọng bên cạnh các chỉ số ví dụ như BLEU, ROUGE, hay human evaluation.

### 8.2 Khuyến nghị
- Dùng STS như một chỉ số nhanh để so sánh prompt version.
- Kết hợp STS với:
  - retrieval score
  - token-level metrics (perplexity, logprobs)
  - human feedback cho trường hợp low similarity.
- Nếu có thể, fine-tune hoặc chọn model STS phù hợp domain để tăng độ nhạy cảm với semantic nuance.

---

## 9. Tùy chọn mở rộng

- So sánh không chỉ `response` mà còn `instruction + context + response` tổng thể.
- Dùng `cross-encoder` để đánh giá chính xác hơn cặp câu `response vs reference_response` trên tập mẫu quan trọng.
- Dùng STS để kích hoạt human review cho response có similarity thấp nhưng importance cao.
