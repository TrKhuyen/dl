# 🚦 Trợ lý Pháp lý AI

Hệ thống hỏi đáp luật giao thông đường bộ Việt Nam, sử dụng **Gemini 2.5 Flash** và **Hybrid RAG** (BM25 + FAISS + CrossEncoder).

---

## 🚀 Chạy ứng dụng

```bash
# 1. Kích hoạt môi trường conda
conda activate dl

# 2. Chạy Streamlit app
streamlit run app.py
```

Mở trình duyệt tại: **http://localhost:8501**

---

## 📋 Yêu cầu

### API Key
Tạo file `.env` ở thư mục gốc:
```
api_key=YOUR_GEMINI_API_KEY
```

### Cài đặt thư viện
```bash
conda activate dl
pip install -r requirements.txt
```

---

## 🏗️ Cấu trúc dự án

```
.
├── app.py                          # Streamlit UI chính
├── .env                            # Gemini API key
├── requirements.txt
├── .streamlit/
│   └── config.toml                 # Dark theme config
└── src/
    ├── ui/
    │   └── gemini_client.py        # Gemini 2.5 Flash wrapper
    ├── rag/
    │   ├── vector_db.py            # Hybrid RAG (BM25 + FAISS + CrossEncoder)
    │   └── data_optimizer.py       # Data chunking
    ├── whisper/
    │   ├── audio_processor.py      # Whisper STT
    │   ├── llm_rewriter.py         # LLM query correction (Ollama)
    │   └── query_rewriter.py       # SymSpell spell checker
    ├── ocr/
    │   ├── ocr_module.py           # VietOCR
    │   └── ocr_paddle.py           # PaddleOCR
    └── data-pre/
        ├── data_loader.py
        └── data_visualization.py
```

---

## ✨ Tính năng

| Tính năng | Mô tả |
|-----------|-------|
| 💬 Chat text | Hỏi đáp luật giao thông bằng tiếng Việt |
| 🤖 Gemini 2.5 Flash | Trả lời streaming, multi-turn conversation |
| 🔍 Hybrid RAG | Tìm context từ CSDL luật (BM25 + FAISS HNSW + CrossEncoder reranker) |
| 📚 Nguồn tham chiếu | Hiển thị đoạn luật gốc được dùng để trả lời |
| 🌙 Dark mode | Giao diện gradient, glassmorphism |
| ❓ Câu hỏi gợi ý | Welcome screen với 4 câu hỏi mẫu |
| 🗑️ Xóa lịch sử | Reset hội thoại |

---

## 🔍 Kích hoạt RAG (tùy chọn)

RAG cần file dữ liệu luật. Nếu chưa có, hệ thống vẫn hoạt động với Gemini thuần.

```bash
conda activate dl

# Bước 1: Chuẩn bị dữ liệu (cần có legal_corpus_chunks.csv)
python src/data-pre/data_loader.py

# Bước 2: Tối ưu chunks
python src/rag/data_optimizer.py

# Bước 3: Chạy app — RAG sẽ tự load
streamlit run app.py
```

Khi RAG active, sidebar hiển thị badge **🟢 RAG: Đang hoạt động**.

---

## 🛠️ Stack

- **UI:** Streamlit 1.35+
- **LLM:** Google Gemini 2.5 Flash (`google-genai` SDK v2)
- **RAG:** FAISS HNSW + BM25Okapi + CrossEncoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`)
- **Embeddings:** `keepitreal/vietnamese-sbert`
- **Môi trường:** Conda (`dl` environment)
