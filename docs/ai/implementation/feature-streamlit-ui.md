---
phase: implementation
title: Implementation Guide — Streamlit Chat UI
feature: streamlit-ui
---

# Implementation Guide — Streamlit Chat UI

## Development Setup

```bash
# 1. Activate conda environment (bắt buộc)
conda activate dl

# 2. Cài thêm dependencies
pip install streamlit google-generativeai

# 3. Chạy app
streamlit run app.py
```

## Code Structure

```
/home/sakana/Code/PTIT/NLP/dl/
├── app.py                          # [NEW] Entry point Streamlit
├── .env                            # Gemini API key (đã có)
├── requirements.txt                # [MODIFY] Thêm streamlit, google-generativeai
├── src/
│   ├── ui/
│   │   └── gemini_client.py        # [NEW] Gemini API wrapper với RAG
│   ├── rag/vector_db.py            # [REUSE] HybridRAGSystem
│   ├── whisper/audio_processor.py  # [REUSE] LegalVoiceAssistant
│   └── ocr/ocr_paddle.py           # [REUSE] PaddleOCR
└── .streamlit/
    └── config.toml                 # [NEW] Theme configuration
```

## Implementation Notes

### Core Features

**1. Gemini Client (`src/ui/gemini_client.py`)**
- Load `api_key` từ `.env` bằng `dotenv` hoặc trực tiếp
- System prompt: "Bạn là trợ lý pháp lý luật giao thông Việt Nam..."
- Nếu có RAG context → thêm vào prompt, yêu cầu trích dẫn

**2. Chat UI (`app.py`)**
- `st.session_state.messages` = list of `{role, content, sources?}`
- Render bubble: user = `st.chat_message("user")`, assistant = `st.chat_message("assistant")`
- Input: `st.chat_input()` + `st.file_uploader()` trong sidebar

**3. Voice Input**
- Nhận `.mp3/.wav` từ `st.file_uploader`
- Lưu tạm vào `tempfile.NamedTemporaryFile`
- Gọi `LegalVoiceAssistant.transcribe(tmp_path)`
- Xóa temp file sau khi transcribe

**4. OCR Input**
- Nhận ảnh từ `st.file_uploader`
- Dùng PaddleOCR hoặc fallback `ocr_paddle.py`
- Extracted text → đưa vào chat như query

## Integration Points

- `.env` → `api_key` → `GeminiLegalAssistant`
- `HybridRAGSystem` → init lazy khi `optimized_corpus.csv` tồn tại
- Whisper → init khi user upload audio (lazy)

## Error Handling

- RAG data missing → warning banner + fallback mode
- Gemini API error → hiển thị lỗi trong chat bubble
- Whisper timeout → thông báo và cho nhập text thủ công
- OCR error → thông báo và cho nhập text thủ công

## Performance Considerations

- `@st.cache_resource` cho RAG system (tránh reload mỗi rerun)
- `@st.cache_resource` cho Whisper model
- Gemini streaming response (nếu muốn typewriter effect)
