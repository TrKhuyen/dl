---
phase: planning
title: Task Breakdown — Streamlit Chat UI
feature: streamlit-ui
---

# Planning — Streamlit Chat UI

## Milestones

- [x] M1: Cài đặt môi trường & skeleton app chạy được
- [x] M2: Chat cơ bản với Gemini API hoạt động
- [x] ~~M3: Tích hợp Whisper (voice input)~~ — SKIPPED (out of scope)
- [x] ~~M4: Tích hợp OCR (image input)~~ — SKIPPED (out of scope)
- [x] M5: RAG integration + polish UI

## Task Breakdown

### Phase 1: Foundation ✅
- [x] 1.1: Thêm `streamlit`, `google-genai`, `python-dotenv` vào `requirements.txt`
- [x] 1.2: Tạo `src/ui/gemini_client.py` — Gemini 2.5 Flash wrapper (streaming + multi-turn)
- [x] 1.3: Tạo `app.py` ở root với skeleton layout

### Phase 2: Core Chat Feature ✅
- [x] 2.1: Implement session state cho lịch sử chat (`st.session_state.messages`)
- [x] 2.2: UI — Header gradient, Sidebar, Chat bubbles, Input bar, Welcome screen
- [x] 2.3: Kết nối Gemini API trả lời câu hỏi (tested live ✅)

### Phase 3: Voice & OCR Input ~~SKIPPED~~
- [x] ~~3.1: Tích hợp Whisper~~ — SKIPPED per user decision
- [x] ~~3.2: Tích hợp PaddleOCR~~ — SKIPPED per user decision

### Phase 4: RAG Integration ✅
- [x] 4.1: Lazy-load `HybridRAGSystem` với `@st.cache_resource` (graceful fallback khi không có data)
- [x] 4.2: Hiển thị "📚 Nguồn tham chiếu" expander trong chat bubble
- [x] 4.3: System prompt tiếng Việt + RAG context injection vào Gemini

### Phase 5: Polish ✅
- [x] 5.1: Custom CSS dark theme — gradient title, glassmorphism, source badges
- [x] 5.2: Error handling — Gemini init error banner, RAG fallback warning, streaming error
- [x] 5.3: README.md hướng dẫn chạy ✅

### Cleanup (post-review) ✅
- [x] C1-C4: Xóa Voice/Image mode, `load_whisper()`, `INPUT_MODES`, `tempfile`/`os` imports thừa
- [x] Sidebar footer cập nhật → "Powered by Gemini 2.5 Flash + Hybrid RAG"
- [x] `process_query()` đơn giản hóa — bỏ `input_type` param

## Dependencies

- `google-genai>=2.0` — đã cài ✅
- `streamlit>=1.35` — đã cài ✅
- `python-dotenv` — đã cài ✅
- `.env` với `api_key` — có sẵn ✅
- Data (optional): `data/processed/optimized_corpus.csv` cho RAG

## Risks & Mitigation (resolved)

| Risk | Mitigation | Status |
|------|-----------|--------|
| RAG data chưa có | Fallback sang Gemini thuần | ✅ Handled |
| `google-generativeai` deprecated | Dùng `google-genai` SDK v2 | ✅ Fixed |
| Model 404 error | Dùng `gemini-2.5-flash` (confirmed available) | ✅ Fixed |
