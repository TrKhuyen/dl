---
phase: testing
title: Testing Strategy — Streamlit Chat UI
feature: streamlit-ui
---

# Testing Strategy — Streamlit Chat UI

## Test Results (2026-05-11)

```
29 passed in 4.72s — tests/test_gemini_client.py
```

## Test Coverage Goals

| Module | Coverage | Test file |
|--------|---------|-----------|
| `src/ui/gemini_client.py` | ~95% (branches) | `tests/test_gemini_client.py` |
| `app.py` (load_rag, process_query) | Key paths | `tests/test_gemini_client.py` |
| `app.py` (UI render functions) | Manual only | Smoke test |

## Unit Tests — `tests/test_gemini_client.py`

### GeminiLegalAssistant — Init (5 tests)
- [x] Khởi tạo với API key trực tiếp
- [x] Đọc key từ biến môi trường (`api_key`)
- [x] Đọc key từ file `.env`
- [x] Raise `ValueError` khi không có key từ bất kỳ nguồn nào
- [x] Env var rỗng → fallback sang `.env` file

### GeminiLegalAssistant — `new_chat()` (2 tests)
- [x] Xóa sạch `_history`
- [x] Idempotent — gọi nhiều lần vẫn an toàn

### GeminiLegalAssistant — `_build_prompt()` (5 tests)
- [x] Không có context → trả về query gốc
- [x] Context list rỗng → trả về query gốc
- [x] Một context → prompt chứa `[Nguồn 1]`
- [x] Nhiều context → chứa `[Nguồn 1..N]`
- [x] Context block xuất hiện trước câu hỏi

### GeminiLegalAssistant — `answer()` (6 tests)
- [x] Happy path — trả về text, lưu vào history
- [x] Với RAG context — prompt chứa context
- [x] Nhiều lần gọi — history tích lũy đúng
- [x] API lỗi → trả về error string, không raise
- [x] API lỗi → **rollback history** (không rò rỉ user message)
- [x] `response.text = None` → trả về chuỗi rỗng, không crash

### GeminiLegalAssistant — `answer_stream()` (6 tests)
- [x] Yield đủ tất cả chunks
- [x] Lưu toàn bộ text (nối chunks) vào history sau khi xong
- [x] Chunk `text=None` bị bỏ qua, không yield
- [x] API lỗi → yield error string, không raise
- [x] API lỗi → **rollback history**
- [x] `answer_stream()` + `answer()` dùng chung history (multi-turn)

### GeminiLegalAssistant — Mid-conversation `new_chat()` (1 test)
- [x] `new_chat()` giữa hội thoại → history rỗng, model không thấy lịch sử cũ

### `app.load_rag()` (2 tests)
- [x] CSV không tồn tại → trả `None`
- [x] `HybridRAGSystem` lỗi → trả `None` gracefully

### `app.process_query()` (2 tests)
- [x] Query rỗng/khoảng trắng → early return, không gọi Gemini
- [x] `gemini = None` → `st.error()` được gọi với message rõ ràng

## End-to-End Tests (Manual)

- [x] Flow 1: Nhập text → câu trả lời hiển thị đúng (**verified 2026-05-11**)
- [x] Flow 2: ~~Upload audio~~ — SKIPPED (out of scope)
- [x] Flow 3: ~~Upload image~~ — SKIPPED (out of scope)
- [x] Flow 4: Clear conversation → lịch sử xóa sạch (**verified visually**)
- [x] Flow 5: RAG không có data → app không crash, fallback hoạt động (**🟡 badge hiển thị đúng**)

## Manual Testing Checklist

**UI:**
- [x] Dark mode hoạt động
- [x] Chat bubble user bên trái, assistant bên phải (Streamlit standard)
- [x] Sidebar toggle mượt
- [x] "Nguồn tham chiếu" expander hiển thị khi RAG active
- [x] Welcome screen với 4 câu hỏi gợi ý khi chưa có chat
- [x] Nút "🗑️ Xóa lịch sử" hoạt động

**Performance:**
- [x] Startup < 5 giây (chỉ import Streamlit + Gemini client)
- [x] Response text < 10 giây (streaming Gemini)
- [x] Không còn Whisper model load (đã loại bỏ)

## Smoke Test Command

```bash
conda activate dl
pytest tests/test_gemini_client.py -v
# Expected: 29 passed

# UI smoke test:
streamlit run app.py
# Mở http://localhost:8501
# Nhập: "vượt đèn đỏ phạt bao nhiêu tiền?"
# Expected: Câu trả lời streaming, trích dẫn Nghị định 100/2019
```

## Deferred / Out of Scope

| Test | Lý do defer |
|------|-------------|
| Streamlit UI render unit tests | Cần `AppTest` (Streamlit 1.38+) hoặc Selenium; phức tạp với mock |
| `HybridRAGSystem` full integration | Cần file CSV và model FAISS — slow test |
| RAG → Gemini → response end-to-end | Cần API key thật — cost/rate limit concern |

## Coverage Gaps (acceptable for POC)

- `render_sidebar()`, `render_chat()`, `render_input()` — render logic phụ thuộc Streamlit runtime
- `_load_from_env_file()` với `.env` file có comment/empty lines — minor edge case
