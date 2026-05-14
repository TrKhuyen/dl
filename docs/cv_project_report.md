# CV Project Report

## CV-ready Entry

**AI Legal Assistant for Vietnamese Law**                                      **May 2026 - Present**

**Solo project in NLP / Deep Learning Subject**

- Built a **Streamlit chat application** for Vietnamese legal question answering, with side-by-side responses from **Gemini 2.5 Flash** and **Qwen 2.5 7B Local**.
- Processed a raw Vietnamese legal corpus into **122K article-level chunks**, optimized it into **61K clause-level contexts**, and prepared **29K fine-tuning samples** plus a **39K-term legal dictionary**.
- Implemented a **Hybrid RAG pipeline** using Vietnamese SBERT embeddings, **FAISS HNSW/IVF** vector search, **BM25** keyword retrieval, embedding caching, and **CrossEncoder reranking**.
- Integrated retrieved legal contexts into LLM prompts, streamed Gemini responses, maintained multi-turn chat history, and displayed cited source snippets in the UI.
- Added multimodal preprocessing modules with **Whisper STT**, **SymSpell/Ollama query rewriting**, **VietOCR**, and **PaddleOCR** to normalize Vietnamese legal questions from audio and document images.
- Wrote **pytest-based unit tests** with mocked Gemini API calls for API-key loading, prompt construction, streaming responses, error handling, chat reset, and lazy RAG loading.

## Short CV Version

**AI Legal Assistant for Vietnamese Law**                                      **May 2026 - Present**

**Solo project in NLP / Deep Learning Subject**

- Built a **Streamlit-based Vietnamese legal Q&A assistant** combining **Gemini 2.5 Flash**, **Qwen 2.5 Local**, and **Hybrid RAG**.
- Processed legal datasets into **122K raw chunks**, **61K optimized retrieval contexts**, **29K finetuning samples**, and a **39K-term legal dictionary**.
- Implemented retrieval with **Vietnamese SBERT**, **FAISS HNSW/IVF**, **BM25**, embedding cache, and **CrossEncoder reranking** to surface relevant legal passages.
- Integrated streaming LLM responses, multi-turn chat history, source references, dark-mode UI, and optional audio/OCR preprocessing with **Whisper**, **VietOCR**, and **PaddleOCR**.

## Vietnamese Version

**AI Legal Assistant for Vietnamese Law**                                      **May 2026 - Present**

**Đồ án cá nhân môn NLP / Deep Learning**

- Xây dựng **ứng dụng chat pháp lý bằng Streamlit** cho câu hỏi tiếng Việt, hiển thị song song câu trả lời từ **Gemini 2.5 Flash** và **Qwen 2.5 7B Local**.
- Xử lý corpus pháp luật Việt Nam thành **122K chunks cấp điều luật**, tối ưu thành **61K ngữ cảnh cấp khoản/điểm**, đồng thời tạo **29K mẫu finetune** và **39K thuật ngữ pháp lý**.
- Triển khai **Hybrid RAG** kết hợp Vietnamese SBERT, **FAISS HNSW/IVF**, **BM25**, cache embedding và **CrossEncoder reranking** để truy xuất đoạn luật liên quan.
- Tích hợp context pháp lý vào prompt LLM, stream câu trả lời Gemini, quản lý hội thoại nhiều lượt và hiển thị nguồn tham chiếu trực tiếp trên giao diện.
- Bổ sung module tiền xử lý đa phương thức gồm **Whisper STT**, **SymSpell/Ollama query rewriting**, **VietOCR** và **PaddleOCR** cho câu hỏi từ âm thanh hoặc ảnh tài liệu.

## Review Notes

- Main stack: Python, Streamlit, Google GenAI SDK, Ollama/Qwen, FAISS, BM25, SentenceTransformers, CrossEncoder, Whisper, VietOCR, PaddleOCR, pandas, pytest.
- The current app focuses on text chat in Streamlit; audio and OCR modules exist as supporting scripts/prototypes.
- Test command used: `conda run -n dl python -m pytest -q`.
- Result on review: **28 passed, 1 failed**. The failing test mocks `st.session_state` as a plain dict, while current `process_query()` accesses it with Streamlit attribute-style state (`st.session_state.messages`).
- Suggested date range is inferred from the repository history in May 2026. Adjust the dates if your actual project timeline is different.
