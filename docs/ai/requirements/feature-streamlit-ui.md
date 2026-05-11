---
phase: requirements
title: Streamlit Chat UI — Giao diện Trợ lý Pháp lý AI
feature: streamlit-ui
description: Giao diện chat kiểu ChatGPT/Gemini cho hệ thống Trợ lý Pháp lý AI bằng Streamlit
---

# Requirements — Streamlit Chat UI

## Problem Statement

Dự án hiện tại có đầy đủ các module backend (RAG, Whisper, OCR) nhưng **chưa có giao diện người dùng**. Người dùng phải chạy từng script Python riêng lẻ, không thể trải nghiệm hệ thống một cách trực quan.

- **Ai bị ảnh hưởng?** Sinh viên, giảng viên, người dùng cuối muốn tra cứu luật giao thông
- **Tình trạng hiện tại:** Chỉ có CLI scripts — không thể demo cho người không biết code
- **API đã sẵn sàng:** Gemini API key (`AIzaSy...`) trong `.env`

## Goals & Objectives

**Mục tiêu chính:**
- Xây dựng giao diện chat kiểu ChatGPT/Gemini bằng Streamlit
- Tích hợp Gemini API (Gemini 2.5 Flash) để trả lời câu hỏi luật giao thông
- Hỗ trợ nhập câu hỏi bằng **text**
- Tích hợp RAG (Hybrid BM25 + FAISS) để cung cấp context từ CSDL luật

**Mục tiêu phụ:**
- Hiển thị nguồn tài liệu tham chiếu (RAG context)
- Lịch sử hội thoại trong session
- Giao diện đẹp, responsive, dark mode

**Ngoài phạm vi:**
- Authentication/login hệ thống
- Lưu trữ lịch sử hội thoại vĩnh viễn (database)
- Triển khai production (Docker, cloud)

## User Stories & Use Cases

- **Người dùng thông thường:** Nhập câu hỏi tiếng Việt → nhận câu trả lời luật giao thông rõ ràng
- **Demo/Presentation:** Giảng viên/sinh viên demo hệ thống mà không cần biết CLI

**Luồng chính:**
1. User mở app → thấy chat interface
2. Nhập câu hỏi (text) → gửi
3. Hệ thống RAG tìm context (nếu có data) → Gemini sinh câu trả lời
4. Hiển thị câu trả lời + nguồn tham chiếu trong chat bubble
5. Lịch sử hội thoại được giữ trong session

**Ngoài phạm vi (đã xác nhận skip):**
- ~~Giọng nói (Whisper)~~ — bỏ qua
- ~~OCR (ảnh)~~ — bỏ qua

## Success Criteria

- [x] App chạy được bằng `conda activate dl && streamlit run app.py`
- [x] Chat interface hiển thị đúng lịch sử hội thoại (trong session)
- [x] Tích hợp Gemini 2.5 Flash API trả lời câu hỏi luật giao thông
- [x] RAG lazy-load: khi có `data/processed/optimized_corpus.csv` → hiển thị nguồn tham chiếu
- [x] Giao diện dark mode, gradient, hiện đại như ChatGPT/Gemini
- [ ] Câu hỏi gợi ý trên màn hình welcome
- [ ] Nút xóa lịch sử hội thoại

## Constraints & Assumptions

- **Technical:** Python phải chạy trong môi trường `conda activate dl`
- **API:** Gemini API key đã có trong `.env` (biến `api_key`)
- **Dependencies:** `streamlit` cần được thêm vào `requirements.txt`
- **Data:** RAG hoạt động khi có file `data/processed/optimized_corpus.csv`
- **Assumption:** Gemini API được dùng thay Ollama local để trả lời (vì API đã có sẵn)

## Questions & Open Items

- RAG system cần file CSV — nếu chưa có, UI fallback sang Gemini thuần ✅ (đã giải quyết)
- Model Gemini: dùng `gemini-2.5-flash` (confirmed available với API key hiện tại) ✅
