import os
import time
import numpy as np
import pandas as pd
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class HybridRAGSystem:
    def __init__(self, csv_path="data/processed/optimized_corpus.csv", cache_file="data/processed/embedding_cache.npy"):
        print("⏳ Đang nạp hệ thống lõi Hybrid RAG...")
        self.csv_path = csv_path
        self.cache_file = cache_file
        
        # 1. Khởi tạo các Mô hình AI
        self.encoder = SentenceTransformer('keepitreal/vietnamese-sbert')
        self.reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        self.dim = self.encoder.get_sentence_embedding_dimension()
        
        # 2. Chạy quy trình xây dựng Dữ liệu
        self._load_and_cache_data()
        self._build_bm25_index()
        self._build_faiss_indices()

    def _load_and_cache_data(self):
        """Đọc dữ liệu và áp dụng CACHE EMBEDDING để không phải tính lại mỗi lần chạy."""
        df = pd.read_csv(self.csv_path)
        self.docs = df[df['context'].str.len() > 50]['context'].astype(str).tolist()
        
        if os.path.exists(self.cache_file):
            print("💾 Tìm thấy file Cache! Đang tải Embeddings thẳng từ ổ cứng (siêu tốc)...")
            self.embeddings = np.load(self.cache_file)
        else:
            print("🧠 Đang mã hóa Embeddings mới (Chỉ chạy 1 lần duy nhất)...")
            # Ép kiểu float32 chuẩn hóa để tương thích 100% với FAISS
            self.embeddings = self.encoder.encode(self.docs, show_progress_bar=True).astype('float32')
            np.save(self.cache_file, self.embeddings)
            print(f"✅ Đã lưu Cache vào {self.cache_file}")

    def _build_bm25_index(self):
        """Xây dựng bộ tìm kiếm từ khóa thô (Sparse Retrieval)."""
        print("🔤 Đang xây dựng chỉ mục BM25 (Keyword Matching)...")
        # Tokenize văn bản đơn giản bằng dấu cách
        tokenized_corpus = [doc.lower().split() for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _build_faiss_indices(self):
        """Xây dựng 2 loại Index của FAISS và thực hiện TUNING."""
        print("🚀 Đang xây dựng FAISS Index...")
        
        # 1. HNSW Index (Nhanh & Chính xác nhất)
        print("  >> Thiết lập HNSW...")
        self.hnsw_index = faiss.IndexHNSWFlat(self.dim, 32) # M=32
        self.hnsw_index.add(self.embeddings)
        # TUNING HNSW: Tăng efSearch để đẩy mạnh độ phủ (Recall)
        self.hnsw_index.hnsw.efSearch = 64 
        
        # 2. IVF-Flat Index (Tối ưu Memory khi Scale)
        print("  >> Thiết lập IVF-Flat...")
        nlist = int(np.sqrt(len(self.docs))) # Công thức chuẩn: nlist = căn bậc 2 của N
        quantizer = faiss.IndexFlatL2(self.dim)
        self.ivf_index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
        self.ivf_index.train(self.embeddings) # Bắt buộc phải train để phân cụm
        self.ivf_index.add(self.embeddings)
        # TUNING IVF: Tăng nprobe để quét nhiều cụm (cluster) hơn
        self.ivf_index.nprobe = min(16, nlist) 
        
        print("✅ Hoàn tất xây dựng bộ máy Tìm kiếm!")

    def hybrid_search(self, query, top_k=3, use_hnsw=True):
        """Thực thi Hybrid Search & Reranking."""
        print(f"\n🔍 [TRUY VẤN]: '{query}'")
        
        # ==============================================================
        # GIAI ĐOẠN 1: TÌM KIẾM HYBRID (BM25 + FAISS VECTOR)
        # ==============================================================
        
        # 1A. Vector Search (Semantic)
        t0 = time.perf_counter()
        q_emb = self.encoder.encode([query]).astype('float32')
        if use_hnsw:
            _, vec_idx = self.hnsw_index.search(q_emb, k=30)
        else:
            _, vec_idx = self.ivf_index.search(q_emb, k=30)
            
        vector_results = [self.docs[i] for i in vec_idx[0]]
        t_vec = (time.perf_counter() - t0) * 1000
        
        # 1B. BM25 Search (Keyword)
        t1 = time.perf_counter()
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_idx = np.argsort(bm25_scores)[::-1][:30]
        bm25_results = [self.docs[i] for i in bm25_idx]
        t_bm25 = (time.perf_counter() - t1) * 1000
        
        # Gộp danh sách và loại bỏ các điều luật trùng lặp
        combined_candidates = list(set(vector_results + bm25_results))
        print(f"⚡ Latency: Vector ({t_vec:.1f}ms) | BM25 ({t_bm25:.1f}ms)")
        print(f"🎯 Đã gộp {len(combined_candidates)} văn bản tiềm năng từ 2 luồng.")
        
        # ==============================================================
        # GIAI ĐOẠN 2: RERANKING BẰNG CROSS-ENCODER
        # ==============================================================
        print("⚖️ Đang chấm điểm logic bằng Cross-Encoder...")
        pairs = [[query, doc] for doc in combined_candidates]
        scores = self.reranker.predict(pairs)
        
        # Sắp xếp từ điểm cao nhất xuống thấp nhất
        ranked_indices = np.argsort(scores)[::-1]
        final_docs = [combined_candidates[i] for i in ranked_indices[:top_k]]
        
        return final_docs

# --- KHỐI TEST ---
if __name__ == "__main__":
    # Đảm bảo file optimized_corpus.csv đã tồn tại trước khi chạy
    rag_system = HybridRAGSystem()
    
    test_queries = [
        "vượt đèn đỏ phạt bao nhiêu tiền",
        "lái xe khi say rượu bị xử lý thế nào"
    ]
    
    print("\n" + "="*50)
    print("TEST 1: TÌM KIẾM VỚI FAISS HNSW + BM25")
    for q in test_queries:
        docs = rag_system.hybrid_search(q, top_k=2, use_hnsw=True)
        for i, doc in enumerate(docs):
            print(f"  [{i+1}] {doc}")
            
    print("\n" + "="*50)
    print("TEST 2: TÌM KIẾM VỚI FAISS IVF-Flat + BM25")
    for q in test_queries:
        docs = rag_system.hybrid_search(q, top_k=2, use_hnsw=False)
        for i, doc in enumerate(docs):
            print(f"  [{i+1}] {doc}")