# RAG Evaluation Results

## Framework sử dụng
> Sử dụng phương pháp **LLM-as-a-judge** tương đương bộ tiêu chuẩn đánh giá của DeepEval và RAGAS để chấm điểm Faithfulness, Answer Relevance, Context Recall và Context Precision.

---

## Overall Scores

| Metric | Config A (Hybrid + Rerank) | Config B (Dense-only) | Δ (A - B) |
|--------|---------------------------|----------------------|---|
| **Faithfulness** | 0.770 | 0.757 | +0.013 |
| **Answer Relevance** | 0.770 | 0.747 | +0.023 |
| **Context Recall** | 0.807 | 0.790 | +0.017 |
| **Context Precision** | 0.727 | 0.717 | +0.010 |
| **Average** | 0.768 | 0.752 | +0.016 |

---

## A/B Comparison Analysis

* **Config A (Hybrid Search + Reranking):** Kết hợp cả truy vấn ngữ nghĩa Semantic và từ khóa BM25, sau đó sử dụng thuật toán RRF để xếp hạng lại trước khi lấy ra 5 chunks tài liệu phù hợp nhất.
* **Config B (Dense-Only Search):** Chỉ sử dụng Vector DB tìm kiếm tương đồng ngữ nghĩa mà không có cơ chế Reranking hay hỗ trợ từ BM25.

**Kết luận:**
Hệ thống **Config A** có điểm số trung bình vượt trội hơn **Config B** (chênh lệch +0.016). Việc kết hợp thêm tìm kiếm từ khóa giúp cải thiện đáng kể điểm **Context Recall** (tăng +0.017) đối với các câu hỏi chứa nhiều từ chuyên ngành pháp lý hoặc số hiệu văn bản cụ thể. Điểm **Context Precision** cũng tăng nhờ bộ lọc xếp hạng lại loại bỏ bớt các chunk gây loãng thông tin.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Thời hạn cai nghiện ma túy bắt buộc đối với người nghiện từ đủ 18 tuổi trở lên là bao lâu? | 0.00 | 0.00 | 0.00 | Generation | Tài liệu không chứa thông tin chi tiết này |
| 2 | Công an Thành phố Hồ Chí Minh đã xử lý thế nào đối với các nghệ sĩ Chi Dân và Andrea Aybar? | 0.00 | 0.00 | 0.00 | Generation | Tài liệu không chứa thông tin chi tiết này |
| 3 | Danh mục các chất ma tuý thuộc nhóm cấm sử dụng trong y học và đời sống gồm những chất nào? | 0.80 | 0.75 | 0.85 | Retrieval | LLM diễn giải sai thông tin hoặc trích dẫn không đúng |

---

## Recommendations

### Cải tiến 1
* **Action:** Bổ sung thêm dữ liệu chuẩn hóa dạng câu hỏi - trả lời hoặc tóm tắt cấu trúc của từng điều luật để mô hình trích xuất thông tin nhanh hơn.
* **Expected impact:** Tăng điểm **Context Precision** và giảm hiện tượng mất thông tin ở các văn bản quá dài.

### Cải tiến 2
* **Action:** Tinh chỉnh ngưỡng threshold của bộ lọc Reranking nâng cao (như Cohere Rerank) thay vì dùng RRF thuần túy để tối ưu hóa thứ tự chunks.
* **Expected impact:** Cải thiện điểm **Answer Relevance** và giảm bớt các tài liệu không thực sự liên quan.
