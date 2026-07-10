# Viettel Memory Benchmark Workspace

Repo này chứa workspace thực nghiệm cho bài toán long-term memory trên LoCoMo, gồm Mem0 local, Mem0 QASE, A-MEM reproduction, RAG baseline, bộ đánh giá unified metrics và demo chat trực quan.

Mục tiêu chính của repo là phục vụ benchmark/reproduction và trình bày kết quả báo cáo, không phải một hệ thống production hoàn chỉnh.

## Thành Phần Chính

- **Mem0 local**: bản Mem0 OSS/local dùng làm baseline và nền để triển khai evaluation wrapper.
- **Mem0 QASE**: lớp truy hồi/chọn evidence query-aware đặt giữa Mem0 semantic retrieval và answer prompt.
- **A-MEM local**: reproduction của A-MEM để so sánh với Mem0 trong cùng điều kiện thực nghiệm.
- **RAG baseline**: baseline chunk-based retrieval với chunk size/k cố định.
- **Unified evaluator**: tính EM, F1, BLEU-1, LLM judge, latency, token và cost estimate.
- **Live demo**: giao diện chat hiển thị retrieved memories, updated memories, rolling summary và chi phí từng lượt trả lời.
- **Planner models**: model nhỏ DistilRoBERTa cho question type và target speaker classification.

## Cấu Trúc Thư Mục

```text
C:\Viettel
├── framework
│   ├── mem0 - Copy
│   │   ├── mem0/                         # Mem0 OSS/local source
│   │   ├── evaluation/
│   │   │   ├── dataset/                  # LoCoMo + dữ liệu train planner
│   │   │   ├── metrics/                  # Judge/metric helpers
│   │   │   ├── models/                   # Planner models, quản lý bằng Git LFS
│   │   │   ├── results/                  # Summary/metrics/checkpoint cần giữ
│   │   │   ├── run_mem0_local_locomo.py  # Mem0 baseline + Mem0 QASE runner
│   │   │   ├── run_amem_local_locomo.py  # A-MEM runner
│   │   │   ├── run_rag_local_locomo.py   # RAG runner
│   │   │   ├── evaluate_unified_memory.py
│   │   │   ├── demo_qase_live_chat.py
│   │   │   ├── mem0_question_aware_budget.py
│   │   │   ├── paper_update_memory.py
│   │   │   └── prompts.py
│   │   └── ...
│   └── A-mem-paper-repro-src             # A-MEM reproduction source
└── README.md
```

## Mem0 QASE

QASE là viết tắt của **Query-Aware & Self-Evaluating**. Trong repo này, QASE không thay lõi Mem0, mà bổ sung một lớp điều phối ở pha QA:

1. Phân loại câu hỏi.
2. Xác định target speaker.
3. Lấy candidate memories bằng Mem0 semantic retriever.
4. Chấm lại candidate pool bằng các tín hiệu như semantic score, lexical BM25, speaker prior và time signal.
5. Cắt evidence pack bằng budget/cutoff theo nhóm câu hỏi.
6. Đưa evidence pack vào answer prompt.

Các nhóm câu hỏi trong bản báo cáo:

- `single_hop`
- `temporal`
- `multi_hop`
- `fallback_ambiguous`

Category 5/adversarial của LoCoMo thường không được đưa vào bảng benchmark chính.

## Evaluation Wrapper Cho Mem0

Wrapper trong `paper_update_memory.py` và `run_mem0_local_locomo.py` bổ sung các thành phần gần với mô tả trong paper:

- xử lý hội thoại theo message pair;
- rolling summary;
- recent context;
- candidate memory extraction;
- tìm memory tương tự trước khi update;
- bốn thao tác `ADD`, `UPDATE`, `DELETE`, `NOOP`;
- chuẩn hóa timestamp/session time;
- answer prompt theo hướng paper-style.

Memory được lưu theo speaker/user namespace trong local JSON store/checkpoint, giúp truy hồi và attribution theo speaker rõ ràng hơn.

## Baseline

### Mem0 Baseline

Mem0 baseline dùng semantic retrieval thuần. Trong báo cáo, cấu hình thường dùng `top_k=7` cho mỗi speaker, tương đương tối đa khoảng 14 memory khi truy hồi cả hai speaker.

### A-MEM Baseline

A-MEM local tổ chức memory thành các memory note thay vì fact text tách riêng theo speaker namespace. Note có thể chứa content, keywords, context, tags, links, timestamp và metadata truy hồi. Embedding của A-MEM được lưu riêng bằng NumPy.

### RAG Baseline

RAG baseline tách hội thoại thành chunk token, embed các chunk, retrieve top-k chunk và đưa chunk vào answer prompt. Bản báo cáo có cấu hình chunk 600, k=1.

## Dataset

Các file dataset nằm trong:

```text
framework/mem0 - Copy/evaluation/dataset/
```

Các file đáng chú ý:

- `locomo10.json`: LoCoMo 10 conversations.
- `locomo10_rag.json`: dữ liệu dùng cho RAG baseline.
- `locomo10_last5.json`: subset 5 conversations cuối, dùng cho planner training.
- `qase_planner_finetune/`: dữ liệu train/test planner.
- `qase_planner_finetune_with_longmemeval/`: dữ liệu planner có bổ sung LongMemEval.

## Planner Models

Planner model nằm trong:

```text
framework/mem0 - Copy/evaluation/models/qase_planner_distilroberta/
```

Gồm hai classifier:

- `qase_question_type_distilroberta`: phân loại loại câu hỏi.
- `qase_target_speaker_distilroberta`: xác định speaker được hỏi chính.

Model weight dùng Git LFS vì file `.safetensors` lớn.

## Results

Kết quả báo cáo nằm trong:

```text
framework/mem0 - Copy/evaluation/results/
```

Các artifact quan trọng:

- `*_unified_metrics.csv`: metric theo từng câu hỏi.
- `*_unified_summary.json`: summary overall/category.
- `mem0_paper_memory_5conv.reuse_checkpoint.json`: checkpoint memory đã ingest để tái sử dụng khi chạy QA-only.

Raw result JSON lớn và judge cache có thể bị loại khỏi repo nếu không cần reproduce chi tiết. Repo ưu tiên giữ các file đủ để đọc bảng kết quả báo cáo.

## Live Demo

File demo:

```text
framework/mem0 - Copy/evaluation/demo_qase_live_chat.py
```

Demo là giao diện chat độc lập với LoCoMo. Mỗi lượt chat:

- truy hồi memory liên quan;
- hiển thị retrieved memories;
- trả lời bằng answer model;
- extract/update memory từ tin nhắn mới;
- hiển thị updated memories;
- cập nhật rolling summary;
- hiển thị memory tokens, QASE retrieval latency và answer latency.

## API Và Runtime

Các script cần biến môi trường:

- `BEEKNOEE_API_KEY`: key cho chat/judge endpoint OpenAI-compatible.
- `BEEKNOEE_BASE_URL`: base URL cho chat/judge endpoint.
- `OPENAI_API_KEY`: key cho embedding OpenAI.
- `MODEL`: answer model mặc định.
- `JUDGE_MODEL`: judge model mặc định.
- `EMBEDDING_MODEL`: embedding model mặc định.

File `.env` local không được commit.

## Artifact Và Git Policy

Không commit:

- `.env`
- `.venv`, `.venv311_gpu`
- `__pycache__`
- log file;
- raw cache hoặc raw output rất lớn nếu không cần báo cáo;
- file chứa key hoặc thông tin private.

Đã dùng Git LFS cho các file model weight lớn. Khi clone repo mới, cần đảm bảo Git LFS được bật để tải đủ model.

## Ghi Chú Về Latency

Latency trong repo là latency của pipeline local. Nó không nên được đối chiếu tuyệt đối với paper gốc vì có thể chịu ảnh hưởng bởi:

- backend local thay vì database/vector database production;
- API/network của model;
- checkpoint/logging;
- warm-up/caching;
- mức độ song song hóa thấp hơn production.

Do đó latency nên được hiểu là so sánh tương đối trong cùng môi trường benchmark.

## Trạng Thái Repo

Repo này đã được dọn để giữ các thành phần cần thiết cho:

- chạy lại benchmark chính;
- đọc kết quả báo cáo;
- demo live chat;
- sử dụng planner model;
- tham chiếu A-MEM/RAG/Mem0 baseline.

Các file runtime/private được giữ local và bị ignore bởi `.gitignore`.
