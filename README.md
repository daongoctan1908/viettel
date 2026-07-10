# Nghiên cứu và Đánh giá Framework bộ nhớ dài hạn cho LLM Agent: Đề xuất pipeline cải tiến cho Mem0

Repo này chứa mã nguồn và kết quả thực nghiệm cho bài toán bộ nhớ dài hạn trong LLM Agent. Trọng tâm là benchmark local trên LoCoMo, so sánh Mem0 local, A-MEM local, RAG baseline và một pipeline cải tiến cho Mem0 ở pha truy hồi memory.

Repo phục vụ nghiên cứu, reproduction và đánh giá thực nghiệm. Đây không phải một deployment production hoàn chỉnh.

## Mục Tiêu

- Nghiên cứu cách các framework long-term memory lưu trữ, cập nhật và truy hồi memory cho LLM Agent.
- Triển khai benchmark local để so sánh Mem0, A-MEM và RAG trong cùng điều kiện thực nghiệm.
- Đề xuất Mem0 QASE, một lớp query-aware retrieval nhằm giảm context dư thừa của Mem0 nhưng vẫn giữ chất lượng trả lời gần baseline.

## Cấu Trúc Repo

```text
C:/Viettel
|-- framework/
|   |-- A-mem-paper-repro-src/             # Source reproduction của A-MEM
|   |-- mem0 - Copy/
|   |   |-- mem0/                          # Source Mem0 OSS/local
|   |   |-- evaluation/
|   |   |   |-- dataset/
|   |   |   |   |-- locomo10.json
|   |   |   |   |-- locomo10_last5.json
|   |   |   |   |-- qase_planner_finetune/
|   |   |   |   `-- qase_planner_finetune_with_longmemeval/
|   |   |   |-- metrics/
|   |   |   |   |-- llm_judge.py
|   |   |   |   `-- utils.py
|   |   |   |-- models/
|   |   |   |   `-- qase_planner_distilroberta/
|   |   |   |-- results/
|   |   |   |   |-- evaluation_mem0_semantic_top7_5conv_judge5/
|   |   |   |   |-- evaluation_mem0_qase_clean_bm25_profile_v1_5conv_judge5_best_result/
|   |   |   |   |-- evaluation_amem_global_linked_top14_openaiemb_5conv_judge5/
|   |   |   |   |-- evaluation_rag_chunk600_k1_5conv_judge5/
|   |   |   |   `-- mem0_paper_memory_5conv.reuse_checkpoint.json
|   |   |   |-- api_clients.py
|   |   |   |-- demo_qase_live_chat.py
|   |   |   |-- evaluate_unified_memory.py
|   |   |   |-- export_qase_planner_finetune_data.py
|   |   |   |-- mem0_question_aware_budget.py
|   |   |   |-- paper_update_memory.py
|   |   |   |-- prompts.py
|   |   |   |-- run_amem_local_locomo.py
|   |   |   |-- run_mem0_local_locomo.py
|   |   |   `-- run_rag_local_locomo.py
|   |   `-- ...
`-- README.md
```

Các file quan trọng trong `evaluation/`:

- `run_mem0_local_locomo.py`: runner cho Mem0 baseline và Mem0 QASE.
- `run_amem_local_locomo.py`: runner cho A-MEM.
- `run_rag_local_locomo.py`: runner cho RAG baseline.
- `evaluate_unified_memory.py`: gom kết quả và tính EM, F1, BLEU-1, LLM judge, latency, token.
- `mem0_question_aware_budget.py`: Question Planner, BM25 reranking, speaker/time bonus, global selector và cutoff của QASE.
- `paper_update_memory.py`: wrapper cập nhật memory theo hướng gần với Mem0 paper.
- `prompts.py`: prompt dùng cho extract/update/answer trong benchmark.
- `api_clients.py`: client OpenAI-compatible cho chat model, judge model và embedding model.
- `demo_qase_live_chat.py`: giao diện chat demo cho Mem0 QASE.

## Các Thành Phần Chính

### Mem0 Local

Mem0 local được dùng làm baseline chính. Source gốc nằm ở `framework/mem0 - Copy/mem0/`, còn phần benchmark và wrapper nằm trong `framework/mem0 - Copy/evaluation/`.

Wrapper trong `paper_update_memory.py` bổ sung các bước: xử lý hội thoại theo message pair, duy trì rolling summary, truyền recent context khi trích xuất memory, tìm memory tương tự trước khi update, và áp dụng bốn thao tác `ADD`, `UPDATE`, `DELETE`, `NOOP`. Memory được lưu theo từng speaker/user namespace và có thể persist bằng checkpoint JSON để tái sử dụng khi chạy QA-only.

### A-MEM Local

A-MEM được chạy từ `framework/A-mem-paper-repro-src/`. Runner ở `run_amem_local_locomo.py` dùng source reproduction này để ingest, retrieve và evaluate trên LoCoMo.

Khác Mem0, A-MEM tổ chức memory thành các memory note thay vì fact ngắn theo speaker namespace. Một note có thể chứa content, keywords, context, tags, links, timestamp và metadata truy hồi. Trong thực nghiệm local, memory note được giữ trong RAM khi chạy, persist thành `memories.pkl`, còn embedding matrix được lưu bằng NumPy.

### RAG Baseline

RAG baseline nằm trong `run_rag_local_locomo.py`. Baseline này tách hội thoại thành chunk token, embed chunk, retrieve top-k chunk và đưa chunk vào answer prompt. Cấu hình chính dùng chunk size khoảng 600 và `k=1`.

### Mem0 QASE

Mem0 QASE là viết tắt của **Query-Aware Self-Evaluating**. QASE không thay lõi lưu trữ/cập nhật memory của Mem0, mà được đặt ở pha QA, giữa semantic retrieval và answer prompt. Phần chính nằm trong `mem0_question_aware_budget.py` và được tích hợp qua `run_mem0_local_locomo.py`.

Pipeline tổng quát:

```text
User Question
-> Question Planner
-> Semantic Candidate Retrieval
-> Evidence Selector
-> Final Memory Pack
-> Answer Prompt
-> LLM Answer
```

QASE có ba ý tưởng chính:

- phân loại câu hỏi để chọn retrieval budget phù hợp;
- chấm lại semantic candidates bằng semantic score, BM25 lexical score, speaker bonus và time bonus;
- cắt final memory pack để giảm context dư thừa trước khi đưa vào prompt.

## Cơ Chế QASE

### Question Planner

Question Planner trong cấu hình QASE chính là rule-based. Planner không dùng LLM, ground-truth answer, judge label, sample id hoặc category id của LoCoMo.

Bốn nhóm câu hỏi:

- `single_hop`: hỏi một fact trực tiếp về một speaker;
- `temporal`: hỏi thời gian, mốc sự kiện hoặc thay đổi theo thời gian;
- `multi_hop`: cần nối nhiều mảnh thông tin, danh sách, so sánh hoặc suy luận;
- `fallback_ambiguous`: không rõ target speaker hoặc không khớp mạnh với các nhóm trên.

Nếu một câu hỏi khớp nhiều nhóm, planner ưu tiên `multi_hop`, sau đó đến `temporal`, `single_hop`, cuối cùng là `fallback_ambiguous`.

### Candidate Retrieval Và Selector

Semantic retrieval vẫn là bước truy hồi chính. QASE lấy semantic candidates từ Mem0 theo từng speaker namespace, gộp candidates của hai speaker vào một pool chung, rồi dùng global selector để chọn final memory pack.

BM25 không được dùng như một retriever độc lập để lấy thêm memory ngoài semantic pool trong cấu hình chính. BM25 chỉ đóng vai trò lexical reranking feature trên các memory đã được semantic retriever lấy ra.

Điểm chọn lọc:

```text
Score(q, m) = S_sem(q, m)
            + 0.30 * S_lex(q, m)
            + B_speaker(q, m)
            + B_time(q, m)
```

Trong đó:

- `S_sem`: semantic similarity do Mem0 retriever trả về;
- `S_lex`: BM25 score đã chuẩn hóa trong phạm vi candidate pool của speaker tương ứng;
- `B_speaker`: cộng nhẹ nếu memory thuộc target speaker;
- `B_time`: cộng nhẹ cho câu hỏi temporal nếu memory có tín hiệu thời gian trong text, hoặc có `event_at`/`event_time_text`.

Sau khi chấm điểm, QASE dùng largest-gap cutoff để cắt final memory pack. Các nhóm đơn giản như single-hop và temporal có ngưỡng cắt thấp hơn để tạo context gọn hơn; multi-hop và fallback/ambiguous giữ evidence rộng hơn.

## Dữ Liệu, Model Và Kết Quả

Dataset chính nằm trong `framework/mem0 - Copy/evaluation/dataset/`. `locomo10.json` chứa 10 conversations của LoCoMo, còn `locomo10_last5.json` là subset 5 conversations cuối dùng cho một số thử nghiệm phụ. Hai thư mục `qase_planner_finetune/` và `qase_planner_finetune_with_longmemeval/` chứa dữ liệu train/test cho planner model thử nghiệm.

Planner model thử nghiệm nằm trong `framework/mem0 - Copy/evaluation/models/qase_planner_distilroberta/`. Model này dùng DistilRoBERTa để thử phân loại question type và target speaker. Tuy nhiên, cấu hình QASE chính vẫn dùng rule-based planner.

Kết quả được giữ trong `framework/mem0 - Copy/evaluation/results/`. Các thư mục kết quả chính:

- `evaluation_mem0_semantic_top7_5conv_judge5/`: Mem0 baseline.
- `evaluation_mem0_qase_clean_bm25_profile_v1_5conv_judge5_best_result/`: Mem0 QASE.
- `evaluation_amem_global_linked_top14_openaiemb_5conv_judge5/`: A-MEM.
- `evaluation_rag_chunk600_k1_5conv_judge5/`: RAG baseline.
- `mem0_paper_memory_5conv.reuse_checkpoint.json`: checkpoint memory đã ingest cho Mem0 local.

Trong mỗi thư mục đánh giá, `*_unified_summary.json` chứa summary overall/category, còn `*_unified_metrics.csv` chứa metric theo từng câu hỏi.

## Thiết Lập Thực Nghiệm

Benchmark chính dùng subset 5 conversations của LoCoMo, gồm 762 câu hỏi sau khi loại category 5/adversarial. Category 5 không được đưa vào benchmark chính vì trong file dataset thực nghiệm, nhóm này không có trường đáp án chuẩn tương ứng như các category còn lại.

Thiết lập chính:

- dataset: LoCoMo 5 conversations;
- answer model: `deepseek-v4-flash`;
- judge model: `deepseek-v4-flash`;
- embedding model: `text-embedding-3-small`;
- judge mode: LLM-as-a-Judge, 5 runs, temperature 0;
- metrics: EM, F1, BLEU-1, LLM judge, latency, retrieved tokens, number of memories.

Latency là số đo local, nên chỉ nên hiểu là so sánh tương đối trong cùng môi trường thực nghiệm, không phải số đo production.

## Kết Quả Chính

### So Sánh Framework

Trong benchmark giữa Mem0, A-MEM và RAG:

- A-MEM đạt overall judge cao nhất, đặc biệt mạnh ở nhóm multi-hop;
- Mem0 tốt hơn A-MEM ở single-hop và temporal, nhờ memory fact ngắn, gọn và theo speaker namespace;
- RAG baseline thấp hơn rõ rệt do chỉ dùng chunk retrieval đơn giản;
- A-MEM đánh đổi chất lượng multi-hop bằng context dài hơn và latency cao hơn.

| Method | Overall Judge |
|---|---:|
| Mem0 local | 64.75 |
| A-MEM local | 71.08 |
| RAG chunk=600, k=1 | 36.75 |

### Mem0 QASE So Với Mem0

Mem0 QASE giữ chất lượng gần baseline trong khi giảm context đưa vào prompt.

| Metric | Mem0 | Mem0 QASE |
|---|---:|---:|
| Overall EM | 19.16 | 21.13 |
| Overall F1 | 43.11 | 43.65 |
| Overall BLEU-1 | 37.25 | 38.02 |
| Overall Judge | 64.75 | 64.99 |
| Avg retrieved tokens | 610 | 463 |
| Avg retrieved memories | 14.00 | 10.37 |

Kết quả chính:

- retrieved tokens giảm khoảng 24.1%;
- số memory trung bình giảm khoảng 25.9%;
- single-hop cải thiện rõ nhất;
- temporal và open-domain cải thiện nhẹ;
- multi-hop giảm nhẹ, cho thấy QASE vẫn cần cơ chế chọn/mở rộng evidence tốt hơn cho câu hỏi cần nối nhiều mảnh thông tin.

## Live Demo

Live demo nằm ở `framework/mem0 - Copy/evaluation/demo_qase_live_chat.py`. Demo là giao diện chat độc lập với LoCoMo, dùng để minh họa quá trình retrieve memory, trả lời, cập nhật memory và cập nhật rolling summary sau mỗi lượt chat.

Demo hiển thị retrieved memories, updated memories, rolling summary, memory tokens, QASE retrieval latency và answer latency. Đây là demo minh họa pipeline, không phải benchmark chính.

## API Và Runtime

Các script dùng API OpenAI-compatible cho chat/judge và OpenAI embedding. Biến môi trường thường dùng:

- `BEEKNOEE_API_KEY`
- `BEEKNOEE_BASE_URL`
- `OPENAI_API_KEY`
- `MODEL`
- `JUDGE_MODEL`
- `EMBEDDING_MODEL`

File `.env` local không được commit.

## Git Policy

Không commit:

- `.env`;
- virtual environment như `.venv`, `.venv311_gpu`;
- `__pycache__`;
- log runtime;
- raw cache hoặc raw output lớn không cần cho việc đối chiếu kết quả;
- file chứa key hoặc thông tin private.

Model weight lớn được quản lý bằng Git LFS khi cần.

## Hạn Chế Và Hướng Phát Triển

Các hạn chế chính:

- benchmark mới chạy trên subset 5 conversations do giới hạn thời gian và chi phí API;
- QASE cải thiện context efficiency nhưng còn yếu ở multi-hop;
- latency local chịu ảnh hưởng bởi runtime, checkpoint/logging, API và môi trường chạy;
- prompt nội bộ của Mem0/A-MEM paper không được công bố đầy đủ nên reproduction không thể khớp tuyệt đối.

Các hướng phát triển:

- thay rule-based planner bằng model phân loại câu hỏi nhỏ được train trên dữ liệu đa dạng hơn;
- cải thiện multi-hop bằng controlled expansion hoặc multi-step retrieval;
- nghiên cứu memory theo nhiều tầng, ví dụ short-term/session/long-term memory;
- đánh giá thêm trên nhiều dataset và nhiều LLM khác nhau.
