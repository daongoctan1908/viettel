# LoCoMo Question Distribution Used In Experiments

Dataset file: `evaluation/dataset/locomo10.json`

Main benchmark setting:

- `max_conversations = 5`
- `skip_category_5 = true`
- conversations: `conv-26`, `conv-30`, `conv-41`, `conv-42`, `conv-43`
- total evaluated questions: `762`

## First 5 Conversations, Category 5 Skipped

| Question type | Category | Count | Share |
|---|---:|---:|---:|
| Single-hop | 1 | 142 | 18.64% |
| Temporal | 2 | 156 | 20.47% |
| Multi-hop | 3 | 46 | 6.04% |
| Open-domain | 4 | 418 | 54.86% |
| Total |  | 762 | 100.00% |

## Breakdown By Conversation

| Conversation | Total | Single-hop | Temporal | Multi-hop | Open-domain |
|---|---:|---:|---:|---:|---:|
| conv-26 | 152 | 32 | 37 | 13 | 70 |
| conv-30 | 81 | 11 | 26 | 0 | 44 |
| conv-41 | 152 | 31 | 27 | 8 | 86 |
| conv-42 | 199 | 37 | 40 | 11 | 111 |
| conv-43 | 178 | 31 | 26 | 14 | 107 |

## First 5 Conversations, Including Category 5

| Question type | Category | Count | Share |
|---|---:|---:|---:|
| Single-hop | 1 | 142 | 14.21% |
| Temporal | 2 | 156 | 15.62% |
| Multi-hop | 3 | 46 | 4.60% |
| Open-domain | 4 | 418 | 41.84% |
| Adversarial / Unanswerable | 5 | 237 | 23.72% |
| Total |  | 999 | 100.00% |

## Full 10 Conversations, Category 5 Skipped

| Question type | Category | Count | Share |
|---|---:|---:|---:|
| Single-hop | 1 | 282 | 18.31% |
| Temporal | 2 | 321 | 20.84% |
| Multi-hop | 3 | 96 | 6.23% |
| Open-domain | 4 | 841 | 54.61% |
| Total |  | 1540 | 100.00% |

## Full 10 Conversations, Including Category 5

| Question type | Category | Count | Share |
|---|---:|---:|---:|
| Single-hop | 1 | 282 | 14.20% |
| Temporal | 2 | 321 | 16.16% |
| Multi-hop | 3 | 96 | 4.83% |
| Open-domain | 4 | 841 | 42.35% |
| Adversarial / Unanswerable | 5 | 446 | 22.46% |
| Total |  | 1986 | 100.00% |

## Question Planner Match Against LoCoMo Categories

This uses the current rule-based QASE Question Planner and treats LoCoMo category labels as a reference:

- category 1 -> `single_hop`
- category 2 -> `temporal`
- category 3 -> `multi_hop`
- category 4 -> `fallback_ambiguous`
- category 5 is skipped in the main benchmark.

Important note: this is not a trained category classifier. The planner is designed to choose retrieval budget/evidence behavior, not to reproduce LoCoMo labels exactly. Open-domain is especially hard for rule-based planning because many category-4 questions still contain surface patterns that look like single-hop, temporal, or multi-hop questions.

### First 5 Conversations, Category 5 Skipped

Overall planner/category agreement:

| Setting | Correct | Total | Accuracy |
|---|---:|---:|---:|
| All categories 1-4 | 269 | 762 | 35.30% |
| Categories 1-3 only | 263 | 344 | 76.45% |

Per-category recall:

| LoCoMo category | Planner label | Correct / Total | Recall |
|---|---|---:|---:|
| 1 | Single-hop | 85 / 142 | 59.86% |
| 2 | Temporal | 142 / 156 | 91.03% |
| 3 | Multi-hop | 36 / 46 | 78.26% |
| 4 | Fallback / Open-domain | 6 / 418 | 1.44% |

Confusion matrix, rows are LoCoMo labels and columns are planner labels:

| LoCoMo \\ Planner | Single-hop | Temporal | Multi-hop | Fallback / Ambiguous |
|---|---:|---:|---:|---:|
| Single-hop | 85 | 6 | 50 | 1 |
| Temporal | 4 | 142 | 10 | 0 |
| Multi-hop | 9 | 0 | 36 | 1 |
| Open-domain | 205 | 105 | 102 | 6 |

### Full 10 Conversations, Category 5 Skipped


Per-category recall:

| LoCoMo category | Planner label | Correct / Total | Recall |
|---|---|---:|---:|
| 1 | Single-hop | 171 / 282 | 60.64% |
| 2 | Temporal | 280 / 321 | 87.23% |
| 3 | Multi-hop | 58 / 96 | 60.42% |


## Mem0 Local Memory Format

In the local Mem0 reproduction, memories are stored by speaker/user namespace:

```text
memories_by_user[user_id] = [memory_1, memory_2, ...]
```

Each memory is a JSON-like dictionary. A compact example:

```json
{
  "id": "pmem_a2c1634fecdf4d23a8fad83d495b028e",
  "user_id": "locomo_conv-26_0_Caroline_20260621_101017",
  "memory": "Caroline attended an LGBTQ support group on 2023-05-07 and found it powerful, especially the transgender stories.",
  "embedding": "<1536 floats>",
  "metadata": {
    "source": "locomo",
    "sample_id": "conv-26",
    "session": "session_1",
    "session_timestamp_raw": "1:56 pm on 8 May, 2023",
    "observed_at": "2023-05-08T13:56:00Z",
    "perspective_speaker": "Caroline",
    "speaker_a": "Caroline",
    "speaker_b": "Melanie",
    "candidate_memory_type": "event",
    "candidate_memory_importance": "medium",
    "event_at": "2023-05-07",
    "event_time_text": "yesterday",
    "memory_update_mode": "paper_update_wrapper",
    "memory_wrapper_store": "json_local"
  },
  "session_timestamp_raw": "1:56 pm on 8 May, 2023",
  "observed_at": "2023-05-08T13:56:00Z",
  "observed_at_epoch": 1683554160,
  "observed_at_timezone_assumption": "UTC",
  "event_at": "2023-05-07",
  "event_time_text": "yesterday",
  "created_at": "2026-06-21T03:10:26.508737+00:00",
  "updated_at": "2026-06-21T03:10:33.555230+00:00",
  "history": [
    {
      "event": "ADD",
      "memory": "Caroline attended an LGBTQ support group on May 7, 2023, and found it powerful."
    },
    {
      "event": "UPDATE",
      "previous_memory": "...",
      "memory": "Caroline attended an LGBTQ support group on 2023-05-07 and found it powerful, especially the transgender stories."
    }
  ]
}
```

Key fields:

| Field | Meaning |
|---|---|
| `id` | Local memory id. |
| `user_id` | Speaker namespace that owns the memory. |
| `memory` | Main fact text used for retrieval and answer prompting. |
| `embedding` | Embedding vector of the memory text, usually 1536 dimensions with `text-embedding-3-small`. |
| `metadata` | Source/session/speaker/batch information and extracted temporal fields. |
| `observed_at` | Canonical timestamp of the conversation/session where the memory was observed. |
| `event_at` | Canonical event time inferred from the content when available. |
| `event_time_text` | Original time expression such as `yesterday`, `last week`, or `January 2024`. |
| `history` | ADD/UPDATE history of the memory. |

In short, a local Mem0 memory is not only a text fact. It includes the fact text, embedding, source metadata, temporal fields, and update history, and is stored under a speaker-specific `user_id`.

## A-MEM Memory Note Format

In the local A-MEM run, memory is stored as Python objects rather than JSON fact records. The robust implementation uses `RobustMemoryNote`, and cached memories are written to `memories.pkl`.

A real note from the cache looks like this:

```text
RobustMemoryNote
- id: f91ba83f-61de-42df-9686-d06da3c04dc7
- content: Speaker Caroline says: Hey Mel! Good to see you! How have you been?
- keywords: [greeting, see, well-being]
- context: Melanie greets Caroline and asks what's new...
- tags: [kids, work, swamped, new, conversation, ...]
- links: []
- importance_score: 1.0
- retrieval_count: 0
- timestamp: 1:56 pm on 8 May, 2023
- last_accessed: 202606141148
- evolution_history: []
- category: Uncategorized
```

Key fields:

| Field | Meaning |
|---|---|
| `content` | Main memory note content, usually a raw dialogue turn or short dialogue-derived text. |
| `keywords` | Keywords extracted for the note. |
| `context` | Additional contextual description around the note. |
| `tags` | Topic/category tags for the note. |
| `links` | Connections to related memory notes. |
| `timestamp` | Session timestamp passed by the runner during ingestion. |
| `importance_score`, `retrieval_count`, `last_accessed` | Metadata used by the memory/retrieval process. |
| `evolution_history` | History of note evolution when available. |

Compared with Mem0 local, A-MEM does not store short JSON facts under speaker-specific namespaces. It stores richer memory notes with content, keywords, context, tags, and links. The embedding vectors are not stored inside each note; they are saved separately in a NumPy `.npy` file.

## A-MEM Original vs Robust Example

Example input turn:

```text
Speaker Caroline says: I went to an LGBTQ support group yesterday.
```

### Original / paper-style implementation

The original implementation asks the LLM to generate structured metadata using JSON-style output:

```json
{
  "keywords": ["LGBTQ support group", "community", "support"],
  "context": "Caroline shared that she attended an LGBTQ support group.",
  "tags": ["identity", "support", "community"]
}
```

The parsed output is then stored as a Python `MemoryNote` object:

```text
MemoryNote
- content: Speaker Caroline says: I went to an LGBTQ support group yesterday.
- timestamp: 1:56 pm on 8 May, 2023
- keywords: [LGBTQ support group, community, support]
- context: Caroline shared that she attended an LGBTQ support group.
- tags: [identity, support, community]
- links: []
```

The memory note itself is not stored as JSON. It is kept as a Python object in memory and cached to disk with pickle (`.pkl`). Embedding vectors are stored separately in NumPy (`.npy`).

### Robust local implementation

The robust implementation keeps the same goal, but does not rely on strict JSON schema output. The LLM can return a section-style response:

```text
KEYWORDS: LGBTQ support group, community, support
CONTEXT: Caroline shared that she attended an LGBTQ support group.
TAGS: identity, support, community
```

A flexible parser extracts the same fields and creates a `RobustMemoryNote`:

```text
RobustMemoryNote
- content: Speaker Caroline says: I went to an LGBTQ support group yesterday.
- timestamp: 1:56 pm on 8 May, 2023
- keywords: [LGBTQ support group, community, support]
- context: Caroline shared that she attended an LGBTQ support group.
- tags: [identity, support, community]
- links: []
```

In short:

```text
Original: raw dialogue -> LLM JSON output -> parse JSON -> MemoryNote -> .pkl
Robust:   raw dialogue -> LLM text output -> flexible parser -> RobustMemoryNote -> .pkl
```

The robust version changes the prompt/parser interface, not the core memory-note structure.

## Compact QA Output Schema

The full result JSON is verbose. For reporting or demo purposes, one QA item can be summarized with the following compact schema:

```json
{
  "question_id": "conv-26_q1",
  "conversation_id": "conv-26",
  "category": "temporal",
  "question": "When did Caroline go to the LGBTQ support group?",
  "gold_answer": "7 May 2023",
  "predicted_answer": "May 7, 2023",
  "retrieved_context": [
    "Caroline attended an LGBTQ support group on 2023-05-07 and found it powerful, especially the transgender stories."
  ],
  "num_retrieved": 1,
  "retrieved_tokens": 23,
  "search_latency_sec": 0.45,
  "generation_latency_sec": 1.61,
  "total_latency_sec": 2.06,
  "method": "Mem0 QASE"
}
```

Mapping from the actual result JSON:

| Compact field | Actual field |
|---|---|
| `question_id` | Derived from `sample_id` and `qa_index`, e.g. `conv-26_q1`. |
| `conversation_id` | `sample_id`. |
| `category` | `qase_question_type` for QASE group, or LoCoMo `category` if using dataset labels. |
| `question` | `question`. |
| `gold_answer` | `ground_truth` / `answer`. |
| `predicted_answer` | `predicted_answer` / `response`. |
| `retrieved_context` | `retrieved_memories[*].memory`. |
| `num_retrieved` | `num_retrieved_memories` or `len(retrieved_memories)`. |
| `retrieved_tokens` | `retrieved_memory_tokens`. |
| `search_latency_sec` | `search_latency_sec`. |
| `generation_latency_sec` | `generation_latency_sec`. |
| `total_latency_sec` | `total_latency_sec`. |
| `method` | Top-level `method` in the result file. |

Important note: `evidence_texts` in the result JSON is the gold evidence from LoCoMo, not the memory retrieved by QASE. For retrieved context, use `retrieved_memories`.
