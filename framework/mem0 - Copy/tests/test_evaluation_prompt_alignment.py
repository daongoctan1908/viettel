from types import SimpleNamespace

from evaluation.paper_update_memory import (
    compact_existing_memory_for_update_prompt,
    decide_memory_operation,
    extract_candidate_memories,
    fallback_noop_decision,
    verify_update_information_content,
)
from evaluation.evaluate_unified_memory import PAPER_BINARY_JUDGE_PROMPT
from evaluation.run_mem0_local_locomo import answer_question, update_rolling_summary


class FakeUsage:
    def model_dump(self):
        return {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


class CapturingCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
            usage=FakeUsage(),
        )


class FakeChatClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=CapturingCompletions(responses))


def test_candidate_prompt_enforces_perspective_ownership_and_keeps_salient_events():
    client = FakeChatClient(['{"memories": []}'])

    extract_candidate_memories(
        client,
        "test-model",
        "Caroline and Melanie discussed their recent activities.",
        [],
        [
            {"role": "user", "content": "Caroline: I attended a support group yesterday."},
            {"role": "assistant", "content": "Melanie: I painted a sunrise last year."},
        ],
        "1:56 pm on 8 May, 2023",
        "2023-05-08T13:56:00Z",
        "Caroline",
        1000,
    )

    call = client.chat.completions.calls[0]
    prompt = call["messages"][1]["content"]
    system = call["messages"][0]["content"]
    assert "TARGET messages authored by Caroline, represented by role=user" in prompt
    assert "Use them only as context; do not" in prompt
    assert "extract memories from them into Caroline's store" in prompt
    assert "meaningful one-off events" in prompt
    assert "assistant-role content belonging to the other speaker" in system


def test_update_prompt_follows_algorithm_one_information_content_rule():
    client = FakeChatClient(
        ['{"operation":"NOOP","target_memory_id":null,"new_memory":null,"reason":"already covered"}']
    )

    decide_memory_operation(
        client,
        "test-model",
        {"memory": "Caroline attends the support group weekly.", "event_at": None},
        [
            {
                "id": "memory-1",
                "memory": "Caroline attended an LGBTQ support group on May 7 and found it empowering.",
            }
        ],
        1000,
    )

    call = client.chat.completions.calls[0]
    prompt = call["messages"][1]["content"]
    system = call["messages"][0]["content"]
    assert "candidate contains more" in prompt
    assert "information than that existing memory" in prompt
    assert "no semantic match -> ADD; contradiction -> DELETE" in prompt
    assert "without inventing facts" in system


def test_update_stores_candidate_fact_instead_of_llm_authored_merge():
    client = FakeChatClient(
        [
            '{"operation":"UPDATE","target_memory_id":"memory-1",'
            '"new_memory":"invented merged text","reason":"candidate is richer"}'
        ]
    )
    candidate = {"memory": "Caroline attends the support group weekly."}

    decision, _ = decide_memory_operation(
        client,
        "test-model",
        candidate,
        [{"id": "memory-1", "memory": "Caroline attended a support group."}],
        1000,
    )

    assert decision["operation"] == "UPDATE"
    assert decision["new_memory"] == candidate["memory"]


def test_information_content_guard_rejects_candidate_that_loses_specific_fact():
    client = FakeChatClient(
        ['{"candidate_is_strictly_richer":false,"reason":"candidate loses counseling and mental health"}']
    )

    result, _ = verify_update_information_content(
        client,
        "test-model",
        {"memory": "Caroline was about to research her career or education."},
        {
            "id": "memory-1",
            "memory": "Caroline is interested in pursuing counseling or mental health as a career.",
        },
        1000,
    )

    prompt = client.chat.completions.calls[0]["messages"][1]["content"]
    assert result["candidate_is_strictly_richer"] is False
    assert "preserves or clearly entails every important factual detail" in prompt
    assert "Do not treat longer wording" in prompt
    assert "as greater information content" in prompt


def test_information_content_guard_accepts_strictly_richer_candidate():
    client = FakeChatClient(
        ['{"candidate_is_strictly_richer":true,"reason":"preserves career and adds field"}']
    )

    result, _ = verify_update_information_content(
        client,
        "test-model",
        {"memory": "Caroline plans a counseling career and is researching mental-health programs."},
        {"id": "memory-1", "memory": "Caroline plans to pursue a career."},
        1000,
    )

    assert result["candidate_is_strictly_richer"] is True


def test_update_failure_fallback_is_noop_instead_of_add():
    decision = fallback_noop_decision("invalid update response")

    assert decision["operation"] == "NOOP"
    assert decision["new_memory"] is None
    assert decision["fallback"] is True


def test_invalid_update_target_is_conservatively_noop():
    client = FakeChatClient(
        ['{"operation":"UPDATE","target_memory_id":"missing","reason":"related"}']
    )

    decision, _ = decide_memory_operation(
        client,
        "test-model",
        {"memory": "Caroline plans to study counseling."},
        [{"id": "memory-1", "memory": "Caroline plans to continue her education."}],
        1000,
    )

    assert decision["operation"] == "NOOP"
    assert decision["fallback"] is True


def test_update_prompt_compacts_retrieved_memories_without_losing_decision_fields():
    memory = {
        "id": "memory-1",
        "memory": "Caroline attended a support group.",
        "metadata": {
            "event_at": "2023-05-07",
            "observed_at": "2023-05-08T13:56:00Z",
            "debug_blob": "x" * 1000,
        },
        "score": 0.91,
        "history": [{"event": "ADD", "large_debug_field": "y" * 1000}],
        "embedding": [0.1, 0.2],
        "updated_at": "2026-06-11T00:00:00Z",
    }

    assert compact_existing_memory_for_update_prompt(memory) == {
        "id": "memory-1",
        "memory": "Caroline attended a support group.",
        "event_at": "2023-05-07",
        "observed_at": "2023-05-08T13:56:00Z",
        "score": 0.91,
    }

    client = FakeChatClient(['{"operation":"NOOP","target_memory_id":null,"reason":"covered"}'])
    decide_memory_operation(
        client,
        "test-model",
        {"memory": "Caroline attended a support group.", "event_at": "2023-05-07"},
        [memory],
        1000,
    )

    prompt = client.chat.completions.calls[0]["messages"][1]["content"]
    assert '"id":"memory-1"' in prompt
    assert '"event_at":"2023-05-07"' in prompt
    assert '"observed_at":"2023-05-08T13:56:00Z"' in prompt
    assert '"score":0.91' in prompt
    assert "debug_blob" not in prompt
    assert "history" not in prompt
    assert "embedding" not in prompt
    assert "updated_at" not in prompt


def test_summary_prompt_preserves_speaker_and_temporal_uncertainty():
    client = FakeChatClient(["Caroline attended a support group the day before the conversation."])

    update_rolling_summary(
        client,
        summary_model="test-model",
        summary_max_tokens=500,
        previous_summary="",
        new_messages=[{"role": "user", "content": "Caroline: I went yesterday."}],
        perspective_speaker="Caroline",
    )

    call = client.chat.completions.calls[0]
    prompt = call["messages"][1]["content"]
    system = call["messages"][0]["content"]
    assert "Do not transfer one speaker's facts to the other speaker" in prompt
    assert "Preserve uncertainty" in prompt
    assert "observation-versus-event time distinctions" in system


def test_first_answer_request_uses_paper_results_generation_prompt_as_system_message():
    client = FakeChatClient(["May 7, 2023"])

    answer, _, prompt = answer_question(
        client,
        llm_model="test-model",
        question="When did Caroline attend the support group?",
        speaker_1_name="Caroline",
        speaker_2_name="Melanie",
        speaker_1_memories=[
            {
                "memory": "Caroline attended an LGBTQ support group.",
                "observed_at": "2023-05-08T13:56:00Z",
                "event_at": "2023-05-07",
            }
        ],
        speaker_2_memories=[],
        answer_max_tokens=500,
    )

    call = client.chat.completions.calls[0]
    system = call["messages"][0]["content"]
    assert answer == "May 7, 2023"
    assert len(call["messages"]) == 1
    assert "No information available" not in system
    assert system == prompt
    assert "Pay special attention to the timestamps to determine the answer" in prompt
    assert "show your work" in prompt


def test_paper_binary_judge_prompt_preserves_appendix_wording():
    assert 'label an answer to a question as \'CORRECT\' or \'WRONG\'' in PAPER_BINARY_JUDGE_PROMPT
    assert "provide a short (one sentence) explanation" in PAPER_BINARY_JUDGE_PROMPT
    assert 'json format with the key as "label"' in PAPER_BINARY_JUDGE_PROMPT
