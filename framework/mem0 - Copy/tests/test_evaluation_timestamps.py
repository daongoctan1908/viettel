from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.paper_update_memory import LocalPaperMemoryStore, normalize_event_at
from evaluation.run_mem0_local_locomo import (
    build_add_metadata,
    build_resume_config,
    format_memories_for_prompt,
    load_resume_checkpoint,
    observation_time_fields,
    parse_locomo_datetime,
    write_run_checkpoint,
)


class FakeEmbeddings:
    def create(self, model, input):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(len(input)), 1.0])])


class FakeEmbeddingClient:
    def __init__(self):
        self.embeddings = FakeEmbeddings()

    def close(self):
        return None


class HeaderTooLargeError(Exception):
    status_code = 431


class FailingEmbeddings:
    def create(self, model, input):
        raise HeaderTooLargeError("Request headers are too large.")


class FailingEmbeddingClient:
    def __init__(self):
        self.embeddings = FailingEmbeddings()
        self.closed = False

    def close(self):
        self.closed = True


def test_parse_locomo_datetime_returns_canonical_utc_time():
    parsed = parse_locomo_datetime("1:56 pm on 8 May, 2023")

    assert parsed == datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)
    assert observation_time_fields("1:56 pm on 8 May, 2023") == {
        "session_timestamp_raw": "1:56 pm on 8 May, 2023",
        "observed_at": "2023-05-08T13:56:00Z",
        "observed_at_epoch": int(parsed.timestamp()),
        "observed_at_timezone_assumption": "UTC",
    }


def test_normalize_event_at_validates_and_canonicalizes_llm_output():
    assert normalize_event_at("2023-05-07") == "2023-05-07"
    assert normalize_event_at("2023-05-07T18:30:00-04:00") == "2023-05-07T22:30:00Z"
    assert normalize_event_at("2023-02-30") is None
    assert normalize_event_at("yesterday") is None
    assert normalize_event_at("null") is None


def test_build_add_metadata_keeps_observation_time_separate_from_created_at():
    metadata = build_add_metadata(
        sample_id="conv-1",
        run_id="run-1",
        session_key="session_1",
        session_index=1,
        session_date="1:56 pm on 8 May, 2023",
        observed_at="2023-05-08T13:56:00Z",
        observed_at_epoch=1683554160,
        perspective_speaker="Caroline",
        speaker_a="Caroline",
        speaker_b="Melanie",
        batch_start=0,
        batch_size=2,
        context_window=10,
        context_start=0,
        num_context_messages_available=0,
        num_context_messages=0,
        num_context_messages_sent=0,
        num_summary_messages_sent=0,
        num_target_messages=2,
        num_messages_added=2,
    )

    assert metadata["session_timestamp_raw"] == "1:56 pm on 8 May, 2023"
    assert metadata["observed_at"] == "2023-05-08T13:56:00Z"
    assert metadata["observed_at_epoch"] == 1683554160
    assert metadata["observed_at_timezone_assumption"] == "UTC"
    assert "created_at" not in metadata


def test_local_paper_store_tracks_observation_and_event_times_in_history():
    store = LocalPaperMemoryStore(
        embedding_model="fake-embedding",
        embedding_client=FakeEmbeddingClient(),
    )
    memory_id = store.add_memory(
        user_id="caroline",
        memory_text="Caroline attended an LGBTQ support group.",
        metadata={
            "session_timestamp_raw": "1:56 pm on 8 May, 2023",
            "observed_at": "2023-05-08T13:56:00Z",
            "observed_at_epoch": 1683554160,
            "event_at": "2023-05-07",
            "event_time_text": "yesterday",
        },
    )

    added = store.get_all("caroline")[0]
    assert added["observed_at"] == "2023-05-08T13:56:00Z"
    assert added["event_at"] == "2023-05-07"
    assert added["created_at"] != added["observed_at"]
    assert added["history"][0]["event_at"] == "2023-05-07"

    assert store.update_memory(
        user_id="caroline",
        memory_id=memory_id,
        new_memory_text="Caroline now attends the support group weekly.",
        metadata_update={
            "session_timestamp_raw": "9:00 am on 15 May, 2023",
            "observed_at": "2023-05-15T09:00:00Z",
            "observed_at_epoch": 1684141200,
            "event_at": None,
            "event_time_text": "now",
        },
    )

    updated = store.get_all("caroline")[0]
    assert updated["observed_at"] == "2023-05-15T09:00:00Z"
    assert updated["event_at"] is None
    assert updated["history"][-1]["observed_at"] == "2023-05-15T09:00:00Z"
    assert updated["history"][-1]["event_at"] is None


def test_local_paper_store_recreates_embedding_client_after_http_431():
    failing_client = FailingEmbeddingClient()
    replacement_client = FakeEmbeddingClient()
    store = LocalPaperMemoryStore(
        embedding_model="fake-embedding",
        embedding_client=failing_client,
        embedding_client_factory=lambda: replacement_client,
    )

    with patch("evaluation.paper_update_memory.time.sleep", return_value=None):
        embedding = store._embed("Caroline attended a support group.")

    assert embedding == [float(len("Caroline attended a support group.")), 1.0]
    assert failing_client.closed
    assert store.embedding_client is replacement_client
    assert store.embedding_recovery_log[0]["status_code"] == 431


def test_prompt_format_distinguishes_observation_time_from_event_time():
    formatted = format_memories_for_prompt(
        [
            {
                "memory": "Caroline attended an LGBTQ support group.",
                "observed_at": "2023-05-08T13:56:00Z",
                "event_at": "2023-05-07",
                "event_time_text": "yesterday",
            }
        ]
    )

    assert formatted == [
        "Observed at 2023-05-08T13:56:00Z; event occurred at 2023-05-07; "
        "source time expression: yesterday: Caroline attended an LGBTQ support group."
    ]


def test_checkpoint_round_trip_and_resume_config_validation():
    args = SimpleNamespace(
        dataset="evaluation/dataset/locomo10.json",
        method="mem0_local_paper",
        max_conversations=5,
        api_max_retries=10,
        api_timeout_sec=120.0,
        checkpoint_path=None,
        output=None,
        resume_checkpoint=None,
    )

    with TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        dataset_path = temporary_path / "locomo10.json"
        dataset_path.write_text("[]", encoding="utf-8")
        output_path = temporary_path / "result.json"
        checkpoint_path = temporary_path / "result.checkpoint.json"
        resume_config = build_resume_config(args, dataset_path)

        write_run_checkpoint(
            checkpoint_path=checkpoint_path,
            run_id="run-1",
            output_path=output_path,
            resume_config=resume_config,
            completed_sample_ids={"conv-26"},
            state={"results": [{"sample_id": "conv-26"}]},
        )

        loaded = load_resume_checkpoint(checkpoint_path, resume_config)
        assert loaded["run_id"] == "run-1"
        assert loaded["completed_sample_ids"] == ["conv-26"]
        assert loaded["state"]["results"] == [{"sample_id": "conv-26"}]
        assert not checkpoint_path.with_name(f"{checkpoint_path.name}.tmp").exists()

        mismatched_config = {**resume_config, "max_conversations": 4}
        try:
            load_resume_checkpoint(checkpoint_path, mismatched_config)
        except ValueError as exc:
            assert "max_conversations" in str(exc)
        else:
            raise AssertionError("Resume accepted a mismatched benchmark configuration.")
