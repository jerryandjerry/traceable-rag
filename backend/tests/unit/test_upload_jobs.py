"""The durable upload queue contract without a live PostgreSQL server."""
from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from visionagent.database.postgres.upload_jobs import (
    InMemoryUploadJobRepository,
    UploadAccountUnavailable,
    UploadDocumentConflict,
    UploadJobRepository,
)
from visionagent.models import (
    AuthorizationScope,
    IngestJob,
    JobIdentity,
    StagedUpload,
    UploadItemState,
    UploadJobState,
)


def _job(user_id: str = "42") -> IngestJob:
    return IngestJob(
        identity=JobIdentity(run_id="run-a", user_id=user_id),
        authorization=AuthorizationScope(can_upload=True),
        file_names=("a.pdf",),
    )


def _repository(items: int = 1) -> InMemoryUploadJobRepository:
    repository = InMemoryUploadJobRepository()
    repository.create(
        "process-a",
        _job(),
        tuple(
            StagedUpload(
                sequence=index,
                original_name="a.pdf",
                part_name=f"part-{index}.pdf",
                storage_key=f"{index}.pdf",
            )
            for index in range(items)
        ),
        ({"step": "upload", "message": "queued"},),
        staging_node_id="node-a",
    )
    return repository


def _make_retry_eligible(
    repository: InMemoryUploadJobRepository, process_id: str = "process-a"
) -> None:
    record = repository.get(process_id)
    assert record is not None
    repository._records[process_id] = record.model_copy(  # noqa: SLF001
        update={"next_attempt_at": datetime.now(UTC) - timedelta(seconds=1)}
    )


def _create_staging(
    repository: InMemoryUploadJobRepository,
    process_id: str = "process-staging",
    *,
    user_id: str = "42",
) -> None:
    repository.create_staging(
        process_id,
        _job(user_id),
        (
            StagedUpload(
                sequence=0,
                original_name=f"{process_id}.pdf",
                part_name=f"{process_id}.pdf",
                storage_key="a.pdf",
            ),
        ),
        ({"step": "upload", "message": "accepted"},),
        staging_node_id="node-a",
    )


def test_staging_ticket_is_active_but_not_dispatchable_until_finalized() -> None:
    repository = InMemoryUploadJobRepository()
    _create_staging(repository)

    record = repository.get("process-staging")
    assert record is not None
    assert record.status is UploadJobState.STAGING
    assert repository.active_ids_for_user("42") == ["process-staging"]
    assert repository.reserve_next("worker-a", "node-a", 45) is None
    assert repository.requeue_expired() == 0
    assert repository.settle_cancellations_for_user("42") == 0

    assert not repository.finalize_staging("process-staging", "99")
    assert repository.finalize_staging("process-staging", "42")
    assert repository.get("process-staging").status is UploadJobState.QUEUED
    assert repository.reserve_next("worker-a", "node-a", 45) == "process-staging"


def test_staging_ticket_cancellation_is_immediate_and_blocks_finalize() -> None:
    repository = InMemoryUploadJobRepository()
    _create_staging(repository)

    assert repository.request_cancel("process-staging", "42") == (
        True,
        True,
        None,
        None,
        None,
    )
    record = repository.get("process-staging")
    assert record is not None
    assert record.status is UploadJobState.CANCELLED
    assert record.cancellation_requested
    assert not repository.finalize_staging("process-staging", "42")


def test_account_deletion_can_discard_a_requested_staging_ticket() -> None:
    repository = InMemoryUploadJobRepository()
    _create_staging(repository)
    # Exercise the defensive crash-recovery state directly: ordinary
    # request_cancel() terminalizes staging immediately, but account erasure
    # must also tolerate a persisted staging row whose flag was set earlier.
    record = repository.get("process-staging")
    assert record is not None
    repository._records["process-staging"] = record.model_copy(  # noqa: SLF001
        update={"cancellation_requested": True}
    )

    assert repository.discard_cancellations_for_user("42") == 1
    discarded = repository.get("process-staging")
    assert discarded is not None
    assert discarded.status is UploadJobState.CANCELLED


def test_staging_failure_is_owner_scoped_and_journalled() -> None:
    repository = InMemoryUploadJobRepository()
    _create_staging(repository)

    assert not repository.fail_staging(
        "process-staging", "99", {"step": "error", "message": "failed"}
    )
    assert repository.fail_staging(
        "process-staging", "42", {"step": "error", "message": "failed"}
    )
    record = repository.get("process-staging")
    assert record is not None
    assert record.status is UploadJobState.FAILED
    assert record.progress[-1].message == "failed"
    assert not repository.fail_staging(
        "process-staging", "42", {"step": "error", "message": "again"}
    )


def test_expired_staging_tickets_fail_in_bounded_batches() -> None:
    repository = InMemoryUploadJobRepository()
    _create_staging(repository, "process-a")
    _create_staging(repository, "process-b")

    assert repository.expire_staging(max_age_seconds=0, limit=1) == 1
    states = {
        process_id: repository.get(process_id).status
        for process_id in ("process-a", "process-b")
    }
    assert set(states.values()) == {UploadJobState.STAGING, UploadJobState.FAILED}
    failed_id = next(
        process_id for process_id, state in states.items() if state is UploadJobState.FAILED
    )
    assert repository.get(failed_id).progress[-1].step == "error"
    assert repository.expire_staging(max_age_seconds=0, limit=100) == 1
    assert repository.expire_staging(max_age_seconds=0, limit=0) == 0


def test_concurrent_same_document_admission_has_exactly_one_winner() -> None:
    repository = InMemoryUploadJobRepository()
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def admit(process_id: str) -> None:
        barrier.wait()
        try:
            repository.create_staging(
                process_id,
                _job(),
                (StagedUpload(
                    sequence=0,
                    original_name="same.pdf",
                    part_name="same.pdf",
                    storage_key=f"{process_id}.pdf",
                ),),
                (),
                staging_node_id="node-a",
            )
        except UploadDocumentConflict:
            outcomes.append("conflict")
        else:
            outcomes.append("accepted")

    threads = [threading.Thread(target=admit, args=(f"process-{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["accepted", "conflict"]


def test_legacy_nfd_active_name_conflicts_with_new_nfc_name() -> None:
    repository = InMemoryUploadJobRepository()
    repository.create_staging(
        "legacy",
        _job(),
        (StagedUpload(
            sequence=0,
            original_name="cafe\N{COMBINING ACUTE ACCENT}.pdf",
            part_name="legacy.pdf",
            storage_key="legacy.pdf",
        ),),
        (),
        staging_node_id="node-a",
    )
    with pytest.raises(UploadDocumentConflict):
        repository.create_staging(
            "new",
            _job(),
            (StagedUpload(
                sequence=0,
                original_name="caf\N{LATIN SMALL LETTER E WITH ACUTE}.pdf",
                part_name="new.pdf",
                storage_key="new.pdf",
            ),),
            (),
            staging_node_id="node-a",
        )


def test_create_compatibility_queues_the_staging_ticket() -> None:
    repository = _repository()

    record = repository.get("process-a")
    assert record is not None
    assert record.status is UploadJobState.QUEUED


def test_only_one_worker_can_claim_a_queued_job() -> None:
    repository = _repository()

    assert repository.reserve_next(
        "worker-a", "node-a", 45, process_id="process-a"
    ) == "process-a"
    assert repository.reserve_next("worker-b", "node-b", 45) is None
    assert repository.activate("process-a", "worker-a", 100, "start-a", 45)

    record = repository.get("process-a")
    assert record is not None
    assert record.status is UploadJobState.PROCESSING
    assert record.lease_owner == "worker-a"


def test_expired_worker_preserves_ambiguous_processing_checkpoint() -> None:
    repository = _repository(items=2)
    assert repository.reserve_next(
        "worker-a", "node-a", -1, process_id="process-a"
    ) == "process-a"
    assert repository.activate("process-a", "worker-a", 100, "start-a", -1)
    assert repository.begin_item("process-a", "worker-a", 0)
    assert repository.finish_item(
        "process-a", "worker-a", 0, indexed_count=3, process_time=1.5, error=None
    )
    assert repository.begin_item("process-a", "worker-a", 1)

    assert repository.requeue_expired() == 1
    record = repository.get("process-a")
    assert record is not None
    assert record.status is UploadJobState.QUEUED
    assert record.items[0].status is UploadItemState.COMPLETED
    assert record.items[0].indexed_count == 3
    assert record.items[1].status is UploadItemState.PROCESSING
    assert any(event.step == "recovery" for event in record.progress)


def test_cancellation_is_owner_scoped_persisted_and_idempotent() -> None:
    repository = _repository()
    assert repository.request_cancel("process-a", "99") == (False, False, None, None, None)

    found, changed, worker_pid, start_token, node_id = repository.request_cancel(
        "process-a", "42"
    )
    assert (found, changed, worker_pid, start_token, node_id) == (
        True, True, None, None, None
    )
    assert repository.request_cancel("process-a", "42") == (
        True, False, None, None, None
    )

    record = repository.get("process-a")
    assert record is not None
    assert record.status is UploadJobState.CANCELLED
    assert record.cancellation_requested
    assert record.progress[-1].step == "cancelled"


def test_progress_is_an_ordered_journal_not_message_deduplication() -> None:
    repository = _repository()
    assert repository.reserve_next(
        "worker-a", "node-a", 45, process_id="process-a"
    ) == "process-a"
    assert repository.activate("process-a", "worker-a", 100, "start-a", 45)
    event = {"step": "encode_pending", "message": "same text", "percent": 10}
    assert repository.append_progress("process-a", "worker-a", event, 45)
    assert repository.append_progress("process-a", "worker-a", event, 45)

    events = repository.events_after("process-a", 0)
    assert [item.id for item in events] == sorted(item.id for item in events)
    assert [item.step for item in events].count("encode_pending") == 2
    assert repository.events_after("process-a", events[-2].id) == (events[-1],)


def test_public_status_excludes_worker_and_staging_internals() -> None:
    repository = _repository()
    assert repository.reserve_next(
        "worker-a", "node-a", 45, process_id="process-a"
    ) == "process-a"
    assert repository.activate("process-a", "worker-a", 100, "start-a", 45)
    record = repository.get("process-a")
    assert record is not None

    public = record.public_dict()
    assert public["user_id"] == "42"
    assert "worker_pid" not in public
    assert "staging_node_id" not in public
    assert "worker_node_id" not in public
    assert "lease_owner" not in public
    assert "items" not in public
    assert "job" not in public


def test_concurrent_dispatchers_reserve_distinct_jobs_before_spawn() -> None:
    repository = _repository()
    repository.create(
        "process-b",
        _job(),
        (
            StagedUpload(
                sequence=0,
                original_name="b.pdf",
                part_name="b.pdf",
                storage_key="b.pdf",
            ),
        ),
        ({"step": "upload", "message": "queued"},),
        staging_node_id="node-a",
    )
    barrier = threading.Barrier(3)
    reserved: list[str | None] = []

    def dispatch(owner: str) -> None:
        barrier.wait()
        reserved.append(repository.reserve_next(owner, "node-a", 45))

    threads = [threading.Thread(target=dispatch, args=(f"worker-{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert set(reserved) == {"process-a", "process-b"}


def test_concurrent_dispatchers_share_one_node_capacity_slot() -> None:
    repository = _repository()
    repository.create(
        "process-b",
        _job(),
        (StagedUpload(
            sequence=0,
            original_name="b.pdf",
            part_name="b.pdf",
            storage_key="b.pdf",
        ),),
        ({"step": "upload", "message": "queued"},),
        staging_node_id="node-a",
    )
    barrier = threading.Barrier(3)
    reserved: list[str | None] = []

    def dispatch(owner: str) -> None:
        barrier.wait()
        reserved.append(
            repository.reserve_next(owner, "node-a", 45, max_workers=1)
        )

    threads = [threading.Thread(target=dispatch, args=(f"worker-{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert len([process_id for process_id in reserved if process_id is not None]) == 1
    assert reserved.count(None) == 1


def test_expired_reservation_does_not_consume_node_capacity() -> None:
    repository = _repository()
    repository.create(
        "process-b",
        _job(),
        (StagedUpload(
            sequence=0,
            original_name="b.pdf",
            part_name="b.pdf",
            storage_key="b.pdf",
        ),),
        ({"step": "upload", "message": "queued"},),
        staging_node_id="node-a",
    )

    assert repository.reserve_next(
        "worker-a", "node-a", -1, max_workers=1
    ) == "process-a"
    assert repository.reserve_next(
        "worker-b", "node-a", 45, process_id="process-b", max_workers=1
    ) == "process-b"


def test_expired_attempt_backs_off_then_queues_reconciliation_at_budget() -> None:
    repository = _repository()
    record = repository.get("process-a")
    assert record is not None
    repository._records["process-a"] = record.model_copy(  # noqa: SLF001
        update={"max_attempts": 2}
    )

    assert repository.reserve_next("worker-a", "node-a", -1) == "process-a"
    assert repository.requeue_expired() == 1
    backed_off = repository.get("process-a")
    assert backed_off is not None
    assert backed_off.status is UploadJobState.QUEUED
    assert backed_off.next_attempt_at is not None
    assert repository.reserve_next("too-soon", "node-a", -1) is None

    _make_retry_eligible(repository)
    assert repository.reserve_next("worker-b", "node-a", -1) == "process-a"
    assert repository.requeue_expired() == 1
    reconcile = repository.get("process-a")
    assert reconcile is not None
    assert reconcile.status is UploadJobState.QUEUED
    assert reconcile.next_attempt_at is None
    assert reconcile.last_failure_class == "worker_lease_expired"


def test_compensated_part_retry_resets_every_sibling_checkpoint() -> None:
    repository = _repository(items=2)
    assert repository.reserve_next("worker", "node", 45) == "process-a"
    assert repository.begin_item("process-a", "worker", 0)
    assert repository.finish_item(
        "process-a", "worker", 0, indexed_count=3, process_time=1, error=None
    )
    assert repository.begin_item("process-a", "worker", 1)

    assert repository.retry_document(
        "process-a",
        "worker",
        "a.pdf",
        {"step": "recovery", "failure_class": "item_timeout"},
    )
    record = repository.get("process-a")
    assert record is not None
    assert [item.status for item in record.items] == [
        UploadItemState.PENDING,
        UploadItemState.PENDING,
    ]
    assert record.total_chunks_inserted == 0


def test_last_attempt_cancellation_terminalizes_without_processing_items() -> None:
    repository = _repository(items=2)
    record = repository.get("process-a")
    assert record is not None
    repository._records["process-a"] = record.model_copy(  # noqa: SLF001
        update={"max_attempts": 1}
    )
    assert repository.reserve_next("worker", "node", 45) == "process-a"
    assert repository.begin_item("process-a", "worker", 0)
    assert repository.request_cancel("process-a", "42")[:2] == (True, True)

    assert not repository.retry_document(
        "process-a",
        "worker",
        "a.pdf",
        {"step": "cancelled", "failure_class": "item_cancelled"},
    )
    assert repository.finish_reconciled(
        "process-a",
        "worker",
        UploadJobState.CANCELLED,
        ("a.pdf",),
        {"step": "cancelled", "failure_class": "item_cancelled"},
    )
    terminal = repository.get("process-a")
    assert terminal is not None
    assert terminal.status is UploadJobState.CANCELLED
    assert all(item.status is not UploadItemState.PROCESSING for item in terminal.items)
    public = terminal.public_dict()
    assert public["attempt_count"] == 1
    assert public["max_attempts"] == 1
    assert public["last_failure_class"] == "item_cancelled"


def test_terminal_reconciliation_is_atomic_per_logical_document() -> None:
    repository = InMemoryUploadJobRepository()
    repository.create(
        "process-a",
        _job(),
        (
            StagedUpload(
                sequence=0,
                original_name="a.pdf",
                part_name="a.pdf",
                storage_key="0.pdf",
            ),
            StagedUpload(
                sequence=1,
                original_name="b.pdf",
                part_name="part-0-b.pdf",
                storage_key="1.pdf",
            ),
            StagedUpload(
                sequence=2,
                original_name="b.pdf",
                part_name="part-1-b.pdf",
                storage_key="2.pdf",
            ),
            StagedUpload(
                sequence=3,
                original_name="c.pdf",
                part_name="c.pdf",
                storage_key="3.pdf",
            ),
        ),
        ({"step": "upload", "message": "queued"},),
        staging_node_id="node-a",
    )
    assert repository.reserve_next("worker", "node", 45) == "process-a"
    assert repository.begin_item("process-a", "worker", 0)
    assert repository.finish_item(
        "process-a", "worker", 0, indexed_count=4, process_time=1, error=None
    )
    assert repository.begin_item("process-a", "worker", 1)
    assert repository.finish_item(
        "process-a", "worker", 1, indexed_count=2, process_time=1, error=None
    )
    assert repository.begin_item("process-a", "worker", 2)
    assert repository.request_cancel("process-a", "42")[:2] == (True, True)

    assert repository.finish_reconciled(
        "process-a",
        "worker",
        UploadJobState.CANCELLED,
        ("b.pdf",),
        {"step": "cancelled", "failure_class": "item_cancelled"},
    )

    terminal = repository.get("process-a")
    assert terminal is not None
    assert terminal.status is UploadJobState.CANCELLED
    assert [item.status for item in terminal.items] == [
        UploadItemState.COMPLETED,
        UploadItemState.FAILED,
        UploadItemState.FAILED,
        UploadItemState.FAILED,
    ]
    assert [item.indexed_count for item in terminal.items] == [4, 0, 0, 0]
    assert terminal.processed_files == terminal.total_files == 4
    assert terminal.total_chunks_inserted == 4


def test_failed_spawn_releases_an_unstarted_reservation() -> None:
    repository = _repository()
    assert repository.reserve_next("worker-a", "node-a", 45) == "process-a"

    assert repository.release_reservation("process-a", "worker-a")
    _make_retry_eligible(repository)
    assert repository.reserve_next("worker-b", "node-a", 45) == "process-a"


def test_active_lease_fences_the_previous_owner_after_transfer() -> None:
    repository = _repository()
    assert repository.reserve_next("worker-a", "node-a", 45) == "process-a"
    assert repository.owns_active_lease("process-a", "worker-a")

    assert repository.activate("process-a", "worker-a", 100, "start-a", -1)
    assert not repository.owns_active_lease("process-a", "worker-a")
    assert repository.requeue_expired() == 1
    _make_retry_eligible(repository)
    assert repository.reserve_next("worker-b", "node-a", 45) == "process-a"

    assert not repository.owns_active_lease("process-a", "worker-a")
    assert repository.owns_active_lease("process-a", "worker-b")


def test_final_spawn_failure_cannot_terminalize_recovery_checkpoint() -> None:
    repository = _repository()
    record = repository.get("process-a")
    assert record is not None
    repository._records["process-a"] = record.model_copy(  # noqa: SLF001
        update={"max_attempts": 1}
    )
    assert repository.reserve_next("worker-a", "node-a", -1) == "process-a"
    assert repository.activate("process-a", "worker-a", 100, "start", -1)
    assert repository.begin_item("process-a", "worker-a", 0)
    assert repository.requeue_expired() == 1
    assert repository.reserve_next("worker-b", "node-a", 45) == "process-a"

    assert repository.release_reservation("process-a", "worker-b")

    queued = repository.get("process-a")
    assert queued is not None
    assert queued.status is UploadJobState.QUEUED
    assert queued.attempt_count == 1
    assert queued.items[0].status is UploadItemState.PROCESSING
    assert not queued.staging_cleaned


def test_any_worker_node_can_reserve_from_shared_staging() -> None:
    repository = _repository()

    assert repository.reserve_next("remote-worker", "node-b", 45) == "process-a"
    record = repository.get("process-a")
    assert record is not None
    assert record.status is UploadJobState.PROCESSING
    assert record.worker_node_id == "node-b"
    assert record.staging_node_id == "node-a"


def test_cancellation_returns_the_node_scoped_process_fingerprint() -> None:
    repository = _repository()
    assert repository.reserve_next("worker-a", "node-a", 45) == "process-a"
    assert repository.activate("process-a", "worker-a", 321, "start-token", 45)

    assert repository.request_cancel("process-a", "42") == (
        True,
        True,
        321,
        "start-token",
        "node-a",
    )
    record = repository.get("process-a")
    assert record is not None
    assert record.status is UploadJobState.PROCESSING
    assert record.cancellation_requested
    assert record.progress[-1].step == "cancellation_pending"


def test_cancellation_atomically_refuses_a_new_pending_item() -> None:
    repository = _repository()
    assert repository.reserve_next("worker-a", "node-a", 45) == "process-a"
    assert repository.activate("process-a", "worker-a", 321, "start-token", 45)
    assert repository.request_cancel("process-a", "42")[:2] == (True, True)

    assert not repository.begin_item("process-a", "worker-a", 0)


def test_cancellation_allows_replay_of_an_existing_processing_checkpoint() -> None:
    repository = _repository()
    assert repository.reserve_next("worker-a", "node-a", 45) == "process-a"
    assert repository.activate("process-a", "worker-a", 321, "start-token", 45)
    assert repository.begin_item("process-a", "worker-a", 0)


def test_a_job_cannot_finish_while_an_item_is_still_processing() -> None:
    repository = _repository()
    assert repository.reserve_next("worker-a", "node-a", 45) == "process-a"
    assert repository.activate("process-a", "worker-a", 321, "start-token", 45)
    assert repository.begin_item("process-a", "worker-a", 0)

    assert not repository.finish(
        "process-a",
        "worker-a",
        UploadJobState.COMPLETED,
        {"step": "complete"},
    )
    assert repository.get("process-a").status is UploadJobState.PROCESSING
    assert repository.request_cancel("process-a", "42")[:2] == (True, True)

    assert repository.begin_item("process-a", "worker-a", 0)


def test_cancelled_inflight_item_is_recovered_to_a_safe_boundary() -> None:
    repository = _repository()
    assert repository.reserve_next("worker-a", "node-a", -1) == "process-a"
    assert repository.activate("process-a", "worker-a", 100, "start-a", -1)
    assert repository.begin_item("process-a", "worker-a", 0)
    assert repository.request_cancel("process-a", "42")[:2] == (True, True)

    assert repository.requeue_expired() == 1
    queued = repository.get("process-a")
    assert queued is not None
    assert queued.status is UploadJobState.QUEUED
    assert queued.items[0].status is UploadItemState.PROCESSING
    _make_retry_eligible(repository)
    assert repository.reserve_next("worker-b", "node-a", 45) == "process-a"
    assert repository.activate("process-a", "worker-b", 101, "start-b", 45)
    assert repository.begin_item("process-a", "worker-b", 0)
    assert repository.finish_item(
        "process-a", "worker-b", 0, indexed_count=2, process_time=1.0, error=None
    )
    assert repository.finish_cancelled(
        "process-a", "worker-b", {"step": "cancelled"}
    )

    terminal = repository.get("process-a")
    assert terminal is not None
    assert terminal.status is UploadJobState.CANCELLED
    assert terminal.items[0].status is UploadItemState.COMPLETED


def test_expired_cancelled_item_keeps_its_replay_checkpoint() -> None:
    repository = _repository()
    assert repository.reserve_next("worker-a", "node-a", -1) == "process-a"
    assert repository.activate("process-a", "worker-a", 100, "start-a", -1)
    assert repository.begin_item("process-a", "worker-a", 0)
    assert repository.request_cancel("process-a", "42")[:2] == (True, True)

    assert repository.requeue_expired() == 1
    record = repository.get("process-a")
    assert record is not None
    assert record.status is UploadJobState.QUEUED
    assert record.items[0].status is UploadItemState.PROCESSING


def test_account_deletion_discards_a_queued_cancelled_recovery_checkpoint() -> None:
    repository = _repository()
    assert repository.reserve_next("worker-a", "node-a", -1) == "process-a"
    assert repository.activate("process-a", "worker-a", 100, "start-a", -1)
    assert repository.begin_item("process-a", "worker-a", 0)
    assert repository.request_cancel("process-a", "42")[:2] == (True, True)
    assert repository.requeue_expired() == 1

    assert repository.discard_cancellations_for_user("42") == 1
    record = repository.get("process-a")
    assert record is not None
    assert record.status is UploadJobState.CANCELLED
    assert record.items[0].status is UploadItemState.FAILED
    assert record.lease_owner is None


def test_terminal_row_cannot_be_deleted_before_staging_acknowledgement() -> None:
    repository = _repository()
    assert repository.request_cancel("process-a", "42")[:2] == (True, True)

    assert not repository.delete_terminal("process-a", "42")
    assert repository.terminal_staging_ids() == ["process-a"]
    assert repository.mark_staging_cleaned("process-a")
    assert repository.delete_terminal("process-a", "42")


def test_tenant_lock_owner_can_settle_an_active_cancellation() -> None:
    repository = _repository()
    assert repository.reserve_next("worker-a", "node-a", 45) == "process-a"
    assert repository.activate("process-a", "worker-a", 100, "start-a", 45)
    assert repository.request_cancel("process-a", "42")[:2] == (True, True)

    assert repository.settle_cancellations_for_user("42") == 1
    record = repository.get("process-a")
    assert record is not None
    assert record.status is UploadJobState.CANCELLED
    assert record.lease_owner is None


def test_postgres_node_capacity_is_serialized_before_reservation() -> None:
    class _Result:
        def __init__(self, row: SimpleNamespace | None = None) -> None:
            self._row = row

        def fetchone(self) -> SimpleNamespace | None:
            return self._row

    class _Session:
        def __init__(self) -> None:
            self.sql: list[str] = []
            self.committed = False

        def execute(self, statement: Any, _params: Any = None) -> _Result:
            sql = str(statement)
            self.sql.append(sql)
            if "COUNT(*) AS live_workers" in sql:
                return _Result(SimpleNamespace(live_workers=1))
            return _Result()

        def commit(self) -> None:
            self.committed = True

        def rollback(self) -> None:
            raise AssertionError("capacity check must not roll back")

        def close(self) -> None:
            pass

    session = _Session()
    repository = UploadJobRepository(lambda: session)

    assert repository.reserve_next(
        "worker-b", "node-a", 45, max_workers=1
    ) is None
    assert "pg_advisory_xact_lock" in session.sql[0]
    assert "COUNT(*) AS live_workers" in session.sql[1]
    assert all("UPDATE upload_jobs" not in sql for sql in session.sql)
    assert session.committed


def test_durable_create_locks_and_honours_the_account_deletion_gate() -> None:
    class _Result:
        def fetchone(self) -> SimpleNamespace:
            return SimpleNamespace(deletion_requested=True)

    class _Session:
        def __init__(self) -> None:
            self.sql: list[str] = []
            self.rolled_back = False

        def execute(self, statement: Any, _params: Any = None) -> _Result:
            self.sql.append(str(statement))
            return _Result()

        def rollback(self) -> None:
            self.rolled_back = True

        def close(self) -> None:
            pass

    session = _Session()
    repository = UploadJobRepository(lambda: session)

    with pytest.raises(UploadAccountUnavailable):
        repository.create(
            "process-gated",
            _job(),
            (StagedUpload(
                sequence=0,
                original_name="a.pdf",
                part_name="a.pdf",
                storage_key="a.pdf",
            ),),
            ({"step": "upload"},),
            staging_node_id="node-a",
        )

    assert "FOR SHARE" in session.sql[0]
    assert session.rolled_back
    assert len(session.sql) == 1, "no job or item may be inserted after the gate is observed"


def test_finalize_staging_rechecks_the_gate_before_queue_visibility() -> None:
    class _Result:
        def __init__(self, row: SimpleNamespace | None) -> None:
            self._row = row

        def fetchone(self) -> SimpleNamespace | None:
            return self._row

    class _Session:
        def __init__(self, *, deletion_requested: bool) -> None:
            self.deletion_requested = deletion_requested
            self.sql: list[str] = []
            self.committed = False
            self.rolled_back = False

        def execute(self, statement: Any, _params: Any = None) -> _Result:
            sql = str(statement)
            self.sql.append(sql)
            if len(self.sql) == 1:
                return _Result(SimpleNamespace(deletion_requested=self.deletion_requested))
            return _Result(SimpleNamespace(process_id="process-staging"))

        def commit(self) -> None:
            self.committed = True

        def rollback(self) -> None:
            self.rolled_back = True

        def close(self) -> None:
            pass

    live = _Session(deletion_requested=False)
    assert UploadJobRepository(lambda: live).finalize_staging("process-staging", "42")
    assert "FOR SHARE" in live.sql[0]
    assert "status = 'staging'" in live.sql[1]
    assert "cancellation_requested = FALSE" in live.sql[1]
    assert live.committed

    gated = _Session(deletion_requested=True)
    with pytest.raises(UploadAccountUnavailable):
        UploadJobRepository(lambda: gated).finalize_staging("process-staging", "42")
    assert len(gated.sql) == 1
    assert gated.rolled_back
