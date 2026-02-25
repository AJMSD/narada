from pathlib import Path

import pytest

from narada.live_notes import IntervalPlanner, SessionSpool, SpoolRecord


def _record(start: int, end: int, sample_rate_hz: int = 4, channels: int = 1) -> SpoolRecord:
    return SpoolRecord(
        start_byte=start,
        end_byte=end,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
    )


def test_session_spool_appends_and_reads_ranges(tmp_path: Path) -> None:
    spool = SessionSpool(base_dir=tmp_path, prefix="narada-test")
    first = spool.append_frame(pcm_bytes=b"\x01\x00" * 4, sample_rate_hz=4, channels=1)
    second = spool.append_frame(pcm_bytes=b"\x02\x00" * 4, sample_rate_hz=4, channels=1)
    payload = spool.read_range(start_byte=first.start_byte, end_byte=second.end_byte)

    assert first.start_byte == 0
    assert first.end_byte == 8
    assert second.start_byte == 8
    assert second.end_byte == 16
    assert payload == (b"\x01\x00" * 4 + b"\x02\x00" * 4)

    directory = spool.directory
    spool.cleanup(keep_files=False)
    assert not directory.exists()


def test_session_spool_compat_flush_mode_flushes_every_append(tmp_path: Path) -> None:
    spool = SessionSpool(
        base_dir=tmp_path,
        prefix="narada-test",
        flush_interval_seconds=0.0,
        flush_bytes=0,
    )
    frame = b"\x01\x00" * 2
    spool.append_frame(pcm_bytes=frame, sample_rate_hz=4, channels=1)

    assert spool.data_path.read_bytes() == frame
    lines = spool.index_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    spool.cleanup(keep_files=False)


def test_session_spool_flushes_when_byte_threshold_reached(tmp_path: Path) -> None:
    spool = SessionSpool(
        base_dir=tmp_path,
        prefix="narada-test",
        flush_interval_seconds=0.0,
        flush_bytes=8,
    )
    first = b"\x01\x00" * 2
    second = b"\x02\x00" * 2
    spool.append_frame(pcm_bytes=first, sample_rate_hz=4, channels=1)
    assert spool.data_path.read_bytes() == b""

    spool.append_frame(pcm_bytes=second, sample_rate_hz=4, channels=1)
    assert spool.data_path.read_bytes() == first + second

    spool.cleanup(keep_files=False)


def test_session_spool_rejects_negative_flush_thresholds(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        SessionSpool(base_dir=tmp_path, flush_interval_seconds=-0.1)
    with pytest.raises(ValueError):
        SessionSpool(base_dir=tmp_path, flush_bytes=-1)


def test_interval_planner_emits_overlap_windows_and_final_tail() -> None:
    planner = IntervalPlanner(interval_seconds=2.0, overlap_seconds=0.5)
    planner.ingest_record(_record(0, 8))
    assert planner.pop_next_ready_task() is None

    planner.ingest_record(_record(8, 16))
    task = planner.pop_next_ready_task()
    assert task is not None
    assert task.start_byte == 0
    assert task.end_byte == 16
    assert not task.is_final

    planner.ingest_record(_record(16, 24))
    assert planner.pop_next_ready_task() is None
    final_tasks = planner.build_final_tasks()
    assert len(final_tasks) == 1
    assert final_tasks[0].is_final
    assert final_tasks[0].start_byte == 12
    assert final_tasks[0].end_byte == 24


def test_interval_planner_flushes_tail_on_format_change() -> None:
    planner = IntervalPlanner(interval_seconds=2.0, overlap_seconds=0.5)
    planner.ingest_record(_record(0, 16, sample_rate_hz=4))
    first = planner.pop_next_ready_task()
    assert first is not None
    assert first.start_byte == 0
    assert first.end_byte == 16

    planner.ingest_record(_record(16, 24, sample_rate_hz=4))
    planner.ingest_record(_record(24, 32, sample_rate_hz=8))

    forced = planner.pop_next_ready_task()
    assert forced is not None
    assert forced.is_final
    assert forced.label == "format-tail"
    assert forced.sample_rate_hz == 4
    assert forced.start_byte == 12
    assert forced.end_byte == 24


def test_interval_planner_pending_backlog_counts_forced_tasks() -> None:
    planner = IntervalPlanner(interval_seconds=2.0, overlap_seconds=0.5)
    planner.ingest_record(_record(0, 16, sample_rate_hz=4))
    _ = planner.pop_next_ready_task()
    planner.ingest_record(_record(16, 24, sample_rate_hz=4))
    planner.ingest_record(_record(24, 32, sample_rate_hz=8))

    backlog_s = planner.pending_backlog_seconds()
    assert backlog_s == pytest.approx(2.0)


def test_interval_planner_build_final_tasks_marks_pending_backlog_as_drained() -> None:
    planner = IntervalPlanner(interval_seconds=2.0, overlap_seconds=0.5)
    planner.ingest_record(_record(0, 8))
    planner.ingest_record(_record(8, 16))

    assert planner.pending_backlog_seconds() > 0.0
    final_tasks = planner.build_final_tasks()

    assert final_tasks
    assert planner.pending_backlog_seconds() == pytest.approx(0.0)
