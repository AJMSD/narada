from pathlib import Path

import pytest

from narada.writer import TranscriptWriter


def test_append_lines_and_resume(tmp_path: Path) -> None:
    path = tmp_path / "transcript.txt"
    with TranscriptWriter(path) as writer:
        writer.append_line("first line")
        writer.append_line("second line")

    with TranscriptWriter(path) as writer:
        writer.append_line("third line")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == ["first line", "second line", "third line"]


def test_periodic_fsync_triggers_on_line_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "periodic-lines.txt"
    sync_calls = {"count": 0}

    def _fake_fsync(_fd: int) -> None:
        sync_calls["count"] += 1

    monkeypatch.setattr("narada.writer.os.fsync", _fake_fsync)
    with TranscriptWriter(path, fsync_mode="periodic", fsync_lines=2, fsync_seconds=0.0) as writer:
        writer.append_line("one")
        writer.append_line("two")
        assert sync_calls["count"] == 1
        writer.append_line("three")
        assert sync_calls["count"] == 1

    # Close always performs a final fsync.
    assert sync_calls["count"] == 2


def test_periodic_fsync_triggers_on_time_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "periodic-time.txt"
    sync_calls = {"count": 0}
    monotonic_values = iter([100.0, 100.0, 101.0, 101.0])

    def _fake_fsync(_fd: int) -> None:
        sync_calls["count"] += 1

    def _fake_monotonic() -> float:
        return next(monotonic_values)

    monkeypatch.setattr("narada.writer.os.fsync", _fake_fsync)
    monkeypatch.setattr("narada.writer.time.monotonic", _fake_monotonic)
    with TranscriptWriter(path, fsync_mode="periodic", fsync_lines=0, fsync_seconds=0.5) as writer:
        writer.append_line("one")
        assert sync_calls["count"] == 0
        writer.append_line("two")
        assert sync_calls["count"] == 1

    assert sync_calls["count"] == 2


def test_writer_rejects_invalid_periodic_configuration(tmp_path: Path) -> None:
    path = tmp_path / "invalid.txt"
    with pytest.raises(ValueError):
        TranscriptWriter(path, fsync_mode="bad")
    with pytest.raises(ValueError):
        TranscriptWriter(path, fsync_mode="periodic", fsync_lines=0, fsync_seconds=0.0)
