from pathlib import Path

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
