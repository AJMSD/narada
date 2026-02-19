from pathlib import Path

from narada.logging_setup import setup_logging


def test_setup_logging_creates_log_file_and_writes(tmp_path: Path) -> None:
    log_file = tmp_path / "narada" / "narada.log"
    logger = setup_logging(debug=True, log_file=log_file)
    logger.info("hello from test")

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "hello from test" in content
