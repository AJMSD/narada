from __future__ import annotations

import http.client
import threading
import urllib.request
from pathlib import Path

from narada.server import TranscriptHandler, TranscriptHTTPServer, start_transcript_server


def _start_server(transcript_path: Path) -> tuple[TranscriptHTTPServer, threading.Thread]:
    stop_event = threading.Event()
    server = TranscriptHTTPServer(("127.0.0.1", 0), TranscriptHandler, transcript_path, stop_event)
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.1},
        daemon=True,
    )
    thread.start()
    return server, thread


def test_transcript_endpoint_returns_raw_file(tmp_path: Path) -> None:
    transcript_path = tmp_path / "session.txt"
    transcript_path.write_text("line-one\nline-two\n", encoding="utf-8")
    server, thread = _start_server(transcript_path)
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/transcript.txt", timeout=5
        ) as response:
            body = response.read().decode("utf-8")
        assert "line-one" in body
        assert "line-two" in body
    finally:
        server.stop_event.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_sse_endpoint_streams_transcript_lines(tmp_path: Path) -> None:
    transcript_path = tmp_path / "session.txt"
    transcript_path.write_text("first-line\n", encoding="utf-8")
    server, thread = _start_server(transcript_path)
    port = server.server_address[1]
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/events")
        response = conn.getresponse()
        assert response.status == 200
        first_event = response.fp.readline().decode("utf-8").strip()
        assert first_event == "data: first-line"
        conn.close()
    finally:
        server.stop_event.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_start_transcript_server_helper_starts_and_stops(tmp_path: Path) -> None:
    transcript_path = tmp_path / "session.txt"
    transcript_path.write_text("line-one\n", encoding="utf-8")
    running = start_transcript_server(
        transcript_path=transcript_path,
        bind="127.0.0.1",
        port=0,
    )
    try:
        assert running.access_url.startswith("http://127.0.0.1:")
        with urllib.request.urlopen(f"{running.access_url}/transcript.txt", timeout=5) as response:
            body = response.read().decode("utf-8")
        assert "line-one" in body
    finally:
        running.stop()
