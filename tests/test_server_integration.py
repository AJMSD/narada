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


def _read_sse_event(response: http.client.HTTPResponse) -> tuple[str, str]:
    event_id = ""
    data_lines: list[str] = []
    while True:
        raw_line = response.fp.readline()
        if not raw_line:
            raise AssertionError("SSE stream ended before a full event was received.")
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if event_id or data_lines:
                return event_id, "\n".join(data_lines)
            continue
        if line.startswith("id:"):
            event_id = line[3:].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())


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
        event_id, data = _read_sse_event(response)
        assert int(event_id) > 0
        assert data == "first-line"
        conn.close()
    finally:
        server.stop_event.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_sse_reconnect_resumes_from_last_event_id_without_duplicates(tmp_path: Path) -> None:
    transcript_path = tmp_path / "session.txt"
    transcript_path.write_text("first-line\n", encoding="utf-8")
    server, thread = _start_server(transcript_path)
    port = server.server_address[1]
    try:
        first_conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        first_conn.request("GET", "/events")
        first_response = first_conn.getresponse()
        assert first_response.status == 200
        first_id, first_data = _read_sse_event(first_response)
        assert first_data == "first-line"
        first_conn.close()

        second_conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        second_conn.request("GET", "/events", headers={"Last-Event-ID": first_id})
        second_response = second_conn.getresponse()
        assert second_response.status == 200

        with transcript_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("second-line\n")
            handle.flush()

        second_id, second_data = _read_sse_event(second_response)
        assert second_data == "second-line"
        assert int(second_id) > int(first_id)
        second_conn.close()
    finally:
        server.stop_event.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_sse_invalid_last_event_id_falls_back_to_replay(tmp_path: Path) -> None:
    transcript_path = tmp_path / "session.txt"
    transcript_path.write_text("first-line\n", encoding="utf-8")
    server, thread = _start_server(transcript_path)
    port = server.server_address[1]
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/events", headers={"Last-Event-ID": "not-a-number"})
        response = conn.getresponse()
        assert response.status == 200
        _event_id, data = _read_sse_event(response)
        assert data == "first-line"
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
