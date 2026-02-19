from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Narada Live Transcript</title>
  <style>
    body {
      font-family: "Segoe UI", Tahoma, sans-serif;
      margin: 24px;
      background: #f2f5f7;
      color: #17212b;
    }
    h1 { margin-top: 0; }
    #log {
      white-space: pre-wrap;
      background: #ffffff;
      border: 1px solid #d8dee4;
      padding: 16px;
      border-radius: 8px;
      min-height: 300px;
    }
  </style>
</head>
<body>
  <h1>Narada Live Transcript</h1>
  <p>
    Streaming updates from <code>/events</code>.
    Raw transcript: <a href="/transcript.txt">/transcript.txt</a>
  </p>
  <div id="log"></div>
  <script>
    const log = document.getElementById("log");
    const source = new EventSource("/events");
    source.onmessage = (event) => {
      log.textContent += event.data + "\\n";
      window.scrollTo(0, document.body.scrollHeight);
    };
  </script>
</body>
</html>
"""


class TranscriptHTTPServer(ThreadingHTTPServer):
    transcript_path: Path
    stop_event: threading.Event

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler: type[BaseHTTPRequestHandler],
        transcript_path: Path,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(server_address, request_handler)
        self.transcript_path = transcript_path
        self.stop_event = stop_event


class TranscriptHandler(BaseHTTPRequestHandler):
    server: TranscriptHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_index()
            return
        if parsed.path == "/transcript.txt":
            self._serve_transcript()
            return
        if parsed.path == "/events":
            self._serve_events()
            return
        self.send_response(404)
        self.end_headers()

    def _serve_index(self) -> None:
        payload = INDEX_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_transcript(self) -> None:
        path = self.server.transcript_path
        if not path.exists():
            payload = b""
        else:
            payload = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        last_size = 0
        while not self.server.stop_event.is_set():
            path = self.server.transcript_path
            try:
                if path.exists():
                    content = path.read_text(encoding="utf-8")
                    current_size = len(content)
                    if current_size < last_size:
                        last_size = 0
                    if current_size > last_size:
                        delta = content[last_size:current_size]
                        for line in delta.splitlines():
                            event = f"data: {line}\n\n"
                            self.wfile.write(event.encode("utf-8"))
                        self.wfile.flush()
                        last_size = current_size
                time.sleep(0.5)
            except (BrokenPipeError, ConnectionResetError):
                break

    def log_message(self, format: str, *args: object) -> None:
        return


def resolve_lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def build_access_url(bind: str, port: int) -> str:
    host = bind
    if bind in {"0.0.0.0", "::"}:
        host = resolve_lan_ip()
    return f"http://{host}:{port}"


def render_ascii_qr(url: str) -> str:
    try:
        import qrcode
    except ImportError:
        return "Install qrcode[pil] to enable ASCII QR rendering."

    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    matrix = qr.get_matrix()
    lines: list[str] = []
    for row in matrix:
        line = "".join("##" if cell else "  " for cell in row)
        lines.append(line)
    return "\n".join(lines)


@dataclass
class RunningTranscriptServer:
    server: TranscriptHTTPServer
    thread: threading.Thread
    stop_event: threading.Event
    access_url: str

    def stop(self) -> None:
        self.stop_event.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


def start_transcript_server(
    transcript_path: Path,
    bind: str,
    port: int,
) -> RunningTranscriptServer:
    stop_event = threading.Event()
    server = TranscriptHTTPServer((bind, port), TranscriptHandler, transcript_path, stop_event)
    access_url = build_access_url(bind, int(server.server_address[1]))
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.5},
        daemon=True,
    )
    thread.start()
    return RunningTranscriptServer(
        server=server,
        thread=thread,
        stop_event=stop_event,
        access_url=access_url,
    )


def serve_transcript_file(
    transcript_path: Path,
    bind: str,
    port: int,
    show_qr: bool,
) -> None:
    running_server = start_transcript_server(transcript_path, bind, port)
    print(f"Serving transcript from {transcript_path}")
    print(f"URL: {running_server.access_url}")
    if bind == "0.0.0.0":
        print("Warning: server bound to all interfaces on local network.")
    if show_qr:
        print(render_ascii_qr(running_server.access_url))

    try:
        while running_server.thread.is_alive():
            running_server.thread.join(timeout=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        running_server.stop()
