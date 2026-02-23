from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

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
    Streaming updates from <code>__EVENTS_URL__</code>.
    Raw transcript: <a href="__TRANSCRIPT_URL__">__TRANSCRIPT_URL__</a>
  </p>
  <div id="log"></div>
  <script>
    const log = document.getElementById("log");
    const source = new EventSource("__EVENTS_URL__");
    let lastRenderedEventId = -1;
    source.onmessage = (event) => {
      const parsedEventId = Number.parseInt(event.lastEventId || "", 10);
      if (Number.isFinite(parsedEventId)) {
        if (parsedEventId <= lastRenderedEventId) {
          return;
        }
        lastRenderedEventId = parsedEventId;
      }
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
    serve_token: str | None

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler: type[BaseHTTPRequestHandler],
        transcript_path: Path,
        stop_event: threading.Event,
        serve_token: str | None,
    ) -> None:
        super().__init__(server_address, request_handler)
        self.transcript_path = transcript_path
        self.stop_event = stop_event
        self.serve_token = serve_token


class TranscriptHandler(BaseHTTPRequestHandler):
    server: TranscriptHTTPServer

    def _is_authorized(self, parsed_path: object) -> bool:
        token = self.server.serve_token
        if token is None:
            return True
        query = parse_qs(getattr(parsed_path, "query", ""), keep_blank_values=True)
        provided = query.get("token")
        return bool(provided and provided[0] == token)

    def _reject_unauthorized(self) -> None:
        payload = b"Unauthorized"
        self.send_response(401)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            if not self._is_authorized(parsed):
                self._reject_unauthorized()
                return
            self._serve_index()
            return
        if parsed.path == "/transcript.txt":
            if not self._is_authorized(parsed):
                self._reject_unauthorized()
                return
            self._serve_transcript()
            return
        if parsed.path == "/events":
            if not self._is_authorized(parsed):
                self._reject_unauthorized()
                return
            self._serve_events()
            return
        self.send_response(404)
        self.end_headers()

    def _serve_index(self) -> None:
        token_suffix = ""
        if self.server.serve_token is not None:
            token_suffix = f"?token={quote(self.server.serve_token, safe='')}"
        events_url = f"/events{token_suffix}"
        transcript_url = f"/transcript.txt{token_suffix}"
        payload = (
            INDEX_HTML.replace("__EVENTS_URL__", events_url).replace(
                "__TRANSCRIPT_URL__", transcript_url
            )
        ).encode("utf-8")
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

        cursor = self._parse_last_event_cursor(self.headers.get("Last-Event-ID"))
        while not self.server.stop_event.is_set():
            path = self.server.transcript_path
            try:
                if path.exists():
                    content = path.read_text(encoding="utf-8")
                    current_size = len(content)
                    if current_size < cursor:
                        cursor = 0
                    sent_any = False
                    while cursor < current_size:
                        newline_index = content.find("\n", cursor)
                        if newline_index == -1:
                            break
                        line = content[cursor:newline_index]
                        cursor = newline_index + 1
                        event = f"id: {cursor}\ndata: {line}\n\n"
                        self.wfile.write(event.encode("utf-8"))
                        sent_any = True
                    if sent_any:
                        self.wfile.flush()
                time.sleep(0.5)
            except (BrokenPipeError, ConnectionResetError):
                break

    @staticmethod
    def _parse_last_event_cursor(raw_value: str | None) -> int:
        if raw_value is None:
            return 0
        try:
            value = int(raw_value.strip())
        except ValueError:
            return 0
        return max(0, value)

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


def build_access_url(bind: str, port: int, serve_token: str | None = None) -> str:
    host = bind
    if bind in {"0.0.0.0", "::"}:
        host = resolve_lan_ip()
    base = f"http://{host}:{port}"
    if serve_token is None:
        return base
    return f"{base}?token={quote(serve_token, safe='')}"


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
    serve_token: str | None = None,
) -> RunningTranscriptServer:
    stop_event = threading.Event()
    server = TranscriptHTTPServer(
        (bind, port),
        TranscriptHandler,
        transcript_path,
        stop_event,
        serve_token,
    )
    access_url = build_access_url(bind, int(server.server_address[1]), serve_token)
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
    serve_token: str | None = None,
) -> None:
    running_server = start_transcript_server(
        transcript_path,
        bind,
        port,
        serve_token=serve_token,
    )
    print(f"Serving transcript from {transcript_path}")
    print(f"URL: {running_server.access_url}")
    if bind == "0.0.0.0":
        print("Warning: server bound to all interfaces on local network.")
        if serve_token is None:
            print("Warning: LAN server is unauthenticated. Set --serve-token to require access.")
    if show_qr:
        print(render_ascii_qr(running_server.access_url))

    try:
        while running_server.thread.is_alive():
            running_server.thread.join(timeout=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        running_server.stop()
