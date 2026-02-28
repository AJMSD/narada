# Narada

## What Narada Is
Narada is a local-first CLI tool for transcribing meeting audio to text files. It is designed to run without cloud upload and to support microphone and system audio workflows across Windows, Linux, and macOS.

## Why I Built It
Many meeting transcription tools require paid APIs or cloud upload of sensitive audio. Narada is built to provide a private local workflow with clear CLI controls and LAN-only sharing when explicitly enabled.

## Features
- Local transcript pipeline with no default network upload path.
- CLI commands:
  - `narada devices`
  - `narada start`
  - `narada serve`
  - `narada doctor`
- Device selection by ID or name with fuzzy matching support.
- Automatic OS-aware device deduplication in `narada devices` (use `--all` for raw endpoints).
- On Windows, default `narada devices` output excludes legacy MME and DirectSound endpoints before deduplication; use `--all` to inspect raw host-API entries.
- On Windows, live `mic` and `system` capture both use the same backend (`PyAudioWPatch`) so listed device IDs match capture runtime behavior.
- Shared logical IDs for combo devices; Narada auto-selects input/output endpoint by command context.
- `--language auto` default with multilingual input support through comma-separated values.
- Automatic model download and cache reuse for supported ASR engines.
- Continuous live capture for mic/system modes with low-latency chunk defaults
  (2.0s chunk, 0.5s overlap) and wall-clock forced flush controls.
- Notes-first live runtime for TTY sessions: audio is continuously spooled to disk,
  interval ASR runs in a background worker, and finalization performs a tail pass.
- Core PII redaction support (excluding names).
- Append-only transcript writing with configurable fsync policy
  (default remains per-line fsync).
- Notes spool flushing is batched by interval/byte thresholds and remains
  durably fsynced on close.
- Automatic hardware channel count detection for system capture; stereo WASAPI loopback devices are opened at their native channel count and downmixed to mono before ASR. If the detected count is rejected by the driver, Narada retries automatically through common fallback values (2, 1) before raising an error.
- Optional LAN serving directly from `narada start --serve`.
- Optional LAN token auth via `--serve-token` for `/`, `/transcript.txt`, and `/events`.
- For faster-whisper on `--compute auto|cuda`, Narada automatically falls back to
  CPU for the current session if GPU runtime/transcription worker errors or timeouts occur.
- Faster-whisper decode presets: `fast`, `balanced` (default), `accurate`.
- For whisper.cpp, Narada probes `whisper-cli` flag support at runtime and logs
  the resolved compute behavior (for example CPU no-GPU flags and backend hints).
- LAN live view endpoints:
  - `/` browser page
  - `/transcript.txt` raw transcript file
  - `/events` SSE stream

## ASR Engine Selection
Narada supports two engine adapters from the start:
- `faster-whisper`
- `whisper-cpp`

`whisper-cpp` runtime requires the `whisper-cli` binary available on your `PATH`.

Select engine with:
```bash
narada start --mode mic --mic 1 --engine faster-whisper
```

Select faster-whisper decode preset with:
```bash
narada start --mode mic --mic 1 --engine faster-whisper --asr-preset balanced
```

## Model Setup
Narada checks local model availability at startup and auto-downloads missing models for the selected engine.

Model sources:
- faster-whisper: `https://huggingface.co/Systran/faster-whisper-small`
- whisper.cpp: `https://huggingface.co/ggerganov/whisper.cpp`

Typical local paths on Windows:
- faster-whisper cache snapshot:
  - `C:\Users\<you>\.cache\huggingface\hub\models--Systran--faster-whisper-small\...`
- whisper.cpp model file:
  - `C:\Users\<you>\AppData\Local\narada\models\whisper-cpp\ggml-small.bin`

One-time setup behavior:
- If a selected model is missing, Narada downloads it into the expected cache/model directory and reuses it on future runs.
- If selected-engine download fails and another engine already has local model files, Narada falls back to that local engine for the run.
- Once model files are present locally, runs are offline unless you choose to fetch/update models.

## Limitations
- System-audio capture depends on OS and backend support. On Windows, Narada uses `PyAudioWPatch` and WASAPI loopback sources; channel count is detected at runtime and downmixed to mono before transcription, with fallback through common values when needed.
- Windows live capture (`--mode mic`, `--mode system`) requires `PyAudioWPatch`.
- Bluetooth HFP devices (Hands-Free Profile) do not expose a standard PCM loopback endpoint on Windows and will produce a descriptive error. If your driver exposes **Stereo Mix**, use it as an input path (`narada start --mode mic --mic <stereo-mix-id>`).
- macOS system capture usually requires a virtual loopback device (for example BlackHole).
- ASR runtime availability depends on optional dependencies being installed.
- Current scaffold focuses on stable interfaces, validation, and quality gates while real-time capture integrations are expanded.
- Redaction is regex-based and best effort. False positives and false negatives are possible.

## Setup Guide
1. Install Python 3.11 or newer.
2. Create and activate a virtual environment.
3. Install project and dev tools:
```bash
pip install -e ".[dev]"
```
4. Install optional ASR and audio extras:
```bash
pip install -e ".[asr,audio,noise]"
```
Optional alternative (single requirements file with tested optional runtime deps):
```bash
pip install -r requirements.txt
```
5. Run verification:
```powershell
./scripts/verify.ps1
```

## Usage Examples
List devices:
```bash
narada devices
```

Show full raw backend endpoints (no dedupe):
```bash
narada devices --all
```

Filter devices:
```bash
narada devices --type input --search yeti
```

Start a session:
```bash
narada start --mode mic --mic 1 --out ./transcripts/session.txt --language auto
```

Start system capture:
```bash
narada start --mode system --system 7 --out ./transcripts/session.txt
```

Start and serve live transcript in one command:
```bash
narada start --mode mic --mic 1 --out ./transcripts/session.txt --serve --bind 0.0.0.0 --port 8787 --qr
```

Tune live wall-clock flush and capture backlog warnings:
```bash
narada start --mode system --system 7 --out ./transcripts/session.txt --wall-flush-seconds 60 --capture-queue-warn-seconds 120
```

Tune notes-first interval scheduling and spool retention:
```bash
narada start --mode mic --mic 1 --out ./transcripts/session.txt --notes-interval-seconds 12 --notes-overlap-seconds 1.5 --notes-commit-holdback-windows 1 --asr-backlog-warn-seconds 45 --keep-spool
```

Tune notes spool flush batching thresholds:
```bash
narada start --mode mic --mic 1 --out ./transcripts/session.txt --spool-flush-interval-seconds 0.25 --spool-flush-bytes 65536
```

Use compatibility mode for spool flush (flush every append):
```bash
narada start --mode mic --mic 1 --out ./transcripts/session.txt --spool-flush-interval-seconds 0 --spool-flush-bytes 0
```

Use periodic transcript fsync mode:
```bash
narada start --mode mic --mic 1 --out ./transcripts/session.txt --writer-fsync-mode periodic --writer-fsync-lines 20 --writer-fsync-seconds 1.0
```

Choose faster-whisper decode presets:
```bash
narada start --mode mic --mic 1 --engine faster-whisper --asr-preset fast
narada start --mode mic --mic 1 --engine faster-whisper --asr-preset balanced
narada start --mode mic --mic 1 --engine faster-whisper --asr-preset accurate
```

Enable debug logging:
```bash
narada --debug --log-file ./logs/narada.log start --mode mic --mic 1
```

Multilingual session:
```bash
narada start --mode mic --mic 1 --language hindi,english --allow-multilingual
```

Serve transcript on LAN:
```bash
narada serve --file ./transcripts/session.txt --port 8787 --qr --bind 0.0.0.0
```

Serve transcript on LAN with token auth:
```bash
narada serve --file ./transcripts/session.txt --port 8787 --bind 0.0.0.0 --serve-token mytoken
```

Start + serve with token auth in one command:
```bash
narada start --mode system --system 7 --out ./transcripts/session.txt --serve --bind 0.0.0.0 --port 8787 --serve-token mytoken
```

Run checks:
```bash
narada doctor --file ./transcripts/session.txt
```

Windows loopback smoke test:
```bash
narada devices
narada start --mode system --system <output-id> --out ./transcripts/system.txt
```
Play audio while running and confirm transcript updates are non-silent.

Structured stdin for `start` (useful for tests or automation):
- Plain text line: treated as transcript text.
- JSON text payload: `{"text":"hello","confidence":0.9}`
- JSON audio payload (mic/system): `{"audio":{"samples":[...],"sample_rate_hz":16000,"channels":1}}`
- Legacy mixed payloads are rejected with migration guidance; run separate `mic` and `system` sessions.

## ASR Presets And Benchmarking
Preset behavior for faster-whisper:
- `fast`: lower latency and CPU, lower decode search.
- `balanced` (default): current baseline behavior.
- `accurate`: stronger context carry-over, highest compute of the three.

Quick local benchmark recipe:
1. Use one fixed 1-3 minute audio sample (same playback path and system load for each run).
2. Run faster-whisper with each preset on the same model.
3. Run whisper-cpp on the same model.
4. Compare final `rtf=...` values in status output and compare transcript quality manually.

Example commands:
```bash
narada start --mode system --system <id> --model small --engine faster-whisper --asr-preset fast --out ./transcripts/fw-fast.txt
narada start --mode system --system <id> --model small --engine faster-whisper --asr-preset balanced --out ./transcripts/fw-balanced.txt
narada start --mode system --system <id> --model small --engine faster-whisper --asr-preset accurate --out ./transcripts/fw-accurate.txt
narada start --mode system --system <id> --model small --engine whisper-cpp --out ./transcripts/wcpp.txt
```

## Privacy And Security Behavior
- No telemetry by default.
- No cloud upload by default.
- LAN serving is opt-in and active only when running `narada serve` or `narada start --serve`.
- Binding to `0.0.0.0` prints a warning because local-network devices can access the endpoint.
- If `--serve-token` is set, all HTTP and SSE endpoints require `?token=<value>`.
- If no token is configured, serving behavior remains backward compatible (no auth).
- Redaction does not attempt named-entity inference, and personal names are intentionally not masked.

## Configuration
Flags override environment values.

Default values for new knobs:
- `--spool-flush-interval-seconds 0.25`
- `--spool-flush-bytes 65536`
- `--writer-fsync-mode line`
- `--writer-fsync-lines 20`
- `--writer-fsync-seconds 1.0`
- `--asr-preset balanced`
- `--serve-token` unset (disabled)

Environment variable map:
- `NARADA_MODE` -> `--mode`
- `NARADA_MIC` -> `--mic`
- `NARADA_SYSTEM` -> `--system`
- `NARADA_OUT` -> `--out`
- `NARADA_MODEL` -> `--model`
- `NARADA_COMPUTE` -> `--compute`
- `NARADA_ENGINE` -> `--engine`
- `NARADA_MODEL_DIR_FASTER_WHISPER` -> `--model-dir-faster-whisper`
- `NARADA_MODEL_DIR_WHISPER_CPP` -> `--model-dir-whisper-cpp`
- `NARADA_LANGUAGE` -> `--language`
- `NARADA_ALLOW_MULTILINGUAL` -> `--allow-multilingual`
- `NARADA_REDACT` -> `--redact`
- `NARADA_NOISE_SUPPRESS` -> `--noise-suppress`
- `NARADA_AGC` -> `--agc`
- `NARADA_GATE` -> `--gate`
- `NARADA_GATE_THRESHOLD_DB` -> `--gate-threshold-db`
- `NARADA_CONFIDENCE_THRESHOLD` -> `--confidence-threshold`
- `NARADA_WALL_FLUSH_SECONDS` -> `--wall-flush-seconds`
- `NARADA_CAPTURE_QUEUE_WARN_SECONDS` -> `--capture-queue-warn-seconds`
- `NARADA_NOTES_INTERVAL_SECONDS` -> `--notes-interval-seconds`
- `NARADA_NOTES_OVERLAP_SECONDS` -> `--notes-overlap-seconds`
- `NARADA_NOTES_COMMIT_HOLDBACK_WINDOWS` -> `--notes-commit-holdback-windows`
- `NARADA_ASR_BACKLOG_WARN_SECONDS` -> `--asr-backlog-warn-seconds`
- `NARADA_KEEP_SPOOL` -> `--keep-spool/--no-keep-spool`
- `NARADA_SPOOL_FLUSH_INTERVAL_SECONDS` -> `--spool-flush-interval-seconds`
- `NARADA_SPOOL_FLUSH_BYTES` -> `--spool-flush-bytes`
- `NARADA_WRITER_FSYNC_MODE` -> `--writer-fsync-mode`
- `NARADA_WRITER_FSYNC_LINES` -> `--writer-fsync-lines`
- `NARADA_WRITER_FSYNC_SECONDS` -> `--writer-fsync-seconds`
- `NARADA_ASR_PRESET` -> `--asr-preset`
- `NARADA_SERVE_TOKEN` -> `--serve-token`
- `NARADA_BIND` -> `--bind`
- `NARADA_PORT` -> `--port`
