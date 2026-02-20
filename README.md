# Narada

## What Narada Is
Narada is a local-first CLI tool for transcribing meeting audio to text files. It is designed to run without cloud upload and to support microphone, system, and mixed audio workflows across Windows, Linux, and macOS.

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
- Shared logical IDs for combo devices; Narada auto-selects input/output endpoint by command context.
- `--language auto` default with multilingual input support through comma-separated values.
- Software mixed mode target (mic + system in Narada).
- Core PII redaction support (excluding names).
- Append-only transcript writing with frequent flush and fsync.
- Automatic hardware channel count detection for system capture; stereo WASAPI loopback devices are opened at their native channel count and downmixed to mono before ASR.
- Optional LAN serving directly from `narada start --serve`.
- LAN live view endpoints:
  - `/` browser page
  - `/transcript.txt` raw transcript file
  - `/events` SSE stream

## ASR Engine Selection
Narada supports two engine adapters from the start:
- `faster-whisper`
- `whisper-cpp`

Select engine with:
```bash
narada start --mode mic --mic 1 --engine faster-whisper
```

## Model Setup
Narada checks local model availability at startup and shows setup guidance when models are missing.

Model sources:
- faster-whisper: `https://huggingface.co/Systran/faster-whisper-small`
- whisper.cpp: `https://huggingface.co/ggerganov/whisper.cpp`

Typical local paths on Windows:
- faster-whisper cache snapshot:
  - `C:\Users\<you>\.cache\huggingface\hub\models--Systran--faster-whisper-small\...`
- whisper.cpp model file:
  - `C:\Users\<you>\AppData\Local\narada\models\whisper-cpp\ggml-small.bin`

One-time setup behavior:
- If a selected model is missing and another local model family is available, Narada reports that and can run with the available one.
- Once model files are present locally, runs are offline unless you choose to fetch/update models.

## Limitations
- System-audio capture depends on OS and backend support. On Windows, WASAPI loopback devices report their channel count at runtime; Narada queries this automatically and downmixes to mono before transcription.
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

Start mixed mode with different microphone and system-output devices:
```bash
narada start --mode mixed --mic 1 --system 7 --out ./transcripts/session.txt
```

Start and serve live transcript in one command:
```bash
narada start --mode mixed --mic 1 --system 7 --out ./transcripts/session.txt --serve --bind 0.0.0.0 --port 8787 --qr
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

Run checks:
```bash
narada doctor --file ./transcripts/session.txt
```

Structured stdin for `start` (useful for tests or automation):
- Plain text line: treated as transcript text.
- JSON text payload: `{"text":"hello","confidence":0.9}`
- JSON audio payload (mic/system): `{"audio":{"samples":[...],"sample_rate_hz":16000,"channels":1}}`
- JSON mixed payload: `{"mic":{...},"system":{...}}` (Narada normalizes and software-mixes before ASR).

## Privacy And Security Behavior
- No telemetry by default.
- No cloud upload by default.
- LAN serving is opt-in and active only when running `narada serve` or `narada start --serve`.
- Binding to `0.0.0.0` prints a warning because local-network devices can access the endpoint.
- Redaction does not attempt named-entity inference, and personal names are intentionally not masked.

## Configuration
Flags override environment values.

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
- `NARADA_BIND` -> `--bind`
- `NARADA_PORT` -> `--port`
