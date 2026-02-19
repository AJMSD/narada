# Narada

## What Narada Is
Narada is a local-first CLI tool for transcribing meeting audio to text files. It is designed to run without cloud upload and to support microphone, system, and mixed audio workflows across Windows, Linux, and macOS.

## Why We Built It
Many meeting transcription tools require paid APIs or cloud upload of sensitive audio. Narada is built to provide a private local workflow with clear CLI controls and LAN-only sharing when explicitly enabled.

## Features
- Local transcript pipeline with no default network upload path.
- CLI commands:
  - `narada devices`
  - `narada start`
  - `narada serve`
  - `narada doctor`
- Device selection by ID or name with fuzzy matching support.
- `--language auto` default with multilingual input support through comma-separated values.
- Software mixed mode target (mic + system in Narada).
- Core PII redaction support (excluding names).
- Append-only transcript writing with frequent flush and fsync.
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

## Limitations
- System-audio capture depends on OS and backend support.
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
5. Run verification:
```powershell
./scripts/verify.ps1
```

## Usage Examples
List devices:
```bash
narada devices
```

Filter devices:
```bash
narada devices --type input --search yeti
```

Start a session:
```bash
narada start --mode mic --mic 1 --out ./transcripts/session.txt --language auto
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

## Privacy And Security Behavior
- No telemetry by default.
- No cloud upload by default.
- LAN serving is opt-in and only active when running `narada serve`.
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
