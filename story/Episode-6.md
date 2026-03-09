# Episode 6

## 1. Context for This Episode
- This episode covered the full performance + security hardening cycle for Narada: planning constraints, implementation in commit batches, verification, and post-implementation benchmarking/debugging.
- The working goal was to deliver non-breaking improvements (spool batching, writer fsync policy, ASR presets, optional serve token), keep mixed mode and no-drop policies intact, and then validate runtime behavior under realistic benchmark conditions.

## 2. Main Problems We Faced

### Issue A: Constraint-heavy scope boundaries
- **Symptoms / signals:** Explicit constraints disallowed queue-drop logic, mixed-mode redesign, and lock-scope refactor while still requiring performance/security gains.
- **Why it mattered:** Typical throughput fixes were off-limits, so every change had to be additive, compatible, and reversible.

### Issue B: Partial server-auth implementation state
- **Symptoms / signals:** Token-auth changes existed in a partial unverified state across CLI/server/tests.
- **Why it mattered:** Regressions were likely unless integration paths were validated before commit.

### Issue C: Brittle CLI assertion under Rich/Typer output wrapping
- **Symptoms / signals:** `test_start_rejects_serve_options_without_serve` failed because expected phrase matching broke on wrapped error formatting.
- **Why it mattered:** CI reliability issue with correct behavior but unstable assertion style.

### Issue D: Formatting/lint failures during closeout
- **Symptoms / signals:** Ruff `E501` line-length failures and later `ruff format --check` failures blocked verify gate.
- **Why it mattered:** Could not claim completion against project quality gates.

### Issue E: Type-check failures after feature landing
- **Symptoms / signals:** `mypy src` errors in `config.py` and `faster_whisper_engine.py`.
- **Why it mattered:** Definition-of-done in checklist required type checks to pass.

### Issue F: "Done or not?" ambiguity from mismatched quality signals
- **Symptoms / signals:** Feature commits were merged, tests passed, but `verify.ps1` initially failed (format and mypy).
- **Why it mattered:** Feature completion and technical completion diverged; needed explicit closeout work.

### Issue G: Benchmark realism gap
- **Symptoms / signals:** Short JFK benchmark produced WER near/at 0, conflicting with real user experience.
- **Why it mattered:** Short/easy samples were not representative enough for model/preset recommendations.

### Issue H: Benchmark runner execution pitfalls on Windows multiprocessing
- **Symptoms / signals:** `spawn` path errors such as `OSError ... '<stdin>'` and runtime bootstrapping issues when launched from stdin-style scripts.
- **Why it mattered:** Could falsely look like ASR failure when actually benchmark harness invocation was invalid.

### Issue I: GPU worker timeout behavior under heavier presets
- **Symptoms / signals:** `faster-whisper` CUDA `fast` could pass, but `balanced`/`accurate` failed with `GPU worker exceeded timeout (12.0s)` and session GPU disable.
- **Why it mattered:** GPU path appeared unstable for realistic long-window decode settings; required code-path diagnosis and targeted fix.

### Issue J: Benchmark data integrity issue
- **Symptoms / signals:** One run reported absurd audio duration (`~134k s`) due WAV frame-count/header mismatch.
- **Why it mattered:** Produced invalid RTF math and misleading extrapolation unless corrected.

## 3. Debugging Path & Options Considered

### Issue A: Constraint-heavy scope
- **Debugging path:** Reconfirmed constraints repeatedly and mapped all changes to opt-in knobs or compatibility-preserving defaults.
- **Options considered:** queue bounding/drop policy, mixed split-mode path, additive tuning-only approach.
- **Outcome:** chose additive tuning-only path.

### Issue B: Partial server-auth state
- **Debugging path:** diff review + targeted tests before commit (`server` + `start` integration paths).
- **Options considered:** commit directly vs validate first.
- **Outcome:** validate first, then commit.

### Issue C: CLI assertion brittleness
- **Debugging path:** reproduced exact CLI output, normalized assertion strategy.
- **Options considered:** exact phrase check vs semantic token checks.
- **Outcome:** semantic token checks.

### Issue D/E/F: Quality gate closeout
- **Debugging path:** run `verify.ps1`, fix formatting first, then mypy, rerun full verify.
- **Options considered:** broad refactor vs minimal surgical fixes.
- **Outcome:** minimal fixes; no behavior drift.

### Issue G: Benchmark realism
- **Debugging path:** moved from short clip to longer reference-backed sample (~5 min LibriSpeech-derived).
- **Options considered:** keep short clip vs build long sample with references.
- **Outcome:** long reference-backed benchmark.

### Issue H: Multiprocessing harness reliability
- **Debugging path:** moved benchmark execution into file-based, `__main__`-guarded scripts to avoid stdin spawn artifacts.
- **Options considered:** inline stdin scripts vs spawn-safe script files.
- **Outcome:** spawn-safe files.

### Issue I: CUDA preset failures
- **Debugging path:** traced worker lifecycle and timeout constants; reproduced startup latency and request timeout boundaries.
- **Options considered:** environment reinstall vs timeout/policy analysis in code.
- **Outcome:** identified code-path timeout policy as primary blocker, not missing CUDA installation.

### Issue J: Wrong audio duration
- **Debugging path:** cross-checked duration source with metadata and switched to robust decode-based duration calculation.
- **Options considered:** trust WAV header vs derive from PCM bytes.
- **Outcome:** derive from decoded PCM bytes.

## 4. Final Solution Used (For This Chat)

### Implementation batches completed
- Added config/CLI/env knobs for spool batching, writer fsync policy, ASR preset, serve token.
- Implemented spool flush batching with close-path durability preserved.
- Added optional periodic writer fsync while keeping default per-line behavior.
- Added faster-whisper presets (`fast`, `balanced`, `accurate`) with balanced default compatibility.
- Added optional token auth for `/`, `/transcript.txt`, `/events` while preserving no-token backward compatibility.
- Updated README docs and examples.

### Closeout fixes completed after initial implementation
- **Formatting pass:** normalized ruff formatting in touched core files.
- **Type-check pass:** resolved mypy issues in `src/narada/config.py` and `src/narada/asr/faster_whisper_engine.py`.
- **Verification pass:** full `scripts/verify.ps1` passing.

### GPU reliability fixes implemented
- Split GPU worker startup timeout from probe timeout in `src/narada/asr/faster_whisper_engine.py` (`_GPU_STARTUP_TIMEOUT_S` introduced, startup wait no longer tied to 4s probe timeout).
- Updated `compute=auto` backend resolution to prefer CUDA when CUDA devices are available; CPU fallback otherwise.
- Added/updated tests in `tests/test_faster_whisper_engine.py` for auto backend selection and CUDA-availability branches.

### Benchmarking outcomes
- Added local untracked benchmarking harness and ran sequential 5-minute tests.
- Confirmed realistic non-zero WER on long sample.
- Confirmed CUDA fast path can run, while heavier CUDA presets can still hit per-request timeout budget in current design.
- Generated CPU/GPU comparison tables and one-hour extrapolation estimates from measured RTF.

## 5. Tools, APIs, and Concepts Used
- **Typer + Rich CLI output:** surfaced assertion fragility in wrapped error text.
- **Pytest integration + unit tests:** used for behavior-locking while patching CLI/server/ASR logic.
- **Ruff (`check`, `format --check`):** enforced style compliance before finalization.
- **mypy:** caught strict typing issues during closeout.
- **`scripts/verify.ps1`:** canonical quality gate pipeline for done/not-done status.
- **`multiprocessing` spawn model (Windows):** required `__main__`-safe benchmark scripts to avoid false failures.
- **faster-whisper / ctranslate2:** used for CUDA capability checks and decode preset behavior.
- **whisper.cpp subprocess path:** used as throughput/accuracy baseline in benchmark comparisons.
- **ffmpeg decode pipeline:** used to normalize audio and fix duration calculation robustness.
- **WER + RTF benchmarking:** used as practical decision metrics for preset recommendations.
- **Git batch commits:** kept changes reviewable and reversible by concern.

## 6. Lessons Learned (For This Episode)
- Feature-complete is not done-complete; always close with full verify pipeline.
- Keep compatibility-sensitive changes additive and default-safe.
- Assert CLI semantics, not formatter-dependent message layout.
- On Windows, treat multiprocessing spawn rules as first-class test harness constraints.
- Benchmark data quality matters as much as model choice; validate duration/reference integrity.
- Use long, reference-backed audio for ASR decisions; short clips can hide real error modes.
- Separate startup timeout and request timeout in worker architectures.
- CUDA availability checks should drive `auto` backend choice instead of static defaults.
- Prefer minimal, targeted fixes when stabilizing late-stage branches.
