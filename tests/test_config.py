from pathlib import Path

import pytest

from narada.config import ConfigError, ConfigOverrides, build_runtime_config, parse_languages


def test_flags_override_environment_values() -> None:
    env = {
        "NARADA_MODE": "mic",
        "NARADA_MIC": "3",
        "NARADA_MODEL": "tiny",
        "NARADA_CONFIDENCE_THRESHOLD": "0.40",
    }
    overrides = ConfigOverrides(mode="system", system="5", model="small", confidence_threshold=0.8)
    config = build_runtime_config(overrides, env=env)
    assert config.mode == "system"
    assert config.system == "5"
    assert config.model == "small"
    assert config.confidence_threshold == 0.8


def test_default_output_path_is_generated_when_missing() -> None:
    env = {"NARADA_MODE": "mic", "NARADA_MIC": "1"}
    config = build_runtime_config(ConfigOverrides(), env=env)
    assert isinstance(config.out, Path)
    assert config.out.name.startswith("narada-")
    assert config.out.suffix == ".txt"
    assert config.wall_flush_seconds == 60.0
    assert config.capture_queue_warn_seconds == 120.0
    assert config.notes_interval_seconds == 12.0
    assert config.notes_overlap_seconds == 1.5
    assert config.notes_commit_holdback_windows == 1
    assert config.asr_backlog_warn_seconds == 45.0
    assert config.keep_spool is False
    assert config.spool_flush_interval_seconds == 0.25
    assert config.spool_flush_bytes == 65536
    assert config.writer_fsync_mode == "line"
    assert config.writer_fsync_lines == 20
    assert config.writer_fsync_seconds == 1.0
    assert config.asr_preset == "balanced"
    assert config.serve_token is None


def test_multilingual_requires_explicit_flag() -> None:
    with pytest.raises(ConfigError):
        parse_languages("hindi,english", allow_multilingual=False)


def test_language_aliases_are_normalized() -> None:
    languages = parse_languages("hindi,english", allow_multilingual=True)
    assert languages == ("hi", "en")


def test_mode_requires_device_selectors() -> None:
    with pytest.raises(ConfigError):
        build_runtime_config(ConfigOverrides(mode="mixed", mic="1"), env={})


def test_model_directory_overrides_are_loaded() -> None:
    env = {
        "NARADA_MODE": "mic",
        "NARADA_MIC": "1",
        "NARADA_MODEL_DIR_FASTER_WHISPER": "C:/models/fw",
        "NARADA_MODEL_DIR_WHISPER_CPP": "C:/models/wc",
    }
    config = build_runtime_config(ConfigOverrides(), env=env)
    assert config.model_dir_faster_whisper == Path("C:/models/fw")
    assert config.model_dir_whisper_cpp == Path("C:/models/wc")


def test_wall_flush_and_queue_warning_threshold_loaded_from_env() -> None:
    env = {
        "NARADA_MODE": "mic",
        "NARADA_MIC": "1",
        "NARADA_WALL_FLUSH_SECONDS": "30.0",
        "NARADA_CAPTURE_QUEUE_WARN_SECONDS": "15.0",
        "NARADA_NOTES_INTERVAL_SECONDS": "9.0",
        "NARADA_NOTES_OVERLAP_SECONDS": "1.0",
        "NARADA_NOTES_COMMIT_HOLDBACK_WINDOWS": "2",
        "NARADA_ASR_BACKLOG_WARN_SECONDS": "50.0",
        "NARADA_KEEP_SPOOL": "true",
        "NARADA_SPOOL_FLUSH_INTERVAL_SECONDS": "0.5",
        "NARADA_SPOOL_FLUSH_BYTES": "32768",
        "NARADA_WRITER_FSYNC_MODE": "periodic",
        "NARADA_WRITER_FSYNC_LINES": "10",
        "NARADA_WRITER_FSYNC_SECONDS": "2.0",
        "NARADA_ASR_PRESET": "fast",
        "NARADA_SERVE_TOKEN": "secret123",
    }
    config = build_runtime_config(ConfigOverrides(), env=env)
    assert config.wall_flush_seconds == 30.0
    assert config.capture_queue_warn_seconds == 15.0
    assert config.notes_interval_seconds == 9.0
    assert config.notes_overlap_seconds == 1.0
    assert config.notes_commit_holdback_windows == 2
    assert config.asr_backlog_warn_seconds == 50.0
    assert config.keep_spool is True
    assert config.spool_flush_interval_seconds == 0.5
    assert config.spool_flush_bytes == 32768
    assert config.writer_fsync_mode == "periodic"
    assert config.writer_fsync_lines == 10
    assert config.writer_fsync_seconds == 2.0
    assert config.asr_preset == "fast"
    assert config.serve_token == "secret123"


def test_wall_flush_and_queue_warning_threshold_validate_ranges() -> None:
    env = {
        "NARADA_MODE": "mic",
        "NARADA_MIC": "1",
    }
    with pytest.raises(ConfigError):
        build_runtime_config(ConfigOverrides(wall_flush_seconds=-1.0), env=env)
    with pytest.raises(ConfigError):
        build_runtime_config(ConfigOverrides(capture_queue_warn_seconds=0.0), env=env)
    with pytest.raises(ConfigError):
        build_runtime_config(ConfigOverrides(notes_interval_seconds=0.0), env=env)
    with pytest.raises(ConfigError):
        build_runtime_config(
            ConfigOverrides(notes_interval_seconds=10.0, notes_overlap_seconds=10.0),
            env=env,
        )
    with pytest.raises(ConfigError):
        build_runtime_config(ConfigOverrides(notes_commit_holdback_windows=-1), env=env)
    with pytest.raises(ConfigError):
        build_runtime_config(ConfigOverrides(asr_backlog_warn_seconds=0.0), env=env)
    with pytest.raises(ConfigError):
        build_runtime_config(ConfigOverrides(spool_flush_interval_seconds=-0.1), env=env)
    with pytest.raises(ConfigError):
        build_runtime_config(ConfigOverrides(spool_flush_bytes=-1), env=env)
    with pytest.raises(ConfigError):
        build_runtime_config(
            ConfigOverrides(
                writer_fsync_mode="periodic",
                writer_fsync_lines=0,
                writer_fsync_seconds=0.0,
            ),
            env=env,
        )
    with pytest.raises(ConfigError):
        build_runtime_config(ConfigOverrides(writer_fsync_mode="bad"), env=env)
    with pytest.raises(ConfigError):
        build_runtime_config(ConfigOverrides(asr_preset="ultra"), env=env)
