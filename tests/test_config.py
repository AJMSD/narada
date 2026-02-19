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
