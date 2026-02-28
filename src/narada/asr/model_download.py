from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from narada.asr.model_discovery import (
    default_hf_hub_dir,
    faster_whisper_model_url,
    resolve_faster_whisper_model_path,
    resolve_whisper_cpp_model_path,
    whisper_cpp_model_url,
)

logger = logging.getLogger("narada.asr.model_download")

MessageEmitter = Callable[[str], None]


class ModelPreparationError(RuntimeError):
    pass


def ensure_engine_model_available(
    *,
    engine_name: str,
    model_name: str,
    faster_whisper_model_dir: Path | None = None,
    whisper_cpp_model_dir: Path | None = None,
    emit: MessageEmitter | None = None,
) -> Path:
    normalized = engine_name.strip().lower()
    if normalized == "faster-whisper":
        return _ensure_faster_whisper_model(
            model_name=model_name,
            model_dir=faster_whisper_model_dir,
            emit=emit,
        )
    if normalized == "whisper-cpp":
        return _ensure_whisper_cpp_model(
            model_name=model_name,
            model_dir=whisper_cpp_model_dir,
            emit=emit,
        )
    raise ModelPreparationError(f"Unsupported engine '{engine_name}'.")


def _load_hf_download_api() -> tuple[Callable[..., str], Callable[..., str]]:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except Exception as exc:  # pragma: no cover - import-path/env specific
        raise ModelPreparationError(
            "Model auto-download requires 'huggingface-hub'. Install optional dependencies: "
            'pip install -e ".[asr]".'
        ) from exc
    return hf_hub_download, snapshot_download


def _emit_message(message: str, *, emit: MessageEmitter | None) -> None:
    logger.info(message)
    if emit is not None:
        emit(message)


def _ensure_faster_whisper_model(
    *,
    model_name: str,
    model_dir: Path | None,
    emit: MessageEmitter | None,
) -> Path:
    model_path = resolve_faster_whisper_model_path(model_name, model_dir)
    if (model_path / "model.bin").exists():
        _emit_message(
            f"Using cached faster-whisper model '{model_name}' at {model_path}.",
            emit=emit,
        )
        return model_path

    _, snapshot_download = _load_hf_download_api()
    repo_id = f"Systran/faster-whisper-{model_name}"

    try:
        if model_dir is not None:
            model_path.mkdir(parents=True, exist_ok=True)
            _emit_message(
                f"Downloading faster-whisper model '{model_name}' to {model_path}.",
                emit=emit,
            )
            _emit_message(
                "Download progress will be shown by huggingface-hub when available.",
                emit=emit,
            )
            downloaded = Path(
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(model_path),
                    local_dir_use_symlinks=False,
                )
            )
            resolved = model_path if (model_path / "model.bin").exists() else downloaded
        else:
            cache_dir = default_hf_hub_dir()
            _emit_message(
                f"Downloading faster-whisper model '{model_name}' to cache at {cache_dir}.",
                emit=emit,
            )
            _emit_message(
                "Download progress will be shown by huggingface-hub when available.",
                emit=emit,
            )
            resolved = Path(snapshot_download(repo_id=repo_id))
    except Exception as exc:  # pragma: no cover - network/fs specific
        _raise_download_failure(
            engine_name="faster-whisper",
            model_name=model_name,
            target_path=model_path,
            source_url=faster_whisper_model_url(model_name),
            exc=exc,
        )

    if not (resolved / "model.bin").exists():
        raise ModelPreparationError(
            "faster-whisper download completed but model files are incomplete at "
            f"{resolved}. Remove the directory and retry."
        )
    _emit_message(
        f"faster-whisper model '{model_name}' is ready at {resolved}.",
        emit=emit,
    )
    return resolved


def _ensure_whisper_cpp_model(
    *,
    model_name: str,
    model_dir: Path | None,
    emit: MessageEmitter | None,
) -> Path:
    model_path = resolve_whisper_cpp_model_path(model_name, model_dir)
    if model_path.exists():
        _emit_message(
            f"Using cached whisper-cpp model '{model_name}' at {model_path}.",
            emit=emit,
        )
        return model_path

    hf_hub_download, _ = _load_hf_download_api()
    repo_id = "ggerganov/whisper.cpp"
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _emit_message(
            f"Downloading whisper-cpp model '{model_name}' to {model_path.parent}.",
            emit=emit,
        )
        _emit_message(
            "Download progress will be shown by huggingface-hub when available.",
            emit=emit,
        )
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=model_path.name,
                local_dir=str(model_path.parent),
                local_dir_use_symlinks=False,
            )
        )
    except Exception as exc:  # pragma: no cover - network/fs specific
        _raise_download_failure(
            engine_name="whisper-cpp",
            model_name=model_name,
            target_path=model_path,
            source_url=whisper_cpp_model_url(model_name),
            exc=exc,
        )

    resolved = model_path if model_path.exists() else downloaded
    if not resolved.exists():
        raise ModelPreparationError(
            "whisper-cpp download completed but model file is missing at "
            f"{model_path}. Remove partial files and retry."
        )
    _emit_message(
        f"whisper-cpp model '{model_name}' is ready at {resolved}.",
        emit=emit,
    )
    return resolved


def _raise_download_failure(
    *,
    engine_name: str,
    model_name: str,
    target_path: Path,
    source_url: str,
    exc: Exception,
) -> None:
    message = str(exc).strip() or exc.__class__.__name__
    raise ModelPreparationError(
        f"Failed to download {engine_name} model '{model_name}' to {target_path}: {message}. "
        "Check internet access and write permissions, or pre-download the model from "
        f"{source_url}."
    ) from exc
