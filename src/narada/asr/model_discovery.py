from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

FASTER_WHISPER_SETUP_URL = "https://github.com/SYSTRAN/faster-whisper"
WHISPER_CPP_SETUP_URL = "https://github.com/ggml-org/whisper.cpp"
WHISPER_CPP_MODEL_REPO_URL = "https://huggingface.co/ggerganov/whisper.cpp"


@dataclass(frozen=True)
class EngineModelAvailability:
    engine: str
    model_name: str
    present: bool
    model_path: Path
    setup_url: str
    model_url: str


@dataclass(frozen=True)
class ModelDiscovery:
    faster_whisper: EngineModelAvailability
    whisper_cpp: EngineModelAvailability

    @property
    def any_present(self) -> bool:
        return self.faster_whisper.present or self.whisper_cpp.present

    @property
    def available_engines(self) -> tuple[str, ...]:
        values: list[str] = []
        if self.faster_whisper.present:
            values.append("faster-whisper")
        if self.whisper_cpp.present:
            values.append("whisper-cpp")
        return tuple(values)


@dataclass(frozen=True)
class StartModelPreflight:
    selected_engine: str
    selected_available: bool
    recommended_engine: str | None
    messages: tuple[str, ...]


def faster_whisper_model_url(model_name: str) -> str:
    return f"https://huggingface.co/Systran/faster-whisper-{model_name}"


def whisper_cpp_model_url(model_name: str) -> str:
    return f"{WHISPER_CPP_MODEL_REPO_URL}/resolve/main/ggml-{model_name}.bin"


def default_hf_hub_dir() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def default_whisper_cpp_model_dir() -> Path:
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "narada" / "models" / "whisper-cpp"
    return Path.home() / ".cache" / "narada" / "models" / "whisper-cpp"


def _latest_snapshot_dir(snapshot_root: Path) -> Path | None:
    if not snapshot_root.exists():
        return None
    snapshots = [path for path in snapshot_root.iterdir() if path.is_dir()]
    if not snapshots:
        return None
    snapshots.sort(key=lambda item: item.name)
    return snapshots[-1]


def resolve_faster_whisper_model_path(model_name: str, model_dir: Path | None = None) -> Path:
    if model_dir is not None:
        candidate_dirs = (
            model_dir,
            model_dir / f"faster-whisper-{model_name}",
            model_dir / model_name,
        )
        for directory in candidate_dirs:
            if (directory / "model.bin").exists():
                return directory
        return model_dir / f"faster-whisper-{model_name}"

    hf_root = default_hf_hub_dir()
    repo_dir = hf_root / f"models--Systran--faster-whisper-{model_name}"
    snapshot = _latest_snapshot_dir(repo_dir / "snapshots")
    if snapshot is not None and (snapshot / "model.bin").exists():
        return snapshot
    return repo_dir / "snapshots" / "latest"


def resolve_whisper_cpp_model_path(model_name: str, model_dir: Path | None = None) -> Path:
    base = model_dir or Path(os.environ.get("NARADA_MODEL_DIR_WHISPER_CPP", ""))
    if str(base) == ".":
        base = default_whisper_cpp_model_dir()
    if not str(base):
        base = default_whisper_cpp_model_dir()
    return base / f"ggml-{model_name}.bin"


def discover_models(
    model_name: str,
    *,
    faster_whisper_model_dir: Path | None = None,
    whisper_cpp_model_dir: Path | None = None,
) -> ModelDiscovery:
    faster_path = resolve_faster_whisper_model_path(model_name, faster_whisper_model_dir)
    whisper_path = resolve_whisper_cpp_model_path(model_name, whisper_cpp_model_dir)
    return ModelDiscovery(
        faster_whisper=EngineModelAvailability(
            engine="faster-whisper",
            model_name=model_name,
            present=faster_path.exists(),
            model_path=faster_path,
            setup_url=FASTER_WHISPER_SETUP_URL,
            model_url=faster_whisper_model_url(model_name),
        ),
        whisper_cpp=EngineModelAvailability(
            engine="whisper-cpp",
            model_name=model_name,
            present=whisper_path.exists(),
            model_path=whisper_path,
            setup_url=WHISPER_CPP_SETUP_URL,
            model_url=whisper_cpp_model_url(model_name),
        ),
    )


def build_start_model_preflight(
    discovery: ModelDiscovery, selected_engine: str
) -> StartModelPreflight:
    selected = selected_engine.strip().lower()
    availability = {
        "faster-whisper": discovery.faster_whisper.present,
        "whisper-cpp": discovery.whisper_cpp.present,
    }
    selected_available = availability.get(selected, False)

    messages: list[str] = []
    recommended_engine: str | None = None

    if not discovery.any_present:
        messages.append("No local ASR model files were detected for faster-whisper or whisper.cpp.")
        messages.append(
            "Narada will try to auto-download the selected engine model before capture starts."
        )
        messages.append(
            "If auto-download fails (for example offline), use these setup links: "
            f"faster-whisper {discovery.faster_whisper.model_url} "
            f"(setup: {discovery.faster_whisper.setup_url}) | "
            f"whisper.cpp {discovery.whisper_cpp.model_url} "
            f"(setup: {discovery.whisper_cpp.setup_url})"
        )
        return StartModelPreflight(
            selected_engine=selected,
            selected_available=False,
            recommended_engine=None,
            messages=tuple(messages),
        )

    if selected_available and len(discovery.available_engines) == 1:
        present_engine = discovery.available_engines[0]
        missing = (
            discovery.whisper_cpp
            if present_engine == "faster-whisper"
            else discovery.faster_whisper
        )
        messages.append(
            f"Only {present_engine} model files are currently available on this device. "
            f"Narada will run using {present_engine}."
        )
        messages.append(
            f"If you want {missing.engine}, setup guide: "
            f"{missing.setup_url} | model: {missing.model_url}"
        )
    elif not selected_available and len(discovery.available_engines) == 1:
        recommended_engine = discovery.available_engines[0]
        messages.append(
            "Selected engine "
            f"'{selected}' does not have local model files for "
            f"'{discovery.faster_whisper.model_name}'."
        )
        messages.append("Narada will try to auto-download the selected model first.")
        messages.append(
            f"Detected {recommended_engine} model files on this device. "
            f"If selected-engine download fails, Narada will run with {recommended_engine}."
        )
        if selected == "faster-whisper":
            missing = discovery.faster_whisper
        else:
            missing = discovery.whisper_cpp
        messages.append(
            f"To use {selected}, setup: {missing.setup_url} | model: {missing.model_url}"
        )

    return StartModelPreflight(
        selected_engine=selected,
        selected_available=selected_available,
        recommended_engine=recommended_engine,
        messages=tuple(messages),
    )
