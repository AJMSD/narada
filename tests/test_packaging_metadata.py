from __future__ import annotations

import tomllib
from pathlib import Path


def test_dev_extra_includes_numpy_dependency() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    optional = data.get("project", {}).get("optional-dependencies", {})
    dev_dependencies = optional.get("dev", [])
    normalized = [str(dep).strip().lower() for dep in dev_dependencies]

    assert any(dep.startswith("numpy") for dep in normalized)
