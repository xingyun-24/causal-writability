"""Mechanical helpers shared by the four thin command-line entry points."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SPRING_EXPERIMENT = "spring_shortcuts_v4"


def bootstrap_repository() -> None:
    """Make the source checkout runnable without a shell-level PYTHONPATH."""
    paths = (REPO_ROOT / "src", REPO_ROOT / "lib" / "diffsynth")
    for path in reversed(paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _set_nested(target: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    if not all(parts):
        raise ValueError(f"Invalid override key: {key!r}")
    current = target
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(
                f"Cannot set {key!r}: {part!r} is not a mapping"
            )
        current = child
    current[parts[-1]] = value


def apply_overrides(
    config: dict[str, Any],
    values: list[str],
) -> dict[str, Any]:
    for item in values:
        if "=" not in item:
            raise ValueError(
                f"Override must use dotted.key=value syntax: {item!r}"
            )
        key, raw = item.split("=", 1)
        _set_nested(config, key, yaml.safe_load(raw))
    return config


def resolve_profile(
    config_file: Path,
    section: str,
    profile: str | None,
    overrides: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    document = load_yaml(config_file)
    profiles = document.get(section, {})
    if profiles is None:
        profiles = {}
    if not isinstance(profiles, dict):
        raise ValueError(f"{section} must be a mapping in {config_file}")
    if profile is None:
        if len(profiles) == 1:
            profile = next(iter(profiles))
        elif profiles:
            raise ValueError(
                f"--profile is required; choose one of: "
                f"{', '.join(sorted(profiles))}"
            )
        selected: dict[str, Any] = {}
    else:
        if profile not in profiles:
            raise ValueError(
                f"Unknown profile {profile!r}; choose one of: "
                f"{', '.join(sorted(profiles))}"
            )
        raw = profiles[profile]
        if not isinstance(raw, dict):
            raise ValueError(
                f"Profile {profile!r} in {config_file} must be a mapping"
            )
        selected = dict(raw)
    selected.setdefault("experiment_id", document.get("experiment_id"))
    selected.setdefault("dataset_id", document.get("dataset_id"))
    selected["profile"] = profile
    return apply_overrides(selected, overrides), document


def experiment_module(experiment: str, role: str):
    module_name = f"sshv2.experiments.{experiment}.{role}"
    return importlib.import_module(module_name)


def training_module(experiment: str):
    from sshv2.wan import trainer

    function = (
        trainer.train_standard
        if experiment == SPRING_EXPERIMENT
        else trainer.train_compat
    )
    return SimpleNamespace(train=function)


def serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def write_resolved_config(
    output_dir: Path,
    *,
    stage: str,
    experiment: str,
    config_file: Path,
    profile: str | None,
    config: dict[str, Any],
    source_config: dict[str, Any],
    parameters: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "resolved_config.yaml"
    payload = {
        "stage": stage,
        "experiment": experiment,
        "profile": profile,
        "config_file": str(config_file),
        "parameters": serializable(parameters),
        "config": serializable(config),
        "source_config": serializable(source_config),
    }
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return destination
