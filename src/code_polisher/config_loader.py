"""Production-grade configuration loader for Code Polish Orchestrator.
Supports persistent project settings via:
- pyproject.toml [tool.code-polish] section (preferred for Python projects)
- .code-polish.toml in repo root (dedicated config)

Loads into OrchestratorConfig, with CLI/env overrides taking precedence (standard 12-factor pattern).
Fully additive: does not affect existing CLI, dry-run, simulation, audit, or agent paths.
Integrates in main.py before creating Orchestrator.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import tomli

from .models import OrchestratorConfig


def load_project_config(target_dir: str | Path) -> dict[str, Any]:
    """Load [tool.code-polish] from pyproject.toml or .code-polish.toml.
    Returns flat dict of settings (e.g. {"model_analyzer": "...", "max_tasks": 20}).
    Returns {} if no config file/section found.
    """
    target = Path(target_dir).resolve()
    candidates = [
        target / "pyproject.toml",
        target / ".code-polish.toml",
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with candidate.open("rb") as f:
                data = tomli.load(f)
            if candidate.name == "pyproject.toml":
                section = data.get("tool", {}).get("code-polish", {})
            else:
                section = data  # root level for dedicated file
            if section:
                # Normalize keys (hyphen to underscore for Pydantic)
                normalized = {k.replace("-", "_"): v for k, v in section.items()}
                return normalized
        except Exception as e:
            # Non-fatal: log warning in caller
            print(f"Warning: Failed to parse {candidate}: {e}")
    return {}


def build_config(target_dir: str | Path, cli_overrides: dict[str, Any] | None = None) -> OrchestratorConfig:
    """Build final OrchestratorConfig by merging:
    1. Defaults (from model)
    2. Project config file (lowest priority)
    3. CLI / env overrides (highest priority, passed from main)
    """
    project_settings = load_project_config(target_dir)
    overrides = cli_overrides or {}

    # Start with project settings, then override with CLI
    merged: dict[str, Any] = {**project_settings, **overrides}

    # Ensure target_dir is set
    merged.setdefault("target_dir", str(Path(target_dir).resolve()))

    # Create config (Pydantic will validate and apply defaults for missing)
    try:
        return OrchestratorConfig(**merged)
    except Exception as e:
        # If bad config, fall back to safe defaults with warning
        print(f"Warning: Invalid project config, using CLI/defaults only. Error: {e}")
        return OrchestratorConfig(target_dir=str(Path(target_dir).resolve()), **overrides)
