"""Repo and per-task path helpers.

Tasks share ``data/`` and ``src/``. Each task owns its own ``scripts/``,
``outputs/``, and ``logs/`` subtree so experiment artifacts do not collide.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TASK_LIVER_DETECT = "liver_detect"
TASK_LIVER_MASS_DETECT = "liver_mass_detect"

KNOWN_TASKS = (TASK_LIVER_DETECT, TASK_LIVER_MASS_DETECT)


def task_scripts(task: str) -> Path:
    return REPO_ROOT / "scripts" / task


def task_outputs(task: str) -> Path:
    return REPO_ROOT / "outputs" / task


def task_logs(task: str) -> Path:
    return REPO_ROOT / "logs" / task


def ensure_task_dirs(task: str) -> None:
    if task not in KNOWN_TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {KNOWN_TASKS}")
    task_outputs(task).mkdir(parents=True, exist_ok=True)
    task_logs(task).mkdir(parents=True, exist_ok=True)
