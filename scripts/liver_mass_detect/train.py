#!/usr/bin/env python3
"""Train a fast liver-mass detection model (scaffold).

Uses the same CT images / patient layout as liver detection. A fusion-series
mask channel will be added once those segmentations are available in ``data/``.

Until then this entrypoint only prepares task-local output/log dirs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path
from src.paths import TASK_LIVER_MASS_DETECT, ensure_task_dirs, task_logs, task_outputs

TASK = TASK_LIVER_MASS_DETECT

sanitize_sys_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--holdout", type=Path, default=ROOT / "data" / "holdout")
    parser.add_argument("--output", type=Path, default=task_outputs(TASK))
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=task_logs(TASK),
        help="Directory for train stdout/stderr logs for this task",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_task_dirs(TASK)
    args.output.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    raise SystemExit(
        "liver_mass_detect training is not implemented yet. "
        "Shared data root is ready; fusion-series masks are not available yet. "
        f"Outputs will go to {args.output} and logs to {args.log_dir}."
    )


if __name__ == "__main__":
    main()
