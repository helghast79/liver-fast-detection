#!/usr/bin/env python3
"""Carve a patient-wise holdout from a downloaded dataset directory.

Moves patient folders and rewrites ``manifest.jsonl`` for both raw and holdout.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Downloaded dataset root")
    parser.add_argument("--raw", type=Path, required=True, help="Destination for train/val pool")
    parser.add_argument("--holdout", type=Path, required=True, help="Destination for holdout")
    parser.add_argument(
        "--n-holdout",
        type=int,
        default=10,
        help="Number of patients to carve for holdout (~10%% is typical)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-patient-id",
        action="append",
        default=[],
        help="Drop these PatientIDs entirely before splitting (repeatable)",
    )
    parser.add_argument(
        "--holdout-patient-id",
        action="append",
        default=[],
        help="Explicit holdout PatientID (repeatable). Overrides --n-holdout/--seed shuffle.",
    )
    parser.add_argument(
        "--holdout-from-json",
        type=Path,
        default=None,
        help="JSON with holdout_patients list (e.g. prior split_holdout.json)",
    )
    return parser.parse_args()


def _load_rows(manifest: Path) -> list[dict]:
    rows: list[dict] = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> int:
    args = parse_args()
    src_manifest = args.src / "manifest.jsonl"
    if not src_manifest.is_file():
        print(f"Missing manifest: {src_manifest}")
        return 1

    exclude = {str(pid) for pid in (args.exclude_patient_id or [])}
    rows = [r for r in _load_rows(src_manifest) if r.get("patient_id") not in exclude]
    patients = sorted({r["patient_id"] for r in rows})
    if len(patients) < 2:
        print(f"Need at least 2 patients to split; found {len(patients)}")
        return 1

    fixed_holdout: list[str] = []
    if args.holdout_from_json is not None:
        payload = json.loads(args.holdout_from_json.read_text(encoding="utf-8"))
        fixed_holdout.extend(str(p) for p in (payload.get("holdout_patients") or []))
    fixed_holdout.extend(str(p) for p in (args.holdout_patient_id or []))

    if fixed_holdout:
        hold_set = set(fixed_holdout)
        missing = sorted(hold_set - set(patients))
        if missing:
            print(f"WARNING: holdout patients missing from src: {missing}")
        holdout_patients = sorted(p for p in patients if p in hold_set)
        if not holdout_patients:
            print("No requested holdout patients present in src")
            return 1
        n_hold = len(holdout_patients)
    else:
        n_hold = min(max(1, args.n_holdout), len(patients) - 1)
        rng = random.Random(args.seed)
        shuffled = patients[:]
        rng.shuffle(shuffled)
        holdout_patients = sorted(shuffled[:n_hold])
        hold_set = set(holdout_patients)
    raw_patients = sorted(p for p in patients if p not in hold_set)

    if args.raw.exists() or args.holdout.exists():
        print(f"Refusing to overwrite existing {args.raw} / {args.holdout}")
        return 1

    args.raw.mkdir(parents=True, exist_ok=True)
    args.holdout.mkdir(parents=True, exist_ok=True)

    for pid in raw_patients:
        src = args.src / pid
        if src.is_dir():
            shutil.move(str(src), str(args.raw / pid))
    for pid in holdout_patients:
        src = args.src / pid
        if src.is_dir():
            shutil.move(str(src), str(args.holdout / pid))

    raw_rows = [r for r in rows if r["patient_id"] not in hold_set]
    hold_rows = [r for r in rows if r["patient_id"] in hold_set]
    _write_manifest(args.raw / "manifest.jsonl", raw_rows)
    _write_manifest(args.holdout / "manifest.jsonl", hold_rows)

    split_info = {
        "seed": args.seed,
        "n_holdout": n_hold,
        "excluded_patient_ids": sorted(exclude),
        "holdout_patients": holdout_patients,
        "raw_patients": raw_patients,
        "n_raw_patients": len(raw_patients),
        "n_holdout_patients": len(holdout_patients),
        "n_raw_slices": len(raw_rows),
        "n_holdout_slices": len(hold_rows),
        "src": str(args.src),
    }
    (args.raw / "split_holdout.json").write_text(
        json.dumps(split_info, indent=2) + "\n", encoding="utf-8"
    )
    (args.holdout / "split_holdout.json").write_text(
        json.dumps(split_info, indent=2) + "\n", encoding="utf-8"
    )

    # Copy download errors if present.
    err = args.src / "download_errors.log"
    if err.is_file():
        shutil.copy2(err, args.raw / "download_errors.log")

    print(
        f"Split seed={args.seed}: raw={len(raw_patients)} patients / {len(raw_rows)} slices, "
        f"holdout={len(holdout_patients)} patients / {len(hold_rows)} slices"
    )
    print(f"Holdout patients: {holdout_patients}")
    print(f"Raw -> {args.raw}")
    print(f"Holdout -> {args.holdout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
