#!/usr/bin/env python3
"""Build a mass-detect dataset with a binary organ-ROI prior channel.

Reuses CT + Mass masks from an existing mass dataset (e.g. raw_v2_mass) and
adds ``liver_masks/`` as the SEG series 300 organ ROI:

  Liver ∪ Mass ∪ Portal vein

(SEG ``Liver`` alone is parenchyma-only and is disjoint from ``Mass``.)
Hardlinks images/masks when possible.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from scripts.download_dataset import (
    choose_stack_z_mapping,
    estimate_slice_spacing_mm,
    find_ct_targets,
    load_ct_series_instances,
    mark_dual_acquisition_passes,
)
from src.dataset_select import select_all_patients
from src.geometry import describe_alignment, group_instances_by_ipp, resample_mask_to_geometry
from src.orthanc_client import OrthancClient
from src.seg_masks import (
    MASK_TARGET_LIVER,
    extract_liver_frames,
    referenced_series_uids,
    resize_mask,
    save_mask_png,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Existing mass dataset root")
    parser.add_argument("--output", type=Path, required=True, help="Destination dataset root")
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--ipp-decimals", type=int, default=3)
    parser.add_argument("--match-tol-mm", type=float, default=1.25)
    parser.add_argument(
        "--require-liver",
        action="store_true",
        default=True,
        help="Drop slices that cannot be matched to a Liver SEG plane",
    )
    parser.add_argument("--no-require-liver", action="store_false", dest="require_liver")
    return parser.parse_args()


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _load_rows(manifest: Path) -> list[dict]:
    rows: list[dict] = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_liver_planes(client: OrthancClient, orthanc_seg_series_id: str) -> tuple[dict[str, dict], list[str]]:
    series = client.get_series(orthanc_seg_series_id)
    plane_masks: dict[str, dict] = {}
    ref_uids: list[str] = []
    for instance_id in series.get("Instances") or []:
        seg_bytes = client.download_instance_file(instance_id)
        frames, seg_ds = extract_liver_frames(
            seg_bytes,
            liver_segment_numbers=[],
            series_number=300,
            series_description="",
            # Organ ROI prior: Liver∪Mass∪Portal (Liver-only is disjoint from Mass).
            mask_target=MASK_TARGET_LIVER,
        )
        for frame in frames:
            prev = plane_masks.get(frame.merge_key)
            if prev is None:
                plane_masks[frame.merge_key] = {
                    "mask": frame.mask,
                    "geometry": frame.geometry,
                    "sop": frame.referenced_sop_instance_uid,
                }
            else:
                prev["mask"] = (prev["mask"].astype(bool) | frame.mask.astype(bool)).astype(np.uint8)
                if prev["sop"] is None and frame.referenced_sop_instance_uid:
                    prev["sop"] = frame.referenced_sop_instance_uid
        if seg_ds is not None:
            ref_uids = referenced_series_uids(seg_ds) or ref_uids
    return plane_masks, ref_uids


def _liver_by_ct_sop(
    client: OrthancClient,
    orthanc_seg_series_id: str,
    image_size: int,
    ipp_decimals: int,
    match_tol_mm: float,
) -> dict[str, np.ndarray]:
    """Return mapping CT SOP UID -> resized uint8 liver mask (0/1)."""
    plane_masks, ref_uids = _extract_liver_planes(client, orthanc_seg_series_id)
    if not plane_masks or not ref_uids:
        return {}

    ct_instances = load_ct_series_instances(client, ref_uids[0])
    if not ct_instances:
        return {}

    by_sop = {inst["sop"]: inst for inst in ct_instances}
    by_ipp = group_instances_by_ipp(ct_instances, decimals=ipp_decimals)
    dual_acq = mark_dual_acquisition_passes(ct_instances, by_ipp)
    match_instances = (
        [inst for inst in ct_instances if inst.get("acquisition_pass", 0) == 1]
        if dual_acq
        else ct_instances
    )
    if not match_instances:
        return {}

    slice_spacing = estimate_slice_spacing_mm(match_instances)
    geom_tol = max(match_tol_mm, 0.6 * slice_spacing)
    plane_list = list(plane_masks.values())
    z_mapping = choose_stack_z_mapping(plane_list, by_sop, match_instances, geom_tol)

    out: dict[str, np.ndarray] = {}
    for plane in plane_list:
        targets, _match_mode = find_ct_targets(
            plane["sop"],
            plane["geometry"].ipp,
            by_sop,
            match_instances,
            by_ipp,
            ipp_decimals,
            geom_tol,
            z_mapping=z_mapping,
            second_pass_only=dual_acq,
        )
        for twin in targets:
            if dual_acq and twin.get("acquisition_pass", 0) != 1:
                continue
            aligned = resample_mask_to_geometry(
                plane["mask"], plane["geometry"], twin["geometry"]
            )
            resized = resize_mask(aligned, image_size)
            prev = out.get(twin["sop"])
            if prev is None:
                out[twin["sop"]] = resized
            else:
                out[twin["sop"]] = (prev.astype(bool) | resized.astype(bool)).astype(np.uint8)
    return out


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    src_manifest = args.src / "manifest.jsonl"
    if not src_manifest.is_file():
        print(f"Missing manifest: {src_manifest}")
        return 1
    if args.output.exists() and any(args.output.iterdir()):
        print(f"Refusing to overwrite non-empty output: {args.output}")
        return 1

    rows = _load_rows(src_manifest)
    by_patient: dict[str, list[dict]] = {}
    for row in rows:
        by_patient.setdefault(row["patient_id"], []).append(row)

    client = OrthancClient()
    # Map patient_id -> orthanc SEG series id via selection (stable for SEG 300).
    choices = {
        c.patient_id: c
        for c in select_all_patients(
            client,
            require_series_number=300,
            mask_target="mass",
        )
    }

    args.output.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict] = []
    missing_liver = 0
    errors: list[str] = []

    for patient_id, patient_rows in tqdm(sorted(by_patient.items()), desc="Patients"):
        choice = choices.get(patient_id)
        if choice is None:
            errors.append(f"{patient_id}: no SEG 300 choice in Orthanc")
            continue
        try:
            liver_by_sop = _liver_by_ct_sop(
                client,
                choice.orthanc_series_id,
                image_size=args.image_size,
                ipp_decimals=args.ipp_decimals,
                match_tol_mm=args.match_tol_mm,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{patient_id}: liver extract failed: {exc}")
            continue

        for row in patient_rows:
            sop = row.get("sop_instance_uid")
            liver = liver_by_sop.get(sop) if sop else None
            if liver is None:
                missing_liver += 1
                if args.require_liver:
                    continue
                liver = np.zeros((args.image_size, args.image_size), dtype=np.uint8)

            src_img = args.src / row["image_path"]
            src_mask = args.src / row["mask_path"]
            if not src_img.is_file() or not src_mask.is_file():
                errors.append(f"{patient_id}: missing src files for {sop}")
                continue

            # Preserve relative layout under patient/study.
            rel_img = Path(row["image_path"])
            rel_mask = Path(row["mask_path"])
            stem = rel_img.stem
            rel_liver = rel_img.parent.parent / "liver_masks" / f"{stem}.png"

            dst_img = args.output / rel_img
            dst_mask = args.output / rel_mask
            dst_liver = args.output / rel_liver
            _link_or_copy(src_img, dst_img)
            _link_or_copy(src_mask, dst_mask)
            dst_liver.parent.mkdir(parents=True, exist_ok=True)
            save_mask_png(liver, str(dst_liver))

            new_row = dict(row)
            new_row["liver_mask_path"] = str(rel_liver)
            new_row["liver_prior"] = True
            new_row["in_channels"] = 2
            new_row["liver_foreground_pixels"] = int(liver.sum())
            # Avoid confusing dual-series loader.
            new_row.pop("image_path_b", None)
            out_rows.append(new_row)

    manifest_path = args.output / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(out_rows)} pairs with liver prior -> {args.output}")
    print(f"Manifest: {manifest_path}")
    if missing_liver:
        print(f"Slices without matched liver plane: {missing_liver}")
    if errors:
        err_path = args.output / "liver_prior_errors.log"
        err_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"{len(errors)} warnings/errors logged to {err_path}")
    return 0 if out_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
