#!/usr/bin/env python3
"""Download dual-channel CT + Mass SEG pairs for liver-mass detect.

Channel rules:
  - Dual-acq merged SEG-referenced series → earlier/later passes as ch0/ch1
  - Else 2–3 CT series → pick two (equal instance-count pair when possible);
    project the Mass mask onto both via geometry even if SEG references only one

Outputs per sample:
  images/{stem}_a.png  (primary / referenced / later)
  images/{stem}_b.png  (secondary / other / earlier, resampled onto primary geom)
  masks/{stem}.png
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

import numpy as np
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from scripts.download_dataset import (
    choose_stack_z_mapping,
    estimate_slice_spacing_mm,
    find_ct_targets,
    load_ct_series_instances,
    mark_dual_acquisition_passes,
)
from src.ct_preprocess import WindowSpec, dicom_to_hu, voi_window_from_dicom
from src.dataset_select import select_all_patients
from src.dual_series import (
    dual_acq_pairs,
    list_ct_series,
    nearest_instance_by_z,
    resample_image_to_geometry,
    select_dual_series,
    series_uid,
)
from src.geometry import describe_alignment, group_instances_by_ipp, resample_mask_to_geometry
from src.orthanc_client import OrthancClient
from src.seg_masks import (
    MASK_TARGET_MASS,
    extract_liver_frames,
    referenced_series_uids,
    resize_mask,
    save_mask_png,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "v2_mass_dual_all_192")
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--window-center", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument("--skip-empty", action="store_true", default=True)
    parser.add_argument("--no-skip-empty", action="store_false", dest="skip_empty")
    parser.add_argument("--limit-patients", type=int, default=None)
    parser.add_argument("--patient-id", type=str, default=None)
    parser.add_argument(
        "--exclude-patient-id",
        action="append",
        default=["HCC_005", "HCC_027", "HCC_088", "HCC_104"],
    )
    parser.add_argument("--ipp-decimals", type=int, default=3)
    parser.add_argument("--match-tol-mm", type=float, default=1.25)
    return parser.parse_args()


def _windowed_array_from_dicom(
    dicom_bytes: bytes,
    window: WindowSpec,
) -> tuple[np.ndarray, dict]:
    """Return float32 [0,1] windowed array at native resolution + meta."""
    import pydicom

    ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
    hu = dicom_to_hu(ds)
    used, source = voi_window_from_dicom(ds, fallback=window)
    # Always use fixed abdomen window for this task (match train recipe).
    used = window
    source = "fixed"
    lo, hi = used.lo, used.hi
    clipped = np.clip(hu, lo, hi)
    gray = ((clipped - lo) / max(hi - lo, 1e-6)).astype(np.float32)
    meta = {
        "window_center": used.center,
        "window_width": used.width,
        "window_source": source,
        "rows": int(gray.shape[0]),
        "cols": int(gray.shape[1]),
    }
    return gray, meta


def _save_float01_png(arr01: np.ndarray, path: Path, size: int) -> None:
    img = Image.fromarray((np.clip(arr01, 0, 1) * 255.0).astype(np.uint8), mode="L")
    if img.width != size or img.height != size:
        img = img.resize((size, size), resample=Image.Resampling.BILINEAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _extract_planes(client: OrthancClient, choice) -> tuple[dict[str, dict], list[int], list[str], list[str]]:
    series = client.get_series(choice.orthanc_series_id)
    instance_ids = series.get("Instances") or []
    plane_masks: dict[str, dict] = {}
    used_segments: list[int] = list(choice.liver_segment_numbers)
    ref_series_uids: list[str] = []
    errors: list[str] = []
    for instance_id in instance_ids:
        try:
            seg_bytes = client.download_instance_file(instance_id)
            frames, seg_ds = extract_liver_frames(
                seg_bytes,
                liver_segment_numbers=choice.liver_segment_numbers,
                series_number=choice.series_number,
                series_description=choice.series_description,
                mask_target=MASK_TARGET_MASS,
            )
            if frames:
                used_segments = frames[0].segment_numbers
            for frame in frames:
                prev = plane_masks.get(frame.merge_key)
                if prev is None:
                    plane_masks[frame.merge_key] = {
                        "mask": frame.mask,
                        "geometry": frame.geometry,
                        "sop": frame.referenced_sop_instance_uid,
                    }
                else:
                    prev["mask"] = (
                        (prev["mask"].astype(bool) | frame.mask.astype(bool)).astype(np.uint8)
                    )
                    if prev["sop"] is None and frame.referenced_sop_instance_uid:
                        prev["sop"] = frame.referenced_sop_instance_uid
            if seg_ds is not None:
                ref_series_uids = referenced_series_uids(seg_ds) or ref_series_uids
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{choice.patient_id} instance {instance_id}: {exc}")
    return plane_masks, used_segments, ref_series_uids, errors


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    image_size = args.image_size or int(os.getenv("IMAGE_SIZE", "192"))
    window = WindowSpec(center=args.window_center, width=args.window_width)

    client = OrthancClient()
    exclude_ids = {str(pid) for pid in (args.exclude_patient_id or [])}
    choices = select_all_patients(
        client,
        require_series_number=300,
        exclude_patient_ids=exclude_ids,
        mask_target=MASK_TARGET_MASS,
    )
    if args.patient_id:
        choices = [c for c in choices if c.patient_id == args.patient_id]
    if args.limit_patients is not None:
        choices = choices[: args.limit_patients]

    print(
        f"Selected {len(choices)} patient(s) for dual-series mass download "
        f"excluded={sorted(exclude_ids)}"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.jsonl"

    total_pairs = 0
    skipped_empty = 0
    skipped_unpaired = 0
    errors: list[str] = []

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for choice in tqdm(choices, desc="Patients"):
            safe_pid = choice.patient_id.replace("/", "_")
            out_dir = args.output / safe_pid / choice.study_instance_uid
            img_dir = out_dir / "images"
            mask_dir = out_dir / "masks"
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            plane_masks, used_segments, ref_series_uids, plane_errors = _extract_planes(
                client, choice
            )
            errors.extend(plane_errors)
            if not plane_masks:
                errors.append(f"{choice.patient_id}: no Mass SEG frames extracted")
                continue
            if not ref_series_uids:
                errors.append(f"{choice.patient_id}: SEG has no ReferencedSeriesSequence")
                continue

            study = client.get_study(choice.orthanc_study_id)
            ct_series = list_ct_series(client, study)
            instances_by_uid: dict[str, list[dict]] = {}
            for series in ct_series:
                uid = series_uid(series)
                if not uid:
                    continue
                instances_by_uid[uid] = load_ct_series_instances(client, uid)

            try:
                dual_choice = select_dual_series(
                    ct_series,
                    ref_series_uids[0],
                    instances_by_uid,
                    ipp_decimals=args.ipp_decimals,
                )
            except ValueError as exc:
                errors.append(f"{choice.patient_id}: dual-series select failed: {exc}")
                continue

            primary_insts = instances_by_uid.get(dual_choice.primary_series_uid) or []
            secondary_insts = instances_by_uid.get(dual_choice.secondary_series_uid) or []
            if not primary_insts or not secondary_insts:
                errors.append(f"{choice.patient_id}: missing primary/secondary CT instances")
                continue

            by_sop = {inst["sop"]: inst for inst in primary_insts}
            by_ipp = group_instances_by_ipp(primary_insts, decimals=args.ipp_decimals)
            dual_acq = mark_dual_acquisition_passes(primary_insts, by_ipp)
            # Match SEG onto the later pass when dual-acq (SEG 300 convention).
            match_instances = (
                [inst for inst in primary_insts if inst.get("acquisition_pass", 0) == 1]
                if dual_choice.mode == "dual_acq"
                else primary_insts
            )
            if not match_instances:
                errors.append(f"{choice.patient_id}: no primary match instances")
                continue

            slice_spacing = estimate_slice_spacing_mm(match_instances)
            geom_tol = max(args.match_tol_mm, 0.6 * slice_spacing)
            plane_list = list(plane_masks.values())
            z_mapping = choose_stack_z_mapping(plane_list, by_sop, match_instances, geom_tol)

            # Map SEG planes → primary CT instances (later pass if dual-acq).
            primary_masks: dict[str, dict] = {}
            unmatched = 0
            for plane in plane_list:
                targets, match_mode = find_ct_targets(
                    plane["sop"],
                    plane["geometry"].ipp,
                    by_sop,
                    match_instances,
                    by_ipp,
                    args.ipp_decimals,
                    geom_tol,
                    z_mapping=z_mapping,
                    second_pass_only=(dual_choice.mode == "dual_acq"),
                )
                if not targets:
                    unmatched += 1
                    continue
                for twin in targets:
                    if dual_choice.mode == "dual_acq" and twin.get("acquisition_pass", 0) != 1:
                        continue
                    aligned = resample_mask_to_geometry(
                        plane["mask"], plane["geometry"], twin["geometry"]
                    )
                    alignment = describe_alignment(plane["geometry"], twin["geometry"])
                    alignment["match_mode"] = match_mode
                    prev = primary_masks.get(twin["sop"])
                    if prev is None:
                        primary_masks[twin["sop"]] = {
                            "mask": aligned,
                            "primary": twin,
                            "alignment": alignment,
                            "source_seg_sop": plane["sop"],
                        }
                    else:
                        prev["mask"] = (
                            (prev["mask"].astype(bool) | aligned.astype(bool)).astype(np.uint8)
                        )

            if unmatched:
                errors.append(
                    f"{choice.patient_id}: {unmatched}/{len(plane_masks)} SEG planes unmatched"
                )
            if not primary_masks:
                errors.append(f"{choice.patient_id}: no primary CT masks after alignment")
                continue

            # Build secondary partner lookup.
            if dual_choice.mode == "dual_acq":
                pairs = {later["sop"]: earlier for earlier, later in dual_acq_pairs(primary_insts, args.ipp_decimals)}
            else:
                pairs = {}

            for sop_uid, info in tqdm(
                primary_masks.items(), desc=f"  {choice.patient_id}", leave=False
            ):
                primary = info["primary"]
                if dual_choice.mode == "dual_acq":
                    secondary = pairs.get(sop_uid)
                    if secondary is None:
                        skipped_unpaired += 1
                        continue
                    pair_mode = "dual_acq_twin"
                    pair_dz = 0.0
                else:
                    secondary, pair_dz = nearest_instance_by_z(
                        secondary_insts, primary["ipp"], geom_tol
                    )
                    if secondary is None:
                        skipped_unpaired += 1
                        continue
                    pair_mode = f"multi_series_z:{pair_dz:.3f}"

                try:
                    prim_bytes = client.download_instance_file(primary["orthanc_id"])
                    sec_bytes = client.download_instance_file(secondary["orthanc_id"])
                    prim_gray, prim_meta = _windowed_array_from_dicom(prim_bytes, window)
                    sec_gray, sec_meta = _windowed_array_from_dicom(sec_bytes, window)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{choice.patient_id}: load CT failed {sop_uid}: {exc}")
                    continue

                # Warp secondary intensities onto primary geometry, then resize both.
                try:
                    sec_on_primary = resample_image_to_geometry(
                        sec_gray, secondary["geometry"], primary["geometry"]
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        f"{choice.patient_id}: secondary resample failed {sop_uid}: {exc}"
                    )
                    continue

                mask_resized = resize_mask(info["mask"], image_size)
                if args.skip_empty and mask_resized.max() == 0:
                    skipped_empty += 1
                    continue

                stem = sop_uid.replace(".", "_")
                image_a = img_dir / f"{stem}_a.png"
                image_b = img_dir / f"{stem}_b.png"
                mask_path = mask_dir / f"{stem}.png"
                _save_float01_png(prim_gray, image_a, image_size)
                _save_float01_png(sec_on_primary, image_b, image_size)
                save_mask_png(mask_resized, str(mask_path))

                record = {
                    "patient_id": choice.patient_id,
                    "orthanc_patient_id": choice.orthanc_patient_id,
                    "orthanc_study_id": choice.orthanc_study_id,
                    "orthanc_seg_series_id": choice.orthanc_series_id,
                    "orthanc_ct_instance_id": primary["orthanc_id"],
                    "orthanc_ct_instance_id_b": secondary["orthanc_id"],
                    "study_instance_uid": choice.study_instance_uid,
                    "seg_series_instance_uid": choice.series_instance_uid,
                    "seg_series_number": choice.series_number,
                    "seg_series_description": choice.series_description,
                    "referenced_series_uids": ref_series_uids,
                    "sop_instance_uid": sop_uid,
                    "sop_instance_uid_b": secondary["sop"],
                    "source_seg_sop_instance_uid": info["source_seg_sop"],
                    "liver_segment_numbers": used_segments,
                    "liver_segment_number": used_segments[0] if used_segments else None,
                    "selection_reason": choice.selection_reason,
                    "dual_series_mode": dual_choice.mode,
                    "dual_series_reason": dual_choice.reason,
                    "primary_series_uid": dual_choice.primary_series_uid,
                    "secondary_series_uid": dual_choice.secondary_series_uid,
                    "dual_acquisition_series": dual_acq,
                    "pair_mode": pair_mode,
                    "pair_dz_mm": float(pair_dz),
                    "acquisition_pass": primary.get("acquisition_pass"),
                    "acquisition_pass_b": secondary.get("acquisition_pass"),
                    "instance_number": primary.get("instance_number"),
                    "instance_number_b": secondary.get("instance_number"),
                    "alignment": info["alignment"],
                    "image_size": image_size,
                    "image_source": "hu",
                    "window_mode": "fixed",
                    "window_center": window.center,
                    "window_width": window.width,
                    "intensity_meta": prim_meta,
                    "intensity_meta_b": sec_meta,
                    "in_channels": 2,
                    "image_path": str(image_a.relative_to(args.output)),
                    "image_path_b": str(image_b.relative_to(args.output)),
                    "mask_path": str(mask_path.relative_to(args.output)),
                    "mask_foreground_pixels": int(mask_resized.sum()),
                }
                manifest.write(json.dumps(record) + "\n")
                total_pairs += 1

    print(f"Wrote {total_pairs} dual-channel pairs to {args.output}")
    print(f"Manifest: {manifest_path}")
    if skipped_empty:
        print(f"Skipped empty masks: {skipped_empty}")
    if skipped_unpaired:
        print(f"Skipped unpaired primary slices: {skipped_unpaired}")
    if errors:
        err_path = args.output / "download_errors.log"
        err_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"{len(errors)} warnings/errors logged to {err_path}")
    return 0 if total_pairs else 1


if __name__ == "__main__":
    raise SystemExit(main())
