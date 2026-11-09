#!/usr/bin/env python3
"""Download paired CT images + liver SEG masks from Orthanc.

Supports:
  - Orthanc /rendered grayscale PNGs (legacy)
  - DICOM → HU → fixed abdomen window → grayscale PNG (preferred)

Aligns SEG masks onto CT geometry (IOP flips, IPP in-plane offsets).

For dual-acquisition CT series with SEG series 300, only the later acquisition
(higher InstanceNumber at each IPP) is kept — SEG 300 was drawn on that pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from src.ct_preprocess import WindowSpec, dicom_bytes_to_windowed_png
from src.dataset_select import select_all_patients
from src.geometry import (
    describe_alignment,
    geometry_from_tags,
    group_instances_by_ipp,
    ipp_key,
    resample_mask_to_geometry,
)
from src.orthanc_client import OrthancClient
from src.seg_masks import (
    extract_liver_frames,
    referenced_series_uids,
    resize_mask,
    save_image_bytes,
    save_mask_png,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--source",
        choices=("hu", "rendered"),
        default="hu",
        help="Image intensity source: DICOM HU+window (hu) or Orthanc /rendered",
    )
    parser.add_argument(
        "--window-center",
        type=float,
        default=40.0,
        help="Fallback/fixed abdomen window center in HU (used with --source hu)",
    )
    parser.add_argument(
        "--window-width",
        type=float,
        default=400.0,
        help="Fallback/fixed abdomen window width in HU (used with --source hu)",
    )
    parser.add_argument(
        "--window-mode",
        choices=("fixed", "dicom"),
        default="fixed",
        help=(
            "With --source hu: 'fixed' uses --window-center/width; "
            "'dicom' uses each instance WindowCenter/WindowWidth "
            "(Orthanc /rendered VOI mimic), falling back to fixed if missing"
        ),
    )
    parser.add_argument("--skip-empty", action="store_true")
    parser.add_argument("--limit-patients", type=int, default=None)
    parser.add_argument("--patient-id", type=str, default=None)
    parser.add_argument(
        "--ipp-decimals",
        type=int,
        default=3,
        help="IPP quantization decimals for dual-acquisition grouping",
    )
    parser.add_argument(
        "--match-tol-mm",
        type=float,
        default=1.25,
        help="Max IPP distance (mm) when falling back to geometric CT matching",
    )
    return parser.parse_args()


def load_ct_series_instances(client: OrthancClient, series_uid: str) -> list[dict]:
    series_id = client.find_series_by_uid(series_uid)
    if not series_id:
        return []
    series = client.get_series(series_id)
    instances: list[dict] = []
    for iid in series.get("Instances") or []:
        tags = client.get_json(f"/instances/{iid}/simplified-tags")
        geom = geometry_from_tags(tags)
        try:
            instance_number = int(float(tags.get("InstanceNumber", 0)))
        except (TypeError, ValueError):
            instance_number = 0
        instances.append(
            {
                "orthanc_id": iid,
                "sop": str(tags["SOPInstanceUID"]),
                "ipp": geom.ipp,
                "iop": geom.iop,
                "geometry": geom,
                "instance_number": instance_number,
                "tags": tags,
            }
        )
    return instances


def mark_dual_acquisition_passes(
    ct_instances: list[dict],
    by_ipp: dict[tuple, list[dict]],
) -> bool:
    """
    Mark each instance with acquisition_pass: 0 = earlier, 1 = later.

    Dual-acq twins are distinguished by InstanceNumber (AcquisitionTime is often
    empty): the higher InstanceNumber at a given IPP is the later pass.
    """
    dual = any(len(group) > 1 for group in by_ipp.values())
    if not dual:
        for inst in ct_instances:
            inst["acquisition_pass"] = 0
        return False

    earlier_nums: list[int] = []
    later_nums: list[int] = []
    for group in by_ipp.values():
        ordered = sorted(group, key=lambda x: (x["instance_number"], x["sop"]))
        if len(ordered) == 1:
            continue
        for i, inst in enumerate(ordered):
            inst["acquisition_pass"] = 1 if i == len(ordered) - 1 else 0
        earlier_nums.append(ordered[0]["instance_number"])
        later_nums.append(ordered[-1]["instance_number"])

    # Lonely IPP slices (no twin): assign pass by InstanceNumber threshold.
    if earlier_nums and later_nums:
        threshold = 0.5 * (float(np.median(earlier_nums)) + float(np.median(later_nums)))
    else:
        threshold = float(np.median([inst["instance_number"] for inst in ct_instances]))
    for group in by_ipp.values():
        if len(group) != 1:
            continue
        inst = group[0]
        inst["acquisition_pass"] = 1 if inst["instance_number"] >= threshold else 0
    return True


def later_acquisition_only(targets: list[dict]) -> list[dict]:
    """Keep only the later dual-acq pass from a twin group."""
    if not targets:
        return []
    if len(targets) == 1:
        return targets
    return [max(targets, key=lambda x: (x.get("acquisition_pass", 0), x["instance_number"]))]


def estimate_slice_spacing_mm(ct_instances: list[dict]) -> float:
    if len(ct_instances) < 2:
        return 2.5
    normal = ct_instances[0]["geometry"].normal
    zs = sorted(float(inst["ipp"] @ normal) for inst in ct_instances)
    diffs = np.diff(zs)
    diffs = diffs[np.abs(diffs) > 1e-4]
    if len(diffs) == 0:
        return 2.5
    return float(np.median(np.abs(diffs)))


def _ct_normal_positions(ct_instances: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    normal = ct_instances[0]["geometry"].normal
    ct_z = np.array([float(inst["ipp"] @ normal) for inst in ct_instances], dtype=np.float64)
    return normal, ct_z


def choose_stack_z_mapping(
    planes: list[dict],
    by_sop: dict[str, dict],
    ct_instances: list[dict],
    match_tol_mm: float,
) -> dict:
    """
    Choose how to map SEG plane positions onto the CT stack.

    Handles:
    - global normal offset
    - inverted SEG stack direction (IPP step sign flipped), using SOP anchors
      when available (seen in HCC_089).
    """
    if not planes or not ct_instances:
        return {"mode": "identity", "offset_mm": 0.0, "anchor_seg_z": None}

    normal, ct_z = _ct_normal_positions(ct_instances)
    seg_z = np.array([float(p["geometry"].ipp @ normal) for p in planes], dtype=np.float64)

    def hit_count(mapped_z: np.ndarray) -> int:
        return int(sum(np.min(np.abs(ct_z - z)) <= match_tol_mm for z in mapped_z))

    # Identity + global offset
    nearest_residuals = np.array(
        [float(z - ct_z[np.argmin(np.abs(ct_z - z))]) for z in seg_z]
    )
    offset = float(np.median(nearest_residuals)) if len(nearest_residuals) else 0.0
    identity_hits = hit_count(seg_z - offset)

    best = {
        "mode": "identity",
        "offset_mm": offset,
        "anchor_seg_z": None,
        "hits": identity_hits,
    }

    # Reflection about each SOP anchor: corrected_z = 2*anchor_seg_z - seg_z
    for plane in planes:
        sop = plane.get("sop")
        if not sop or sop not in by_sop:
            continue
        anchor_seg_z = float(plane["geometry"].ipp @ normal)
        mapped = 2.0 * anchor_seg_z - seg_z
        # small residual after reflection
        refl_offset = float(
            np.median([float(z - ct_z[np.argmin(np.abs(ct_z - z))]) for z in mapped])
        )
        mapped = mapped - refl_offset
        hits = hit_count(mapped)
        if hits > best["hits"]:
            best = {
                "mode": "reflect",
                "offset_mm": refl_offset,
                "anchor_seg_z": anchor_seg_z,
                "hits": hits,
            }

    return best


def map_seg_z(seg_z: float, mapping: dict) -> float:
    if mapping["mode"] == "reflect" and mapping["anchor_seg_z"] is not None:
        return 2.0 * float(mapping["anchor_seg_z"]) - float(seg_z) - float(mapping["offset_mm"])
    return float(seg_z) - float(mapping["offset_mm"])


def find_ct_targets(
    frame_sop: str | None,
    seg_ipp: np.ndarray,
    by_sop: dict[str, dict],
    ct_instances: list[dict],
    by_ipp: dict[tuple, list[dict]],
    ipp_decimals: int,
    match_tol_mm: float,
    z_mapping: dict | None = None,
    second_pass_only: bool = False,
) -> tuple[list[dict], str]:
    """
    Resolve CT instance(s) for a SEG plane.

    Prefer referenced SOP. For dual-acq + SEG 300 (second_pass_only), remap to
    the later acquisition twin and do not keep the earlier pass.
    """
    if frame_sop and frame_sop in by_sop:
        ct_inst = by_sop[frame_sop]
        twins = by_ipp.get(ipp_key(ct_inst["ipp"], ipp_decimals), [ct_inst])
        if second_pass_only:
            return later_acquisition_only(twins), "sop_second_pass"
        return twins, "sop"

    # Geometric matching uses only eligible instances (already filtered if needed).
    if not ct_instances:
        return [], "none"

    normal, ct_z = _ct_normal_positions(ct_instances)
    raw_z = float(np.asarray(seg_ipp) @ normal)
    seg_z = map_seg_z(raw_z, z_mapping or {"mode": "identity", "offset_mm": 0.0, "anchor_seg_z": None})
    best_i = int(np.argmin(np.abs(ct_z - seg_z)))
    best_dz = float(abs(ct_z[best_i] - seg_z))
    if best_dz > match_tol_mm:
        return [], f"geom_miss:{best_dz:.3f}"

    ct_inst = ct_instances[best_i]
    twins = by_ipp.get(ipp_key(ct_inst["ipp"], ipp_decimals), [ct_inst])
    mode = (z_mapping or {}).get("mode", "identity")
    if second_pass_only:
        return later_acquisition_only(twins), f"geom_z:{mode}:second_pass:{best_dz:.3f}"
    return twins, f"geom_z:{mode}:{best_dz:.3f}"


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    image_size = args.image_size or int(os.getenv("IMAGE_SIZE", "128"))

    client = OrthancClient()
    print(f"Connecting to {client.base_url} ...")
    choices = select_all_patients(client)
    if args.patient_id:
        choices = [c for c in choices if c.patient_id == args.patient_id]
    if args.limit_patients is not None:
        choices = choices[: args.limit_patients]

    print(f"Selected {len(choices)} patient(s) with usable SEG series")
    window = WindowSpec(center=args.window_center, width=args.window_width)
    print(
        f"Image source={args.source} size={image_size}"
        + (
            f" window_mode={args.window_mode} "
            f"fallback=C{window.center}/W{window.width}"
            if args.source == "hu"
            else ""
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.jsonl"

    total_pairs = 0
    skipped_empty = 0
    errors: list[str] = []

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for choice in tqdm(choices, desc="Patients"):
            safe_pid = choice.patient_id.replace("/", "_")
            out_dir = args.output / safe_pid / choice.study_instance_uid
            img_dir = out_dir / "images"
            mask_dir = out_dir / "masks"
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            series = client.get_series(choice.orthanc_series_id)
            instance_ids = series.get("Instances") or []
            if not instance_ids:
                errors.append(f"{choice.patient_id}: SEG series has no instances")
                continue

            plane_masks: dict[str, dict] = {}
            used_segments = list(choice.liver_segment_numbers)
            ref_series_uids: list[str] = []

            for instance_id in instance_ids:
                try:
                    seg_bytes = client.download_instance_file(instance_id)
                    frames, seg_ds = extract_liver_frames(
                        seg_bytes,
                        liver_segment_numbers=choice.liver_segment_numbers,
                        series_number=choice.series_number,
                        series_description=choice.series_description,
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
                                (prev["mask"].astype(bool) | frame.mask.astype(bool)).astype(
                                    np.uint8
                                )
                            )
                            if prev["sop"] is None and frame.referenced_sop_instance_uid:
                                prev["sop"] = frame.referenced_sop_instance_uid
                    if seg_ds is not None:
                        ref_series_uids = referenced_series_uids(seg_ds) or ref_series_uids
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{choice.patient_id} instance {instance_id}: {exc}")

            if not plane_masks:
                errors.append(f"{choice.patient_id}: no liver SEG frames extracted")
                continue

            if not ref_series_uids:
                errors.append(f"{choice.patient_id}: SEG has no ReferencedSeriesSequence")
                continue

            ct_instances = load_ct_series_instances(client, ref_series_uids[0])
            if not ct_instances:
                errors.append(
                    f"{choice.patient_id}: referenced CT series not found: {ref_series_uids[0]}"
                )
                continue

            by_sop = {inst["sop"]: inst for inst in ct_instances}
            by_ipp = group_instances_by_ipp(ct_instances, decimals=args.ipp_decimals)
            dual_acq = mark_dual_acquisition_passes(ct_instances, by_ipp)
            # SEG series 300 on dual-acq was drawn on the later pass only.
            second_pass_only = dual_acq and choice.series_number == 300
            match_instances = (
                [inst for inst in ct_instances if inst.get("acquisition_pass", 0) == 1]
                if second_pass_only
                else ct_instances
            )
            if second_pass_only and not match_instances:
                errors.append(f"{choice.patient_id}: dual-acq but no later-pass CT instances")
                continue

            slice_spacing = estimate_slice_spacing_mm(match_instances)
            geom_tol = max(args.match_tol_mm, 0.6 * slice_spacing)
            plane_list = list(plane_masks.values())
            z_mapping = choose_stack_z_mapping(plane_list, by_sop, match_instances, geom_tol)

            ct_masks: dict[str, dict] = {}
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
                    second_pass_only=second_pass_only,
                )
                if not targets:
                    unmatched += 1
                    continue

                for twin in targets:
                    if second_pass_only and twin.get("acquisition_pass", 0) != 1:
                        continue
                    aligned = resample_mask_to_geometry(
                        plane["mask"], plane["geometry"], twin["geometry"]
                    )
                    alignment = describe_alignment(plane["geometry"], twin["geometry"])
                    alignment["match_mode"] = match_mode
                    alignment["acquisition_pass"] = twin.get("acquisition_pass")
                    prev = ct_masks.get(twin["sop"])
                    if prev is None:
                        ct_masks[twin["sop"]] = {
                            "mask": aligned,
                            "orthanc_id": twin["orthanc_id"],
                            "alignment": alignment,
                            "source_seg_sop": plane["sop"],
                            "acquisition_pass": twin.get("acquisition_pass"),
                            "instance_number": twin.get("instance_number"),
                        }
                    else:
                        prev["mask"] = (
                            (prev["mask"].astype(bool) | aligned.astype(bool)).astype(np.uint8)
                        )

            if unmatched:
                errors.append(
                    f"{choice.patient_id}: {unmatched}/{len(plane_masks)} SEG planes unmatched to CT"
                )
            if not ct_masks:
                errors.append(f"{choice.patient_id}: no CT masks after alignment")
                continue

            for sop_uid, info in tqdm(ct_masks.items(), desc=f"  {choice.patient_id}", leave=False):
                orthanc_instance = info["orthanc_id"]
                intensity_meta: dict = {}
                try:
                    if args.source == "hu":
                        dicom_bytes = client.download_instance_file(orthanc_instance)
                        png_bytes, intensity_meta = dicom_bytes_to_windowed_png(
                            dicom_bytes,
                            size=image_size,
                            window=window,
                            window_mode=args.window_mode,
                        )
                    else:
                        png_bytes = client.download_rendered(
                            orthanc_instance, width=image_size
                        )
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        f"{choice.patient_id}: {args.source} failed {sop_uid}: {exc}"
                    )
                    continue

                mask_resized = resize_mask(info["mask"], image_size)
                if args.skip_empty and mask_resized.max() == 0:
                    skipped_empty += 1
                    continue

                stem = sop_uid.replace(".", "_")
                image_path = img_dir / f"{stem}.png"
                mask_path = mask_dir / f"{stem}.png"
                save_image_bytes(png_bytes, str(image_path), size=image_size)
                save_mask_png(mask_resized, str(mask_path))

                record = {
                    "patient_id": choice.patient_id,
                    "orthanc_patient_id": choice.orthanc_patient_id,
                    "orthanc_study_id": choice.orthanc_study_id,
                    "orthanc_seg_series_id": choice.orthanc_series_id,
                    "orthanc_ct_instance_id": orthanc_instance,
                    "study_instance_uid": choice.study_instance_uid,
                    "seg_series_instance_uid": choice.series_instance_uid,
                    "seg_series_number": choice.series_number,
                    "seg_series_description": choice.series_description,
                    "referenced_series_uids": ref_series_uids,
                    "sop_instance_uid": sop_uid,
                    "source_seg_sop_instance_uid": info["source_seg_sop"],
                    "liver_segment_numbers": used_segments,
                    "liver_segment_number": used_segments[0] if used_segments else None,
                    "selection_reason": choice.selection_reason,
                    "dual_acquisition_series": dual_acq,
                    "dual_acq_second_pass_only": second_pass_only,
                    "acquisition_pass": info.get("acquisition_pass"),
                    "instance_number": info.get("instance_number"),
                    "alignment": info["alignment"],
                    "image_size": image_size,
                    "image_source": args.source,
                    "window_mode": args.window_mode if args.source == "hu" else None,
                    "window_center": (
                        (intensity_meta or {}).get("window_center", window.center)
                        if args.source == "hu"
                        else None
                    ),
                    "window_width": (
                        (intensity_meta or {}).get("window_width", window.width)
                        if args.source == "hu"
                        else None
                    ),
                    "intensity_meta": intensity_meta or None,
                    "image_path": str(image_path.relative_to(args.output)),
                    "mask_path": str(mask_path.relative_to(args.output)),
                    "mask_foreground_pixels": int(mask_resized.sum()),
                }
                manifest.write(json.dumps(record) + "\n")
                total_pairs += 1

    print(f"Wrote {total_pairs} pairs to {args.output}")
    print(f"Manifest: {manifest_path}")
    if skipped_empty:
        print(f"Skipped empty masks: {skipped_empty}")
    if errors:
        err_path = args.output / "download_errors.log"
        err_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"{len(errors)} warnings/errors logged to {err_path}")
    return 0 if total_pairs else 1


if __name__ == "__main__":
    raise SystemExit(main())
