#!/usr/bin/env python3
"""Download paired CT rendered images + liver SEG masks from Orthanc."""

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

from src.dataset_select import select_all_patients
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
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "raw",
        help="Output directory for image/mask pairs",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Rendered square size (default: IMAGE_SIZE from .env or 128)",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip slices whose liver mask is empty",
    )
    parser.add_argument(
        "--limit-patients",
        type=int,
        default=None,
        help="Optional cap on number of patients to download",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Only download a single PatientID (exact match)",
    )
    return parser.parse_args()


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

            # Prefer a single multi-frame SEG instance; process all if needed.
            sop_to_mask: dict[str, object] = {}
            seg_ds = None
            used_segment = choice.liver_segment_number
            ref_series_uids: list[str] = []

            for instance_id in instance_ids:
                try:
                    seg_bytes = client.download_instance_file(instance_id)
                    frames, seg_ds = extract_liver_frames(
                        seg_bytes,
                        liver_segment_number=choice.liver_segment_number,
                        series_number=choice.series_number,
                    )
                    if frames:
                        used_segment = frames[0].segment_number
                    for frame in frames:
                        # If multiple frames map to same SOP, OR the masks.
                        prev = sop_to_mask.get(frame.referenced_sop_instance_uid)
                        if prev is None:
                            sop_to_mask[frame.referenced_sop_instance_uid] = frame.mask
                        else:
                            sop_to_mask[frame.referenced_sop_instance_uid] = (
                                (prev.astype(bool) | frame.mask.astype(bool)).astype(np.uint8)
                            )
                    if seg_ds is not None:
                        ref_series_uids = referenced_series_uids(seg_ds) or ref_series_uids
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{choice.patient_id} instance {instance_id}: {exc}")

            if not sop_to_mask:
                errors.append(f"{choice.patient_id}: no liver SEG frames extracted")
                continue

            for sop_uid, mask in tqdm(
                sop_to_mask.items(),
                desc=f"  {choice.patient_id}",
                leave=False,
            ):
                orthanc_instance = client.find_instance_by_sop_uid(sop_uid)
                if not orthanc_instance:
                    errors.append(f"{choice.patient_id}: CT SOP not in Orthanc: {sop_uid}")
                    continue

                try:
                    rendered = client.download_rendered(orthanc_instance, width=image_size)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{choice.patient_id}: render failed {sop_uid}: {exc}")
                    continue

                mask_resized = resize_mask(mask, image_size)
                if args.skip_empty and mask_resized.max() == 0:
                    skipped_empty += 1
                    continue

                stem = sop_uid.replace(".", "_")
                image_path = img_dir / f"{stem}.png"
                mask_path = mask_dir / f"{stem}.png"
                save_image_bytes(rendered, str(image_path), size=image_size)
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
                    "liver_segment_number": used_segment,
                    "selection_reason": choice.selection_reason,
                    "image_size": image_size,
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
