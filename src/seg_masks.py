"""Extract per-SOP liver binary masks from DICOM SEG instances."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from PIL import Image


@dataclass
class FrameMask:
    referenced_sop_instance_uid: str
    referenced_series_instance_uid: str | None
    segment_number: int
    mask: np.ndarray  # uint8 HxW, values 0/1


def _get_segment_label(seg_ds: Dataset, segment_number: int) -> str:
    for item in getattr(seg_ds, "SegmentSequence", []) or []:
        if int(getattr(item, "SegmentNumber", -1)) == segment_number:
            return str(getattr(item, "SegmentLabel", "")).strip()
    return ""


def resolve_liver_segment_number(seg_ds: Dataset, preferred: int, series_number: int) -> int:
    """
    Prefer the planned segment number; for series 999 also accept a segment
    whose label contains 'liver' (case-insensitive).
    """
    segments = list(getattr(seg_ds, "SegmentSequence", []) or [])
    numbers = [int(getattr(s, "SegmentNumber", 0)) for s in segments]

    if preferred in numbers:
        if series_number == 999:
            label = _get_segment_label(seg_ds, preferred).lower()
            if "liver" in label or preferred == 2:
                return preferred
            for seg in segments:
                if "liver" in str(getattr(seg, "SegmentLabel", "")).lower():
                    return int(seg.SegmentNumber)
        return preferred

    for seg in segments:
        if "liver" in str(getattr(seg, "SegmentLabel", "")).lower():
            return int(seg.SegmentNumber)

    if numbers:
        # Fallback: first segment for 300-style, second if available for 999.
        if series_number == 999 and len(numbers) >= 2:
            return numbers[1]
        return numbers[0]

    raise ValueError("SEG has no SegmentSequence entries")


def _referenced_sop_from_frame(frame_fg: Dataset) -> tuple[str | None, str | None]:
    """Return (SOPInstanceUID, SeriesInstanceUID) referenced by a SEG frame."""
    sop_uid: str | None = None
    series_uid: str | None = None

    der = getattr(frame_fg, "DerivationImageSequence", None)
    if der:
        for der_item in der:
            src = getattr(der_item, "SourceImageSequence", None)
            if not src:
                continue
            for src_item in src:
                sop_uid = str(getattr(src_item, "ReferencedSOPInstanceUID", "") or "") or sop_uid
                # Series UID is often only at ReferencedSeriesSequence level.

    # Some SEGs put refs under PurposeOfReference / ReferencedImageSequence
    ref_img = getattr(frame_fg, "ReferencedImageSequence", None)
    if ref_img and sop_uid is None:
        for item in ref_img:
            sop_uid = str(getattr(item, "ReferencedSOPInstanceUID", "") or "") or sop_uid

    return sop_uid, series_uid


def _series_uid_from_shared(seg_ds: Dataset) -> str | None:
    shared = getattr(seg_ds, "SharedFunctionalGroupsSequence", None)
    if shared:
        # Rarely holds series refs; prefer top-level ReferencedSeriesSequence.
        pass
    refs = getattr(seg_ds, "ReferencedSeriesSequence", None)
    if refs:
        return str(getattr(refs[0], "SeriesInstanceUID", "") or "") or None
    return None


def _pixel_array_frames(seg_ds: Dataset) -> np.ndarray:
    """Return SEG pixel data as (num_frames, H, W) uint8/bool-friendly array."""
    arr = seg_ds.pixel_array
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 4:
        # (frames, H, W, samples) — take first sample
        arr = arr[..., 0]
    return np.asarray(arr)


def extract_liver_frames(
    seg_bytes: bytes,
    liver_segment_number: int,
    series_number: int,
) -> tuple[list[FrameMask], Dataset]:
    seg_ds = pydicom.dcmread(io.BytesIO(seg_bytes))
    segment_number = resolve_liver_segment_number(seg_ds, liver_segment_number, series_number)
    series_uid = _series_uid_from_shared(seg_ds)
    series_sops = referenced_sop_uids_in_series(seg_ds, series_uid)

    # Prefer highdicom when available (handles fragmented / multi-frame SEG cleanly).
    try:
        frames = _extract_with_highdicom(seg_bytes, segment_number, series_uid)
        if frames:
            return frames, seg_ds
    except Exception:
        pass

    pixels = _pixel_array_frames(seg_ds)
    per_frame = list(getattr(seg_ds, "PerFrameFunctionalGroupsSequence", []) or [])
    if len(per_frame) != pixels.shape[0]:
        n = min(len(per_frame), pixels.shape[0])
        per_frame = per_frame[:n]
        pixels = pixels[:n]

    frames = []
    liver_frame_i = 0
    for idx, frame_fg in enumerate(per_frame):
        seg_id = getattr(frame_fg, "SegmentIdentificationSequence", None)
        frame_seg_num = None
        if seg_id:
            frame_seg_num = int(getattr(seg_id[0], "ReferencedSegmentNumber", -1))
        if frame_seg_num != segment_number:
            continue

        sop_uid, _ = _referenced_sop_from_frame(frame_fg)
        if not sop_uid and liver_frame_i < len(series_sops):
            # Fallback: some SEGs only list SOPs under ReferencedSeriesSequence.
            sop_uid = series_sops[liver_frame_i]
        liver_frame_i += 1
        if not sop_uid:
            continue

        mask = (pixels[idx] > 0).astype(np.uint8)
        frames.append(
            FrameMask(
                referenced_sop_instance_uid=sop_uid,
                referenced_series_instance_uid=series_uid,
                segment_number=segment_number,
                mask=mask,
            )
        )

    return frames, seg_ds


def _extract_with_highdicom(
    seg_bytes: bytes,
    segment_number: int,
    series_uid: str | None,
) -> list[FrameMask]:
    from highdicom.seg.sop import Segmentation

    seg = Segmentation.from_dataset(pydicom.dcmread(io.BytesIO(seg_bytes)))
    source_uids = seg.get_source_image_uids()
    # Each entry is (StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID)
    sop_uids = [str(uids[2]) for uids in source_uids]
    if not sop_uids:
        return []

    pixels = seg.get_pixels_by_source_instance(
        source_sop_instance_uids=sop_uids,
        segment_numbers=[segment_number],
        combine_segments=True,
    )
    arr = np.asarray(pixels)
    # Expected shapes: (N, H, W) or (N, H, W, 1)
    if arr.ndim == 4:
        arr = arr[..., 0]
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]

    frames: list[FrameMask] = []
    for sop_uid, frame in zip(sop_uids, arr):
        mask = (frame > 0).astype(np.uint8)
        frames.append(
            FrameMask(
                referenced_sop_instance_uid=sop_uid,
                referenced_series_instance_uid=series_uid,
                segment_number=segment_number,
                mask=mask,
            )
        )
    return frames


def referenced_series_uids(seg_ds: Dataset) -> list[str]:
    uids: list[str] = []
    for item in getattr(seg_ds, "ReferencedSeriesSequence", []) or []:
        uid = str(getattr(item, "SeriesInstanceUID", "") or "")
        if uid and uid not in uids:
            uids.append(uid)
    return uids


def referenced_sop_uids_in_series(seg_ds: Dataset, series_uid: str | None = None) -> list[str]:
    uids: list[str] = []
    for item in getattr(seg_ds, "ReferencedSeriesSequence", []) or []:
        item_uid = str(getattr(item, "SeriesInstanceUID", "") or "")
        if series_uid and item_uid != series_uid:
            continue
        for img in getattr(item, "ReferencedInstanceSequence", []) or []:
            sop = str(getattr(img, "ReferencedSOPInstanceUID", "") or "")
            if sop and sop not in uids:
                uids.append(sop)
    return uids


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbor resize to size x size, binary 0/1."""
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.resize((size, size), resample=Image.Resampling.NEAREST)
    return (np.asarray(img) > 127).astype(np.uint8)


def save_mask_png(mask: np.ndarray, path: str) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_image_bytes(image_bytes: bytes, path: str, size: int | None = None) -> None:
    """Save Orthanc rendered bytes; optionally force square size."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    if size is not None and (img.width != size or img.height != size):
        img = img.resize((size, size), resample=Image.Resampling.BILINEAR)
    img.save(path)
