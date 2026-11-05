"""Extract per-SOP / per-plane liver binary masks from DICOM SEG instances."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from PIL import Image

from .geometry import SliceGeometry, ipp_key


@dataclass
class FrameMask:
    referenced_sop_instance_uid: str | None
    referenced_series_instance_uid: str | None
    segment_numbers: list[int]
    mask: np.ndarray  # uint8 HxW, values 0/1 at SEG native resolution
    geometry: SliceGeometry
    merge_key: str  # stable key used to OR-merge segments on the same plane


def resolve_segment_numbers(
    seg_ds: Dataset,
    preferred: list[int],
    series_number: int,
    series_description: str = "",
) -> list[int]:
    segments = list(getattr(seg_ds, "SegmentSequence", []) or [])
    available = {int(getattr(s, "SegmentNumber", 0)) for s in segments}
    if not available:
        raise ValueError("SEG has no SegmentSequence entries")

    chosen = [n for n in preferred if n in available]
    if chosen:
        return chosen

    liver_labeled = [
        int(s.SegmentNumber)
        for s in segments
        if "liver" in str(getattr(s, "SegmentLabel", "")).lower()
    ]
    if liver_labeled:
        return liver_labeled

    numbers = sorted(available)
    if series_number == 999 and "Spleen" in series_description and len(numbers) >= 2:
        return [numbers[1]]
    if series_number == 300:
        return numbers[:3]
    return [numbers[0]]


def _referenced_sop_from_frame(frame_fg: Dataset) -> str | None:
    der = getattr(frame_fg, "DerivationImageSequence", None)
    if der:
        for der_item in der:
            src = getattr(der_item, "SourceImageSequence", None)
            if not src:
                continue
            for src_item in src:
                sop = str(getattr(src_item, "ReferencedSOPInstanceUID", "") or "")
                if sop:
                    return sop

    ref_img = getattr(frame_fg, "ReferencedImageSequence", None)
    if ref_img:
        for item in ref_img:
            sop = str(getattr(item, "ReferencedSOPInstanceUID", "") or "")
            if sop:
                return sop
    return None


def _series_uid_from_shared(seg_ds: Dataset) -> str | None:
    refs = getattr(seg_ds, "ReferencedSeriesSequence", None)
    if refs:
        return str(getattr(refs[0], "SeriesInstanceUID", "") or "") or None
    return None


def _pixel_array_frames(seg_ds: Dataset) -> np.ndarray:
    arr = seg_ds.pixel_array
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 4:
        arr = arr[..., 0]
    return np.asarray(arr)


def _shared_or_frame_geometry(seg_ds: Dataset, frame_fg: Dataset) -> SliceGeometry:
    shared = getattr(seg_ds, "SharedFunctionalGroupsSequence", [None])[0]

    def _from(seq_owner: Dataset | None, attr: str, field: str):
        if seq_owner is None:
            return None
        seq = getattr(seq_owner, attr, None)
        if not seq:
            return None
        return getattr(seq[0], field, None)

    iop = _from(frame_fg, "PlaneOrientationSequence", "ImageOrientationPatient")
    if iop is None:
        iop = _from(shared, "PlaneOrientationSequence", "ImageOrientationPatient")
    ipp = _from(frame_fg, "PlanePositionSequence", "ImagePositionPatient")
    if ipp is None:
        ipp = _from(shared, "PlanePositionSequence", "ImagePositionPatient")
    spacing = _from(frame_fg, "PixelMeasuresSequence", "PixelSpacing")
    if spacing is None:
        spacing = _from(shared, "PixelMeasuresSequence", "PixelSpacing")

    if iop is None or ipp is None or spacing is None:
        raise ValueError("SEG frame missing IOP/IPP/PixelSpacing")

    return SliceGeometry(
        ipp=np.array([float(x) for x in ipp], dtype=np.float64),
        iop=np.array([float(x) for x in iop], dtype=np.float64),
        spacing=np.array([float(x) for x in spacing], dtype=np.float64),
        rows=int(seg_ds.Rows),
        cols=int(seg_ds.Columns),
    )


def extract_liver_frames(
    seg_bytes: bytes,
    liver_segment_numbers: list[int],
    series_number: int,
    series_description: str = "",
) -> tuple[list[FrameMask], Dataset]:
    """
    Extract per-plane masks, OR-merging requested segments.

    Frames are merged by referenced SOP when available, otherwise by quantized IPP
    so SEGs with broken DerivationImageSequence still produce one mask per plane.
    """
    seg_ds = pydicom.dcmread(io.BytesIO(seg_bytes))
    segment_numbers = resolve_segment_numbers(
        seg_ds,
        preferred=liver_segment_numbers,
        series_number=series_number,
        series_description=series_description,
    )
    series_uid = _series_uid_from_shared(seg_ds)

    pixels = _pixel_array_frames(seg_ds)
    per_frame = list(getattr(seg_ds, "PerFrameFunctionalGroupsSequence", []) or [])
    if len(per_frame) != pixels.shape[0]:
        n = min(len(per_frame), pixels.shape[0])
        per_frame = per_frame[:n]
        pixels = pixels[:n]

    wanted = set(segment_numbers)
    merged: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}

    for idx, frame_fg in enumerate(per_frame):
        seg_id = getattr(frame_fg, "SegmentIdentificationSequence", None)
        frame_seg_num = None
        if seg_id:
            frame_seg_num = int(getattr(seg_id[0], "ReferencedSegmentNumber", -1))
        if frame_seg_num not in wanted:
            continue

        try:
            geometry = _shared_or_frame_geometry(seg_ds, frame_fg)
        except ValueError:
            continue

        sop_uid = _referenced_sop_from_frame(frame_fg)
        if sop_uid:
            key = f"sop:{sop_uid}"
        else:
            key = "ipp:" + ",".join(str(x) for x in ipp_key(geometry.ipp, decimals=3))

        mask = (pixels[idx] > 0).astype(np.uint8)
        prev = merged.get(key)
        if prev is None:
            merged[key] = mask
            meta[key] = {"sop": sop_uid, "geometry": geometry}
        else:
            merged[key] = (prev.astype(bool) | mask.astype(bool)).astype(np.uint8)
            # Prefer keeping a real SOP if a later segment frame provides one.
            if meta[key]["sop"] is None and sop_uid:
                meta[key]["sop"] = sop_uid

    frames = [
        FrameMask(
            referenced_sop_instance_uid=meta[key]["sop"],
            referenced_series_instance_uid=series_uid,
            segment_numbers=segment_numbers,
            mask=mask,
            geometry=meta[key]["geometry"],
            merge_key=key,
        )
        for key, mask in merged.items()
    ]
    return frames, seg_ds


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
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.resize((size, size), resample=Image.Resampling.NEAREST)
    return (np.asarray(img) > 127).astype(np.uint8)


def save_mask_png(mask: np.ndarray, path: str) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_image_bytes(image_bytes: bytes, path: str, size: int | None = None) -> None:
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    if size is not None and (img.width != size or img.height != size):
        img = img.resize((size, size), resample=Image.Resampling.BILINEAR)
    img.save(path)
