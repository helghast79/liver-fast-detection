"""Select and pair dual CT channels for mass-detect training.

Rules:
  - If the SEG-referenced CT series is a dual-acquisition merge (≈2 instances
    per IPP), use the earlier/later passes as the two channels.
  - Otherwise, among CT series in the study, pick 2: if exactly 2 exist use
    both; if 3+, prefer the pair that shares the same instance count (and
    includes the referenced series when possible).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import SliceGeometry, group_instances_by_ipp, ipp_key
from .orthanc_client import OrthancClient


@dataclass
class DualSeriesChoice:
    mode: str  # "dual_acq" | "multi_series"
    primary_series_uid: str
    secondary_series_uid: str
    reason: str


def _main_tags(resource: dict[str, Any]) -> dict[str, Any]:
    return resource.get("MainDicomTags") or {}


def list_ct_series(client: OrthancClient, study: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for series_id in study.get("Series") or []:
        series = client.get_series(series_id)
        tags = _main_tags(series)
        if str(tags.get("Modality", "")).upper() != "CT":
            continue
        out.append(series)
    return out


def series_uid(series: dict[str, Any]) -> str:
    return str(_main_tags(series).get("SeriesInstanceUID", "") or "")


def is_dual_acquisition_series(
    instances: list[dict],
    ipp_decimals: int = 3,
    min_twin_fraction: float = 0.4,
) -> bool:
    """True when a substantial fraction of IPPs have ≥2 instances."""
    if len(instances) < 4:
        return False
    by_ipp = group_instances_by_ipp(instances, decimals=ipp_decimals)
    if not by_ipp:
        return False
    twin_keys = sum(1 for g in by_ipp.values() if len(g) >= 2)
    return (twin_keys / len(by_ipp)) >= min_twin_fraction


def select_dual_series(
    ct_series: list[dict[str, Any]],
    ref_series_uid: str,
    instances_by_uid: dict[str, list[dict]],
    ipp_decimals: int = 3,
) -> DualSeriesChoice:
    """Choose primary/secondary CT sources for dual-channel export."""
    if not ct_series:
        raise ValueError("No CT series in study")

    by_uid = {series_uid(s): s for s in ct_series if series_uid(s)}
    if ref_series_uid not in by_uid:
        raise ValueError(f"Referenced CT series not among study CT series: {ref_series_uid}")

    ref_instances = instances_by_uid.get(ref_series_uid) or []
    if is_dual_acquisition_series(ref_instances, ipp_decimals=ipp_decimals):
        return DualSeriesChoice(
            mode="dual_acq",
            primary_series_uid=ref_series_uid,
            secondary_series_uid=ref_series_uid,
            reason=(
                "Referenced CT is dual-acquisition merge; "
                "channels = later/earlier passes at each IPP"
            ),
        )

    if len(ct_series) == 1:
        raise ValueError("Only one CT series and it is not dual-acquisition")

    if len(ct_series) == 2:
        other_uid = next(uid for uid in by_uid if uid != ref_series_uid)
        return DualSeriesChoice(
            mode="multi_series",
            primary_series_uid=ref_series_uid,
            secondary_series_uid=other_uid,
            reason="Exactly two CT series; primary=SEG-referenced, secondary=other",
        )

    # 3+: prefer two series with equal instance counts, including the referenced one.
    count_map: dict[int, list[str]] = {}
    for uid, insts in instances_by_uid.items():
        if uid not in by_uid:
            continue
        count_map.setdefault(len(insts), []).append(uid)

    # Prefer counts matching the referenced series if that count has a twin.
    ref_n = len(ref_instances)
    candidates = count_map.get(ref_n, [])
    if len(candidates) >= 2 and ref_series_uid in candidates:
        other = next(uid for uid in candidates if uid != ref_series_uid)
        return DualSeriesChoice(
            mode="multi_series",
            primary_series_uid=ref_series_uid,
            secondary_series_uid=other,
            reason=(
                f"Picked two CT series with n={ref_n} instances "
                f"(includes SEG-referenced)"
            ),
        )

    # Else any count that appears ≥2 times (largest count first).
    for n, uids in sorted(count_map.items(), key=lambda kv: (-kv[0], -len(kv[1]))):
        if len(uids) < 2:
            continue
        if ref_series_uid in uids:
            other = next(uid for uid in uids if uid != ref_series_uid)
            primary, secondary = ref_series_uid, other
        else:
            primary, secondary = uids[0], uids[1]
        return DualSeriesChoice(
            mode="multi_series",
            primary_series_uid=primary,
            secondary_series_uid=secondary,
            reason=f"Picked two CT series with equal n={n} instances",
        )

    # Fallback: referenced + largest other.
    others = sorted(
        ((uid, len(insts)) for uid, insts in instances_by_uid.items() if uid != ref_series_uid),
        key=lambda x: -x[1],
    )
    if not others:
        raise ValueError("Could not find a secondary CT series")
    return DualSeriesChoice(
        mode="multi_series",
        primary_series_uid=ref_series_uid,
        secondary_series_uid=others[0][0],
        reason="Fallback: SEG-referenced + largest other CT series",
    )


def nearest_instance_by_z(
    instances: list[dict],
    ipp: np.ndarray,
    match_tol_mm: float,
) -> tuple[dict | None, float]:
    if not instances:
        return None, float("inf")
    normal = instances[0]["geometry"].normal
    target_z = float(np.asarray(ipp, dtype=np.float64) @ normal)
    best = None
    best_dz = float("inf")
    for inst in instances:
        z = float(inst["ipp"] @ normal)
        dz = abs(z - target_z)
        if dz < best_dz:
            best_dz = dz
            best = inst
    if best is None or best_dz > match_tol_mm:
        return None, best_dz
    return best, best_dz


def resample_image_to_geometry(
    image: np.ndarray,
    src: SliceGeometry,
    dst: SliceGeometry,
) -> np.ndarray:
    """Bilinear resample of a 2D float/uint image from src frame onto dst frame."""
    from scipy import ndimage

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    d0 = dst.ipp - src.ipp
    a00 = float((dst.spacing[0] * dst.col_dir) @ src.col_dir / src.spacing[0])
    a01 = float((dst.spacing[1] * dst.row_dir) @ src.col_dir / src.spacing[0])
    t0 = float(d0 @ src.col_dir / src.spacing[0])
    a10 = float((dst.spacing[0] * dst.col_dir) @ src.row_dir / src.spacing[1])
    a11 = float((dst.spacing[1] * dst.row_dir) @ src.row_dir / src.spacing[1])
    t1 = float(d0 @ src.row_dir / src.spacing[1])
    matrix = np.array([[a00, a01], [a10, a11]], dtype=np.float64)
    offset = np.array([t0, t1], dtype=np.float64)

    if (
        np.allclose(matrix, np.eye(2), atol=1e-6)
        and np.allclose(offset, 0, atol=1e-3)
        and image.shape == (dst.rows, dst.cols)
    ):
        return image.astype(np.float32)

    warped = ndimage.affine_transform(
        image,
        matrix=matrix,
        offset=offset,
        output_shape=(dst.rows, dst.cols),
        order=1,
        mode="nearest",
        prefilter=False,
    )
    return warped.astype(np.float32)


def dual_acq_pairs(
    instances: list[dict],
    ipp_decimals: int = 3,
) -> list[tuple[dict, dict]]:
    """Return (earlier, later) pairs for IPPs that have both acquisition passes."""
    by_ipp = group_instances_by_ipp(instances, decimals=ipp_decimals)
    dual = any(len(g) > 1 for g in by_ipp.values())
    if not dual:
        return []
    for group in by_ipp.values():
        ordered = sorted(group, key=lambda x: (x["instance_number"], x["sop"]))
        if len(ordered) >= 2:
            for i, inst in enumerate(ordered):
                inst["acquisition_pass"] = 1 if i == len(ordered) - 1 else 0
        elif ordered:
            ordered[0]["acquisition_pass"] = 0

    pairs: list[tuple[dict, dict]] = []
    for group in by_ipp.values():
        ordered = sorted(
            group,
            key=lambda x: (x.get("acquisition_pass", 0), x["instance_number"], x["sop"]),
        )
        if len(ordered) < 2:
            continue
        pairs.append((ordered[0], ordered[-1]))
    return pairs
