"""Slice geometry helpers: IOP/IPP alignment and mask resampling onto CT."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class SliceGeometry:
    """DICOM axial-frame geometry (patient LPS)."""

    ipp: np.ndarray  # (3,) ImagePositionPatient
    iop: np.ndarray  # (6,) ImageOrientationPatient
    spacing: np.ndarray  # (2,) PixelSpacing [row_spacing, col_spacing]
    rows: int
    cols: int

    @property
    def row_dir(self) -> np.ndarray:
        """Direction cosines of the image rows (increasing column index)."""
        return self.iop[:3]

    @property
    def col_dir(self) -> np.ndarray:
        """Direction cosines of the image columns (increasing row index)."""
        return self.iop[3:]

    @property
    def normal(self) -> np.ndarray:
        n = np.cross(self.row_dir, self.col_dir)
        return n / np.linalg.norm(n)

    def pixel_to_patient(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Map pixel centers (row, col) -> patient coords (N, 3)."""
        rows = np.asarray(rows, dtype=np.float64)
        cols = np.asarray(cols, dtype=np.float64)
        return (
            self.ipp[None, :]
            + cols[:, None] * self.spacing[1] * self.row_dir[None, :]
            + rows[:, None] * self.spacing[0] * self.col_dir[None, :]
        )

    def patient_to_pixel(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map patient coords (N, 3) -> (rows, cols) in this frame."""
        d = np.asarray(points, dtype=np.float64) - self.ipp[None, :]
        cols = (d @ self.row_dir) / self.spacing[1]
        rows = (d @ self.col_dir) / self.spacing[0]
        return rows, cols


def parse_float_vector(value) -> np.ndarray:
    if isinstance(value, str):
        parts = value.split("\\")
        return np.array([float(x) for x in parts], dtype=np.float64)
    return np.array([float(x) for x in value], dtype=np.float64)


def geometry_from_tags(tags: dict) -> SliceGeometry:
    return SliceGeometry(
        ipp=parse_float_vector(tags["ImagePositionPatient"]),
        iop=parse_float_vector(tags["ImageOrientationPatient"]),
        spacing=parse_float_vector(tags["PixelSpacing"]),
        rows=int(tags["Rows"]),
        cols=int(tags["Columns"]),
    )


def iop_is_180_flip(a: np.ndarray, b: np.ndarray, atol: float = 1e-3) -> bool:
    """True if both in-plane axes are negated (180° rotation about slice normal)."""
    return bool(np.allclose(a, -b, atol=atol)) and not bool(np.allclose(a, b, atol=atol))


def inplane_offset_mm(src: SliceGeometry, dst: SliceGeometry) -> tuple[float, float, float]:
    """
    Residual origin offset of src relative to dst, expressed in dst row/col/normal mm.
    For a 180° IOP flip, first map src IPP to the equivalent dst-corner origin.
    """
    src_ipp = src.ipp.copy()
    if iop_is_180_flip(src.iop, dst.iop):
        # SEG(0,0) corresponds to CT(rows-1, cols-1) under a pure 180 in-plane flip.
        src_ipp = (
            src.ipp
            - (dst.cols - 1) * dst.spacing[1] * dst.row_dir
            - (dst.rows - 1) * dst.spacing[0] * dst.col_dir
        )
    delta = src_ipp - dst.ipp
    return (
        float(delta @ dst.row_dir),
        float(delta @ dst.col_dir),
        float(delta @ dst.normal),
    )


def resample_mask_to_geometry(
    mask: np.ndarray,
    src: SliceGeometry,
    dst: SliceGeometry,
) -> np.ndarray:
    """
    Nearest-neighbor resample of a binary mask from src frame onto dst frame
    using patient coordinates (handles IOP flips and IPP in-plane offsets).
    """
    mask = np.asarray(mask)
    if mask.shape != (src.rows, src.cols):
        # Allow masks already resized only if geometries match shape; otherwise require native.
        if mask.shape[0] == 0:
            return np.zeros((dst.rows, dst.cols), dtype=np.uint8)

    # Affine mapping: dst pixel (r,c) -> src continuous (r,c) for ndimage.
    # patient = dst.ipp + c*dst.sp_col*dst.row_dir + r*dst.sp_row*dst.col_dir
    # r_s = dot(patient-src.ipp, src.col_dir)/src.sp_row
    # c_s = dot(patient-src.ipp, src.row_dir)/src.sp_col
    d0 = dst.ipp - src.ipp
    # r_s = A00*r + A01*c + t0
    # c_s = A10*r + A11*c + t1
    a00 = float((dst.spacing[0] * dst.col_dir) @ src.col_dir / src.spacing[0])
    a01 = float((dst.spacing[1] * dst.row_dir) @ src.col_dir / src.spacing[0])
    t0 = float(d0 @ src.col_dir / src.spacing[0])
    a10 = float((dst.spacing[0] * dst.col_dir) @ src.row_dir / src.spacing[1])
    a11 = float((dst.spacing[1] * dst.row_dir) @ src.row_dir / src.spacing[1])
    t1 = float(d0 @ src.row_dir / src.spacing[1])

    matrix = np.array([[a00, a01], [a10, a11]], dtype=np.float64)
    offset = np.array([t0, t1], dtype=np.float64)

    # Identity shortcut.
    if (
        np.allclose(matrix, np.eye(2), atol=1e-6)
        and np.allclose(offset, 0, atol=1e-3)
        and mask.shape == (dst.rows, dst.cols)
    ):
        return (mask > 0).astype(np.uint8)

    warped = ndimage.affine_transform(
        (mask > 0).astype(np.float32),
        matrix=matrix,
        offset=offset,
        output_shape=(dst.rows, dst.cols),
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return (warped > 0.5).astype(np.uint8)


def ipp_key(ipp: np.ndarray, decimals: int = 3) -> tuple[float, float, float]:
    """Quantized IPP key for dual-acquisition grouping."""
    rounded = np.round(np.asarray(ipp, dtype=np.float64), decimals=decimals)
    return (float(rounded[0]), float(rounded[1]), float(rounded[2]))


def group_instances_by_ipp(instances: list[dict], decimals: int = 3) -> dict[tuple, list[dict]]:
    """
    Group CT instance dicts (must contain 'ipp' ndarray) by quantized IPP.
    Dual-acquisition series typically have multiplicity 2 per key.
    """
    groups: dict[tuple, list[dict]] = {}
    for inst in instances:
        key = ipp_key(inst["ipp"], decimals=decimals)
        groups.setdefault(key, []).append(inst)
    return groups


def describe_alignment(src: SliceGeometry, dst: SliceGeometry) -> dict:
    flip = iop_is_180_flip(src.iop, dst.iop)
    d_row, d_col, d_n = inplane_offset_mm(src, dst)
    return {
        "iop_180_flip": flip,
        "iop_same": bool(np.allclose(src.iop, dst.iop, atol=1e-3)),
        "offset_row_mm": d_row,
        "offset_col_mm": d_col,
        "offset_normal_mm": d_n,
        "offset_inplane_mm": float(np.hypot(d_row, d_col)),
    }
