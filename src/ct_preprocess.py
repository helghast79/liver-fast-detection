"""CT intensity preprocessing: DICOM → HU → window → uint8.

Supports:
  - fixed abdomen window (C40/W400)
  - Orthanc-/rendered mimic: use each instance's DICOM WindowCenter/WindowWidth
    (VOI LUT), which is what Orthanc REST /instances/{id}/rendered applies by
    default when no explicit window query param is passed.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from PIL import Image


# Common soft-tissue / abdomen window (HU).
DEFAULT_WINDOW_CENTER = 40.0
DEFAULT_WINDOW_WIDTH = 400.0


@dataclass(frozen=True)
class WindowSpec:
    center: float = DEFAULT_WINDOW_CENTER
    width: float = DEFAULT_WINDOW_WIDTH

    @property
    def lo(self) -> float:
        return self.center - self.width / 2.0

    @property
    def hi(self) -> float:
        return self.center + self.width / 2.0


def _first_float(value: Any) -> float | None:
    """DICOM WC/WW may be a scalar or MultiValue list."""
    if value is None:
        return None
    if isinstance(value, pydicom.multival.MultiValue):
        if len(value) == 0:
            return None
        return float(value[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def voi_window_from_dicom(
    ds: Dataset,
    fallback: WindowSpec | None = None,
) -> tuple[WindowSpec, str]:
    """
    Read WindowCenter/WindowWidth from DICOM (Orthanc /rendered default VOI).

    Returns (window, source) where source is 'dicom' or 'fallback'.
    """
    fallback = fallback or WindowSpec()
    center = _first_float(getattr(ds, "WindowCenter", None))
    width = _first_float(getattr(ds, "WindowWidth", None))
    if center is None or width is None or width <= 0:
        return fallback, "fallback"
    return WindowSpec(center=center, width=width), "dicom"


def dicom_to_hu(ds: Dataset) -> np.ndarray:
    """Convert DICOM pixel data to Hounsfield units (float32)."""
    pixels = np.asarray(ds.pixel_array)
    if pixels.ndim == 3:
        # Multi-frame: take first frame (axial CT instances are usually single-frame).
        pixels = pixels[0]
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    hu = pixels.astype(np.float32) * slope + intercept

    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper()
    if photometric == "MONOCHROME1":
        # Higher values are darker; invert so brighter = higher HU-like intensity.
        hu = -hu
    return hu


def apply_window(hu: np.ndarray, window: WindowSpec) -> np.ndarray:
    """Map HU to float32 in [0, 1] with a fixed window (Orthanc-style linear clip)."""
    clipped = np.clip(hu, window.lo, window.hi)
    return ((clipped - window.lo) / max(window.width, 1e-6)).astype(np.float32)


def hu_to_uint8_image(
    hu: np.ndarray,
    size: int,
    window: WindowSpec | None = None,
) -> Image.Image:
    """Window HU, scale to uint8 grayscale, resize with bilinear."""
    window = window or WindowSpec()
    norm = apply_window(hu, window)
    gray = (norm * 255.0).round().astype(np.uint8)
    img = Image.fromarray(gray, mode="L")
    if img.width != size or img.height != size:
        img = img.resize((size, size), resample=Image.Resampling.BILINEAR)
    return img


def dicom_bytes_to_windowed_png(
    dicom_bytes: bytes,
    size: int,
    window: WindowSpec | None = None,
    window_mode: str = "fixed",
) -> tuple[bytes, dict]:
    """
    Decode DICOM bytes → windowed grayscale PNG bytes.

    window_mode:
      - "fixed": use ``window`` (default C40/W400)
      - "dicom": use instance WindowCenter/WindowWidth (Orthanc /rendered mimic);
        falls back to ``window`` if tags missing
    """
    fallback = window or WindowSpec()
    ds = pydicom.dcmread(io.BytesIO(dicom_bytes), force=True)
    hu = dicom_to_hu(ds)
    if window_mode == "dicom":
        used, win_src = voi_window_from_dicom(ds, fallback=fallback)
    elif window_mode == "fixed":
        used, win_src = fallback, "fixed"
    else:
        raise ValueError(f"Unknown window_mode={window_mode!r}")
    img = hu_to_uint8_image(hu, size=size, window=used)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    meta = {
        "window_center": used.center,
        "window_width": used.width,
        "window_mode": window_mode,
        "window_source": win_src,
        "hu_min": float(np.min(hu)),
        "hu_max": float(np.max(hu)),
        "hu_mean": float(np.mean(hu)),
        "rows": int(getattr(ds, "Rows", hu.shape[0])),
        "columns": int(getattr(ds, "Columns", hu.shape[1])),
        "rescale_slope": float(getattr(ds, "RescaleSlope", 1.0) or 1.0),
        "rescale_intercept": float(getattr(ds, "RescaleIntercept", 0.0) or 0.0),
    }
    return buf.getvalue(), meta
