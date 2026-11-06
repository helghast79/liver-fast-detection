"""CT intensity preprocessing: DICOM → HU → fixed abdomen window → uint8."""

from __future__ import annotations

import io
from dataclasses import dataclass

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
    """Map HU to float32 in [0, 1] with a fixed window."""
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
) -> tuple[bytes, dict]:
    """
    Decode DICOM bytes → abdomen-windowed grayscale PNG bytes.

    Returns (png_bytes, meta) where meta includes window and HU stats.
    """
    window = window or WindowSpec()
    ds = pydicom.dcmread(io.BytesIO(dicom_bytes), force=True)
    hu = dicom_to_hu(ds)
    img = hu_to_uint8_image(hu, size=size, window=window)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    meta = {
        "window_center": window.center,
        "window_width": window.width,
        "hu_min": float(np.min(hu)),
        "hu_max": float(np.max(hu)),
        "hu_mean": float(np.mean(hu)),
        "rows": int(getattr(ds, "Rows", hu.shape[0])),
        "columns": int(getattr(ds, "Columns", hu.shape[1])),
        "rescale_slope": float(getattr(ds, "RescaleSlope", 1.0) or 1.0),
        "rescale_intercept": float(getattr(ds, "RescaleIntercept", 0.0) or 0.0),
    }
    return buf.getvalue(), meta
