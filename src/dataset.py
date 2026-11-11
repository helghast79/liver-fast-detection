"""PyTorch dataset over downloaded CT/mask PNG pairs."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset


def load_manifest(data_root: Path) -> list[dict]:
    manifest_path = data_root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def patient_split(
    rows: list[dict],
    val_ratio: float = 0.2,
    seed: int = 42,
    n_val_patients: int | None = None,
) -> tuple[list[dict], list[dict], list[str]]:
    """
    Split rows by patient ID.

    If n_val_patients is set, hold out exactly that many random patients for
    validation; otherwise use val_ratio.
    Returns (train_rows, val_rows, val_patient_ids).
    """
    patients = sorted({r["patient_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(patients)

    if n_val_patients is not None:
        n_val = min(max(0, n_val_patients), max(0, len(patients) - 1))
    else:
        n_val = max(1, int(round(len(patients) * val_ratio))) if len(patients) > 1 else 0

    val_patient_ids = sorted(patients[:n_val])
    val_patients = set(val_patient_ids)
    train_rows = [r for r in rows if r["patient_id"] not in val_patients]
    val_rows = [r for r in rows if r["patient_id"] in val_patients]
    if not train_rows:
        train_rows, val_rows, val_patient_ids = rows, [], []
    return train_rows, val_rows, val_patient_ids


def patient_kfold_splits(
    rows: list[dict],
    n_val_patients: int = 20,
    seed: int = 42,
) -> list[tuple[list[dict], list[dict], list[str]]]:
    """
    Patient-wise cross-validation folds with ~n_val_patients per fold.

    Builds as many non-overlapping folds of exactly n_val_patients as fit.
    Leftover patients remain in the training set for every fold.
    Returns list of (train_rows, val_rows, val_patient_ids) per fold.
    """
    patients = sorted({r["patient_id"] for r in rows})
    if len(patients) < 2:
        return [(rows, [], [])]

    n_val = min(max(1, n_val_patients), len(patients) - 1)
    n_folds = max(1, len(patients) // n_val)
    rng = random.Random(seed)
    shuffled = patients[:]
    rng.shuffle(shuffled)

    folds: list[tuple[list[dict], list[dict], list[str]]] = []
    for fold_i in range(n_folds):
        start = fold_i * n_val
        val_ids = sorted(shuffled[start : start + n_val])
        val_set = set(val_ids)
        train_rows = [r for r in rows if r["patient_id"] not in val_set]
        val_rows = [r for r in rows if r["patient_id"] in val_set]
        if not train_rows or not val_rows:
            continue
        folds.append((train_rows, val_rows, val_ids))
    return folds


def filter_existing_pairs(data_root: Path, rows: list[dict]) -> list[dict]:
    """Drop manifest rows whose image/mask files are missing on disk."""
    kept: list[dict] = []
    for row in rows:
        image_path = data_root / row["image_path"]
        mask_path = data_root / row["mask_path"]
        if not (image_path.is_file() and mask_path.is_file()):
            continue
        path_b = row.get("image_path_b")
        if path_b and not (data_root / path_b).is_file():
            continue
        liver_path = row.get("liver_mask_path")
        if liver_path and not (data_root / liver_path).is_file():
            continue
        kept.append(row)
    return kept


def filter_nonempty_masks(rows: list[dict], data_root: Path | None = None) -> list[dict]:
    """Keep only slices with a non-empty SEG mask.

    Prefers ``mask_foreground_pixels`` from the manifest; if missing/zero-looking,
    optionally inspects the PNG on disk when ``data_root`` is provided.
    """
    kept: list[dict] = []
    for row in rows:
        fg = row.get("mask_foreground_pixels")
        if fg is not None:
            if int(fg) > 0:
                kept.append(row)
            continue
        if data_root is None:
            kept.append(row)
            continue
        mask_path = data_root / row["mask_path"]
        if not mask_path.is_file():
            continue
        mask = np.asarray(Image.open(mask_path).convert("L"))
        if mask.max() > 0:
            kept.append(row)
    return kept


def series_key(row: dict) -> tuple:
    """Group key for axial ordering within one CT acquisition stack."""
    refs = row.get("referenced_series_uids") or []
    ref0 = refs[0] if refs else ""
    return (
        row.get("patient_id", ""),
        row.get("study_instance_uid", ""),
        ref0,
        row.get("acquisition_pass", 0),
    )


def sort_key_instance(row: dict) -> tuple:
    return (
        int(row.get("instance_number") or 0),
        str(row.get("sop_instance_uid") or ""),
    )


def build_neighbor_index(rows: list[dict], context: int = 1) -> list[tuple[int, ...]]:
    """
    For each row index, return indices of [prev..., center, next...] with clamp.

    Neighbors are among kept slices in the same series, sorted by InstanceNumber.
    context=1 -> 3 channels (prev, center, next).
    """
    context = max(0, int(context))
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[series_key(row)].append(i)

    neighbor_of = [tuple([i] * (2 * context + 1)) for i in range(len(rows))]
    for idxs in groups.values():
        idxs_sorted = sorted(idxs, key=lambda j: sort_key_instance(rows[j]))
        n = len(idxs_sorted)
        for pos, center_i in enumerate(idxs_sorted):
            stack: list[int] = []
            for d in range(-context, context + 1):
                j = min(max(pos + d, 0), n - 1)
                stack.append(idxs_sorted[j])
            neighbor_of[center_i] = tuple(stack)
    return neighbor_of


def _approx_hu_from_windowed(gray01: np.ndarray, center: float = 40.0, width: float = 400.0) -> np.ndarray:
    """Invert fixed abdomen window mapping (approx.) to HU for re-window jitter."""
    lo = center - width / 2.0
    return gray01.astype(np.float32) * width + lo


def _apply_hu_window(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    lo = center - width / 2.0
    hi = center + width / 2.0
    return np.clip((hu - lo) / max(width, 1e-6), 0.0, 1.0).astype(np.float32)


def spatial_channels(context_slices: int = 0) -> int:
    return 2 * max(0, int(context_slices)) + 1


def total_in_channels(
    context_slices: int = 0,
    dual_window: bool = False,
    dual_series: bool = False,
    liver_prior: bool = False,
) -> int:
    if dual_series or liver_prior:
        # Dual CT series OR CT + liver-prior bool occupy 2 spatial channels.
        n = 2
    else:
        n = spatial_channels(context_slices)
    return n * 2 if dual_window else n


def rows_have_dual_series(rows: list[dict]) -> bool:
    return any(bool(r.get("image_path_b")) and not r.get("liver_prior") for r in rows)


def rows_have_liver_prior(rows: list[dict]) -> bool:
    return any(bool(r.get("liver_mask_path")) or bool(r.get("liver_prior")) for r in rows)


class LiverSegDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        rows: list[dict],
        augment: bool = False,
        seed: int = 0,
        context_slices: int = 0,
        strong_aug: bool = False,
        base_window_center: float = 40.0,
        base_window_width: float = 400.0,
        dual_window: bool = False,
        bone_window_center: float = 400.0,
        bone_window_width: float = 1500.0,
        dual_series: bool | None = None,
        liver_prior: bool | None = None,
    ) -> None:
        """
        context_slices=0 -> 1-channel center only.
        context_slices=1 -> 3-channel stack [prev, center, next] with edge clamp.
        dual_window appends a second HU window (bone/soft) per spatial slice:
          [abd_prev..abd_next, bone_prev..bone_next] (or 2ch if context=0).
        dual_series stacks image_path + image_path_b as 2 spatial channels
        (requires context_slices=0); secondary is already resampled onto primary.
        liver_prior stacks [CT, liver_bool] from image_path + liver_mask_path
        (requires context_slices=0). Prefer this over dual_series when both set.
        Bone channels are derived by approx-inverting the stored abdomen PNG
        (assumed C40/W400) then re-windowing — no Orthanc /rendered.
        strong_aug adds CT-aware augs: small affine/elastic, HU window jitter,
        gaussian noise, mild blur (on top of flip/rot/brightness).
        """
        self.data_root = Path(data_root)
        self.rows = rows
        self.augment = augment
        self.strong_aug = bool(strong_aug)
        self.dual_window = bool(dual_window)
        self.liver_prior = (
            rows_have_liver_prior(rows) if liver_prior is None else bool(liver_prior)
        )
        self.dual_series = (
            False
            if self.liver_prior
            else (rows_have_dual_series(rows) if dual_series is None else bool(dual_series))
        )
        if (self.dual_series or self.liver_prior) and int(context_slices) > 0:
            raise ValueError("dual_series/liver_prior currently require context_slices=0")
        self.rng = random.Random(seed)
        self.context_slices = (
            0 if (self.dual_series or self.liver_prior) else max(0, int(context_slices))
        )
        self.n_spatial = (
            2 if (self.dual_series or self.liver_prior) else spatial_channels(self.context_slices)
        )
        self.in_channels = total_in_channels(
            self.context_slices,
            self.dual_window,
            dual_series=self.dual_series,
            liver_prior=self.liver_prior,
        )
        self.neighbor_index = build_neighbor_index(rows, context=self.context_slices)
        self.base_window_center = float(base_window_center)
        self.base_window_width = float(base_window_width)
        self.bone_window_center = float(bone_window_center)
        self.bone_window_width = float(bone_window_width)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_gray(self, rel_path: str) -> np.ndarray:
        path = self.data_root / rel_path
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        return arr

    def _affine_elastic(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Small affine + optional mild elastic warp; shared across channels."""
        _, h, w = image.shape
        angle = self.rng.uniform(-12.0, 12.0)
        scale = self.rng.uniform(0.92, 1.08)
        tx = self.rng.uniform(-0.04, 0.04) * w
        ty = self.rng.uniform(-0.04, 0.04) * h
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad) * scale, np.sin(rad) * scale
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        # scipy affine_transform uses (row, col) = (y, x)
        a00, a01 = cos_a, sin_a
        a10, a11 = -sin_a, cos_a
        o_y = cy - a00 * (cy + ty) - a01 * (cx + tx)
        o_x = cx - a10 * (cy + ty) - a11 * (cx + tx)
        mat = np.array([[a00, a01], [a10, a11]], dtype=np.float64)
        off = np.array([o_y, o_x], dtype=np.float64)

        image_w = np.stack(
            [
                ndimage.affine_transform(
                    image[c], matrix=mat, offset=off, order=1, mode="nearest", prefilter=False
                )
                for c in range(image.shape[0])
            ],
            axis=0,
        ).astype(np.float32)
        mask_w = ndimage.affine_transform(
            mask, matrix=mat, offset=off, order=0, mode="nearest", prefilter=False
        ).astype(np.float32)

        if self.rng.random() < 0.5:
            rs = np.random.RandomState(self.rng.randint(0, 1_000_000))
            sigma = self.rng.uniform(4.0, 7.0)
            alpha = self.rng.uniform(2.0, 6.0)
            dy = ndimage.gaussian_filter(rs.uniform(-1, 1, size=(h, w)).astype(np.float32), sigma=sigma, mode="nearest") * alpha
            dx = ndimage.gaussian_filter(rs.uniform(-1, 1, size=(h, w)).astype(np.float32), sigma=sigma, mode="nearest") * alpha
            yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            coords = [np.clip(yy + dy, 0, h - 1), np.clip(xx + dx, 0, w - 1)]
            image_w = np.stack(
                [
                    ndimage.map_coordinates(image_w[c], coords, order=1, mode="nearest")
                    for c in range(image_w.shape[0])
                ],
                axis=0,
            ).astype(np.float32)
            mask_w = ndimage.map_coordinates(mask_w, coords, order=0, mode="nearest").astype(np.float32)

        return image_w, mask_w

    def _bone_from_abdomen(self, abd01: np.ndarray) -> np.ndarray:
        hu = _approx_hu_from_windowed(
            abd01, self.base_window_center, self.base_window_width
        )
        return _apply_hu_window(hu, self.bone_window_center, self.bone_window_width)

    def _stack_windows(self, abd_spatial: np.ndarray) -> np.ndarray:
        """(n_spatial,H,W) abdomen -> optional dual stack."""
        if not self.dual_window:
            return abd_spatial
        bone = np.stack(
            [self._bone_from_abdomen(abd_spatial[i]) for i in range(abd_spatial.shape[0])],
            axis=0,
        )
        return np.concatenate([abd_spatial, bone], axis=0).astype(np.float32)

    def _hu_window_jitter(self, image: np.ndarray) -> np.ndarray:
        """
        Jitter abdomen (+ bone if dual) from approx HU of abdomen spatial channels.
        Assumes channel layout [abd_spatial..., bone_spatial...].
        """
        n = self.n_spatial
        abd = image[:n]
        d_c = self.rng.uniform(-25.0, 25.0)
        d_w = self.rng.uniform(-80.0, 80.0)
        abd_c = self.base_window_center + d_c
        abd_w = max(150.0, self.base_window_width + d_w)
        out_abd = []
        hus = []
        for i in range(n):
            hu = _approx_hu_from_windowed(
                abd[i], self.base_window_center, self.base_window_width
            )
            hus.append(hu)
            out_abd.append(_apply_hu_window(hu, abd_c, abd_w))
        out_abd_a = np.stack(out_abd, axis=0).astype(np.float32)
        if not self.dual_window:
            return out_abd_a
        bone_c = self.bone_window_center + self.rng.uniform(-40.0, 40.0)
        bone_w = max(600.0, self.bone_window_width + self.rng.uniform(-200.0, 200.0))
        out_bone = np.stack(
            [_apply_hu_window(hu, bone_c, bone_w) for hu in hus], axis=0
        ).astype(np.float32)
        return np.concatenate([out_abd_a, out_bone], axis=0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[idx]
        if self.liver_prior:
            liver_rel = row.get("liver_mask_path")
            if not liver_rel:
                raise KeyError(
                    f"liver_prior row missing liver_mask_path: {row.get('sop_instance_uid')}"
                )
            ct = self._load_gray(row["image_path"]) / 255.0
            liver = (self._load_gray(liver_rel) > 127).astype(np.float32)
            abd = np.stack([ct, liver], axis=0).astype(np.float32)
        elif self.dual_series:
            path_b = row.get("image_path_b")
            if not path_b:
                raise KeyError(f"dual_series row missing image_path_b: {row.get('sop_instance_uid')}")
            abd = np.stack(
                [
                    self._load_gray(row["image_path"]) / 255.0,
                    self._load_gray(path_b) / 255.0,
                ],
                axis=0,
            ).astype(np.float32)
        else:
            neighbor_idxs = self.neighbor_index[idx]
            abd_channels = [
                self._load_gray(self.rows[j]["image_path"]) / 255.0 for j in neighbor_idxs
            ]
            # (n_spatial, H, W) abdomen, then optional bone append
            abd = np.stack(abd_channels, axis=0).astype(np.float32)
        image = self._stack_windows(abd)
        mask = (self._load_gray(row["mask_path"]) > 127).astype(np.float32)

        if self.augment:
            if self.rng.random() < 0.5:
                image = np.flip(image, axis=2).copy()
                mask = np.fliplr(mask).copy()
            if self.rng.random() < 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flipud(mask).copy()
            if self.rng.random() < 0.3:
                k = self.rng.randint(1, 3)
                image = np.rot90(image, k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k).copy()
            if self.rng.random() < 0.5:
                scale = self.rng.uniform(0.85, 1.15)
                shift = self.rng.uniform(-0.05, 0.05)
                image = np.clip(image * scale + shift, 0.0, 1.0)

            if self.strong_aug:
                if self.rng.random() < 0.55:
                    image, mask = self._affine_elastic(image, mask)
                if self.rng.random() < 0.5:
                    image = self._hu_window_jitter(image)
                if self.rng.random() < 0.4:
                    noise = np.random.RandomState(self.rng.randint(0, 1_000_000)).normal(
                        0.0, self.rng.uniform(0.01, 0.04), size=image.shape
                    ).astype(np.float32)
                    image = np.clip(image + noise, 0.0, 1.0)
                if self.rng.random() < 0.35:
                    sigma = self.rng.uniform(0.3, 0.9)
                    image = np.stack(
                        [
                            ndimage.gaussian_filter(image[c], sigma=sigma, mode="nearest")
                            for c in range(image.shape[0])
                        ],
                        axis=0,
                    ).astype(np.float32)

        image_t = torch.from_numpy(np.ascontiguousarray(image))
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
        return {
            "image": image_t,
            "mask": mask_t,
            "patient_id": row["patient_id"],
            "sop_instance_uid": row["sop_instance_uid"],
        }
