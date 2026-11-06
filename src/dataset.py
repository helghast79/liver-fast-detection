"""PyTorch dataset over downloaded CT/mask PNG pairs."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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
        if image_path.is_file() and mask_path.is_file():
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


class LiverSegDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        rows: list[dict],
        augment: bool = False,
        seed: int = 0,
        context_slices: int = 0,
    ) -> None:
        """
        context_slices=0 -> 1-channel center only.
        context_slices=1 -> 3-channel stack [prev, center, next] with edge clamp.
        """
        self.data_root = Path(data_root)
        self.rows = rows
        self.augment = augment
        self.rng = random.Random(seed)
        self.context_slices = max(0, int(context_slices))
        self.in_channels = 2 * self.context_slices + 1
        self.neighbor_index = build_neighbor_index(rows, context=self.context_slices)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_gray(self, rel_path: str) -> np.ndarray:
        path = self.data_root / rel_path
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        return arr

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[idx]
        neighbor_idxs = self.neighbor_index[idx]
        channels = [
            self._load_gray(self.rows[j]["image_path"]) / 255.0 for j in neighbor_idxs
        ]
        # (C, H, W)
        image = np.stack(channels, axis=0).astype(np.float32)
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

        image_t = torch.from_numpy(np.ascontiguousarray(image))
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return {
            "image": image_t,
            "mask": mask_t,
            "patient_id": row["patient_id"],
            "sop_instance_uid": row["sop_instance_uid"],
        }
