"""PyTorch dataset over downloaded CT/mask PNG pairs."""

from __future__ import annotations

import json
import random
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
) -> tuple[list[dict], list[dict]]:
    patients = sorted({r["patient_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_val = max(1, int(round(len(patients) * val_ratio))) if len(patients) > 1 else 0
    val_patients = set(patients[:n_val])
    train_rows = [r for r in rows if r["patient_id"] not in val_patients]
    val_rows = [r for r in rows if r["patient_id"] in val_patients]
    if not train_rows:
        # Tiny datasets: keep everything in train, empty val handled by caller.
        train_rows, val_rows = rows, []
    return train_rows, val_rows


class LiverSegDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        rows: list[dict],
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.data_root = Path(data_root)
        self.rows = rows
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_gray(self, rel_path: str) -> np.ndarray:
        path = self.data_root / rel_path
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        return arr

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[idx]
        image = self._load_gray(row["image_path"]) / 255.0
        mask = (self._load_gray(row["mask_path"]) > 127).astype(np.float32)

        if self.augment:
            if self.rng.random() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            if self.rng.random() < 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()
            if self.rng.random() < 0.3:
                k = self.rng.randint(1, 3)
                image = np.rot90(image, k).copy()
                mask = np.rot90(mask, k).copy()
            if self.rng.random() < 0.5:
                scale = self.rng.uniform(0.85, 1.15)
                shift = self.rng.uniform(-0.05, 0.05)
                image = np.clip(image * scale + shift, 0.0, 1.0)

        image_t = torch.from_numpy(image).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return {
            "image": image_t,
            "mask": mask_t,
            "patient_id": row["patient_id"],
            "sop_instance_uid": row["sop_instance_uid"],
        }
