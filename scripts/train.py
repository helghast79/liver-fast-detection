#!/usr/bin/env python3
"""Train TinyUNet on downloaded liver CT/mask pairs and export ONNX."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import LiverSegDataset, load_manifest, patient_split
from src.model import TinyUNet, bce_dice_loss, soft_dice, soft_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--export-onnx", action="store_true", default=True)
    parser.add_argument("--no-export-onnx", action="store_false", dest="export_onnx")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: TinyUNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    if len(loader) == 0:
        return {"loss": float("nan"), "dice": float("nan"), "iou": float("nan")}
    losses, dices, ious = [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        losses.append(float(bce_dice_loss(logits, masks).item()))
        dices.append(soft_dice(logits, masks))
        ious.append(soft_iou(logits, masks))
    return {
        "loss": float(np.mean(losses)),
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
    }


@torch.no_grad()
def save_samples(
    model: TinyUNet,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    max_samples: int = 8,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        preds = (torch.sigmoid(logits) > 0.5).float()
        for i in range(images.size(0)):
            if saved >= max_samples:
                return
            stem = str(batch["sop_instance_uid"][i]).replace(".", "_")
            img = (images[i, 0].cpu().numpy() * 255).astype(np.uint8)
            gt = (masks[i, 0].cpu().numpy() * 255).astype(np.uint8)
            pr = (preds[i, 0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img, mode="L").save(out_dir / f"{stem}_image.png")
            Image.fromarray(gt, mode="L").save(out_dir / f"{stem}_mask.png")
            Image.fromarray(pr, mode="L").save(out_dir / f"{stem}_pred.png")
            saved += 1


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    set_seed(args.seed)

    image_size = args.image_size or int(os.getenv("IMAGE_SIZE", "128"))
    args.output.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(args.data)
    if not rows:
        print(f"No rows in manifest under {args.data}")
        return 1

    train_rows, val_rows = patient_split(rows, val_ratio=args.val_ratio, seed=args.seed)
    print(
        f"Slices: {len(rows)} total | train={len(train_rows)} val={len(val_rows)} | "
        f"patients train={len({r['patient_id'] for r in train_rows})} "
        f"val={len({r['patient_id'] for r in val_rows})}"
    )

    train_ds = LiverSegDataset(args.data, train_rows, augment=True, seed=args.seed)
    val_ds = LiverSegDataset(args.data, val_rows, augment=False, seed=args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet(in_channels=1, out_channels=1, base=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict] = []
    best_dice = -1.0
    best_path = ckpt_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = bce_dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            pbar.set_postfix(loss=float(np.mean(epoch_losses[-20:])))

        train_metrics = {
            "loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
        }
        val_metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(
            f"epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f}"
        )

        score = val_metrics["dice"]
        if np.isnan(score):
            score = -train_metrics["loss"]
        if score > best_dice:
            best_dice = score
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val": val_metrics,
                    "image_size": image_size,
                    "base_channels": args.base_channels,
                },
                best_path,
            )
            print(f"  saved best checkpoint -> {best_path}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "image_size": image_size,
                "base_channels": args.base_channels,
            },
            ckpt_dir / "last.pt",
        )

    metrics = {
        "best_dice": best_dice,
        "best_checkpoint": str(best_path),
        "history": history,
        "image_size": image_size,
        "base_channels": args.base_channels,
        "num_train": len(train_rows),
        "num_val": len(val_rows),
    }
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {metrics_path}")

    # Reload best for samples + export
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])

    sample_loader = val_loader if len(val_ds) else train_loader
    save_samples(model, sample_loader, device, args.output / "samples")

    if args.export_onnx and best_path.exists():
        onnx_path = args.output / f"liver_unet_{image_size}.onnx"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "export_onnx.py"),
            "--checkpoint",
            str(best_path),
            "--output",
            str(onnx_path),
            "--image-size",
            str(image_size),
            "--base-channels",
            str(args.base_channels),
        ]
        print("Exporting ONNX...")
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPATH"] = str(ROOT)
        subprocess.check_call(cmd, env=env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
