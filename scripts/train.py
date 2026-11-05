#!/usr/bin/env python3
"""Train TinyUNet on liver CT/mask pairs (single GPU, patient-wise val split)."""

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

from src.dataset import LiverSegDataset, filter_existing_pairs, load_manifest, patient_split
from src.model import TinyUNet, bce_dice_loss, soft_dice, soft_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--val-patients",
        type=int,
        default=10,
        help="Number of random patients held out for validation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Physical GPU index to use (only this one is visible)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stop after this many epochs without val_loss improvement",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum val_loss decrease to count as improvement",
    )
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
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
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
    max_samples: int = 16,
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
    args = parse_args()
    # Pin to a single GPU before CUDA initializes.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    load_dotenv(ROOT / ".env")
    set_seed(args.seed)

    image_size = args.image_size or int(os.getenv("IMAGE_SIZE", "128"))
    args.output.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rows = filter_existing_pairs(args.data, load_manifest(args.data))
    if not rows:
        print(f"No usable pairs under {args.data}")
        return 1

    train_rows, val_rows, val_patients = patient_split(
        rows,
        n_val_patients=args.val_patients,
        seed=args.seed,
    )
    split_path = args.output / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "val_patients": val_patients,
                "n_train_patients": len({r["patient_id"] for r in train_rows}),
                "n_val_patients": len(val_patients),
                "n_train_slices": len(train_rows),
                "n_val_slices": len(val_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Slices: {len(rows)} usable | train={len(train_rows)} val={len(val_rows)}\n"
        f"Patients: train={len({r['patient_id'] for r in train_rows})} "
        f"val={len(val_patients)} -> {val_patients}\n"
        f"GPU: physical {args.gpu} (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})"
    )

    train_ds = LiverSegDataset(args.data, train_rows, augment=True, seed=args.seed)
    val_ds = LiverSegDataset(args.data, val_rows, augment=False, seed=args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — training on CPU")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        print(f"Using device {device} ({torch.cuda.get_device_name(0)})")

    model = TinyUNet(in_channels=1, out_channels=1, base=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict] = []
    best_val_loss = float("inf")
    best_dice = -1.0
    best_epoch = 0
    epochs_without_improve = 0
    best_path = ckpt_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = bce_dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            pbar.set_postfix(loss=float(np.mean(epoch_losses[-20:])))

        train_metrics = {"loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan")}
        val_metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(
            f"epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f}"
        )

        improved = False
        if not np.isnan(val_metrics["loss"]) and (
            best_val_loss - val_metrics["loss"] > args.min_delta
        ):
            improved = True
            best_val_loss = val_metrics["loss"]
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val": val_metrics,
                    "image_size": image_size,
                    "base_channels": args.base_channels,
                    "val_patients": val_patients,
                },
                best_path,
            )
            print(f"  saved best checkpoint (val_loss={best_val_loss:.4f}) -> {best_path}")
        else:
            epochs_without_improve += 1
            print(
                f"  no val_loss improvement ({epochs_without_improve}/{args.patience})"
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "image_size": image_size,
                "base_channels": args.base_channels,
            },
            ckpt_dir / "last.pt",
        )

        if epochs_without_improve >= args.patience:
            print(
                f"Early stopping at epoch {epoch}: "
                f"no val_loss improvement for {args.patience} epochs "
                f"(best epoch {best_epoch}, val_loss={best_val_loss:.4f})"
            )
            break

    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_dice": best_dice,
        "best_checkpoint": str(best_path),
        "early_stopped": epochs_without_improve >= args.patience,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "val_patients": val_patients,
        "history": history,
        "image_size": image_size,
        "base_channels": args.base_channels,
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "gpu": args.gpu,
    }
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {metrics_path}")

    if best_path.exists():
        state = torch.load(best_path, map_location=device, weights_only=False)
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
