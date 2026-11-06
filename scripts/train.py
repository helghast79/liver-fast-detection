#!/usr/bin/env python3
"""Train TinyUNet on liver CT/mask pairs (single GPU, patient-wise val split)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
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
from src.train_monitor import MetricsStore, start_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--holdout", type=Path, default=ROOT / "data" / "holdout")
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
        default=15,
        help="Early-stop after this many epochs without val_loss improvement",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=5e-5,
        help="Minimum val_loss decrease to count as improvement (smaller => longer runs)",
    )
    parser.add_argument("--export-onnx", action="store_true", default=True)
    parser.add_argument("--no-export-onnx", action="store_false", dest="export_onnx")
    parser.add_argument("--dashboard-host", type=str, default="0.0.0.0")
    parser.add_argument("--dashboard-port", type=int, default=8767)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Short label for experiment_log.md (default: timestamped baseline)",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Free-text description of this run / improvement being tested",
    )
    parser.add_argument("--experiment-log", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: TinyUNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    if len(loader) == 0:
        return {
            "loss": float("nan"),
            "dice": float("nan"),
            "iou": float("nan"),
            "hard_dice": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
        }
    losses, dices, ious = [], [], []
    tp = fp = fn = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        losses.append(float(bce_dice_loss(logits, masks).item()))
        dices.append(soft_dice(logits, masks))
        ious.append(soft_iou(logits, masks))
        preds = (torch.sigmoid(logits) > 0.5).float()
        tp += int((preds * masks).sum().item())
        fp += int((preds * (1 - masks)).sum().item())
        fn += int(((1 - preds) * masks).sum().item())
    hard_dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    return {
        "loss": float(np.mean(losses)),
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "hard_dice": float(hard_dice),
        "precision": float(precision),
        "recall": float(recall),
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


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}" if u != "B" else f"{int(x)} {u}"
        x /= 1024
    return f"{n} B"


def benchmark_onnx_holdout(
    onnx_path: Path,
    holdout_root: Path,
    rows: list[dict],
    image_size: int,
) -> dict:
    """Average wall-clock time to infer one patient (all slices) with ONNX Runtime CPU."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_patient[row["patient_id"]].append(row)

    case_times_s: dict[str, float] = {}
    slice_times_ms: list[float] = []
    # warmup
    dummy = np.zeros((1, 1, image_size, image_size), dtype=np.float32)
    for _ in range(3):
        session.run(None, {"input": dummy})

    for patient_id, patient_rows in sorted(by_patient.items()):
        t_case0 = time.perf_counter()
        for row in patient_rows:
            image = np.asarray(Image.open(holdout_root / row["image_path"]).convert("L"), dtype=np.float32)
            if image.shape != (image_size, image_size):
                image = np.asarray(
                    Image.fromarray(image.astype(np.uint8)).resize((image_size, image_size), Image.BILINEAR),
                    dtype=np.float32,
                )
            x = (image / 255.0)[None, None, ...].astype(np.float32)
            t0 = time.perf_counter()
            session.run(None, {"input": x})
            slice_times_ms.append((time.perf_counter() - t0) * 1000.0)
        case_times_s[patient_id] = time.perf_counter() - t_case0

    values = list(case_times_s.values())
    return {
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "onnx_size_human": human_bytes(onnx_path.stat().st_size),
        "n_cases": len(values),
        "n_slices": len(slice_times_ms),
        "avg_case_seconds": float(np.mean(values)) if values else float("nan"),
        "median_case_seconds": float(np.median(values)) if values else float("nan"),
        "avg_slice_ms": float(np.mean(slice_times_ms)) if slice_times_ms else float("nan"),
        "case_times_seconds": case_times_s,
    }


def append_experiment_log(
    log_path: Path,
    *,
    run_name: str,
    description: str,
    params: dict,
    val_metrics: dict,
    holdout_metrics: dict | None,
    inference: dict | None,
    notes: list[str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"\n## {run_name}",
        f"- **When:** {ts}",
        f"- **Description:** {description or '(none)'}",
        "",
        "### Main considerations",
    ]
    for note in notes:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "### Train parameters",
            "```json",
            json.dumps(params, indent=2),
            "```",
            "",
            "### Validation (best checkpoint)",
            "```json",
            json.dumps(val_metrics, indent=2),
            "```",
            "",
            "### Holdout",
            "```json",
            json.dumps(holdout_metrics if holdout_metrics is not None else {"error": "not evaluated"}, indent=2),
            "```",
            "",
            "### Inference artifact",
            "```json",
            json.dumps(inference if inference is not None else {"error": "no onnx"}, indent=2),
            "```",
            "",
            "---",
            "",
        ]
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


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
    experiment_log = args.experiment_log or (args.output / "experiment_log.md")
    started = datetime.now(timezone.utc)
    run_name = args.run_name or f"baseline-retrain-{started.strftime('%Y%m%d-%H%M%S')}"
    description = args.description or (
        "Retrain after carving a fixed 10-patient holdout from data/raw. "
        "Same model/data pipeline as first run (rendered 128 PNG, TinyUNet base=8, BCE+Dice), "
        "with softer early-stopping (patience=15, min_delta=5e-5) and live metrics dashboard."
    )

    store = MetricsStore()
    dashboard = None
    if not args.no_dashboard:
        dashboard = start_dashboard(args.dashboard_host, args.dashboard_port, store)
        print(
            f"Training dashboard: http://{args.dashboard_host}:{args.dashboard_port}/ "
            f"(bind {args.dashboard_host})"
        )

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

    holdout_rows: list[dict] = []
    holdout_patients: list[str] = []
    if args.holdout.exists() and (args.holdout / "manifest.jsonl").exists():
        holdout_rows = filter_existing_pairs(args.holdout, load_manifest(args.holdout))
        holdout_patients = sorted({r["patient_id"] for r in holdout_rows})

    print(
        f"Slices: {len(rows)} usable | train={len(train_rows)} val={len(val_rows)}\n"
        f"Patients: train={len({r['patient_id'] for r in train_rows})} "
        f"val={len(val_patients)} -> {val_patients}\n"
        f"Holdout: {len(holdout_rows)} slices / {len(holdout_patients)} patients -> {holdout_patients}\n"
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
        device_name = "cpu"
    else:
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
        print(f"Using device {device} ({device_name})")

    model = TinyUNet(in_channels=1, out_channels=1, base=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict] = []
    best_val_loss = float("inf")
    best_dice = -1.0
    best_val_metrics: dict[str, float] = {}
    best_epoch = 0
    epochs_without_improve = 0
    best_path = ckpt_dir / "best.pt"

    store.update(
        status="running",
        run_name=run_name,
        max_epochs=args.epochs,
        patience=args.patience,
        device=device_name,
        history=[],
        epoch=0,
        best_epoch=0,
        best_val_loss=None,
        best_dice=None,
        epochs_without_improve=0,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )

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
            f"val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f} "
            f"hard_dice={val_metrics['hard_dice']:.4f} "
            f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f}"
        )

        improved = False
        if not np.isnan(val_metrics["loss"]) and (
            best_val_loss - val_metrics["loss"] > args.min_delta
        ):
            improved = True
            best_val_loss = val_metrics["loss"]
            best_dice = val_metrics["dice"]
            best_val_metrics = dict(val_metrics)
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

        store.update(
            status="running",
            epoch=epoch,
            latest=record,
            history=history,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss if best_epoch else None,
            best_dice=best_dice if best_epoch else None,
            epochs_without_improve=epochs_without_improve,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

        if epochs_without_improve >= args.patience:
            print(
                f"Early stopping at epoch {epoch}: "
                f"no val_loss improvement for {args.patience} epochs "
                f"(best epoch {best_epoch}, val_loss={best_val_loss:.4f})"
            )
            break

    metrics = {
        "run_name": run_name,
        "description": description,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_dice": best_dice,
        "best_val_metrics": best_val_metrics,
        "best_checkpoint": str(best_path),
        "early_stopped": epochs_without_improve >= args.patience,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "val_patients": val_patients,
        "holdout_patients": holdout_patients,
        "history": history,
        "image_size": image_size,
        "base_channels": args.base_channels,
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "num_holdout": len(holdout_rows),
        "gpu": args.gpu,
        "device_name": device_name,
    }
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {metrics_path}")

    if best_path.exists():
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])

    sample_loader = val_loader if len(val_ds) else train_loader
    save_samples(model, sample_loader, device, args.output / "samples")

    holdout_metrics = None
    if holdout_rows:
        holdout_ds = LiverSegDataset(args.holdout, holdout_rows, augment=False, seed=args.seed)
        holdout_loader = DataLoader(
            holdout_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print("Evaluating holdout…")
        holdout_metrics = evaluate(model, holdout_loader, device)
        holdout_metrics["n_slices"] = len(holdout_rows)
        holdout_metrics["n_patients"] = len(holdout_patients)
        holdout_metrics["patients"] = holdout_patients
        print(
            f"holdout: loss={holdout_metrics['loss']:.4f} "
            f"dice={holdout_metrics['dice']:.4f} iou={holdout_metrics['iou']:.4f} "
            f"hard_dice={holdout_metrics['hard_dice']:.4f} "
            f"P={holdout_metrics['precision']:.4f} R={holdout_metrics['recall']:.4f}"
        )
        metrics["holdout"] = holdout_metrics
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    inference_info = None
    onnx_path = args.output / f"liver_unet_{image_size}.onnx"
    quant_path = args.output / f"liver_unet_{image_size}.quant.onnx"
    if args.export_onnx and best_path.exists():
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
        print("Exporting ONNX…")
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPATH"] = str(ROOT)
        subprocess.check_call(cmd, env=env)

        # Prefer quantized model for browser inference timing/size reporting.
        infer_onnx = quant_path if quant_path.exists() else onnx_path
        if holdout_rows and infer_onnx.exists():
            print(f"Benchmarking ONNX inference on holdout ({infer_onnx.name})…")
            inference_info = benchmark_onnx_holdout(
                infer_onnx, args.holdout, holdout_rows, image_size
            )
            inference_info["float_onnx_path"] = str(onnx_path)
            inference_info["float_onnx_size_bytes"] = (
                onnx_path.stat().st_size if onnx_path.exists() else None
            )
            inference_info["float_onnx_size_human"] = (
                human_bytes(onnx_path.stat().st_size) if onnx_path.exists() else None
            )
            print(
                f"ONNX {infer_onnx.name}: {inference_info['onnx_size_human']}, "
                f"avg case={inference_info['avg_case_seconds']:.3f}s "
                f"({inference_info['n_cases']} cases), "
                f"avg slice={inference_info['avg_slice_ms']:.2f}ms"
            )
            metrics["inference"] = inference_info
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    params = {
        "run_name": run_name,
        "data": str(args.data),
        "holdout": str(args.holdout),
        "epochs_max": args.epochs,
        "epochs_ran": history[-1]["epoch"] if history else 0,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "val_patients_n": args.val_patients,
        "val_patients": val_patients,
        "base_channels": args.base_channels,
        "image_size": image_size,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "loss": "BCE + soft Dice",
        "optimizer": "Adam",
        "augmentations": "flip/rot90/brightness-contrast",
        "n_train_slices": len(train_rows),
        "n_val_slices": len(val_rows),
        "n_holdout_slices": len(holdout_rows),
        "gpu": args.gpu,
        "device_name": device_name,
    }
    notes = [
        f"Training inputs at {image_size}×{image_size} (see run description for intensity pipeline).",
        "SEG labels merge all segments except Abdominal aorta.",
        f"2D TinyUNet base={args.base_channels} [c,2c,4c,8c]; BCE+Dice; patient-wise val split.",
        "Fixed holdout under data/holdout excluded from train/val.",
        f"Early stop on val_loss with patience={args.patience}, min_delta={args.min_delta}.",
        "Inference artifact for browser: INT8 dynamically quantized ONNX (CPU ORT timing).",
    ]
    append_experiment_log(
        experiment_log,
        run_name=run_name,
        description=description,
        params=params,
        val_metrics={
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_dice": best_dice,
            **best_val_metrics,
        },
        holdout_metrics=holdout_metrics,
        inference=inference_info,
        notes=notes,
    )
    print(f"Appended experiment log -> {experiment_log}")

    store.update(
        status="finished",
        epoch=history[-1]["epoch"] if history else 0,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss if best_epoch else None,
        best_dice=best_dice if best_epoch else None,
        holdout=holdout_metrics,
        inference=inference_info,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    if dashboard is not None:
        print(
            f"Dashboard still serving final metrics at "
            f"http://{args.dashboard_host}:{args.dashboard_port}/ (Ctrl+C to stop if attached)"
        )
        # Keep process alive briefly so the user can open the finished dashboard,
        # but do not block forever when launched non-interactively.
        time.sleep(2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
