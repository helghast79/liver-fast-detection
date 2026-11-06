#!/usr/bin/env python3
"""Train TinyUNet on liver CT/mask pairs (single GPU, patient-wise val / CV)."""

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

from src.dataset import (
    LiverSegDataset,
    filter_existing_pairs,
    load_manifest,
    patient_kfold_splits,
    patient_split,
)
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
        help="Number of random patients held out for validation (per fold if --cv)",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Patient-wise cross-validation with --val-patients per fold",
    )
    parser.add_argument(
        "--compare-to",
        type=str,
        default="",
        help="Prior run_name in experiment_log.md to cite in the comparison section",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument(
        "--context-slices",
        type=int,
        default=0,
        help="Axial neighbors per side as extra channels (1 => prev/center/next = 3ch 2.5D)",
    )
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
            # Center channel for multi-channel (2.5D) inputs.
            center = images.shape[1] // 2
            img = (images[i, center].cpu().numpy() * 255).astype(np.uint8)
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
    context_slices: int = 0,
) -> dict:
    """Average wall-clock time to infer one patient (all slices) with ONNX Runtime CPU."""
    import onnxruntime as ort

    from src.dataset import LiverSegDataset

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_channels = 2 * max(0, context_slices) + 1
    ds = LiverSegDataset(
        holdout_root,
        rows,
        augment=False,
        seed=0,
        context_slices=context_slices,
    )

    by_patient: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_patient[row["patient_id"]].append(i)

    case_times_s: dict[str, float] = {}
    slice_times_ms: list[float] = []
    dummy = np.zeros((1, in_channels, image_size, image_size), dtype=np.float32)
    for _ in range(3):
        session.run(None, {"input": dummy})

    for patient_id, idxs in sorted(by_patient.items()):
        t_case0 = time.perf_counter()
        for i in idxs:
            image = ds[i]["image"].numpy()
            if image.shape[-2:] != (image_size, image_size):
                # rare; resize each channel
                chans = []
                for c in range(image.shape[0]):
                    chans.append(
                        np.asarray(
                            Image.fromarray((image[c] * 255).astype(np.uint8)).resize(
                                (image_size, image_size), Image.BILINEAR
                            ),
                            dtype=np.float32,
                        )
                        / 255.0
                    )
                image = np.stack(chans, axis=0)
            x = image[None, ...].astype(np.float32)
            t0 = time.perf_counter()
            session.run(None, {"input": x})
            slice_times_ms.append((time.perf_counter() - t0) * 1000.0)
        case_times_s[patient_id] = time.perf_counter() - t_case0

    values = list(case_times_s.values())
    return {
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "onnx_size_human": human_bytes(onnx_path.stat().st_size),
        "in_channels": in_channels,
        "context_slices": context_slices,
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
    comparison: dict | None = None,
    cv_summary: dict | None = None,
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
        ]
    )
    if cv_summary is not None:
        lines.extend(
            [
                "",
                "### Cross-validation summary",
                "```json",
                json.dumps(cv_summary, indent=2),
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "### Holdout",
            "```json",
            json.dumps(holdout_metrics if holdout_metrics is not None else {"error": "not evaluated"}, indent=2),
            "```",
        ]
    )
    if comparison is not None:
        lines.extend(
            [
                "",
                "### Comparison vs prior run",
                "```json",
                json.dumps(comparison, indent=2),
                "```",
            ]
        )
    lines.extend(
        [
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


def load_prior_run_from_log(log_path: Path, run_name: str) -> dict | None:
    """Parse holdout + val JSON blocks for a named ## section in experiment_log.md."""
    if not log_path.exists() or not run_name:
        return None
    text = log_path.read_text(encoding="utf-8")
    marker = f"## {run_name}"
    start = text.find(marker)
    if start < 0:
        return None
    rest = text[start + len(marker) :]
    nxt = rest.find("\n## ")
    section = rest if nxt < 0 else rest[:nxt]

    def _extract(heading: str) -> dict | None:
        h = f"### {heading}"
        i = section.find(h)
        if i < 0:
            return None
        j = section.find("```json", i)
        if j < 0:
            return None
        k = section.find("```", j + 7)
        if k < 0:
            return None
        try:
            return json.loads(section[j + 7 : k].strip())
        except json.JSONDecodeError:
            return None

    return {
        "run_name": run_name,
        "val": _extract("Validation (best checkpoint)"),
        "holdout": _extract("Holdout"),
        "inference": _extract("Inference artifact"),
        "params": _extract("Train parameters"),
    }


def metric_delta(new: float | None, old: float | None) -> dict | None:
    if new is None or old is None:
        return None
    return {"new": new, "old": old, "delta": float(new) - float(old)}


def build_comparison(prior: dict | None, holdout_metrics: dict | None, val_metrics: dict) -> dict | None:
    if prior is None:
        return None
    prior_h = prior.get("holdout") or {}
    prior_v = prior.get("val") or {}
    out: dict = {"prior_run": prior.get("run_name"), "holdout": {}, "validation": {}}
    if holdout_metrics:
        for key in ("loss", "dice", "iou", "hard_dice", "precision", "recall"):
            d = metric_delta(holdout_metrics.get(key), prior_h.get(key))
            if d is not None:
                out["holdout"][key] = d
    for key in ("best_val_loss", "best_dice", "loss", "dice", "iou", "hard_dice", "precision", "recall"):
        d = metric_delta(val_metrics.get(key), prior_v.get(key))
        if d is not None:
            out["validation"][key] = d
    return out


def make_loaders(
    data_root: Path,
    train_rows: list[dict],
    val_rows: list[dict],
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    context_slices: int = 0,
) -> tuple[DataLoader, DataLoader, LiverSegDataset, LiverSegDataset]:
    train_ds = LiverSegDataset(
        data_root, train_rows, augment=True, seed=seed, context_slices=context_slices
    )
    val_ds = LiverSegDataset(
        data_root, val_rows, augment=False, seed=seed, context_slices=context_slices
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, train_ds, val_ds


def train_one_fold(
    *,
    train_rows: list[dict],
    val_rows: list[dict],
    val_patients: list[str],
    data_root: Path,
    device: torch.device,
    args: argparse.Namespace,
    image_size: int,
    ckpt_dir: Path,
    best_path: Path,
    store: MetricsStore,
    run_label: str,
    fold_tag: str = "",
) -> dict:
    """Train one split; returns fold result dict including best metrics/paths."""
    in_channels = 2 * max(0, args.context_slices) + 1
    train_loader, val_loader, train_ds, val_ds = make_loaders(
        data_root,
        train_rows,
        val_rows,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        context_slices=args.context_slices,
    )
    model = TinyUNet(
        in_channels=in_channels, out_channels=1, base=args.base_channels
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict] = []
    best_val_loss = float("inf")
    best_dice = -1.0
    best_val_metrics: dict[str, float] = {}
    best_epoch = 0
    epochs_without_improve = 0
    last_path = ckpt_dir / ("last.pt" if not fold_tag else f"last_{fold_tag}.pt")

    store.update(
        status="running",
        run_name=run_label,
        max_epochs=args.epochs,
        patience=args.patience,
        history=[],
        epoch=0,
        best_epoch=0,
        best_val_loss=None,
        best_dice=None,
        epochs_without_improve=0,
        fold=fold_tag or None,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        desc = f"{fold_tag + ' ' if fold_tag else ''}Epoch {epoch}/{args.epochs}"
        pbar = tqdm(train_loader, desc=desc)
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
            f"{fold_tag + ' ' if fold_tag else ''}epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f} "
            f"hard_dice={val_metrics['hard_dice']:.4f} "
            f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f}"
        )

        if not np.isnan(val_metrics["loss"]) and (
            best_val_loss - val_metrics["loss"] > args.min_delta
        ):
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
                    "in_channels": in_channels,
                    "context_slices": args.context_slices,
                    "val_patients": val_patients,
                    "fold": fold_tag or None,
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
                "in_channels": in_channels,
                "context_slices": args.context_slices,
            },
            last_path,
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
            fold=fold_tag or None,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

        if epochs_without_improve >= args.patience:
            print(
                f"Early stopping at epoch {epoch}: "
                f"no val_loss improvement for {args.patience} epochs "
                f"(best epoch {best_epoch}, val_loss={best_val_loss:.4f})"
            )
            break

    if best_path.exists():
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])

    return {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_dice": best_dice,
        "best_val_metrics": best_val_metrics,
        "best_path": best_path,
        "early_stopped": epochs_without_improve >= args.patience,
        "val_patients": val_patients,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "epochs_ran": history[-1]["epoch"] if history else 0,
    }


def export_and_benchmark(
    *,
    best_path: Path,
    output: Path,
    image_size: int,
    base_channels: int,
    holdout_root: Path,
    holdout_rows: list[dict],
    context_slices: int = 0,
    in_channels: int = 1,
) -> dict | None:
    suffix = f"_{image_size}" if context_slices <= 0 else f"_{image_size}_2p5d"
    onnx_path = output / f"liver_unet{suffix}.onnx"
    quant_path = output / f"liver_unet{suffix}.quant.onnx"
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
        str(base_channels),
        "--in-channels",
        str(in_channels),
    ]
    print("Exporting ONNX…")
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    subprocess.check_call(cmd, env=env)

    infer_onnx = quant_path if quant_path.exists() else onnx_path
    if not holdout_rows or not infer_onnx.exists():
        return {
            "onnx_path": str(infer_onnx if infer_onnx.exists() else onnx_path),
            "float_onnx_path": str(onnx_path),
            "in_channels": in_channels,
            "context_slices": context_slices,
        }
    print(f"Benchmarking ONNX inference on holdout ({infer_onnx.name})…")
    inference_info = benchmark_onnx_holdout(
        infer_onnx,
        holdout_root,
        holdout_rows,
        image_size,
        context_slices=context_slices,
    )
    inference_info["float_onnx_path"] = str(onnx_path)
    inference_info["float_onnx_size_bytes"] = onnx_path.stat().st_size if onnx_path.exists() else None
    inference_info["float_onnx_size_human"] = (
        human_bytes(onnx_path.stat().st_size) if onnx_path.exists() else None
    )
    print(
        f"ONNX {infer_onnx.name}: {inference_info['onnx_size_human']}, "
        f"avg case={inference_info['avg_case_seconds']:.3f}s "
        f"({inference_info['n_cases']} cases), "
        f"avg slice={inference_info['avg_slice_ms']:.2f}ms"
    )
    return inference_info


def summarize_cv_folds(fold_results: list[dict]) -> dict:
    keys = ["best_val_loss", "best_dice", "hard_dice", "precision", "recall", "iou"]
    per_fold = []
    for i, fr in enumerate(fold_results):
        m = dict(fr["best_val_metrics"])
        per_fold.append(
            {
                "fold": i,
                "best_epoch": fr["best_epoch"],
                "best_val_loss": fr["best_val_loss"],
                "best_dice": fr["best_dice"],
                "val_patients": fr["val_patients"],
                "n_train": fr["n_train"],
                "n_val": fr["n_val"],
                "epochs_ran": fr["epochs_ran"],
                **{k: m.get(k) for k in ("loss", "dice", "iou", "hard_dice", "precision", "recall")},
            }
        )
    summary: dict = {"n_folds": len(fold_results), "folds": per_fold, "mean": {}, "std": {}}
    for key in keys:
        vals = []
        for fr in fold_results:
            if key in ("best_val_loss", "best_dice"):
                vals.append(fr[key])
            else:
                vals.append(fr["best_val_metrics"].get(key, float("nan")))
        vals_a = np.asarray(vals, dtype=np.float64)
        summary["mean"][key] = float(np.nanmean(vals_a))
        summary["std"][key] = float(np.nanstd(vals_a))
    return summary


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
    # Keep the shared experiment log at repo outputs/ even when --output is a run subdir.
    experiment_log = args.experiment_log or (ROOT / "outputs" / "experiment_log.md")
    started = datetime.now(timezone.utc)
    run_name = args.run_name or f"baseline-retrain-{started.strftime('%Y%m%d-%H%M%S')}"
    description = args.description or (
        "Retrain after carving a fixed 10-patient holdout from data/raw. "
        "Same model/data pipeline as first run (rendered 128 PNG, TinyUNet base=8, BCE+Dice), "
        "with softer early-stopping (patience=15, min_delta=5e-5) and live metrics dashboard."
    )
    compare_to = args.compare_to or ""

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

    holdout_rows: list[dict] = []
    holdout_patients: list[str] = []
    if args.holdout.exists() and (args.holdout / "manifest.jsonl").exists():
        holdout_rows = filter_existing_pairs(args.holdout, load_manifest(args.holdout))
        holdout_patients = sorted({r["patient_id"] for r in holdout_rows})

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — training on CPU")
        device = torch.device("cpu")
        device_name = "cpu"
    else:
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
        print(f"Using device {device} ({device_name})")

    store.update(device=device_name)

    fold_results: list[dict] = []
    selected: dict

    if args.cv:
        folds = patient_kfold_splits(rows, n_val_patients=args.val_patients, seed=args.seed)
        if not folds:
            print("Could not build CV folds")
            return 1
        split_info = {
            "mode": "patient_kfold",
            "seed": args.seed,
            "val_patients_per_fold": args.val_patients,
            "n_folds": len(folds),
            "folds": [
                {
                    "fold": i,
                    "val_patients": vp,
                    "n_train_patients": len({r["patient_id"] for r in tr}),
                    "n_val_patients": len(vp),
                    "n_train_slices": len(tr),
                    "n_val_slices": len(vr),
                }
                for i, (tr, vr, vp) in enumerate(folds)
            ],
        }
        (args.output / "split.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
        print(
            f"CV: {len(folds)} folds × {args.val_patients} val patients | "
            f"{len(rows)} slices | holdout={len(holdout_patients)} patients {holdout_patients}\n"
            f"GPU: physical {args.gpu}"
        )
        for i, (train_rows, val_rows, val_patients) in enumerate(folds):
            fold_tag = f"fold{i}"
            print(
                f"\n===== {fold_tag}/{len(folds) - 1} "
                f"train={len(train_rows)} val={len(val_rows)} patients={val_patients} ====="
            )
            fr = train_one_fold(
                train_rows=train_rows,
                val_rows=val_rows,
                val_patients=val_patients,
                data_root=args.data,
                device=device,
                args=args,
                image_size=image_size,
                ckpt_dir=ckpt_dir,
                best_path=ckpt_dir / f"best_{fold_tag}.pt",
                store=store,
                run_label=f"{run_name}/{fold_tag}",
                fold_tag=fold_tag,
            )
            fold_results.append(fr)

        # Select fold with lowest validation loss for deployment / holdout eval.
        selected = min(fold_results, key=lambda fr: fr["best_val_loss"])
        selected_idx = fold_results.index(selected)
        # Promote selected fold checkpoint to best.pt for convenience.
        best_path = ckpt_dir / "best.pt"
        if selected["best_path"].exists():
            state = torch.load(selected["best_path"], map_location="cpu", weights_only=False)
            torch.save(state, best_path)
            selected["best_path"] = best_path
        print(
            f"\nSelected fold{selected_idx} "
            f"(val_loss={selected['best_val_loss']:.4f}, dice={selected['best_dice']:.4f})"
        )
        cv_summary = summarize_cv_folds(fold_results)
        cv_summary["selected_fold"] = selected_idx
    else:
        train_rows, val_rows, val_patients = patient_split(
            rows,
            n_val_patients=args.val_patients,
            seed=args.seed,
        )
        (args.output / "split.json").write_text(
            json.dumps(
                {
                    "mode": "single_split",
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
            f"Holdout: {len(holdout_rows)} slices / {len(holdout_patients)} patients -> {holdout_patients}\n"
            f"GPU: physical {args.gpu} (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})"
        )
        selected = train_one_fold(
            train_rows=train_rows,
            val_rows=val_rows,
            val_patients=val_patients,
            data_root=args.data,
            device=device,
            args=args,
            image_size=image_size,
            ckpt_dir=ckpt_dir,
            best_path=ckpt_dir / "best.pt",
            store=store,
            run_label=run_name,
        )
        fold_results = [selected]
        cv_summary = None

    model = selected["model"]
    val_loader = selected["val_loader"]
    val_ds = selected["val_ds"]
    train_loader = selected["train_loader"]
    history = selected["history"]
    best_path = Path(selected["best_path"])
    best_epoch = selected["best_epoch"]
    best_val_loss = selected["best_val_loss"]
    best_dice = selected["best_dice"]
    best_val_metrics = selected["best_val_metrics"]
    val_patients = selected["val_patients"]

    metrics = {
        "run_name": run_name,
        "description": description,
        "cross_validation": bool(args.cv),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_dice": best_dice,
        "best_val_metrics": best_val_metrics,
        "best_checkpoint": str(best_path),
        "early_stopped": selected["early_stopped"],
        "patience": args.patience,
        "min_delta": args.min_delta,
        "val_patients": val_patients,
        "holdout_patients": holdout_patients,
        "history": history,
        "image_size": image_size,
        "base_channels": args.base_channels,
        "context_slices": args.context_slices,
        "in_channels": 2 * max(0, args.context_slices) + 1,
        "num_train": selected["n_train"],
        "num_val": selected["n_val"],
        "num_holdout": len(holdout_rows),
        "gpu": args.gpu,
        "device_name": device_name,
        "cv_summary": cv_summary,
    }
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {metrics_path}")

    sample_loader = val_loader if len(val_ds) else train_loader
    save_samples(model, sample_loader, device, args.output / "samples")

    holdout_metrics = None
    if holdout_rows:
        holdout_ds = LiverSegDataset(
            args.holdout,
            holdout_rows,
            augment=False,
            seed=args.seed,
            context_slices=args.context_slices,
        )
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
        if args.cv:
            holdout_metrics["selected_fold"] = cv_summary["selected_fold"] if cv_summary else None
        print(
            f"holdout: loss={holdout_metrics['loss']:.4f} "
            f"dice={holdout_metrics['dice']:.4f} iou={holdout_metrics['iou']:.4f} "
            f"hard_dice={holdout_metrics['hard_dice']:.4f} "
            f"P={holdout_metrics['precision']:.4f} R={holdout_metrics['recall']:.4f}"
        )
        metrics["holdout"] = holdout_metrics
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    inference_info = None
    in_channels = 2 * max(0, args.context_slices) + 1
    if args.export_onnx and best_path.exists():
        inference_info = export_and_benchmark(
            best_path=best_path,
            output=args.output,
            image_size=image_size,
            base_channels=args.base_channels,
            holdout_root=args.holdout,
            holdout_rows=holdout_rows,
            context_slices=args.context_slices,
            in_channels=in_channels,
        )
        if inference_info is not None:
            metrics["inference"] = inference_info
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    val_log = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_dice": best_dice,
        **best_val_metrics,
    }
    if args.cv and cv_summary is not None:
        val_log["selected_fold"] = cv_summary["selected_fold"]
        val_log["cv_mean_val_loss"] = cv_summary["mean"]["best_val_loss"]
        val_log["cv_std_val_loss"] = cv_summary["std"]["best_val_loss"]
        val_log["cv_mean_dice"] = cv_summary["mean"]["best_dice"]
        val_log["cv_std_dice"] = cv_summary["std"]["best_dice"]

    prior = load_prior_run_from_log(experiment_log, compare_to) if compare_to else None
    comparison = build_comparison(prior, holdout_metrics, val_log)

    params = {
        "run_name": run_name,
        "data": str(args.data),
        "holdout": str(args.holdout),
        "output": str(args.output),
        "cross_validation": bool(args.cv),
        "epochs_max": args.epochs,
        "epochs_ran": selected["epochs_ran"],
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "val_patients_n": args.val_patients,
        "val_patients": val_patients,
        "base_channels": args.base_channels,
        "context_slices": args.context_slices,
        "in_channels": in_channels,
        "image_size": image_size,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "loss": "BCE + soft Dice",
        "optimizer": "Adam",
        "augmentations": "flip/rot90/brightness-contrast (shared across 2.5D channels)",
        "n_train_slices": selected["n_train"],
        "n_val_slices": selected["n_val"],
        "n_holdout_slices": len(holdout_rows),
        "gpu": args.gpu,
        "device_name": device_name,
        "compare_to": compare_to or None,
    }
    if args.cv and cv_summary is not None:
        params["n_folds"] = cv_summary["n_folds"]
        params["selected_fold"] = cv_summary["selected_fold"]

    notes = [
        f"Training inputs at {image_size}×{image_size} (see run description for intensity pipeline).",
        "SEG labels merge all segments except Abdominal aorta.",
        (
            f"TinyUNet base={args.base_channels} [c,2c,4c,8c], in_channels={in_channels}; BCE+Dice; "
            + (
                f"patient-wise {cv_summary['n_folds']}-fold CV with {args.val_patients} val patients/fold."
                if args.cv and cv_summary
                else "patient-wise val split."
            )
        ),
        (
            f"2.5D: context_slices={args.context_slices} "
            f"(channels=[prev..center..next], series-sorted by InstanceNumber, edge clamp)."
            if args.context_slices > 0
            else "2D single-slice input (1 channel)."
        ),
        "Fixed holdout under data/holdout excluded from train/val.",
        f"Early stop on val_loss with patience={args.patience}, min_delta={args.min_delta}.",
        "Inference artifact for browser: INT8 dynamically quantized ONNX (CPU ORT timing).",
    ]
    if args.cv and cv_summary is not None:
        notes.append(
            f"Deployment checkpoint = fold{cv_summary['selected_fold']} "
            f"(lowest fold val_loss); CV mean±std logged separately."
        )
    if comparison is not None:
        notes.append(f"Logged deltas vs prior run `{compare_to}`.")

    append_experiment_log(
        experiment_log,
        run_name=run_name,
        description=description,
        params=params,
        val_metrics=val_log,
        holdout_metrics=holdout_metrics,
        inference=inference_info,
        notes=notes,
        comparison=comparison,
        cv_summary=cv_summary,
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
        cv_summary=cv_summary,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    if dashboard is not None:
        print(
            f"Dashboard still serving final metrics at "
            f"http://{args.dashboard_host}:{args.dashboard_port}/ (Ctrl+C to stop if attached)"
        )
        time.sleep(2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
