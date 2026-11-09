#!/usr/bin/env python3
"""Evaluate holdout with raised thresholds and/or largest-CC postprocess (no retrain)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path
from src.paths import TASK_LIVER_DETECT, task_outputs

TASK = TASK_LIVER_DETECT

sanitize_sys_path()

import numpy as np
import torch
from dotenv import load_dotenv
from scipy import ndimage
from torch.utils.data import DataLoader

from src.dataset import LiverSegDataset, filter_existing_pairs, load_manifest
from src.model import TinyUNet, bce_dice_loss, soft_dice, soft_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=task_outputs(TASK) / "checkpoints" / "best.pt",
    )
    parser.add_argument("--holdout", type=Path, default=ROOT / "data" / "holdout")
    parser.add_argument("--output", type=Path, default=task_outputs(TASK))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="hu-window-192-threshold-lcc",
    )
    parser.add_argument(
        "--parent-run",
        type=str,
        default="hu-window-192",
        help="Base trained model this postprocess builds on",
    )
    parser.add_argument(
        "--context-slices",
        type=int,
        default=None,
        help="Axial neighbors per side (default: read from checkpoint)",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="Quant/float ONNX path to cite in inference config (optional)",
    )
    parser.add_argument(
        "--parent-metrics",
        type=Path,
        default=None,
        help="Parent metrics.json for carrying timing (optional)",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Experiment log description override",
    )
    parser.add_argument("--experiment-log", type=Path, default=None)
    return parser.parse_args()


def keep_largest_cc(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected component; empty stays empty."""
    labeled, n = ndimage.label(mask.astype(bool))
    if n <= 1:
        return mask.astype(np.uint8)
    counts = ndimage.sum(mask.astype(bool), labeled, index=np.arange(1, n + 1))
    keep = int(np.argmax(counts)) + 1
    return (labeled == keep).astype(np.uint8)


@torch.no_grad()
def collect_probs(
    model: TinyUNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float]]:
    """Return per-slice probability/gt maps plus soft metrics (threshold-independent)."""
    model.eval()
    probs: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    losses, dices, ious = [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        losses.append(float(bce_dice_loss(logits, masks).item()))
        dices.append(soft_dice(logits, masks))
        ious.append(soft_iou(logits, masks))
        prob = torch.sigmoid(logits)[:, 0].cpu().numpy()
        gt = masks[:, 0].cpu().numpy()
        for i in range(prob.shape[0]):
            probs.append(prob[i])
            gts.append(gt[i])
    soft = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "dice": float(np.mean(dices)) if dices else float("nan"),
        "iou": float(np.mean(ious)) if ious else float("nan"),
    }
    return probs, gts, soft


def hard_metrics(
    probs: list[np.ndarray],
    gts: list[np.ndarray],
    threshold: float,
    largest_cc: bool,
) -> dict[str, float]:
    tp = fp = fn = 0
    for prob, gt in zip(probs, gts):
        pred = (prob >= threshold).astype(np.uint8)
        if largest_cc:
            pred = keep_largest_cc(pred)
        g = gt.astype(bool)
        p = pred.astype(bool)
        tp += int(np.logical_and(p, g).sum())
        fp += int(np.logical_and(p, np.logical_not(g)).sum())
        fn += int(np.logical_and(np.logical_not(p), g).sum())
    hard_dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    return {
        "threshold": threshold,
        "largest_cc": largest_cc,
        "hard_dice": float(hard_dice),
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def pick_best(rows: list[dict], baseline: dict, max_dice_drop: float = 0.02) -> dict:
    """
    FP-oriented selection: among configs with hard_dice within max_dice_drop of
    baseline, maximize precision (then hard_dice, then recall).
    Falls back to max hard_dice if none stay within the budget.
    """
    floor = baseline["hard_dice"] - max_dice_drop
    eligible = [r for r in rows if r["hard_dice"] >= floor]
    pool = eligible or rows
    return max(pool, key=lambda r: (r["precision"], r["hard_dice"], r["recall"]))


def append_log(
    log_path: Path,
    *,
    run_name: str,
    parent_run: str,
    description: str,
    params: dict,
    soft: dict,
    baseline: dict,
    best: dict,
    sweep: list[dict],
    inference: dict,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    in_ch = params.get("in_channels", 1)
    ctx = params.get("context_slices", 0)
    notes = [
        f"No retrain — postprocess on checkpoint from `{parent_run}`.",
        "Sweep probability thresholds with optional largest connected-component filter.",
        "Goal: cut false positives while preserving holdout hard Dice.",
        (
            f"Same HU abdomen window (C40/W400) @ {params.get('image_size', 192)}; "
            f"in_channels={in_ch}"
            + (f" (2.5D context_slices={ctx})." if ctx else ".")
        ),
        "ONNX weights unchanged; viewer should apply chosen threshold + LCC after sigmoid.",
    ]
    body = [
        f"\n## {run_name}",
        f"- **When:** {ts}",
        f"- **Description:** {description}",
        "",
        "### Main considerations",
        *[f"- {n}" for n in notes],
        "",
        "### Train / postprocess parameters",
        "```json",
        json.dumps(params, indent=2),
        "```",
        "",
        "### Soft metrics (threshold-independent, same checkpoint)",
        "```json",
        json.dumps(soft, indent=2),
        "```",
        "",
        "### Holdout baseline (thr=0.5, no LCC)",
        "```json",
        json.dumps(baseline, indent=2),
        "```",
        "",
        "### Holdout selected postprocess",
        "```json",
        json.dumps(best, indent=2),
        "```",
        "",
        "### Full sweep",
        "```json",
        json.dumps(sweep, indent=2),
        "```",
        "",
        "### Inference artifact",
        "```json",
        json.dumps(inference, indent=2),
        "```",
        "",
        "---",
        "",
    ]
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(body))


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    load_dotenv(ROOT / ".env")

    holdout_rows = filter_existing_pairs(args.holdout, load_manifest(args.holdout))
    if not holdout_rows:
        print(f"No holdout pairs under {args.holdout}")
        return 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    image_size = int(state.get("image_size", 192))
    base = int(state.get("base_channels", 8))
    if args.context_slices is not None:
        context_slices = int(args.context_slices)
    elif state.get("context_slices") is not None:
        context_slices = int(state["context_slices"])
    elif state.get("in_channels") is not None:
        # in_channels = 2*context + 1
        context_slices = max(0, (int(state["in_channels"]) - 1) // 2)
    else:
        context_slices = 0
    in_channels = 2 * context_slices + 1
    model = TinyUNet(in_channels=in_channels, out_channels=1, base=base).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    ds = LiverSegDataset(
        args.holdout,
        holdout_rows,
        augment=False,
        seed=0,
        context_slices=context_slices,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(
        f"Collecting probabilities on {len(holdout_rows)} holdout slices "
        f"(in_channels={in_channels}, context_slices={context_slices})…"
    )
    probs, gts, soft = collect_probs(model, loader, device)

    sweep: list[dict] = []
    for thr in args.thresholds:
        for lcc in (False, True):
            row = hard_metrics(probs, gts, thr, lcc)
            sweep.append(row)
            print(
                f"thr={thr:.2f} lcc={lcc}: "
                f"hard_dice={row['hard_dice']:.4f} "
                f"P={row['precision']:.4f} R={row['recall']:.4f} iou={row['iou']:.4f}"
            )

    baseline = next(r for r in sweep if r["threshold"] == 0.5 and not r["largest_cc"])
    # Improvement under test: raised threshold and/or largest-CC (exclude plain 0.5 baseline)
    candidates = [r for r in sweep if r["threshold"] > 0.5 or r["largest_cc"]]
    best = pick_best(candidates, baseline=baseline, max_dice_drop=0.02)

    out_dir = args.output / "postprocess"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "run_name": args.run_name,
        "parent_run": args.parent_run,
        "checkpoint": str(args.checkpoint),
        "image_size": image_size,
        "base_channels": base,
        "in_channels": in_channels,
        "context_slices": context_slices,
        "soft": soft,
        "baseline": baseline,
        "selected": best,
        "sweep": sweep,
        "n_holdout_slices": len(holdout_rows),
        "n_holdout_patients": len({r["patient_id"] for r in holdout_rows}),
    }
    result_path = out_dir / f"{args.run_name}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Inference config for the viewer (same ONNX weights + postprocess knobs)
    suffix = f"_{image_size}" if context_slices <= 0 else f"_{image_size}_2p5d"
    quant = args.output / f"liver_unet{suffix}.quant.onnx"
    float_onnx = args.output / f"liver_unet{suffix}.onnx"
    if args.onnx is not None:
        onnx_path = args.onnx
    elif quant.exists():
        onnx_path = quant
    else:
        onnx_path = float_onnx
    infer_cfg = {
        "parent_run": args.parent_run,
        "run_name": args.run_name,
        "onnx_path": str(onnx_path),
        "float_onnx_path": str(float_onnx) if float_onnx.exists() else None,
        "image_size": image_size,
        "in_channels": in_channels,
        "context_slices": context_slices,
        "image_source": "hu",
        "window_center": 40.0,
        "window_width": 400.0,
        "threshold": best["threshold"],
        "keep_largest_cc": best["largest_cc"],
        "holdout_hard_dice": best["hard_dice"],
        "holdout_precision": best["precision"],
        "holdout_recall": best["recall"],
    }
    cfg_path = out_dir / f"{args.run_name}_inference_config.json"
    cfg_path.write_text(json.dumps(infer_cfg, indent=2), encoding="utf-8")

    inference = {
        "onnx_path": infer_cfg["onnx_path"],
        "onnx_size_bytes": Path(infer_cfg["onnx_path"]).stat().st_size
        if Path(infer_cfg["onnx_path"]).exists()
        else None,
        "in_channels": in_channels,
        "context_slices": context_slices,
        "note": (
            "Weights unchanged from parent run; apply threshold + optional largest-CC "
            "after sigmoid. Model forward time same as parent; postprocess is CPU/numpy."
        ),
        "recommended_threshold": best["threshold"],
        "recommended_keep_largest_cc": best["largest_cc"],
        "inference_config": str(cfg_path),
    }
    parent_metrics = args.parent_metrics or (args.output / "metrics.json")
    if parent_metrics.exists():
        try:
            pm = json.loads(parent_metrics.read_text())
            if "inference" in pm:
                inference["parent_avg_case_seconds"] = pm["inference"].get("avg_case_seconds")
                inference["parent_avg_slice_ms"] = pm["inference"].get("avg_slice_ms")
                inference["onnx_size_human"] = pm["inference"].get("onnx_size_human")
                inference["onnx_size_bytes"] = inference.get("onnx_size_bytes") or pm[
                    "inference"
                ].get("onnx_size_bytes")
        except Exception:  # noqa: BLE001
            pass

    params = {
        "run_name": args.run_name,
        "parent_run": args.parent_run,
        "retrain": False,
        "checkpoint": str(args.checkpoint),
        "thresholds_swept": args.thresholds,
        "largest_cc_options": [False, True],
        "selection_rule": (
            "among thr>0.5 or LCC, maximize precision subject to hard_dice "
            "within 0.02 of baseline; else max hard_dice"
        ),
        "image_size": image_size,
        "base_channels": base,
        "in_channels": in_channels,
        "context_slices": context_slices,
        "n_holdout_slices": len(holdout_rows),
    }

    log_path = args.experiment_log or (task_outputs(TASK) / "experiment_log.md")
    description = args.description or (
        f"Postprocess on `{args.parent_run}`: raise decision threshold and/or "
        "keep largest connected component to cut false positives (no retrain)."
    )
    append_log(
        log_path,
        run_name=args.run_name,
        parent_run=args.parent_run,
        description=description,
        params=params,
        soft=soft,
        baseline=baseline,
        best=best,
        sweep=sweep,
        inference=inference,
    )

    print(f"\nBaseline thr=0.5: hard_dice={baseline['hard_dice']:.4f} P={baseline['precision']:.4f}")
    print(
        f"Selected thr={best['threshold']:.2f} lcc={best['largest_cc']}: "
        f"hard_dice={best['hard_dice']:.4f} P={best['precision']:.4f} R={best['recall']:.4f}"
    )
    print(f"Wrote {result_path}")
    print(f"Wrote {cfg_path}")
    print(f"Appended {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
