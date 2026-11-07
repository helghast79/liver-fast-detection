#!/usr/bin/env python3
"""Parse experiment_log.md and rebuild comparison tables/charts for all runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "outputs" / "experiment_log.md"
DEFAULT_OUT = ROOT / "outputs" / "liver_unet_variable_width_training"

# Display order and short labels for charts.
RUN_META = [
    {
        "run_name": "baseline-after-holdout",
        "label": "render-128",
        "family": "Orthanc /rendered",
        "input": "rendered 128",
        "in_channels": 1,
    },
    {
        "run_name": "render-192",
        "label": "render-192",
        "family": "Orthanc /rendered",
        "input": "rendered 192",
        "in_channels": 1,
    },
    {
        "run_name": "render-256",
        "label": "render-256",
        "family": "Orthanc /rendered",
        "input": "rendered 256",
        "in_channels": 1,
    },
    {
        "run_name": "hu-window-192",
        "label": "HU-192",
        "family": "DICOM HU window",
        "input": "HU C40/W400 @192",
        "in_channels": 1,
    },
    {
        "run_name": "hu-window-192-threshold-lcc",
        "label": "HU-192+LCC",
        "family": "postprocess",
        "input": "HU@192 + thr0.8+LCC",
        "in_channels": 1,
        "is_postprocess": True,
    },
    {
        "run_name": "hu-window-192-cv20",
        "label": "HU-192-CV20",
        "family": "DICOM HU + CV",
        "input": "HU@192, 4×20 CV",
        "in_channels": 1,
    },
    {
        "run_name": "hu-window-192-cv20-2p5d",
        "label": "HU-192-CV20-2.5D",
        "family": "DICOM HU + CV + 2.5D",
        "input": "HU@192 3ch prev/c/next, 4×20 CV",
        "in_channels": 3,
    },
    {
        "run_name": "hu-window-192-cv20-2p5d-threshold-lcc",
        "label": "HU-2.5D+LCC",
        "family": "postprocess",
        "input": "2.5D CV + thr/LCC",
        "in_channels": 3,
        "is_postprocess": True,
        "parent_run": "hu-window-192-cv20-2p5d",
    },
    {
        "run_name": "hu-window-192-2p5d",
        "label": "HU-192-2.5D",
        "family": "DICOM HU + 2.5D",
        "input": "HU@192 3ch, single 10-val split",
        "in_channels": 3,
    },
    {
        "run_name": "hu-window-192-2p5d-base12",
        "label": "HU-2.5D-b12",
        "family": "DICOM HU + 2.5D capacity",
        "input": "HU@192 3ch, base=12",
        "in_channels": 3,
    },
    {
        "run_name": "hu-window-192-2p5d-base12-threshold-lcc",
        "label": "HU-b12+LCC",
        "family": "postprocess",
        "input": "base12 2.5D + thr/LCC",
        "in_channels": 3,
        "is_postprocess": True,
        "parent_run": "hu-window-192-2p5d-base12",
    },
    {
        "run_name": "hu-window-192-2p5d-base16",
        "label": "HU-2.5D-b16",
        "family": "DICOM HU + 2.5D capacity",
        "input": "HU@192 3ch, base=16",
        "in_channels": 3,
    },
]


def extract_json_block(section: str, heading: str) -> dict | None:
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


def parse_experiment_log(log_path: Path) -> dict[str, dict]:
    text = log_path.read_text(encoding="utf-8")
    parts = re.split(r"\n(?=## )", text)
    by_name: dict[str, dict] = {}
    for part in parts:
        m = re.match(r"## ([^\n]+)", part)
        if not m:
            continue
        name = m.group(1).strip()
        by_name[name] = {
            "run_name": name,
            "params": extract_json_block(part, "Train parameters")
            or extract_json_block(part, "Train / postprocess parameters"),
            "val": extract_json_block(part, "Validation (best checkpoint)"),
            "holdout": extract_json_block(part, "Holdout"),
            "soft": extract_json_block(
                part, "Soft metrics (threshold-independent, same checkpoint)"
            ),
            "selected_postprocess": extract_json_block(part, "Holdout selected postprocess"),
            "inference": extract_json_block(part, "Inference artifact"),
            "cv": extract_json_block(part, "Cross-validation summary"),
        }
    return by_name


def kb(n_bytes: float | int | None) -> float | None:
    if n_bytes is None:
        return None
    return round(float(n_bytes) / 1024.0, 2)


def build_rows(parsed: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for meta in RUN_META:
        name = meta["run_name"]
        block = parsed.get(name)
        if not block:
            print(f"WARNING: missing run in log: {name}")
            continue
        params = block.get("params") or {}
        val = block.get("val") or {}
        hold = block.get("holdout") or {}
        infer = block.get("inference") or {}
        cv = block.get("cv")

        is_pp = bool(meta.get("is_postprocess"))
        if is_pp:
            parent_name = meta.get("parent_run") or params.get("parent_run") or "hu-window-192"
            parent = parsed.get(parent_name) or {}
            soft = block.get("soft") or extract_holdout_soft_from_pp(block, parent)
            hard = block.get("selected_postprocess") or {}
            holdout_soft_dice = soft.get("dice")
            holdout_hard_dice = hard.get("hard_dice")
            # Soft IoU from parent; hard P/R/Dice from selected postprocess recipe.
            holdout_precision = hard.get("precision")
            holdout_recall = hard.get("recall")
            holdout_iou = soft.get("iou")
            holdout_loss = soft.get("loss")
            parent_inf = parent.get("inference") or {}
            avg_case = infer.get("parent_avg_case_seconds") or parent_inf.get("avg_case_seconds")
            avg_slice = infer.get("parent_avg_slice_ms") or parent_inf.get("avg_slice_ms")
            quant_bytes = infer.get("onnx_size_bytes") or parent_inf.get("onnx_size_bytes")
            float_bytes = parent_inf.get("float_onnx_size_bytes")
            image_size = params.get("image_size") or 192
            epochs_ran = None
            best_epoch = None
            parent_val = parent.get("val") or {}
            val_soft = parent_val.get("best_dice") or parent_val.get("dice")
            val_hard = parent_val.get("hard_dice")
            val_prec = parent_val.get("precision")
            val_rec = parent_val.get("recall")
            val_loss = parent_val.get("best_val_loss") or parent_val.get("loss")
            parent_hold = parent.get("holdout") or {}
            hold = {
                "n_patients": parent_hold.get("n_patients"),
                "n_slices": parent_hold.get("n_slices") or params.get("n_holdout_slices"),
            }
            thr = hard.get("threshold")
            lcc = hard.get("largest_cc")
            note = f"No retrain; thr={thr} LCC={lcc} on `{parent_name}`"
        else:
            holdout_soft_dice = hold.get("dice")
            holdout_hard_dice = hold.get("hard_dice")
            holdout_precision = hold.get("precision")
            holdout_recall = hold.get("recall")
            holdout_iou = hold.get("iou")
            holdout_loss = hold.get("loss")
            avg_case = infer.get("avg_case_seconds")
            avg_slice = infer.get("avg_slice_ms")
            quant_bytes = infer.get("onnx_size_bytes")
            float_bytes = infer.get("float_onnx_size_bytes")
            image_size = params.get("image_size")
            epochs_ran = params.get("epochs_ran")
            best_epoch = val.get("best_epoch")
            val_soft = val.get("best_dice") or val.get("dice")
            val_hard = val.get("hard_dice")
            val_prec = val.get("precision")
            val_rec = val.get("recall")
            val_loss = val.get("best_val_loss") or val.get("loss")
            note = meta.get("input", "")
            if cv:
                note += f"; selected fold{cv.get('selected_fold')}"

        row = {
            "run_name": name,
            "label": meta["label"],
            "family": meta["family"],
            "input": meta["input"],
            "image_size": image_size,
            "in_channels": meta.get("in_channels", params.get("in_channels", 1)),
            "context_slices": params.get("context_slices", 0 if meta.get("in_channels", 1) == 1 else 1),
            "cross_validation": bool(params.get("cross_validation")),
            "is_postprocess": is_pp,
            "epochs_ran": epochs_ran,
            "best_epoch": best_epoch,
            "val_loss": val_loss,
            "val_soft_dice": val_soft,
            "val_hard_dice": val_hard,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "holdout_soft_dice": holdout_soft_dice,
            "holdout_hard_dice": holdout_hard_dice,
            "holdout_iou": holdout_iou,
            "holdout_precision": holdout_precision,
            "holdout_recall": holdout_recall,
            "holdout_loss": holdout_loss,
            "avg_case_s": avg_case,
            "avg_slice_ms": avg_slice,
            "quant_kb": kb(quant_bytes),
            "float_kb": kb(float_bytes),
            "n_holdout_patients": hold.get("n_patients"),
            "n_holdout_slices": hold.get("n_slices"),
            "cv_mean_dice": (cv or {}).get("mean", {}).get("best_dice") if cv else None,
            "cv_std_dice": (cv or {}).get("std", {}).get("best_dice") if cv else None,
            "note": note,
        }
        rows.append(row)
    return rows


def extract_holdout_soft_from_pp(block: dict, parent: dict | None) -> dict:
    """Soft metrics for postprocess runs come from parent holdout (threshold-independent)."""
    if parent and parent.get("holdout"):
        return parent["holdout"]
    # fallback: some logs embed soft under holdout
    return block.get("holdout") or {}


def write_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "label",
        "run_name",
        "family",
        "input",
        "image_size",
        "in_channels",
        "cross_validation",
        "is_postprocess",
        "best_epoch",
        "epochs_ran",
        "val_soft_dice",
        "val_hard_dice",
        "val_precision",
        "val_recall",
        "val_loss",
        "holdout_soft_dice",
        "holdout_hard_dice",
        "holdout_iou",
        "holdout_precision",
        "holdout_recall",
        "holdout_loss",
        "avg_case_s",
        "avg_slice_ms",
        "quant_kb",
        "n_holdout_patients",
        "n_holdout_slices",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_readme(rows: list[dict], path: Path) -> None:
    lines = [
        "# Progressive TinyUNet experiment comparison",
        "",
        "Holdout-fixed comparison across Orthanc render widths, HU windowing, "
        "postprocess, patient-wise CV (20 val/fold), and 2.5D (prev/center/next).",
        "",
        "TinyUNet (`base=8`), BCE+Dice, early-stop patience=15 / min_delta=5e-5.",
        "",
        "| Run | Input | Soft Dice | Hard Dice | P | R | IoU | Case (s) | Quant ONNX |",
        "|-----|-------|----------:|----------:|--:|--:|----:|---------:|-----------:|",
    ]
    for r in rows:
        soft = r["holdout_soft_dice"]
        hard = r["holdout_hard_dice"]
        p = r["holdout_precision"]
        rec = r["holdout_recall"]
        iou = r["holdout_iou"]
        case = r["avg_case_s"]
        q = r["quant_kb"]
        lines.append(
            f"| {r['label']} | {r['input']} | "
            f"{soft:.3f} | {hard:.3f} | {p:.3f} | {rec:.3f} | {iou:.3f} | "
            f"{case:.2f} | {q:.1f} KB |"
            if all(x is not None for x in (soft, hard, p, rec, iou, case, q))
            else f"| {r['label']} | {r['input']} | — | — | — | — | — | — | — |"
        )
    best_soft = max(rows, key=lambda r: r["holdout_soft_dice"] or -1)
    best_hard = max(rows, key=lambda r: r["holdout_hard_dice"] or -1)
    lines.extend(
        [
            "",
            "## Highlights",
            f"- Best holdout **soft Dice**: `{best_soft['label']}` ({best_soft['holdout_soft_dice']:.3f})",
            f"- Best holdout **hard Dice**: `{best_hard['label']}` ({best_hard['holdout_hard_dice']:.3f})",
            "- Holdout patients are fixed (11 after HCC_027 restored; render-128 logged with 10).",
            "- `HU-192+LCC` / `HU-2.5D+LCC` / `HU-b12+LCC` are postprocess only (thr+LCC); same ONNX as parent.",
            "- All 2.5D ONNX inputs are `[1,3,192,192]` (prev/center/next, edge clamp).",
            "- Latency: `HU-2.5D-b12` ~1.86s/case (kept); `HU-2.5D-b16` ~3.06s (over ~2s budget).",
            "",
            "## Artifacts",
            "- `comparison_metrics.csv` — tabular metrics",
            "- `comparison_summary.json` — machine-readable summary",
            "- `figures/comparison_charts.png` — holdout metrics + latency",
            "- `figures/comparison_table.png` — summary table image",
            "- `figures/dice_vs_run.png` — Dice across runs",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{digits}f}"


def plot_charts(rows: list[dict], out_path: Path) -> None:
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    soft = [r["holdout_soft_dice"] or np.nan for r in rows]
    hard = [r["holdout_hard_dice"] or np.nan for r in rows]
    prec = [r["holdout_precision"] or np.nan for r in rows]
    rec = [r["holdout_recall"] or np.nan for r in rows]
    case_t = [r["avg_case_s"] or np.nan for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), constrained_layout=True)
    ax = axes[0]
    width = 0.2
    ax.bar(x - 1.5 * width, soft, width, label="Soft Dice", color="#3b6ea5")
    ax.bar(x - 0.5 * width, hard, width, label="Hard Dice", color="#2a9d8f")
    ax.bar(x + 0.5 * width, prec, width, label="Precision", color="#e9c46a")
    ax.bar(x + 1.5 * width, rec, width, label="Recall", color="#f4a261")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.65, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Holdout segmentation metrics")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    colors = ["#6c757d" if not r.get("is_postprocess") else "#adb5bd" for r in rows]
    bars = ax2.bar(x, case_t, color=colors, width=0.65)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right")
    ax2.set_ylabel("Seconds / case (CPU ORT, quant ONNX)")
    ax2.set_title("Holdout average inference time")
    ax2.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, case_t):
        if v == v:  # not nan
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.03,
                f"{v:.2f}s",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("Liver TinyUNet — progressive holdout comparison", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_dice_vs_run(rows: list[dict], out_path: Path) -> None:
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    ax.plot(x, [r["holdout_soft_dice"] for r in rows], "o-", color="#1d3557", label="Holdout soft Dice")
    ax.plot(x, [r["holdout_hard_dice"] for r in rows], "s-", color="#2a9d8f", label="Holdout hard Dice")
    ax.plot(
        x,
        [r["val_soft_dice"] for r in rows],
        "o--",
        color="#457b9d",
        alpha=0.75,
        label="Val soft Dice",
    )
    ax.plot(
        x,
        [r["val_hard_dice"] for r in rows],
        "s--",
        color="#80cbc4",
        alpha=0.75,
        label="Val hard Dice",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Dice")
    ax.set_title("Dice across progressive experiments")
    ax.set_ylim(0.65, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_table(rows: list[dict], out_path: Path) -> None:
    cols = [
        ("Run", lambda r: r["label"]),
        ("Input", lambda r: r["input"]),
        ("Soft Dice", lambda r: _fmt(r["holdout_soft_dice"])),
        ("Hard Dice", lambda r: _fmt(r["holdout_hard_dice"])),
        ("P", lambda r: _fmt(r["holdout_precision"])),
        ("R", lambda r: _fmt(r["holdout_recall"])),
        ("IoU", lambda r: _fmt(r["holdout_iou"])),
        ("Val soft", lambda r: _fmt(r["val_soft_dice"])),
        ("Case (s)", lambda r: _fmt(r["avg_case_s"], 2)),
        ("ONNX KB", lambda r: _fmt(r["quant_kb"], 1)),
    ]
    cell_text = [[fn(r) for _, fn in cols] for r in rows]
    col_labels = [c[0] for c in cols]

    # highlight best soft/hard dice cells
    soft_vals = [r["holdout_soft_dice"] if r["holdout_soft_dice"] is not None else -1 for r in rows]
    hard_vals = [r["holdout_hard_dice"] if r["holdout_hard_dice"] is not None else -1 for r in rows]
    best_soft_i = int(np.argmax(soft_vals))
    best_hard_i = int(np.argmax(hard_vals))

    fig_h = 1.2 + 0.45 * len(rows)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (row_i, col_i), cell in table.get_celld().items():
        if row_i == 0:
            cell.set_facecolor("#1d3557")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f8f9fa" if row_i % 2 else "white")
            # soft dice col index 2, hard dice col index 3 (0-based data cols)
            data_row = row_i - 1
            if col_i == 2 and data_row == best_soft_i:
                cell.set_facecolor("#d8f3dc")
            if col_i == 3 and data_row == best_hard_i:
                cell.set_facecolor("#d8f3dc")
    ax.set_title("Main metrics — progressive holdout comparison", pad=12, fontsize=13)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    parsed = parse_experiment_log(args.log)
    rows = build_rows(parsed)
    if not rows:
        print("No runs parsed")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)

    write_csv(rows, args.out / "comparison_metrics.csv")
    summary = {
        "experiment": "progressive TinyUNet holdout comparison",
        "model": "TinyUNet base=8, BCE+Dice, patience=15, min_delta=5e-5",
        "source_log": str(args.log),
        "n_runs": len(rows),
        "runs": rows,
    }
    (args.out / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_readme(rows, args.out / "README.md")

    plot_charts(rows, args.out / "figures" / "comparison_charts.png")
    plot_table(rows, args.out / "figures" / "comparison_table.png")
    plot_dice_vs_run(rows, args.out / "figures" / "dice_vs_run.png")
    # Keep legacy filename as a copy of the new dice chart for compatibility.
    plot_dice_vs_run(rows, args.out / "figures" / "dice_vs_width.png")

    print(f"Wrote comparison for {len(rows)} runs -> {args.out}")
    for r in rows:
        print(
            f"  {r['label']}: soft={r['holdout_soft_dice']:.3f} "
            f"hard={r['holdout_hard_dice']:.3f} case={r['avg_case_s']:.2f}s"
            if r["holdout_soft_dice"] is not None and r["avg_case_s"] is not None
            else f"  {r['label']}: incomplete metrics"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
