#!/usr/bin/env python3
"""Browse model predictions on held-out validation patients.

Shows CT, ground-truth mask, prediction, and overlays. Default: web viewer on
0.0.0.0:8766 using the best checkpoint and val patients from
outputs/liver_detect/split.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path
from src.paths import TASK_LIVER_DETECT, task_outputs

TASK = TASK_LIVER_DETECT

sanitize_sys_path()

import numpy as np
import torch
from matplotlib.colors import to_rgba
from PIL import Image

from src.dataset import filter_existing_pairs, load_manifest
from src.model import TinyUNet

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Liver prediction viewer</title>
  <style>
    :root { color-scheme: dark; }
    body {
      margin: 0; font-family: ui-sans-serif, system-ui, sans-serif;
      background: #111; color: #eee; padding: 1rem;
    }
    h1 { font-size: 1rem; font-weight: 600; margin: 0 0 .5rem; }
    .meta { color: #aaa; font-size: .85rem; white-space: pre-wrap; margin-bottom: 1rem; }
    .row { display: flex; gap: 1rem; flex-wrap: wrap; }
    figure { margin: 0; text-align: center; }
    figcaption { margin-top: .4rem; color: #bbb; font-size: .85rem; }
    img {
      background: #000; image-rendering: pixelated;
      width: min(280px, 18vw); height: auto; border: 1px solid #333;
    }
    .help { margin-top: 1rem; color: #888; font-size: .8rem; }
    button, select {
      background: #333; color: #eee; border: 1px solid #555;
      padding: .4rem .8rem; margin-right: .4rem; cursor: pointer; border-radius: 4px;
    }
    button:hover, select:hover { background: #444; }
  </style>
</head>
<body>
  <h1>Liver prediction viewer (validation cases)</h1>
  <div>
    <button id="prev" type="button">Prev</button>
    <button id="next" type="button">Next</button>
    <button id="fg" type="button">Only GT foreground</button>
    <label>Patient
      <select id="patient"><option value="">All val patients</option></select>
    </label>
  </div>
  <p class="meta" id="meta">Loading…</p>
  <div class="row">
    <figure><img id="ct" alt="CT" /><figcaption>CT</figcaption></figure>
    <figure><img id="gt" alt="GT" /><figcaption>GT mask</figcaption></figure>
    <figure><img id="pred" alt="Pred" /><figcaption>Prediction</figcaption></figure>
    <figure><img id="gt_ov" alt="GT overlay" /><figcaption>GT overlay</figcaption></figure>
    <figure><img id="pred_ov" alt="Pred overlay" /><figcaption>Pred overlay</figcaption></figure>
  </div>
  <p class="help">Keys: ←/→ or a/d · [ / ] ±10 · Home/End · f toggle empty GT · patient dropdown filters</p>
  <script>
    let index = START_INDEX;
    let onlyFg = ONLY_FG;
    let patientId = PATIENT_FILTER;
    let total = 0;
    const patients = PATIENTS;

    const sel = document.getElementById('patient');
    for (const p of patients) {
      const opt = document.createElement('option');
      opt.value = p; opt.textContent = p;
      if (p === patientId) opt.selected = true;
      sel.appendChild(opt);
    }

    async function refresh() {
      const q = new URLSearchParams({
        index: String(index),
        only_foreground: onlyFg ? '1' : '0',
      });
      if (patientId) q.set('patient_id', patientId);
      const info = await fetch('/api/item?' + q.toString()).then(r => r.json());
      if (info.error) { document.getElementById('meta').textContent = info.error; return; }
      index = info.index;
      total = info.total;
      document.getElementById('meta').textContent =
        '[' + (index + 1) + '/' + total + ']  patient=' + info.patient_id +
        '  dice=' + info.dice.toFixed(3) +
        '  gt_fg=' + info.gt_fg + '  pred_fg=' + info.pred_fg +
        '\\nsop=' + info.sop_instance_uid;
      const bust = Date.now();
      const imgQ = q.toString() + '&t=' + bust;
      document.getElementById('ct').src = '/api/image?' + imgQ;
      document.getElementById('gt').src = '/api/gt?' + imgQ;
      document.getElementById('pred').src = '/api/pred?' + imgQ;
      document.getElementById('gt_ov').src = '/api/gt_overlay?' + imgQ;
      document.getElementById('pred_ov').src = '/api/pred_overlay?' + imgQ;
      document.getElementById('fg').textContent = onlyFg ? 'Show all' : 'Only GT foreground';
    }

    function move(delta) {
      if (!total) return;
      index = (index + delta + total) % total;
      refresh();
    }

    document.getElementById('prev').onclick = () => move(-1);
    document.getElementById('next').onclick = () => move(1);
    document.getElementById('fg').onclick = () => { onlyFg = !onlyFg; index = 0; refresh(); };
    sel.onchange = () => { patientId = sel.value; index = 0; refresh(); };

    window.addEventListener('keydown', (e) => {
      const k = e.key.toLowerCase();
      if (k === 'arrowright' || k === 'd' || k === 'n' || k === ' ') move(1);
      else if (k === 'arrowleft' || k === 'a' || k === 'p') move(-1);
      else if (k === 'home') { index = 0; refresh(); }
      else if (k === 'end') { index = Math.max(0, total - 1); refresh(); }
      else if (k === 'pageup' || k === '[') move(-10);
      else if (k === 'pagedown' || k === ']') move(10);
      else if (k === 'f') { onlyFg = !onlyFg; index = 0; refresh(); }
    });

    refresh();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--split", type=Path, default=task_outputs(TASK) / "split.json")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=task_outputs(TASK) / "checkpoints" / "best.pt",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--patient-id", type=str, default=None)
    parser.add_argument("--only-foreground", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gt-color", type=str, default="#2aff2a")
    parser.add_argument("--pred-color", type=str, default="#ff2a2a")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


def load_val_patients(split_path: Path) -> list[str]:
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    data = json.loads(split_path.read_text(encoding="utf-8"))
    patients = list(data.get("val_patients") or [])
    if not patients:
        raise ValueError(f"No val_patients in {split_path}")
    return patients


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: str, alpha: float) -> np.ndarray:
    rgb = np.stack([image, image, image], axis=-1).astype(np.float32) / 255.0
    out = rgb.copy()
    color_rgb = np.array(to_rgba(color)[:3], dtype=np.float32)
    m = mask.astype(bool)
    if m.any():
        out[m] = (1.0 - alpha) * out[m] + alpha * color_rgb
    return np.clip(out, 0.0, 1.0)


def png_bytes(arr: np.ndarray, mode: str = "L") -> bytes:
    if mode == "RGB":
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="RGB")
    else:
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def load_model(checkpoint: Path, device: torch.device) -> tuple[TinyUNet, dict]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    base = int(state.get("base_channels", 8))
    model = TinyUNet(in_channels=1, out_channels=1, base=base).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state


@torch.no_grad()
def predict_mask(
    model: TinyUNet,
    image: np.ndarray,
    device: torch.device,
    threshold: float,
) -> np.ndarray:
    x = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return (prob >= threshold).astype(np.uint8)


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    return float((2 * inter + eps) / (p.sum() + g.sum() + eps))


class PredHandler(BaseHTTPRequestHandler):
    data_root: Path = ROOT / "data" / "raw"
    rows: list[dict] = []
    patients: list[str] = []
    model: TinyUNet | None = None
    device: torch.device = torch.device("cpu")
    threshold: float = 0.5
    alpha: float = 0.5
    gt_color: str = "#2aff2a"
    pred_color: str = "#ff2a2a"
    start_index: int = 0
    only_foreground_default: bool = False
    patient_filter_default: str = ""
    pred_cache: dict[int, np.ndarray] = {}

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _filtered_rows(self, only_fg: bool, patient_id: str | None) -> list[tuple[int, dict]]:
        out: list[tuple[int, dict]] = []
        for i, row in enumerate(self.rows):
            if patient_id and row["patient_id"] != patient_id:
                continue
            if only_fg and int(row.get("mask_foreground_pixels", 0)) <= 0:
                continue
            out.append((i, row))
        return out

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _bool(self, qs: dict[str, list[str]], name: str, default: bool = False) -> bool:
        vals = qs.get(name)
        if not vals:
            return default
        return vals[0] not in {"0", "false", "False", ""}

    def _get_pred(self, global_idx: int, row: dict, image: np.ndarray) -> np.ndarray:
        cached = self.pred_cache.get(global_idx)
        if cached is not None:
            return cached
        assert self.model is not None
        pred = predict_mask(self.model, image, self.device, self.threshold)
        self.pred_cache[global_idx] = pred
        return pred

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        only_fg = self._bool(qs, "only_foreground", self.only_foreground_default)
        patient_id = (qs.get("patient_id") or [self.patient_filter_default])[0] or None

        if parsed.path in {"/", "/index.html"}:
            page = (
                HTML_PAGE.replace("START_INDEX", str(self.start_index))
                .replace("ONLY_FG", "true" if only_fg else "false")
                .replace("PATIENT_FILTER", json.dumps(patient_id or ""))
                .replace("PATIENTS", json.dumps(self.patients))
            )
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            return

        filtered = self._filtered_rows(only_fg, patient_id)
        if parsed.path == "/api/item":
            if not filtered:
                self._send(
                    200,
                    json.dumps({"error": "No slices match filters"}).encode(),
                    "application/json",
                )
                return
            index = int(qs.get("index", ["0"])[0]) % len(filtered)
            global_idx, row = filtered[index]
            image = np.asarray(Image.open(self.data_root / row["image_path"]).convert("L"))
            gt = (np.asarray(Image.open(self.data_root / row["mask_path"]).convert("L")) > 127).astype(
                np.uint8
            )
            pred = self._get_pred(global_idx, row, image)
            payload = {
                "index": index,
                "total": len(filtered),
                "patient_id": row["patient_id"],
                "sop_instance_uid": row.get("sop_instance_uid"),
                "dice": dice_score(pred, gt),
                "gt_fg": int(gt.sum()),
                "pred_fg": int(pred.sum()),
            }
            self._send(200, json.dumps(payload).encode(), "application/json")
            return

        endpoints = {
            "/api/image",
            "/api/gt",
            "/api/pred",
            "/api/gt_overlay",
            "/api/pred_overlay",
        }
        if parsed.path in endpoints:
            if not filtered:
                self._send(404, b"no rows", "text/plain")
                return
            index = int(qs.get("index", ["0"])[0]) % len(filtered)
            global_idx, row = filtered[index]
            image = np.asarray(Image.open(self.data_root / row["image_path"]).convert("L"))
            gt = (np.asarray(Image.open(self.data_root / row["mask_path"]).convert("L")) > 127).astype(
                np.uint8
            )
            pred = self._get_pred(global_idx, row, image)

            if parsed.path == "/api/image":
                body = png_bytes(image, mode="L")
            elif parsed.path == "/api/gt":
                body = png_bytes(gt * 255, mode="L")
            elif parsed.path == "/api/pred":
                body = png_bytes(pred * 255, mode="L")
            elif parsed.path == "/api/gt_overlay":
                body = png_bytes(overlay_mask(image, gt, self.gt_color, self.alpha), mode="RGB")
            else:
                body = png_bytes(overlay_mask(image, pred, self.pred_color, self.alpha), mode="RGB")
            self._send(200, body, "image/png")
            return

        self._send(404, b"not found", "text/plain")


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    val_patients = load_val_patients(args.split)
    if args.patient_id:
        if args.patient_id not in val_patients:
            print(f"Patient {args.patient_id} is not in val split: {val_patients}")
            return 1
        val_patients = [args.patient_id]

    rows = filter_existing_pairs(args.data, load_manifest(args.data))
    rows = [r for r in rows if r["patient_id"] in set(val_patients)]
    rows.sort(key=lambda r: (r["patient_id"], r.get("sop_instance_uid", "")))
    if not rows:
        print("No validation pairs found on disk.")
        return 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, state = load_model(args.checkpoint, device)
    print(
        f"Loaded {args.checkpoint} (epoch={state.get('epoch')}) on {device}\n"
        f"Val patients ({len(val_patients)}): {val_patients}\n"
        f"Slices: {len(rows)}",
        flush=True,
    )

    PredHandler.data_root = args.data
    PredHandler.rows = rows
    PredHandler.patients = val_patients
    PredHandler.model = model
    PredHandler.device = device
    PredHandler.threshold = args.threshold
    PredHandler.alpha = args.alpha
    PredHandler.gt_color = args.gt_color
    PredHandler.pred_color = args.pred_color
    PredHandler.start_index = max(0, min(args.start, len(rows) - 1))
    PredHandler.only_foreground_default = args.only_foreground
    PredHandler.patient_filter_default = args.patient_id or ""
    PredHandler.pred_cache = {}

    server = ThreadingHTTPServer((args.host, args.port), PredHandler)
    print(f"Prediction viewer listening on http://{args.host}:{args.port}/", flush=True)
    if args.host in {"0.0.0.0", "::"}:
        print(f"On this machine: http://127.0.0.1:{args.port}/", flush=True)
        print(f"From another machine: http://<this-host-ip>:{args.port}/", flush=True)
    print("Ctrl+C to stop.", flush=True)

    if args.open_browser:
        open_url = f"http://127.0.0.1:{args.port}/"
        threading.Timer(0.5, lambda: webbrowser.open(open_url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
