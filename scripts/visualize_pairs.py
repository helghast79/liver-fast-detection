#!/usr/bin/env python3
"""Browse CT / liver-mask pairs with a semi-transparent overlay before training.

Default mode for remote SSH: local web viewer (open the printed URL in a browser;
Cursor Remote will usually offer to forward the port).

Optional GUI mode (needs DISPLAY + Tk/Qt): --gui
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

import numpy as np
from matplotlib.colors import to_rgba
from PIL import Image

from src.dataset import load_manifest

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Liver pair viewer</title>
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
      width: min(360px, 30vw); height: auto; border: 1px solid #333;
    }
    .help { margin-top: 1rem; color: #888; font-size: .8rem; }
    button {
      background: #333; color: #eee; border: 1px solid #555;
      padding: .4rem .8rem; margin-right: .4rem; cursor: pointer; border-radius: 4px;
    }
    button:hover { background: #444; }
  </style>
</head>
<body>
  <h1>Liver pair viewer</h1>
  <div>
    <button id="prev" type="button">Prev</button>
    <button id="next" type="button">Next</button>
    <button id="fg" type="button">Only foreground</button>
  </div>
  <p class="meta" id="meta">Loading…</p>
  <div class="row">
    <figure><img id="ct" alt="CT" /><figcaption>CT</figcaption></figure>
    <figure><img id="mask" alt="Mask" /><figcaption>Mask</figcaption></figure>
    <figure><img id="overlay" alt="Overlay" /><figcaption>Overlay</figcaption></figure>
  </div>
  <p class="help">Keys: ←/→ or a/d prev/next · [ / ] ±10 · Home/End · f toggle empty-only · q ignored in browser</p>
  <script>
    let index = START_INDEX;
    let onlyFg = ONLY_FG;
    let total = 0;

    async function refresh() {
      const q = onlyFg ? '&only_foreground=1' : '';
      const info = await fetch('/api/item?index=' + index + q).then(r => r.json());
      if (info.error) { document.getElementById('meta').textContent = info.error; return; }
      index = info.index;
      total = info.total;
      document.getElementById('meta').textContent =
        '[' + (index + 1) + '/' + total + ']  patient=' + info.patient_id +
        '  seg=' + info.seg_series_number + '  liver_seg=' + info.liver_segment_number +
        '  fg_pixels=' + info.mask_foreground_pixels +
        '\\nsop=' + info.sop_instance_uid;
      const bust = Date.now();
      document.getElementById('ct').src = '/api/image?index=' + index + q + '&t=' + bust;
      document.getElementById('mask').src = '/api/mask?index=' + index + q + '&t=' + bust;
      document.getElementById('overlay').src = '/api/overlay?index=' + index + q + '&t=' + bust;
      document.getElementById('fg').textContent = onlyFg ? 'Show all' : 'Only foreground';
    }

    function move(delta) {
      if (!total) return;
      index = (index + delta + total) % total;
      refresh();
    }

    document.getElementById('prev').onclick = () => move(-1);
    document.getElementById('next').onclick = () => move(1);
    document.getElementById('fg').onclick = () => { onlyFg = !onlyFg; index = 0; refresh(); };

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
    parser.add_argument("--patient-id", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--only-foreground", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--color", type=str, default="#ff2a2a")
    parser.add_argument(
        "--mode",
        choices=("auto", "web", "gui"),
        default="auto",
        help="auto: web unless an interactive GUI backend is available",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Try to open the web viewer in a local browser",
    )
    return parser.parse_args()


def filter_rows(rows: list[dict], patient_id: str | None, only_foreground: bool) -> list[dict]:
    if patient_id:
        rows = [r for r in rows if r["patient_id"] == patient_id]
    if only_foreground:
        rows = [r for r in rows if int(r.get("mask_foreground_pixels", 0)) > 0]
    return rows


def load_overlay(data_root: Path, row: dict, alpha: float, color: str) -> np.ndarray:
    image = np.asarray(Image.open(data_root / row["image_path"]).convert("L"), dtype=np.float32) / 255.0
    mask = np.asarray(Image.open(data_root / row["mask_path"]).convert("L")) > 127
    rgb = np.stack([image, image, image], axis=-1)
    overlay = rgb.copy()
    color_rgb = np.array(to_rgba(color)[:3], dtype=np.float32)
    if mask.any():
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color_rgb
    return np.clip(overlay, 0.0, 1.0)


def png_bytes_from_array(arr: np.ndarray, mode: str = "L") -> bytes:
    from io import BytesIO

    if mode == "RGB":
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="RGB")
    else:
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def try_enable_gui_backend() -> str | None:
    """Return backend name if an interactive GUI backend can be selected."""
    import os

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return None

    import matplotlib

    for backend in ("TkAgg", "QtAgg", "Qt5Agg", "GTK4Agg", "GTK3Agg"):
        try:
            matplotlib.use(backend, force=True)
            # Import after use() to validate bindings exist.
            import matplotlib.pyplot as plt  # noqa: F401

            fig = plt.figure()
            interactive = hasattr(fig.canvas, "manager") and fig.canvas.manager is not None
            plt.close(fig)
            if interactive and "agg" not in matplotlib.get_backend().lower():
                return matplotlib.get_backend()
        except Exception:
            continue
    return None


def run_gui(args: argparse.Namespace, rows: list[dict]) -> int:
    import matplotlib.pyplot as plt

    index = max(0, min(args.start, len(rows) - 1))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("Liver pair viewer")
    state = {"index": index}

    def redraw() -> None:
        i = state["index"]
        row = rows[i]
        image = np.asarray(Image.open(args.data / row["image_path"]).convert("L"))
        mask = np.asarray(Image.open(args.data / row["mask_path"]).convert("L"))
        overlay = load_overlay(args.data, row, args.alpha, args.color)
        fg = int(row.get("mask_foreground_pixels", int((mask > 127).sum())))
        views = [
            (axes[0], image, "CT", "gray"),
            (axes[1], mask, "Mask", "gray"),
            (axes[2], overlay, f"Overlay α={args.alpha:.2f}", None),
        ]
        for ax, data, title, cmap in views:
            ax.clear()
            if cmap:
                ax.imshow(data, cmap=cmap, vmin=0, vmax=255)
            else:
                ax.imshow(data)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(
            f"[{i + 1}/{len(rows)}]  patient={row['patient_id']}  "
            f"seg={row.get('seg_series_number')}  liver_seg={row.get('liver_segment_number')}  "
            f"fg_pixels={fg}\n"
            f"sop={row.get('sop_instance_uid', '')}\n"
            f"←/→ or a/d: prev/next   Home/End   q: quit",
            fontsize=10,
        )
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        key = (event.key or "").lower()
        n = len(rows)
        if key in {"right", "d", "n", " "}:
            state["index"] = (state["index"] + 1) % n
        elif key in {"left", "a", "p"}:
            state["index"] = (state["index"] - 1) % n
        elif key == "home":
            state["index"] = 0
        elif key == "end":
            state["index"] = n - 1
        elif key == "pageup":
            state["index"] = max(0, state["index"] - 10)
        elif key == "pagedown":
            state["index"] = min(n - 1, state["index"] + 10)
        elif key in {"q", "escape"}:
            plt.close(fig)
            return
        else:
            return
        redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.tight_layout()
    print(
        f"GUI backend={plt.get_backend()} | {len(rows)} pairs\n"
        "Keys: ←/→ (or a/d) | PageUp/PageDown | Home/End | q"
    )
    plt.show()
    return 0


class PairHandler(BaseHTTPRequestHandler):
    data_root: Path = ROOT / "data" / "raw"
    all_rows: list[dict] = []
    patient_id: str | None = None
    alpha: float = 0.5
    color: str = "#ff2a2a"
    start_index: int = 0
    only_foreground_default: bool = False

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _rows(self, only_foreground: bool) -> list[dict]:
        return filter_rows(self.all_rows, self.patient_id, only_foreground)

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _query_bool(self, qs: dict[str, list[str]], name: str, default: bool = False) -> bool:
        vals = qs.get(name)
        if not vals:
            return default
        return vals[0] not in {"0", "false", "False", ""}

    def do_GET(self) -> None:  # noqa: N802
        from urllib.parse import parse_qs

        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        only_fg = self._query_bool(qs, "only_foreground", self.only_foreground_default)
        rows = self._rows(only_fg)

        if parsed.path in {"/", "/index.html"}:
            page = (
                HTML_PAGE.replace("START_INDEX", str(min(self.start_index, max(len(rows) - 1, 0))))
                .replace("ONLY_FG", "true" if only_fg else "false")
            )
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            return

        if parsed.path == "/api/item":
            if not rows:
                self._send(200, json.dumps({"error": "No pairs match filters"}).encode(), "application/json")
                return
            index = int(qs.get("index", ["0"])[0])
            index = index % len(rows)
            row = rows[index]
            payload = {
                "index": index,
                "total": len(rows),
                "patient_id": row.get("patient_id"),
                "seg_series_number": row.get("seg_series_number"),
                "liver_segment_number": row.get("liver_segment_number"),
                "mask_foreground_pixels": row.get("mask_foreground_pixels"),
                "sop_instance_uid": row.get("sop_instance_uid"),
            }
            self._send(200, json.dumps(payload).encode(), "application/json")
            return

        if parsed.path in {"/api/image", "/api/mask", "/api/overlay"}:
            if not rows:
                self._send(404, b"no rows", "text/plain")
                return
            index = int(qs.get("index", ["0"])[0]) % len(rows)
            row = rows[index]
            if parsed.path == "/api/image":
                data = np.asarray(Image.open(self.data_root / row["image_path"]).convert("L"))
                body = png_bytes_from_array(data, mode="L")
            elif parsed.path == "/api/mask":
                data = np.asarray(Image.open(self.data_root / row["mask_path"]).convert("L"))
                body = png_bytes_from_array(data, mode="L")
            else:
                overlay = load_overlay(self.data_root, row, self.alpha, self.color)
                body = png_bytes_from_array(overlay, mode="RGB")
            self._send(200, body, "image/png")
            return

        self._send(404, b"not found", "text/plain")


def run_web(args: argparse.Namespace, rows: list[dict]) -> int:
    del rows  # browser can toggle foreground; load full patient-filtered set
    PairHandler.data_root = args.data
    all_rows = load_manifest(args.data)
    if args.patient_id:
        all_rows = [r for r in all_rows if r["patient_id"] == args.patient_id]
    PairHandler.all_rows = all_rows
    PairHandler.patient_id = None
    PairHandler.alpha = args.alpha
    PairHandler.color = args.color
    PairHandler.start_index = args.start
    PairHandler.only_foreground_default = args.only_foreground

    server = ThreadingHTTPServer((args.host, args.port), PairHandler)
    bind_url = f"http://{args.host}:{args.port}/"
    print(f"Web viewer listening on {bind_url}")
    if args.host in {"0.0.0.0", "::"}:
        print(f"On this machine: http://127.0.0.1:{args.port}/")
        print(f"From another machine: http://<this-host-ip>:{args.port}/")
    print(f"Pairs available: {len(PairHandler.all_rows)} (patient={args.patient_id!r})")
    print("Ctrl+C to stop.")

    if args.open_browser:
        open_url = f"http://127.0.0.1:{args.port}/" if args.host in {"0.0.0.0", "::"} else bind_url
        threading.Timer(0.5, lambda: webbrowser.open(open_url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


def main() -> int:
    args = parse_args()
    rows = filter_rows(load_manifest(args.data), args.patient_id, args.only_foreground)
    if not rows and not (args.mode in {"auto", "web"} and args.only_foreground):
        # Web mode can still start with only-foreground off toggleable in UI,
        # but empty patient filter should fail.
        all_rows = load_manifest(args.data)
        if args.patient_id:
            all_rows = [r for r in all_rows if r["patient_id"] == args.patient_id]
        if not all_rows:
            print("No pairs to show (check --data / filters).")
            return 1
        rows = all_rows

    if not rows:
        print("No pairs to show (check --data / filters).")
        return 1

    mode = args.mode
    if mode == "auto":
        backend = try_enable_gui_backend()
        mode = "gui" if backend else "web"
        if backend:
            print(f"Using GUI backend: {backend}")
        else:
            print("No interactive matplotlib backend available — starting web viewer.")
            print("(On this host: install python3-tk or pip install PySide6 for --mode gui with ssh -X)")

    if mode == "gui":
        backend = try_enable_gui_backend()
        if not backend:
            print(
                "GUI requested but no interactive backend found.\n"
                "Install one of: sudo apt install python3-tk   OR   pip install PySide6\n"
                "Then reconnect with: ssh -X ...\n"
                "Falling back to web viewer."
            )
            return run_web(args, rows)
        return run_gui(args, rows)

    return run_web(args, rows)


if __name__ == "__main__":
    raise SystemExit(main())
