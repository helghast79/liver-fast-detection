"""Live browser dashboard for training metrics."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Liver U-Net training</title>
<style>
  :root {
    --bg: #0f1419;
    --panel: #1a222c;
    --text: #e7eef7;
    --muted: #8b9bb0;
    --accent: #3db8ff;
    --good: #3dd68c;
    --warn: #f0b429;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    background: radial-gradient(1200px 600px at 10% -10%, #1b2a3a 0%, var(--bg) 55%);
    color: var(--text); min-height: 100vh;
  }
  header { padding: 1.25rem 1.5rem 0.5rem; }
  h1 { margin: 0; font-size: 1.35rem; letter-spacing: 0.02em; }
  .sub { color: var(--muted); margin-top: 0.35rem; font-size: 0.92rem; }
  main { padding: 1rem 1.5rem 2rem; display: grid; gap: 1rem; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem; }
  .card {
    background: var(--panel); border: 1px solid #273242; border-radius: 10px;
    padding: 0.85rem 1rem;
  }
  .card .label { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
  .card .value { font-size: 1.45rem; margin-top: 0.25rem; font-variant-numeric: tabular-nums; }
  .panel {
    background: var(--panel); border: 1px solid #273242; border-radius: 10px; padding: 1rem;
  }
  canvas { width: 100%; height: 280px; display: block; }
  table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  th, td { padding: 0.4rem 0.5rem; text-align: right; border-bottom: 1px solid #273242; }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--muted); font-weight: 600; }
  .status-ok { color: var(--good); }
  .status-run { color: var(--accent); }
  .status-done { color: var(--warn); }
  .mono { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.85rem; color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Liver TinyUNet — live training</h1>
  <div class="sub" id="subtitle">connecting…</div>
</header>
<main>
  <section class="cards" id="cards"></section>
  <section class="panel">
    <canvas id="chart" width="1200" height="280"></canvas>
  </section>
  <section class="panel">
    <table>
      <thead>
        <tr>
          <th>Epoch</th><th>Train loss</th><th>Val loss</th><th>Val Dice</th><th>Val IoU</th><th>Best?</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </section>
</main>
<script>
const COLORS = {
  train: '#3db8ff',
  val: '#f0b429',
  dice: '#3dd68c',
};

function fmt(x, d=4) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return Number(x).toFixed(d);
}

function drawChart(history) {
  const canvas = document.getElementById('chart');
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!history.length) {
    ctx.fillStyle = '#8b9bb0';
    ctx.fillText('Waiting for first epoch…', 24, h/2);
    return;
  }
  const pad = {l: 48, r: 16, t: 16, b: 28};
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;
  const losses = history.flatMap(r => [r.train.loss, r.val.loss]).filter(x => Number.isFinite(x));
  const dices = history.map(r => r.val.dice).filter(x => Number.isFinite(x));
  const minL = Math.min(...losses);
  const maxL = Math.max(...losses);
  const minD = Math.min(0, ...dices);
  const maxD = Math.max(1, ...dices);
  const n = history.length;
  const xAt = i => pad.l + (n === 1 ? plotW/2 : (i / (n-1)) * plotW);
  const yL = v => pad.t + (1 - ((v - minL) / (maxL - minL || 1))) * plotH;
  const yD = v => pad.t + (1 - ((v - minD) / (maxD - minD || 1))) * plotH;

  ctx.strokeStyle = '#273242';
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const y = pad.t + (g/4) * plotH;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
  }

  function series(key, color, yFn) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    history.forEach((r, i) => {
      const v = key === 'dice' ? r.val.dice : (key === 'train' ? r.train.loss : r.val.loss);
      const x = xAt(i), y = yFn(v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  series('train', COLORS.train, yL);
  series('val', COLORS.val, yL);
  series('dice', COLORS.dice, yD);

  ctx.fillStyle = '#8b9bb0';
  ctx.font = '12px sans-serif';
  ctx.fillText('loss ' + fmt(maxL, 3), 4, pad.t + 10);
  ctx.fillText('loss ' + fmt(minL, 3), 4, h - pad.b);
  ctx.fillText('dice right axis', w - 110, pad.t + 12);

  const legend = [
    ['train loss', COLORS.train],
    ['val loss', COLORS.val],
    ['val dice', COLORS.dice],
  ];
  legend.forEach(([name, c], i) => {
    const x = pad.l + i * 120;
    ctx.fillStyle = c; ctx.fillRect(x, 4, 12, 12);
    ctx.fillStyle = '#e7eef7'; ctx.fillText(name, x + 16, 14);
  });
}

function render(data) {
  const status = data.status || 'idle';
  const statusClass = status === 'running' ? 'status-run' : (status === 'finished' ? 'status-done' : 'status-ok');
  document.getElementById('subtitle').innerHTML =
    `<span class="${statusClass}">${status}</span>` +
    ` · ${data.run_name || ''}` +
    ` · epoch ${data.epoch || 0}/${data.max_epochs || '?'}` +
    ` · best @ ${data.best_epoch || '—'} (val_loss ${fmt(data.best_val_loss)})` +
    ` · <span class="mono">${data.updated_at || ''}</span>`;

  const cards = [
    ['Epoch', `${data.epoch || 0} / ${data.max_epochs || '?'}`],
    ['Train loss', fmt(data.latest?.train?.loss)],
    ['Val loss', fmt(data.latest?.val?.loss)],
    ['Val Dice', fmt(data.latest?.val?.dice)],
    ['Val IoU', fmt(data.latest?.val?.iou)],
    ['Best Dice', fmt(data.best_dice)],
    ['No improve', `${data.epochs_without_improve || 0} / ${data.patience || '?'}`],
    ['GPU', data.device || '—'],
  ];
  document.getElementById('cards').innerHTML = cards.map(([label, value]) =>
    `<div class="card"><div class="label">${label}</div><div class="value">${value}</div></div>`
  ).join('');

  const history = data.history || [];
  drawChart(history);
  const bestEpoch = data.best_epoch;
  document.getElementById('rows').innerHTML = [...history].reverse().map(r => `
    <tr>
      <td>${r.epoch}</td>
      <td>${fmt(r.train.loss)}</td>
      <td>${fmt(r.val.loss)}</td>
      <td>${fmt(r.val.dice)}</td>
      <td>${fmt(r.val.iou)}</td>
      <td>${r.epoch === bestEpoch ? '★' : ''}</td>
    </tr>`).join('');
}

async function tick() {
  try {
    const res = await fetch('/api/metrics', {cache: 'no-store'});
    const data = await res.json();
    render(data);
  } catch (e) {
    document.getElementById('subtitle').textContent = 'waiting for trainer…';
  }
}
tick();
setInterval(tick, 1500);
</script>
</body>
</html>
"""


class MetricsStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {
            "status": "idle",
            "history": [],
            "epoch": 0,
        }

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            self._data.update(kwargs)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._data))


class _Handler(BaseHTTPRequestHandler):
    store: MetricsStore

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/metrics":
            body = json.dumps(self.store.snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()


def start_dashboard(host: str, port: int, store: MetricsStore) -> ThreadingHTTPServer:
    handler = type("TrainDashHandler", (_Handler,), {"store": store})
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
