"""Flask live dashboard on :5050 — same port/convention as the FortyK project.

Reads training_out.txt for [step ...] log lines and charts:
  - win% (live policy)
  - entropy
  - policy loss / value loss
  - KL divergence
  - steps-per-second
  - episode length

Run alongside training:
  nohup python -m threnody_rl.dashboard > dashboard.log 2>&1 &

Then visit http://localhost:5050 — or tunnel from the GPU box.
"""

from __future__ import annotations

import argparse
import re
import os
from dataclasses import dataclass
from pathlib import Path

from flask import Flask, jsonify, render_template_string


LOG_LINE_RE = re.compile(
    r"\[step\s+(?P<step>\d+)\]\s+"
    r"wr_live=(?P<wr>[-\d\.nan]+)\s+"
    r"ep_len=(?P<ep>[-\d\.nan]+)\s+"
    r"pol_loss=(?P<pol>[+\-\d\.]+)\s+"
    r"val_loss=(?P<val>[-\d\.]+)\s+"
    r"ent=(?P<ent>[-\d\.]+)\s+"
    r"kl=(?P<kl>[+\-\d\.]+)\s+"
    r"clip_frac=(?P<clip>[-\d\.]+)\s+"
    r"fps=(?P<fps>[-\d\.]+)"
)


@dataclass
class Point:
    step: int
    wr: float
    ep_len: float
    pol: float
    val: float
    ent: float
    kl: float
    clip: float
    fps: float


def parse_log(path: Path) -> list[Point]:
    if not path.exists():
        return []
    out: list[Point] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = LOG_LINE_RE.search(line)
            if not m:
                continue
            try:
                out.append(Point(
                    step=int(m["step"]),
                    wr=float(m["wr"]),
                    ep_len=float(m["ep"]),
                    pol=float(m["pol"]),
                    val=float(m["val"]),
                    ent=float(m["ent"]),
                    kl=float(m["kl"]),
                    clip=float(m["clip"]),
                    fps=float(m["fps"]),
                ))
            except ValueError:
                continue
    return out


INDEX_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Threnody RL Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body { font-family: system-ui, sans-serif; background: #111; color: #ddd; margin: 0; padding: 20px; }
  h1 { margin-top: 0; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .card { background: #1b1b24; padding: 12px; border-radius: 8px; }
  .summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }
  .stat { background: #1b1b24; padding: 12px; border-radius: 6px; }
  .stat .label { color: #888; font-size: 12px; text-transform: uppercase; }
  .stat .value { font-size: 22px; margin-top: 4px; }
  canvas { max-height: 240px; }
</style>
</head>
<body>
<h1>Threnody RL — live</h1>
<div class="summary" id="summary"></div>
<div class="grid">
  <div class="card"><canvas id="wr"></canvas></div>
  <div class="card"><canvas id="ent"></canvas></div>
  <div class="card"><canvas id="pol"></canvas></div>
  <div class="card"><canvas id="val"></canvas></div>
  <div class="card"><canvas id="kl"></canvas></div>
  <div class="card"><canvas id="ep"></canvas></div>
</div>
<script>
const charts = {};
function mkChart(id, label, color) {
  const el = document.getElementById(id).getContext('2d');
  charts[id] = new Chart(el, {
    type: 'line',
    data: { labels: [], datasets: [{ label, data: [], borderColor: color, tension: 0.2,
            pointRadius: 0, borderWidth: 1.5, backgroundColor: color+'22', fill: true }] },
    options: {
      animation: false,
      plugins: { legend: { labels: { color: '#bbb' } } },
      scales: {
        x: { ticks: { color: '#888' }, grid: { color: '#333' } },
        y: { ticks: { color: '#888' }, grid: { color: '#333' } }
      }
    }
  });
}
mkChart('wr',  'live win%', '#4af');
mkChart('ent', 'entropy',   '#fa4');
mkChart('pol', 'policy loss', '#f44');
mkChart('val', 'value loss',  '#4f4');
mkChart('kl',  'approx KL',   '#f4f');
mkChart('ep',  'episode len', '#ff4');

async function refresh() {
  const r = await fetch('/data');
  const pts = await r.json();
  if (!pts.length) return;
  const steps = pts.map(p => p.step);
  const fields = { wr: 'wr', ent: 'ent', pol: 'pol', val: 'val', kl: 'kl', ep: 'ep_len' };
  for (const [cid, k] of Object.entries(fields)) {
    const c = charts[cid];
    c.data.labels = steps;
    c.data.datasets[0].data = pts.map(p => p[k]);
    c.update();
  }
  const latest = pts[pts.length - 1];
  document.getElementById('summary').innerHTML = `
    <div class="stat"><div class="label">step</div><div class="value">${latest.step.toLocaleString()}</div></div>
    <div class="stat"><div class="label">win% (live)</div><div class="value">${(latest.wr*100).toFixed(1)}%</div></div>
    <div class="stat"><div class="label">entropy</div><div class="value">${latest.ent.toFixed(3)}</div></div>
    <div class="stat"><div class="label">fps</div><div class="value">${latest.fps.toFixed(0)}</div></div>`;
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


def create_app(log_path: str) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(INDEX_HTML)

    @app.route("/data")
    def data():
        pts = parse_log(Path(log_path))
        return jsonify([p.__dict__ for p in pts])

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default="training_out.txt")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5050)
    args = ap.parse_args()
    app = create_app(args.log)
    print(f"[dashboard] reading {args.log}, serving on {args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
