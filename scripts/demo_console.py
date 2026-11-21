#!/usr/bin/env python3
"""Minimal stdlib 1-Click Silent Demo Console + Live SSE Telemetry Strip (http://127.0.0.1:8771).

No Streamlit, zero external web frameworks. Strictly subordinate to physical TonyPi + MuJoCo Studio.
All actions run with ROBOHARNESS_MUTE_AUDIO=1 (zero listening, zero spoken audio).
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SIM_MCP_URL = os.environ.get("ROBOHARNESS_SIM_MCP_URL", "http://127.0.0.1:8765/mcp")
REAL_MCP_URL = os.environ.get("ROBOHARNESS_REAL_MCP_URL", "http://127.0.0.1:8766/mcp")
STUDIO_URL = os.environ.get("ROBOHARNESS_STUDIO_URL", "http://localhost:8080/")
PORT = int(os.environ.get("ROBOHARNESS_DEMO_PORT", "8771"))

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

ACTIONS: dict[str, dict[str, Any]] = {
    "full_demo": {
        "label": "Run Full 60s Demo",
        "desc": "Sim Floss -> Real Floss -> Real Wave -> Backflip 0-Token Refusal -> 5 Sigma",
        "cmd": ["bash", "scripts/demo_60s.sh", "full"],
    },
    "floss_real": {
        "label": "Floss (Real TonyPi)",
        "desc": "SHA-256 upper_body_adapted skill + live Decisions-1 audit (1.6 deg tilt)",
        "cmd": ["bash", "scripts/demo_60s.sh", "floss-real"],
    },
    "wave_real": {
        "label": "Judge Wave (Real TonyPi)",
        "desc": "SHA-256 full_body wave + live ER-2 spatial check + Decisions-1 audit",
        "cmd": ["bash", "scripts/demo_60s.sh", "wave-real"],
    },
    "orange_real": {
        "label": "Orange Justice (Real TonyPi)",
        "desc": "High-energy upper_body_adapted dance on real TonyPi",
        "cmd": ["bash", "scripts/demo_60s.sh", "orange-real"],
    },
    "floss_sim": {
        "label": "Floss (MuJoCo 4.0.1 Sim)",
        "desc": "Execute live on MuJoCo 4.0.1 Studio (:8080)",
        "cmd": ["bash", "scripts/demo_60s.sh", "floss-sim"],
    },
    "backflip_gate": {
        "label": "Backflip Safety Gate (0-Token)",
        "desc": "Live gemini-decision-1 refusal on real TonyPi camera+IMU (0 servos moved)",
        "cmd": ["bash", "scripts/demo_60s.sh", "backflip-gate"],
    },
    "webcam_mimic": {
        "label": "Live Webcam Mimic (5s)",
        "desc": "Record 5s webcam -> Hydrogen -> Managed Agent Neon -> Decisions-1 -> Real TonyPi",
        "cmd": ["bash", "scripts/demo_60s.sh", "webcam"],
    },
    "preflight": {
        "label": "5/5 Preflight Check",
        "desc": "Verify all 4 Gemini models + live physical TonyPi MCP",
        "cmd": ["bash", "scripts/demo_60s.sh", "check"],
    },
    "unlatch": {
        "label": "Unlatch STOP",
        "desc": "Clear tonyd stop latch over SSH",
        "cmd": [
            "ssh",
            "-o",
            "ConnectTimeout=5",
            "pi@tonypi.local",
            '~/tonypi/.venv/bin/python -c \'from tonypi.tonyd.client import TonydClient; c = TonydClient(); print(c.call("unlatch")); c.close()\'',
        ],
    },
    "get_up": {
        "label": "Stand Up (get_up)",
        "desc": "Run TonyPi get_up recovery over SSH if knocked over",
        "cmd": [
            "ssh",
            "-o",
            "ConnectTimeout=5",
            "pi@tonypi.local",
            '~/tonypi/.venv/bin/python -c \'from tonypi.tonyd.client import TonydClient; c = TonydClient(); print(c.call("skill", name="get_up", count=1, wait=True)); c.close()\'',
        ],
    },
}


class DemoRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._current_action: str | None = None
        self._running = False
        self._history: deque[dict[str, Any]] = deque(maxlen=250)
        self._subs: set[queue.Queue[dict[str, Any]]] = set()

    def _broadcast(self, event: str, payload: dict[str, Any]) -> None:
        msg = {"event": event, "ts": time.strftime("%H:%M:%S"), **payload}
        with self._lock:
            self._history.append(msg)
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=300)
        with self._lock:
            for item in self._history:
                q.put_nowait(item)
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._subs.discard(q)

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "action": self._current_action,
                "sim_url": SIM_MCP_URL,
                "real_url": REAL_MCP_URL,
                "studio_url": STUDIO_URL,
                "audio_muted": True,
                "voice_listening": False,
            }

    def stop_current(self) -> None:
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            self._broadcast("log", {"line": "[stopped by operator]"})

    def start(self, action_id: str) -> tuple[bool, str]:
        if action_id not in ACTIONS:
            return False, f"Unknown action: {action_id}"
        with self._lock:
            if self._running:
                return False, f"Busy running {self._current_action}"
            self._running = True
            self._current_action = action_id
        t = threading.Thread(target=self._run_worker, args=(action_id,), daemon=True)
        t.start()
        return True, f"Started {action_id}"

    def _run_worker(self, action_id: str) -> None:
        spec = ACTIONS[action_id]
        env = os.environ.copy()
        env["ROBOHARNESS_MUTE_AUDIO"] = "1"
        env["ROBOHARNESS_SIM_MCP_URL"] = SIM_MCP_URL
        env["ROBOHARNESS_REAL_MCP_URL"] = REAL_MCP_URL
        env["PYTHONUNBUFFERED"] = "1"
        env["COLUMNS"] = "100"

        self._broadcast("status", {"running": True, "action": action_id, "label": spec["label"]})
        self._broadcast("log", {"line": f"=== {spec['label']} (audio muted, mic disabled) ==="})
        t0 = time.perf_counter()
        rc = -1
        try:
            proc = subprocess.Popen(
                spec["cmd"],
                cwd=str(REPO_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with self._lock:
                self._proc = proc
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                clean = ANSI_RE.sub("", raw_line).rstrip("\r\n")
                if clean:
                    self._broadcast("log", {"line": clean})
            rc = proc.wait()
        except Exception as exc:  # noqa: BLE001
            self._broadcast("log", {"line": f"[error] {exc}"})
        finally:
            elapsed = time.perf_counter() - t0
            with self._lock:
                self._proc = None
                self._running = False
                self._current_action = None
            status_word = "DONE" if rc == 0 else f"EXIT {rc}"
            self._broadcast("log", {"line": f"=== {spec['label']}: {status_word} ({elapsed:.1f}s) ==="})
            self._broadcast(
                "status",
                {"running": False, "action": None, "last_action": action_id, "rc": rc, "elapsed_s": round(elapsed, 1)},
            )


RUNNER = DemoRunner()

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RoboHarness — 1-Click Live Demo Console</title>
<style>
  :root {
    --bg: #0b0f17; --card: #121826; --border: #232d42; --text: #e6edf7; --muted: #8ea0bd;
    --green: #22c55e; --cyan: #38bdf8; --amber: #f59e0b; --red: #ef4444; --purple: #a78bfa;
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 12px 14px; background: var(--bg); color: var(--text); font: 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  .top { display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
  .brand { font-size: 16px; font-weight: 700; letter-spacing: 0.2px; display: flex; align-items: center; gap: 8px; }
  .pills { display: flex; gap: 6px; flex-wrap: wrap; }
  .pill { padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; border: 1px solid var(--border); background: var(--card); color: var(--muted); }
  .pill.ok { color: var(--green); border-color: rgba(34,197,94,.45); }
  .pill.warn { color: var(--amber); border-color: rgba(245,158,11,.45); }
  .pill.cyan { color: var(--cyan); border-color: rgba(56,189,248,.45); }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 8px; margin-bottom: 10px; }
  button {
    appearance: none; border: 1px solid var(--border); background: var(--card); color: var(--text);
    padding: 9px 10px; border-radius: 8px; cursor: pointer; text-align: left; transition: transform .06s, border-color .15s, background .15s;
  }
  button:hover:not(:disabled) { border-color: var(--cyan); background: #172033; }
  button:active:not(:disabled) { transform: scale(0.98); }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  button.primary { background: linear-gradient(135deg, #0ea5e9, #2563eb); border-color: #38bdf8; color: #fff; grid-column: span 2; }
  button.primary:hover:not(:disabled) { filter: brightness(1.08); }
  button.danger { border-color: rgba(239,68,68,.55); color: #fecaca; }
  button.safety { border-color: rgba(245,158,11,.55); }
  .btitle { font-weight: 700; font-size: 13px; display: block; }
  .bsub { font-size: 11px; color: var(--muted); display: block; margin-top: 2px; }
  button.primary .bsub { color: rgba(255,255,255,.88); }
  .bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 11px; color: var(--muted); }
  .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 10px; }
  .mcard { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 7px 9px; }
  .mval { font-size: 15px; font-weight: 700; color: var(--green); }
  .mlbl { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .4px; }
  pre#log {
    margin: 0; padding: 10px 12px; background: #070a10; border: 1px solid var(--border); border-radius: 8px;
    height: 300px; overflow-y: auto; font: 12px/1.38 ui-monospace, SFMono-Regular, Menlo, monospace; color: #dbeafe;
    white-space: pre-wrap; word-break: break-word;
  }
  a { color: var(--cyan); text-decoration: none; }
</style>
</head>
<body>
  <div class="top">
    <div class="brand">🤖 RoboHarness <span class="pill cyan">Gemini 4 + Managed Agents + Decisions-1</span></div>
    <div class="pills">
      <span class="pill ok">MIC: OFF</span>
      <span class="pill ok">AUDIO: MUTED</span>
      <span class="pill cyan">Real TonyPi: :8766</span>
      <span class="pill cyan"><a href="http://localhost:8080/" target="_blank">MuJoCo 4.0.1 Studio: :8080 ↗</a></span>
      <span id="runBadge" class="pill ok">IDLE — READY</span>
    </div>
  </div>

  <div class="metrics">
    <div class="mcard"><div class="mval">33.3% → 91.7%</div><div class="mlbl">5 Sigma Safety Pass (+58.3 pp)</div></div>
    <div class="mcard"><div class="mval">6 → 0 Falls</div><div class="mlbl">Real MuJoCo 3D Physics</div></div>
    <div class="mcard"><div class="mval">2/2 (100%)</div><div class="mlbl">0.25x Low-Friction Slip Survival</div></div>
    <div class="mcard"><div class="mval">0 Tokens</div><div class="mlbl">gemini-decision-1 Safety Gate</div></div>
  </div>

  <div class="grid">
    <button class="primary" data-act="full_demo"><span class="btitle">▶ Run Full 60s Demo Sequence</span><span class="bsub">Sim Floss (:8080) → Real Floss (1.6° tilt) → Real Wave → Backflip 0-Token Refusal → 5 Sigma</span></button>
    <button data-act="floss_real"><span class="btitle">🕺 Floss (Real TonyPi)</span><span class="bsub">SHA-256 signed • 1.6° peak tilt</span></button>
    <button data-act="wave_real"><span class="btitle">👋 Judge Wave (Real TonyPi)</span><span class="bsub">Spatial ER-2 check • 5.8° tilt</span></button>
    <button data-act="orange_real"><span class="btitle">⚡ Orange Justice (Real)</span><span class="bsub">Upper-body adapted dance</span></button>
    <button data-act="floss_sim"><span class="btitle">🖥️ Floss (MuJoCo Studio)</span><span class="bsub">Move 3D robot on :8080</span></button>
    <button class="safety" data-act="backflip_gate"><span class="btitle">🚫 Backflip 0-Token Gate</span><span class="bsub">Live Decisions-1 safe refusal</span></button>
    <button data-act="webcam_mimic"><span class="btitle">🎬 Live Webcam Mimic (5s)</span><span class="bsub">Record → Forge → Real TonyPi</span></button>
    <button data-act="preflight"><span class="btitle">✅ 5/5 Preflight Check</span><span class="bsub">4 Gemini models + TonyPi MCP</span></button>
    <button class="danger" data-act="unlatch"><span class="btitle">🔓 Unlatch STOP</span><span class="bsub">Clear tonyd latch if needed</span></button>
    <button class="danger" data-act="get_up"><span class="btitle">⬆️ Stand Up</span><span class="bsub">Run get_up recovery</span></button>
  </div>

  <div class="bar">
    <span>Live Pipeline Stream (SSE — zero audio, zero listening)</span>
    <span>
      <button id="stopBtn" style="padding:3px 8px;font-size:11px;display:inline-block" class="danger">■ Stop Current</button>
      <button id="clearBtn" style="padding:3px 8px;font-size:11px;display:inline-block">Clear Log</button>
    </span>
  </div>
  <pre id="log">Ready. Click "▶ Run Full 60s Demo Sequence" or any single button above.\n</pre>

<script>
const logEl = document.getElementById('log');
const badge = document.getElementById('runBadge');
const actBtns = [...document.querySelectorAll('button[data-act]')];

function setRunning(running, label) {
  actBtns.forEach(b => b.disabled = running);
  badge.textContent = running ? ('RUNNING: ' + (label || '...')) : 'IDLE — READY';
  badge.className = 'pill ' + (running ? 'warn' : 'ok');
}

function appendLine(ts, line) {
  logEl.textContent += `[${ts}] ${line}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

actBtns.forEach(btn => {
  btn.addEventListener('click', async () => {
    const act = btn.dataset.act;
    const resp = await fetch('/api/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: act})
    });
    const data = await resp.json();
    if (!data.ok) appendLine('--:--:--', '[busy] ' + data.message);
  });
});

document.getElementById('stopBtn').addEventListener('click', () => fetch('/api/stop', {method: 'POST'}));
document.getElementById('clearBtn').addEventListener('click', () => { logEl.textContent = ''; });

const es = new EventSource('/events');
es.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.event === 'log') appendLine(msg.ts, msg.line);
  if (msg.event === 'status') setRunning(!!msg.running, msg.label || msg.action);
};
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, code: int, body: dict[str, Any]) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            raw = HTML_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, {"ok": True, **RUNNER.state()})
            return
        if self.path == "/events":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = RUNNER.subscribe()
            try:
                while True:
                    try:
                        msg = q.get(timeout=15)
                        payload = f"data: {json.dumps(msg)}\n\n".encode()
                    except queue.Empty:
                        payload = b": keepalive\n\n"
                    self.wfile.write(payload)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                RUNNER.unsubscribe(q)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path == "/api/stop":
            RUNNER.stop_current()
            self._send_json(HTTPStatus.OK, {"ok": True})
            return
        if self.path == "/api/run":
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            ok, msg = RUNNER.start(str(body.get("action", "")))
            self._send_json(HTTPStatus.OK if ok else HTTPStatus.CONFLICT, {"ok": ok, "message": msg})
            return
        self.send_error(HTTPStatus.NOT_FOUND)


class QuietServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    server = QuietServer(("127.0.0.1", PORT), Handler)
    print(f"RoboHarness 1-Click Silent Demo Console listening at http://127.0.0.1:{PORT}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
