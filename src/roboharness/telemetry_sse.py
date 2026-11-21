"""Minimal stdlib ThreadingHTTPServer live SSE telemetry strip for RoboHarness.

Zero Streamlit, zero external web frameworks, zero CDN dependencies. Provides a compact
dark-theme horizontal status strip strictly subordinate to the physical TonyPi robot.
"""

from __future__ import annotations

import json
import queue
import threading
from collections import deque
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any, Self

from roboharness.config import (
    DECISION_CONFIDENCE_THRESHOLD,
    DECISION_MODEL,
    FORGE_AGENT_ID,
    FORGE_MODEL,
    MAX_SERVO_TEMP_C,
    MAX_TORSO_TILT_DEG,
    MIN_BATTERY_V,
    MOTION_MODEL,
    SPATIAL_MODEL,
)


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class TelemetryHub:
    """Thread-safe in-memory pub/sub hub with a 50-event ring buffer for SSE clients."""

    def __init__(self, max_history: int = 50) -> None:
        self._max_history = max_history
        self._lock = threading.Lock()
        self._history: deque[dict[str, Any]] = deque(maxlen=max_history)
        self._subscribers: set[queue.Queue[dict[str, Any] | None]] = set()
        self._seq: int = 0
        self._closed: bool = False

    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Publish an event to the ring buffer and all connected SSE subscribers."""
        with self._lock:
            self._seq += 1
            envelope: dict[str, Any] = {
                "id": self._seq,
                "event": str(event_type),
                "ts": datetime.now(UTC).isoformat(),
                "data": dict(data),
            }
            self._history.append(envelope)
            subscribers = list(self._subscribers)

        for sub in subscribers:
            try:
                sub.put_nowait(envelope)
            except queue.Full:
                try:
                    sub.get_nowait()
                except queue.Empty:
                    pass
                try:
                    sub.put_nowait(envelope)
                except queue.Full:
                    pass

    def history(self) -> list[dict[str, Any]]:
        """Return a snapshot list of the most recent buffered events (up to 50)."""
        with self._lock:
            return list(self._history)

    def subscribe(
        self,
        *,
        replay_history: bool = True,
        maxsize: int = 256,
    ) -> queue.Queue[dict[str, Any] | None]:
        """Register a new subscriber queue, optionally seeded with buffered history."""
        q: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=maxsize)
        with self._lock:
            if replay_history:
                for item in self._history:
                    try:
                        q.put_nowait(item)
                    except queue.Full:
                        break
            if self._closed:
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass
            else:
                self._subscribers.add(q)
        return q

    def unsubscribe(self, q: queue.Queue[dict[str, Any] | None]) -> None:
        """Remove a subscriber queue."""
        with self._lock:
            self._subscribers.discard(q)

    def reopen(self) -> None:
        """Mark hub open for new subscribers if reused across server restarts."""
        with self._lock:
            self._closed = False

    def close(self) -> None:
        """Signal all active subscribers to terminate promptly."""
        with self._lock:
            self._closed = True
            subscribers = list(self._subscribers)
            self._subscribers.clear()
        for sub in subscribers:
            try:
                sub.put_nowait(None)
            except queue.Full:
                try:
                    sub.get_nowait()
                except queue.Empty:
                    pass
                try:
                    sub.put_nowait(None)
                except queue.Full:
                    pass


def format_sse_frame(envelope: dict[str, Any]) -> bytes:
    """Format a TelemetryHub event envelope as a standard `text/event-stream` frame."""
    seq = envelope.get("id", 0)
    event_type = str(envelope.get("event", "message")).replace("\n", " ").replace("\r", "")
    payload = json.dumps(envelope, separators=(",", ":"))
    frame = f"id: {seq}\nevent: {event_type}\ndata: {payload}\n\n"
    return frame.encode("utf-8")


def _render_strip_html() -> str:
    """Render the self-contained dark-theme horizontal telemetry strip HTML."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>RoboHarness Live Telemetry</title>
<style>
  :root {{
    --bg: #090d14;
    --panel: #111827;
    --border: #1f2937;
    --text: #e5e7eb;
    --muted: #94a3b8;
    --cyan: #38bdf8;
    --green: #34d399;
    --amber: #fbbf24;
    --red: #f87171;
    --purple: #a78bfa;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 12px;
    padding: 8px 12px;
  }}
  .strip {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px 12px;
  }}
  .brand {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding-right: 10px;
    border-right: 1px solid var(--border);
  }}
  .dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--amber);
    box-shadow: 0 0 8px var(--amber);
  }}
  .dot.live {{ background: var(--green); box-shadow: 0 0 8px var(--green); }}
  .dot.err {{ background: var(--red); box-shadow: 0 0 8px var(--red); }}
  .title {{ font-weight: 700; letter-spacing: 0.06em; color: var(--cyan); }}
  .stage {{
    background: #1e293b;
    color: var(--cyan);
    border: 1px solid #334155;
    border-radius: 999px;
    padding: 2px 9px;
    font-weight: 700;
    text-transform: uppercase;
    font-size: 11px;
  }}
  .stage.aborted {{ color: var(--red); border-color: var(--red); }}
  .models {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding-right: 10px;
    border-right: 1px solid var(--border);
  }}
  .pill {{
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: #0f172a;
    color: var(--muted);
    font-size: 11px;
    transition: all 0.15s ease;
  }}
  .pill.active {{ border-color: var(--cyan); color: #e0f2fe; background: #0c2438; }}
  .pill.done {{ border-color: var(--green); color: #d1fae5; }}
  .pill.error {{ border-color: var(--red); color: #fee2e2; }}
  .gauges {{
    display: flex;
    gap: 10px;
    padding-right: 10px;
    border-right: 1px solid var(--border);
  }}
  .gauge {{ display: flex; flex-direction: column; min-width: 68px; }}
  .g-label {{ color: var(--muted); font-size: 10px; text-transform: uppercase; }}
  .g-val {{ font-weight: 700; font-size: 13px; color: var(--green); }}
  .g-val.warn {{ color: var(--amber); }}
  .g-val.danger {{ color: var(--red); }}
  .decision {{
    display: flex;
    flex-direction: column;
    gap: 3px;
    min-width: 170px;
    padding-right: 10px;
    border-right: 1px solid var(--border);
  }}
  .dec-head {{ display: flex; justify-content: space-between; font-size: 10px; color: var(--muted); }}
  .dec-action {{ color: var(--purple); font-weight: 700; }}
  .bar-wrap {{
    position: relative;
    height: 8px;
    background: #1e293b;
    border-radius: 4px;
    overflow: hidden;
  }}
  .bar-fill {{
    height: 100%;
    width: 0%;
    background: var(--green);
    transition: width 0.2s ease;
  }}
  .bar-fill.low {{ background: var(--amber); }}
  .bar-thresh {{
    position: absolute;
    left: {int(DECISION_CONFIDENCE_THRESHOLD * 100)}%;
    top: 0; bottom: 0;
    width: 2px;
    background: #f8fafc;
    opacity: 0.7;
  }}
  .log {{
    flex: 1 1 200px;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .log strong {{ color: var(--text); }}
</style>
</head>
<body>
  <div class="strip" role="status" aria-live="polite">
    <div class="brand">
      <span id="conn-dot" class="dot" title="SSE connection"></span>
      <span class="title">ROBOHARNESS</span>
      <span id="stage-badge" class="stage">IDLE</span>
    </div>

    <div class="models">
      <span id="m-hydrogen" class="pill">{MOTION_MODEL}</span>
      <span id="m-forge" class="pill">{FORGE_AGENT_ID} ({FORGE_MODEL})</span>
      <span id="m-decision" class="pill">{DECISION_MODEL}</span>
      <span id="m-spatial" class="pill">{SPATIAL_MODEL}</span>
    </div>

    <div class="gauges">
      <div class="gauge">
        <span class="g-label">Battery (&ge;{MIN_BATTERY_V:.1f}V)</span>
        <span id="g-bat" class="g-val">--.-V</span>
      </div>
      <div class="gauge">
        <span class="g-label">Temp (&lt;{MAX_SERVO_TEMP_C:.0f}C)</span>
        <span id="g-tmp" class="g-val">--.-C</span>
      </div>
      <div class="gauge">
        <span class="g-label">Tilt (&lt;{MAX_TORSO_TILT_DEG:.0f}&deg;)</span>
        <span id="g-tilt" class="g-val">--.-&deg;</span>
      </div>
    </div>

    <div class="decision">
      <div class="dec-head">
        <span>DECISION CONF <span id="dec-pct">--%</span></span>
        <span id="dec-act" class="dec-action">--</span>
      </div>
      <div class="bar-wrap">
        <div id="dec-bar" class="bar-fill"></div>
        <div class="bar-thresh" title="0.70 auto-branch threshold"></div>
      </div>
    </div>

    <div id="event-log" class="log">Waiting for telemetry stream...</div>
  </div>

<script>
(function() {{
  const dot = document.getElementById("conn-dot");
  const stageBadge = document.getElementById("stage-badge");
  const gBat = document.getElementById("g-bat");
  const gTmp = document.getElementById("g-tmp");
  const gTilt = document.getElementById("g-tilt");
  const decPct = document.getElementById("dec-pct");
  const decAct = document.getElementById("dec-act");
  const decBar = document.getElementById("dec-bar");
  const logEl = document.getElementById("event-log");

  const modelEls = {{
    "{MOTION_MODEL}": document.getElementById("m-hydrogen"),
    "hydrogen": document.getElementById("m-hydrogen"),
    "{FORGE_AGENT_ID}": document.getElementById("m-forge"),
    "{FORGE_MODEL}": document.getElementById("m-forge"),
    "forge": document.getElementById("m-forge"),
    "{DECISION_MODEL}": document.getElementById("m-decision"),
    "decision": document.getElementById("m-decision"),
    "{SPATIAL_MODEL}": document.getElementById("m-spatial"),
    "spatial": document.getElementById("m-spatial"),
  }};

  function setGauge(el, text, cls) {{
    el.textContent = text;
    el.className = "g-val" + (cls ? " " + cls : "");
  }}

  function handleEnvelope(env) {{
    const ev = env.event || "message";
    const d = env.data || {{}};

    if (d.stage) {{
      stageBadge.textContent = String(d.stage).toUpperCase();
      stageBadge.className = "stage" + (String(d.stage).toLowerCase().includes("abort") ? " aborted" : "");
    }}
    if (d.model && modelEls[d.model]) {{
      const st = d.status || "active";
      modelEls[d.model].className = "pill " + st;
    }}
    if (typeof d.battery_v === "number") {{
      const cls = d.battery_v < {MIN_BATTERY_V} ? "danger" : (d.battery_v < 11.2 ? "warn" : "");
      setGauge(gBat, d.battery_v.toFixed(2) + "V", cls);
    }}
    if (typeof d.max_servo_temp_c === "number") {{
      const cls = d.max_servo_temp_c >= {MAX_SERVO_TEMP_C} ? "danger" : (d.max_servo_temp_c >= 60 ? "warn" : "");
      setGauge(gTmp, d.max_servo_temp_c.toFixed(1) + "C", cls);
    }}
    if (typeof d.tilt_deg === "number" || typeof d.torso_pitch_deg === "number") {{
      const tilt = typeof d.tilt_deg === "number"
        ? d.tilt_deg
        : Math.hypot(d.torso_pitch_deg || 0, d.torso_roll_deg || 0);
      const cls = tilt >= {MAX_TORSO_TILT_DEG} ? "danger" : (tilt >= 22 ? "warn" : "");
      setGauge(gTilt, tilt.toFixed(1) + "\\u00b0", cls);
    }}
    if (typeof d.min_confidence === "number" || typeof d.confidence === "number") {{
      const conf = typeof d.min_confidence === "number" ? d.min_confidence : d.confidence;
      const pct = Math.max(0, Math.min(100, Math.round(conf * 100)));
      decPct.textContent = pct + "%";
      decBar.style.width = pct + "%";
      decBar.className = "bar-fill" + (conf < {DECISION_CONFIDENCE_THRESHOLD} ? " low" : "");
    }}
    if (d.effective_action || d.route) {{
      decAct.textContent = String(d.effective_action || d.route);
    }}

    const summary = d.message || d.summary || d.abort_reason || JSON.stringify(d);
    logEl.textContent = "";
    const tag = document.createElement("strong");
    tag.textContent = "[" + ev + "] ";
    logEl.appendChild(tag);
    logEl.appendChild(document.createTextNode(String(summary).slice(0, 140)));
  }}

  const es = new EventSource("/events");
  es.onopen = () => {{ dot.className = "dot live"; }};
  es.onerror = () => {{ dot.className = "dot err"; }};

  const eventNames = [
    "message", "stage", "model", "telemetry", "decision",
    "spatial", "execution", "safety", "log", "status", "error"
  ];
  eventNames.forEach((name) => {{
    es.addEventListener(name, (evt) => {{
      try {{
        handleEnvelope(JSON.parse(evt.data));
      }} catch (_e) {{}}
    }});
  }});
}})();
</script>
</body>
</html>
"""


class TelemetrySSEServer:
    """Lightweight stdlib ThreadingHTTPServer serving the live SSE telemetry strip."""

    def __init__(
        self,
        hub: TelemetryHub,
        host: str = "127.0.0.1",
        port: int = 8770,
    ) -> None:
        self.hub = hub
        self.host = host
        self.port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._url: str | None = None
        self._html_bytes: bytes = _render_strip_html().encode("utf-8")

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    @property
    def url(self) -> str | None:
        """Base HTTP URL once started, else None."""
        return self._url

    def start(self) -> str:
        """Start the background HTTP/SSE server and return its `http://host:port` URL."""
        if self._httpd is not None and self._url is not None:
            return self._url

        self.hub.reopen()
        hub_ref = self.hub
        html_bytes = self._html_bytes

        class _SSEHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/health":
                    body = json.dumps(
                        {"ok": True, "buffered_events": len(hub_ref.history())}
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html_bytes)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(html_bytes)
                    return

                if path == "/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()

                    sub_q = hub_ref.subscribe(replay_history=True)
                    try:
                        self.wfile.write(b": connected\n\n")
                        self.wfile.flush()
                        while True:
                            try:
                                item = sub_q.get(timeout=15.0)
                            except queue.Empty:
                                self.wfile.write(b": keepalive\n\n")
                                self.wfile.flush()
                                continue
                            if item is None:
                                break
                            self.wfile.write(format_sse_frame(item))
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                    finally:
                        hub_ref.unsubscribe(sub_q)
                    return

                body = json.dumps({"error": "not_found"}).encode("utf-8")
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._httpd = _QuietThreadingHTTPServer((self.host, self.port), _SSEHandler)
        bound_port = self._httpd.server_address[1]
        self._url = f"http://{self.host}:{bound_port}"
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
            name="roboharness-telemetry-sse",
        )
        self._thread.start()
        return self._url

    def stop(self) -> None:
        """Gracefully close all SSE streams and stop the HTTP server."""
        self.hub.close()
        httpd = self._httpd
        thread = self._thread
        self._httpd = None
        self._thread = None
        self._url = None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
