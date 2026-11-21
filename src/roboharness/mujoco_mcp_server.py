"""Clean-room MuJoCo Streamable HTTP MCP server for the Hiwonder TonyPi model.

Loads only the public MIT `pab47/TonyPi` MuJoCo XML asset (`scene.xml`, overridable via
`ROBOHARNESS_MUJOCO_XML`) using `mujoco >= 3.x` and exposes a Streamable HTTP MCP
endpoint (`POST /mcp`) plus live camera endpoints (`GET /frame.jpg`, `GET /stream.mjpeg`).

Offscreen OpenGL rendering runs exclusively on a single dedicated worker thread to
satisfy macOS CGL context thread-affinity requirements safely under `ThreadingHTTPServer`.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import queue
import threading
import time
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import httpx
from PIL import Image

from roboharness.schemas import JOINTS_BY_ID, JOINTS_BY_NAME, NEUTRAL_POSE, TONYPI_JOINTS

DEFAULT_MUJOCO_XML: Path = (
    Path.home() / ".local" / "share" / "tonypi" / "simulation" / "PAB47-TonyPi" / "model" / "scene.xml"
)
DEG_PER_PULSE: float = 0.24
RAD_PER_PULSE: float = math.radians(DEG_PER_PULSE)
RENDER_WIDTH: int = 480
RENDER_HEIGHT: int = 360


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def resolve_mujoco_xml_path(xml_path: Path | str | None = None) -> Path:
    """Resolve the TonyPi MuJoCo `scene.xml` asset path."""
    if xml_path is not None:
        return Path(xml_path).expanduser()
    env_override = os.environ.get("ROBOHARNESS_MUJOCO_XML", "").strip()
    if env_override:
        return Path(env_override).expanduser()
    return DEFAULT_MUJOCO_XML


def pulse_to_rad(pulse: float, low: float, high: float) -> float:
    """Convert TonyPi servo pulse `0..1000` to radians clamped to actuator range."""
    rad = (float(pulse) - 500.0) * RAD_PER_PULSE
    return max(float(low), min(float(high), rad))


def rad_to_pulse(rad: float) -> int:
    """Convert actuator radian angle back to canonical `0..1000` servo pulse."""
    return max(0, min(1000, round(500.0 + math.degrees(float(rad)) / DEG_PER_PULSE)))


def _quat_to_euler_and_tilt_deg(quat_wxyz: Any) -> tuple[float, float, float]:
    """Compute `(pitch_deg, roll_deg, total_tilt_deg)` from torso `[w, x, y, z]` quaternion."""
    w, x, y, z = (float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]))
    # Z-component of body up vector in world frame
    zz = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
    tilt_deg = math.degrees(math.acos(zz))

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll_deg = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch_deg = math.degrees(math.asin(sinp))
    return round(pitch_deg, 2), round(roll_deg, 2), round(tilt_deg, 2)


def _normalize_servo_targets(raw: Any) -> dict[str, int]:
    """Normalize canonical joint names or string/int servo IDs (`'1'..'16'`) to joint->pulse."""
    out: dict[str, int] = {}
    if isinstance(raw, dict):
        for raw_key, raw_val in raw.items():
            try:
                pulse = round(float(raw_val))
            except (TypeError, ValueError):
                continue
            key_s = str(raw_key).strip()
            if key_s in JOINTS_BY_NAME:
                out[key_s] = pulse
            elif key_s.isdigit() and int(key_s) in JOINTS_BY_ID:
                out[JOINTS_BY_ID[int(key_s)].name] = pulse
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            val = item.get("pulse", item.get("position"))
            if val is None:
                continue
            try:
                pulse = round(float(val))
            except (TypeError, ValueError):
                continue
            name = item.get("name")
            if isinstance(name, str) and name in JOINTS_BY_NAME:
                out[name] = pulse
            elif item.get("id") is not None:
                try:
                    spec = JOINTS_BY_ID.get(int(item["id"]))
                except (TypeError, ValueError):
                    spec = None
                if spec is not None:
                    out[spec.name] = pulse
    return out


class _DedicatedRenderWorker:
    """Single-thread offscreen MuJoCo renderer keeping OpenGL context affinity intact on macOS."""

    def __init__(self, model: Any, *, width: int = RENDER_WIDTH, height: int = RENDER_HEIGHT) -> None:
        self._model = model
        self._width = width
        self._height = height
        self._queue: queue.Queue[tuple[Any, Future[bytes]] | None] = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="mujoco-offscreen-renderer",
        )
        self._thread.start()

    def _loop(self) -> None:
        import mujoco

        renderer = mujoco.Renderer(self._model, height=self._height, width=self._width)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.0, 0.0, 0.22]
        cam.distance = 0.85
        cam.azimuth = 135.0
        cam.elevation = -15.0
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                data, fut = item
                if fut.cancelled():
                    continue
                try:
                    renderer.update_scene(data, camera=cam)
                    rgb = renderer.render()
                    buf = io.BytesIO()
                    Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
                    fut.set_result(buf.getvalue())
                except Exception as exc:  # noqa: BLE001
                    fut.set_exception(exc)
        finally:
            renderer.close()

    def render_jpeg(self, data: Any, *, timeout: float = 10.0) -> bytes:
        """Render `data` on the dedicated OpenGL thread and return JPEG bytes."""
        if self._closed:
            raise RuntimeError("MuJoCo render worker has been closed.")
        fut: Future[bytes] = Future()
        self._queue.put((data, fut))
        return fut.result(timeout=timeout)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)


class MujocoTonypiMCPServer:
    """Clean-room MuJoCo 3.x physics + Streamable HTTP MCP server for TonyPi."""

    def __init__(
        self,
        *,
        xml_path: Path | str | None = None,
        friction: float = 1.0,
        session_id: str = "mujoco-tonypi-session-001",
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.xml_path = resolve_mujoco_xml_path(xml_path)
        if not self.xml_path.is_file():
            raise FileNotFoundError(
                f"MuJoCo TonyPi scene XML not found at '{self.xml_path}'. "
                "Set ROBOHARNESS_MUJOCO_XML or install the public MIT pab47/TonyPi model."
            )

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.session_id = session_id

        self._actuator_ids: dict[str, int] = {}
        for i in range(self.model.nu):
            act_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if act_name:
                self._actuator_ids[str(act_name)] = i

        missing = [j.name for j in TONYPI_JOINTS if j.name not in self._actuator_ids]
        if missing:
            raise ValueError(f"MuJoCo model missing expected TonyPi actuators: {missing}")

        self._floor_geom_id = 0
        self._base_floor_friction = float(self.model.geom_friction[self._floor_geom_id, 0])
        self._friction_scale = float(friction)
        self._battery_v = 11.7
        self._max_servo_temp_c = 36.5

        self.spoken_texts: list[str] = []
        self.call_log: list[dict[str, Any]] = []

        self.lock = threading.Lock()
        self._frame_cond = threading.Condition()
        self._latest_jpeg: bytes = b""
        self._frame_seq: int = 0
        self._closed: bool = False
        self._http_running: bool = False

        self._renderer = _DedicatedRenderWorker(self.model)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._base_url: str | None = None

        with self.lock:
            self._reset_locked(friction=self._friction_scale)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    @property
    def mcp_url(self) -> str | None:
        return f"{self._base_url}/mcp" if self._base_url else None

    @property
    def stream_url(self) -> str | None:
        return f"{self._base_url}/stream.mjpeg" if self._base_url else None

    @property
    def frame_url(self) -> str | None:
        return f"{self._base_url}/frame.jpg" if self._base_url else None

    def latest_jpeg(self) -> bytes:
        """Return the most recently rendered 480x360 MuJoCo JPEG frame."""
        with self._frame_cond:
            return self._latest_jpeg

    def _update_latest_jpeg_locked(self) -> bytes:
        jpeg = self._renderer.render_jpeg(self.data)
        with self._frame_cond:
            self._latest_jpeg = jpeg
            self._frame_seq += 1
            self._frame_cond.notify_all()
        return jpeg

    def _current_servo_positions_locked(self) -> dict[str, int]:
        positions: dict[str, int] = dict(NEUTRAL_POSE)
        for spec in TONYPI_JOINTS:
            act_id = self._actuator_ids[spec.name]
            positions[spec.name] = rad_to_pulse(float(self.data.ctrl[act_id]))
        return positions

    def _status_locked(self) -> dict[str, Any]:
        pitch_deg, roll_deg, tilt_deg = _quat_to_euler_and_tilt_deg(self.data.qpos[3:7])
        base_z = round(float(self.data.qpos[2]), 4)
        standing = bool(tilt_deg < 45.0 and base_z > -0.10)
        return {
            "battery_v": round(self._battery_v, 3),
            "max_servo_temp_c": round(self._max_servo_temp_c, 2),
            "torso_pitch_deg": pitch_deg,
            "torso_roll_deg": roll_deg,
            "tilt_deg": tilt_deg,
            "standing": standing,
            "base_z": base_z,
            "contacts": int(self.data.ncon),
            "servo_positions": self._current_servo_positions_locked(),
            "sim": True,
            "friction": round(self._friction_scale, 3),
        }

    def _reset_locked(self, *, friction: float = 1.0) -> dict[str, Any]:
        mujoco = self._mujoco
        self._friction_scale = max(0.01, float(friction))
        self.model.geom_friction[self._floor_geom_id, 0] = (
            self._base_floor_friction * self._friction_scale
        )
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self._battery_v = 11.7
        self._max_servo_temp_c = 36.5
        mujoco.mj_forward(self.model, self.data)

        settle_steps = max(1, round(0.3 / float(self.model.opt.timestep)))
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)

        self._update_latest_jpeg_locked()
        return self._status_locked()

    def _play_frames_locked(
        self,
        frames: list[dict[str, Any]],
        *,
        hold_s: float = 0.5,
        guard_tilt_deg: float | None = 30.0,
        render_final: bool = True,
    ) -> dict[str, Any]:
        mujoco = self._mujoco
        dt = float(self.model.opt.timestep)
        render_interval_steps = max(1, round(0.08 / dt))

        _, _, init_tilt = _quat_to_euler_and_tilt_deg(self.data.qpos[3:7])
        peak_tilt = init_tilt
        max_contacts = int(self.data.ncon)
        frames_executed = 0
        aborted = False
        abort_reason: str | None = None
        fell = False
        steps_since_render = 0

        for idx, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue
            raw_ms = frame.get("ms", frame.get("duration_ms", 400))
            try:
                duration_ms = max(20, round(float(raw_ms)))
            except (TypeError, ValueError):
                duration_ms = 400
            raw_pos = (
                frame.get("pos")
                if "pos" in frame
                else frame.get("servos", frame.get("positions", frame.get("joints", {})))
            )
            targets = _normalize_servo_targets(raw_pos)

            start_ctrl = self.data.ctrl.copy()
            target_ctrl = start_ctrl.copy()
            for name, pulse in targets.items():
                act_id = self._actuator_ids.get(name)
                if act_id is None:
                    continue
                low, high = self.model.actuator_ctrlrange[act_id]
                target_ctrl[act_id] = pulse_to_rad(pulse, float(low), float(high))

            n_steps = max(1, round((duration_ms / 1000.0) / dt))
            for step in range(1, n_steps + 1):
                alpha = step / n_steps
                self.data.ctrl[:] = start_ctrl + alpha * (target_ctrl - start_ctrl)
                mujoco.mj_step(self.model, self.data)
                steps_since_render += 1

                _, _, tilt_deg = _quat_to_euler_and_tilt_deg(self.data.qpos[3:7])
                base_z = float(self.data.qpos[2])
                peak_tilt = max(peak_tilt, tilt_deg)
                max_contacts = max(max_contacts, int(self.data.ncon))
                if tilt_deg >= 45.0 or base_z <= -0.10:
                    fell = True

                if steps_since_render >= render_interval_steps:
                    self._update_latest_jpeg_locked()
                    steps_since_render = 0

                if guard_tilt_deg is not None and tilt_deg >= float(guard_tilt_deg):
                    aborted = True
                    abort_reason = (
                        f"MuJoCo torso tilt {tilt_deg:.1f} deg >= guard {guard_tilt_deg:.1f} deg "
                        f"at frame {idx}"
                    )
                    break

            frames_executed += 1
            self._max_servo_temp_c = round(min(69.0, self._max_servo_temp_c + 0.08), 2)
            self._battery_v = round(max(10.0, self._battery_v - 0.002), 3)
            if aborted:
                break

        if not aborted and hold_s > 0.0:
            hold_steps = max(1, round(float(hold_s) / dt))
            for _ in range(hold_steps):
                mujoco.mj_step(self.model, self.data)
                steps_since_render += 1
                _, _, tilt_deg = _quat_to_euler_and_tilt_deg(self.data.qpos[3:7])
                base_z = float(self.data.qpos[2])
                peak_tilt = max(peak_tilt, tilt_deg)
                max_contacts = max(max_contacts, int(self.data.ncon))
                if tilt_deg >= 45.0 or base_z <= -0.10:
                    fell = True
                if steps_since_render >= render_interval_steps:
                    self._update_latest_jpeg_locked()
                    steps_since_render = 0
                if guard_tilt_deg is not None and tilt_deg >= float(guard_tilt_deg):
                    aborted = True
                    abort_reason = (
                        f"MuJoCo torso tilt {tilt_deg:.1f} deg >= guard {guard_tilt_deg:.1f} deg "
                        "during settle"
                    )
                    break

        frame_b64: str | None = None
        if render_final:
            jpeg = self._update_latest_jpeg_locked()
            frame_b64 = base64.b64encode(jpeg).decode("ascii")

        status = self._status_locked()
        if not status["standing"]:
            fell = True

        return {
            "ok": not aborted and not fell,
            "frames_executed": frames_executed,
            "peak_tilt_deg": round(peak_tilt, 2),
            "end_tilt_deg": status["tilt_deg"],
            "aborted": aborted,
            "abort_reason": abort_reason,
            "fell": fell,
            "standing": status["standing"],
            "base_z": status["base_z"],
            "contacts": max_contacts,
            "friction": status["friction"],
            "status": status,
            "frame_b64": frame_b64,
        }

    def _build_tools(self) -> list[dict[str, Any]]:
        pos_schema = {
            "type": "object",
            "description": "Mapping of canonical TonyPi joint names or string servo IDs '1'..'16' to pulse 0..1000.",
            "additionalProperties": {"type": "integer"},
        }
        frame_item_schema = {
            "type": "object",
            "properties": {
                "ms": {"type": "integer"},
                "pos": pos_schema,
            },
            "required": ["ms", "pos"],
        }
        return [
            {
                "name": "status",
                "description": "Read MuJoCo TonyPi IMU tilt, standing state, contacts, friction, and 16-DOF servo positions.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "perceive",
                "description": "Render a 480x360 JPEG snapshot from the MuJoCo 3D camera.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"mode": {"type": "string", "default": "snapshot"}},
                },
            },
            {
                "name": "say",
                "description": "Log a spoken announcement from TonyPi.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
            {
                "name": "play_frames",
                "description": "Smoothly interpolate and step a timed sequence of 16-DOF keyframes in MuJoCo.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "frames": {"type": "array", "items": frame_item_schema},
                        "hold_s": {"type": "number", "default": 0.5},
                        "guard_tilt_deg": {"type": ["number", "null"], "default": 30.0},
                        "wait": {"type": "boolean", "default": True},
                    },
                    "required": ["frames"],
                },
            },
            {
                "name": "set_pose",
                "description": "Move MuJoCo TonyPi smoothly to a single target pose over `ms` milliseconds.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "positions": pos_schema,
                        "ms": {"type": "integer", "default": 500},
                        "hold_s": {"type": "number", "default": 0.5},
                    },
                    "required": ["positions"],
                },
            },
            {
                "name": "simulate_motion",
                "description": "Reset to standing home and simulate keyframes in MuJoCo without tilt-guard abort.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "frames": {"type": "array", "items": frame_item_schema},
                        "start": {"type": "string", "default": "home"},
                        "settle_s": {"type": "number", "default": 0.8},
                        "render": {"type": "boolean", "default": True},
                        "friction": {"type": "number"},
                    },
                    "required": ["frames"],
                },
            },
            {
                "name": "reset",
                "description": "Reset MuJoCo TonyPi to standing home with configurable floor sliding friction scale.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"friction": {"type": "number", "default": 1.0}},
                },
            },
        ]

    def _dispatch_tool_locked(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "status":
            payload = self._status_locked()
            return {
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "structuredContent": payload,
            }

        if name == "perceive":
            jpeg = self._update_latest_jpeg_locked()
            b64 = base64.b64encode(jpeg).decode("ascii")
            return {
                "content": [
                    {
                        "type": "image",
                        "data": b64,
                        "mimeType": "image/jpeg",
                        "mime_type": "image/jpeg",
                    }
                ]
            }

        if name == "say":
            text = str(arguments.get("text", ""))
            self.spoken_texts.append(text)
            payload = {"ok": True, "text": text}
            return {
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "structuredContent": payload,
            }

        if name == "play_frames":
            raw_frames = arguments.get("frames", arguments.get("keyframes", []))
            frames = list(raw_frames) if isinstance(raw_frames, list) else []
            hold_s = float(arguments.get("hold_s", 0.5))
            guard_raw = arguments.get("guard_tilt_deg", 30.0)
            guard_tilt = None if guard_raw is None else float(guard_raw)
            payload = self._play_frames_locked(
                frames,
                hold_s=hold_s,
                guard_tilt_deg=guard_tilt,
                render_final=True,
            )
            return {
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "structuredContent": payload,
            }

        if name == "set_pose":
            raw_pos = arguments.get("positions", arguments.get("pos", arguments.get("servos", {})))
            ms = int(arguments.get("ms", arguments.get("duration_ms", 500)))
            hold_s = float(arguments.get("hold_s", 0.5))
            payload = self._play_frames_locked(
                [{"ms": ms, "pos": raw_pos}],
                hold_s=hold_s,
                guard_tilt_deg=30.0,
                render_final=True,
            )
            return {
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "structuredContent": payload,
            }

        if name == "simulate_motion":
            raw_frames = arguments.get("frames", arguments.get("keyframes", []))
            frames = list(raw_frames) if isinstance(raw_frames, list) else []
            settle_s = float(arguments.get("settle_s", 0.8))
            render = bool(arguments.get("render", True))
            friction = (
                float(arguments["friction"])
                if "friction" in arguments and arguments["friction"] is not None
                else self._friction_scale
            )
            self._reset_locked(friction=friction)
            payload = self._play_frames_locked(
                frames,
                hold_s=settle_s,
                guard_tilt_deg=None,
                render_final=render,
            )
            return {
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "structuredContent": payload,
            }

        if name == "reset":
            friction = float(arguments.get("friction", 1.0))
            status = self._reset_locked(friction=friction)
            payload = {"ok": True, **status}
            return {
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "structuredContent": payload,
            }

        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
        }

    def handle_jsonrpc_bytes(
        self,
        body: bytes,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        """Process one Streamable HTTP MCP JSON-RPC 2.0 POST request."""
        req_headers = {k.lower(): v for k, v in (headers or {}).items()}
        try:
            msg = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            err = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
            ).encode()
            return 400, {"Content-Type": "application/json"}, err

        method = str(msg.get("method", ""))
        req_id = msg.get("id")
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}

        with self.lock:
            entry: dict[str, Any] = {
                "method": method,
                "session_id": req_headers.get("mcp-session-id"),
            }
            if method == "tools/call":
                entry["tool"] = params.get("name")
                entry["arguments"] = params.get("arguments", {})
            self.call_log.append(entry)

            if method == "notifications/initialized":
                return 202, {"Mcp-Session-Id": self.session_id}, b""

            if method == "initialize":
                rpc_resp: dict[str, Any] = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": "roboharness-mujoco-mcp", "version": "0.1.0"},
                    },
                }
            elif method == "ping":
                rpc_resp = {"jsonrpc": "2.0", "id": req_id, "result": {}}
            elif method == "tools/list":
                rpc_resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"tools": self._build_tools()},
                }
            elif method == "tools/call":
                tool_name = str(params.get("name", ""))
                tool_args = params.get("arguments")
                if not isinstance(tool_args, dict):
                    tool_args = {}
                tool_res = self._dispatch_tool_locked(tool_name, tool_args)
                rpc_resp = {"jsonrpc": "2.0", "id": req_id, "result": tool_res}
            else:
                rpc_resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

        resp_headers = {
            "Content-Type": "application/json",
            "Mcp-Session-Id": self.session_id,
        }
        return 200, resp_headers, json.dumps(rpc_resp).encode()

    def as_httpx_transport(self) -> httpx.MockTransport:
        """Return an in-process `httpx.MockTransport` bound directly to this MuJoCo server."""

        def _handler(request: httpx.Request) -> httpx.Response:
            status, headers, content = self.handle_jsonrpc_bytes(
                request.content,
                dict(request.headers.items()),
            )
            return httpx.Response(status_code=status, headers=headers, content=content)

        return httpx.MockTransport(_handler)

    def start_http(self, host: str = "127.0.0.1", port: int = 8765) -> str:
        """Start the local `ThreadingHTTPServer` exposing `/mcp`, `/frame.jpg`, `/stream.mjpeg`."""
        if self._httpd is not None and self._base_url is not None:
            return f"{self._base_url}/mcp"

        server_ref = self

        class _MujocoMCPHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_POST(self) -> None:
                if self.path.split("?", 1)[0].rstrip("/") not in ("", "/mcp"):
                    self.send_response(404)
                    self.end_headers()
                    return
                content_len = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_len) if content_len > 0 else b""
                status, resp_headers, resp_body = server_ref.handle_jsonrpc_bytes(
                    raw_body,
                    dict(self.headers.items()),
                )
                self.send_response(status)
                for k, v in resp_headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                if resp_body:
                    self.wfile.write(resp_body)

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/health":
                    body = json.dumps(
                        {"ok": True, "sim": True, "xml": str(server_ref.xml_path)}
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if path == "/frame.jpg":
                    jpeg = server_ref.latest_jpeg()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                    return

                if path == "/stream.mjpeg":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                    )
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    last_seq = -1
                    try:
                        while server_ref._http_running and not server_ref._closed:
                            with server_ref._frame_cond:
                                if server_ref._frame_seq == last_seq:
                                    server_ref._frame_cond.wait(timeout=0.5)
                                if not server_ref._http_running or server_ref._closed:
                                    break
                                jpeg = server_ref._latest_jpeg
                                last_seq = server_ref._frame_seq
                            if not jpeg:
                                continue
                            part = (
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n"
                                + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                                + jpeg
                                + b"\r\n"
                            )
                            self.wfile.write(part)
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                    return

                self.send_response(404)
                self.end_headers()

        self._http_running = True
        self._httpd = _QuietThreadingHTTPServer((host, port), _MujocoMCPHandler)
        bound_port = self._httpd.server_address[1]
        self._base_url = f"http://{host}:{bound_port}"
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
            name="roboharness-mujoco-mcp-http",
        )
        self._thread.start()
        return f"{self._base_url}/mcp"

    def stop_http(self) -> None:
        """Stop the local HTTP server without closing the offscreen MuJoCo renderer."""
        with self._frame_cond:
            self._http_running = False
            self._frame_cond.notify_all()
        httpd = self._httpd
        thread = self._thread
        self._httpd = None
        self._thread = None
        self._base_url = None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def stop(self) -> None:
        """Stop the HTTP server, wake MJPEG listeners, and close the MuJoCo renderer safely."""
        with self._frame_cond:
            self._closed = True
            self._frame_cond.notify_all()
        self.stop_http()
        self._renderer.close()


def run_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    viewer: bool = False,
    friction: float = 1.0,
    xml_path: Path | str | None = None,
) -> None:
    """Run the clean-room MuJoCo TonyPi MCP server (with optional 3D passive viewer)."""
    server = MujocoTonypiMCPServer(xml_path=xml_path, friction=friction)
    mcp_url = server.start_http(host=host, port=port)
    print(f"RoboHarness MuJoCo MCP server listening on {mcp_url}", flush=True)
    print(f"Live MJPEG stream: {server.stream_url}", flush=True)
    print(f"Latest snapshot:   {server.frame_url}", flush=True)

    try:
        if viewer:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(server.model, server.data) as v:
                while v.is_running():
                    with server.lock:
                        v.sync()
                    time.sleep(1.0 / 30.0)
        else:
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for `python -m roboharness.mujoco_mcp_server`."""
    parser = argparse.ArgumentParser(
        description="Run the clean-room MuJoCo Streamable HTTP MCP server for TonyPi."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open interactive 3D MuJoCo passive viewer on main thread (use `mjpython` on macOS).",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=1.0,
        help="Initial floor sliding friction multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="Optional override path to TonyPi MuJoCo scene.xml.",
    )
    args = parser.parse_args(argv)
    run_server(
        host=args.host,
        port=args.port,
        viewer=args.viewer,
        friction=args.friction,
        xml_path=args.xml,
    )


if __name__ == "__main__":
    main()
