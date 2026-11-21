"""Deterministic offline MockTonypiMCPServer for testing and offline evals.

Implements Streamable HTTP MCP JSON-RPC 2.0 (`initialize`, `notifications/initialized`,
`tools/list`, `tools/call`) usable both in-process via `httpx.MockTransport` and over
local loopback HTTP via stdlib `ThreadingHTTPServer`.
"""

from __future__ import annotations

import base64
import io
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any, Literal, Self

import httpx
from PIL import Image, ImageDraw

from roboharness.config import MAX_TORSO_TILT_DEG
from roboharness.safety import DEG_PER_PULSE, estimate_frame_tilt_and_com
from roboharness.schemas import JOINTS_BY_ID, JOINTS_BY_NAME, NEUTRAL_POSE

MockServerMode = Literal["batch", "frame_only", "action_only"]


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _normalize_incoming_servos(raw: Any) -> dict[str, int]:
    """Accept canonical joint names, string/int servo IDs, or list[{id, pulse}]."""
    out: dict[str, int] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                pulse = round(float(v))
            except (TypeError, ValueError):
                continue
            key_s = str(k).strip()
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


class MockTonypiMCPServer:
    """Deterministic offline MCP mock server emulating a 16-DOF TonyPi robot."""

    def __init__(
        self,
        *,
        mode: MockServerMode = "batch",
        battery_v: float = 11.6,
        servo_temp_c: float = 38.0,
        table_slip_multiplier: float = 1.0,
        edge_of_table: bool = False,
        standing: bool = True,
        torso_pitch_deg: float = 0.0,
        torso_roll_deg: float = 0.0,
        sse_responses: bool = False,
        use_servo_ids: bool = False,
        expose_say: bool = True,
        session_id: str = "mock-tonypi-session-001",
    ) -> None:
        self.mode: MockServerMode = mode
        self.battery_v: float = float(battery_v)
        self.servo_temp_c: float = float(servo_temp_c)
        self.table_slip_multiplier: float = float(table_slip_multiplier)
        self.edge_of_table: bool = bool(edge_of_table)
        self.standing: bool = bool(standing)
        self.torso_pitch_deg: float = float(torso_pitch_deg)
        self.torso_roll_deg: float = float(torso_roll_deg)
        self.sse_responses: bool = bool(sse_responses)
        self.use_servo_ids: bool = bool(use_servo_ids)
        self.expose_say: bool = bool(expose_say)
        self.session_id: str = session_id

        self.servo_positions: dict[str, int] = dict(NEUTRAL_POSE)
        self.saved_actions: dict[str, list[dict[str, Any]]] = {}
        self.spoken_texts: list[str] = []
        self.call_log: list[dict[str, Any]] = []

        self._lock = threading.Lock()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._url: str | None = None

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
    def url(self) -> str | None:
        """Local HTTP endpoint URL when running via `start_http()`, else None."""
        return self._url

    def _servo_map_schema(self) -> dict[str, Any]:
        if self.use_servo_ids:
            return {
                "type": "object",
                "description": "Mapping of integer servo IDs (1..16) to pulse 0..1000",
                "patternProperties": {"^[0-9]+$": {"type": "integer"}},
            }
        return {
            "type": "object",
            "description": "Mapping of canonical TonyPi joint names to pulse 0..1000",
            "additionalProperties": {"type": "integer"},
        }

    def _build_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = [
            {
                "name": "get_status",
                "description": "Read TonyPi battery voltage, servo temps, IMU tilt, and 16-DOF positions.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "perceive",
                "description": "Capture a 320x240 JPEG frame from the TonyPi camera.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]
        if self.expose_say:
            tools.append(
                {
                    "name": "say",
                    "description": "Speak a short utterance through TonyPi speaker.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                }
            )

        servo_schema = self._servo_map_schema()
        if self.mode == "batch":
            tools.append(
                {
                    "name": "run_keyframes",
                    "description": "Execute a timed sequence of 16-DOF servo keyframes.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "speed_scale": {"type": "number"},
                            "keyframes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "duration_ms": {"type": "integer"},
                                        "servos": servo_schema,
                                        "note": {"type": "string"},
                                    },
                                    "required": ["duration_ms", "servos"],
                                },
                            },
                        },
                        "required": ["keyframes"],
                    },
                }
            )
        elif self.mode == "frame_only":
            tools.append(
                {
                    "name": "set_servos",
                    "description": "Move TonyPi servos to target pulse positions over duration_ms.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "servos": servo_schema,
                            "duration_ms": {"type": "integer"},
                        },
                        "required": ["servos"],
                    },
                }
            )
        elif self.mode == "action_only":
            tools.extend(
                [
                    {
                        "name": "save_action",
                        "description": "Upload and store a named action group on TonyPi.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "keyframes": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "duration_ms": {"type": "integer"},
                                            "servos": servo_schema,
                                        },
                                        "required": ["duration_ms", "servos"],
                                    },
                                },
                            },
                            "required": ["name", "keyframes"],
                        },
                    },
                    {
                        "name": "run_action",
                        "description": "Play a previously saved TonyPi action group.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "speed_scale": {"type": "number"},
                            },
                            "required": ["name"],
                        },
                    },
                ]
            )
        return tools

    def _apply_frame(self, raw_servos: Any) -> float:
        """Apply servo positions and update simulated IMU tilt, temp, and battery."""
        updates = _normalize_incoming_servos(raw_servos)
        total_delta = 0
        for name, pulse in updates.items():
            prev = self.servo_positions.get(name, 500)
            total_delta += abs(pulse - prev)
            self.servo_positions[name] = pulse

        base_tilt_deg, _ = estimate_frame_tilt_and_com(self.servo_positions)
        effective_tilt = round(base_tilt_deg * self.table_slip_multiplier, 2)

        # Compute deterministic signed sagittal vs lateral directional split
        pose = self.servo_positions
        avg_hip_p = 0.5 * ((pose["r_hip_pitch"] - 500) + (pose["l_hip_pitch"] - 500)) * DEG_PER_PULSE
        avg_ank_p = (
            0.5 * ((pose["r_ankle_pitch"] - 500) + (pose["l_ankle_pitch"] - 500)) * DEG_PER_PULSE
        )
        avg_knee = 0.5 * ((pose["r_knee"] - 500) + (pose["l_knee"] - 500)) * DEG_PER_PULSE
        sag_signed = avg_ank_p + 0.55 * avg_hip_p - 0.35 * avg_knee

        avg_ank_r = (
            0.5 * ((pose["r_ankle_roll"] - 500) + (pose["l_ankle_roll"] - 500)) * DEG_PER_PULSE
        )
        avg_hip_r = 0.5 * ((pose["r_hip_roll"] - 500) + (pose["l_hip_roll"] - 500)) * DEG_PER_PULSE
        lat_signed = avg_ank_r + 0.45 * avg_hip_r + 0.22 * (pose["r_knee"] - pose["l_knee"]) * DEG_PER_PULSE

        raw_mag = math.hypot(sag_signed, lat_signed)
        if effective_tilt == 0.0:
            self.torso_pitch_deg = 0.0
            self.torso_roll_deg = 0.0
        elif raw_mag > 1e-6:
            scale = effective_tilt / raw_mag
            self.torso_pitch_deg = round(sag_signed * scale, 2)
            self.torso_roll_deg = round(lat_signed * scale, 2)
        else:
            # Arm-only tilt bias splits predominantly sagittal
            self.torso_pitch_deg = round(effective_tilt, 2)
            self.torso_roll_deg = 0.0

        # Realistic gentle thermal rise and battery drain proportional to servo work
        self.servo_temp_c = round(self.servo_temp_c + 0.15 + min(0.5, total_delta / 4000.0), 2)
        self.battery_v = round(max(0.0, self.battery_v - 0.004), 3)
        return effective_tilt

    def _status_payload(self) -> dict[str, Any]:
        return {
            "battery_v": self.battery_v,
            "max_servo_temp_c": self.servo_temp_c,
            "torso_pitch_deg": self.torso_pitch_deg,
            "torso_roll_deg": self.torso_roll_deg,
            "standing": self.standing,
            "servo_positions": dict(self.servo_positions),
        }

    def render_camera_jpeg(self) -> bytes:
        """Render a deterministic 320x240 JPEG of TonyPi stick figure and table bounds."""
        width, height = 320, 240
        img = Image.new("RGB", (width, height), (18, 22, 30))
        draw = ImageDraw.Draw(img)

        # Table geometry
        table_left = 36
        table_right = 284
        table_top = 172
        table_bottom = 228

        # Drop-off void zones
        draw.rectangle([0, table_top, table_left - 1, height], fill=(10, 12, 16))
        draw.rectangle([table_right + 1, table_top, width, height], fill=(10, 12, 16))

        # Tabletop surface and crisp boundary lines
        draw.rectangle([table_left, table_top, table_right, table_bottom], fill=(44, 52, 66))
        edge_color = (248, 113, 113) if self.edge_of_table else (56, 189, 248)
        draw.line([(table_left, table_top), (table_right, table_top)], fill=(56, 189, 248), width=2)
        draw.line([(table_left, table_top), (table_left, table_bottom)], fill=(56, 189, 248), width=2)
        draw.line([(table_right, table_top), (table_right, table_bottom)], fill=edge_color, width=3)

        # Center placement vs edge-of-table risk placement
        cx = 264 if self.edge_of_table else 160
        foot_y = 180

        pose = self.servo_positions
        r_knee_deg = (pose["r_knee"] - 500) * DEG_PER_PULSE
        l_knee_deg = (pose["l_knee"] - 500) * DEG_PER_PULSE
        r_hip_r_deg = (pose["r_hip_roll"] - 500) * DEG_PER_PULSE
        l_hip_r_deg = (pose["l_hip_roll"] - 500) * DEG_PER_PULSE
        r_sh_p_deg = (pose["r_shoulder_pitch"] - 500) * DEG_PER_PULSE
        r_sh_r_deg = (pose["r_shoulder_roll"] - 500) * DEG_PER_PULSE
        r_elb_deg = (pose["r_elbow"] - 500) * DEG_PER_PULSE
        l_sh_p_deg = (pose["l_shoulder_pitch"] - 500) * DEG_PER_PULSE
        l_sh_r_deg = (pose["l_shoulder_roll"] - 500) * DEG_PER_PULSE
        l_elb_deg = (pose["l_elbow"] - 500) * DEG_PER_PULSE

        crouch = int(max(-10.0, min(14.0, 0.5 * (r_knee_deg + l_knee_deg) * 0.25)))
        hip_y = 132 + crouch

        lean_deg = max(-45.0, min(45.0, self.torso_roll_deg + 0.25 * self.torso_pitch_deg))
        lean_rad = math.radians(lean_deg)
        torso_len = 46
        sh_cx = cx + round(torso_len * math.sin(lean_rad))
        sh_cy = hip_y - round(torso_len * math.cos(lean_rad))

        # Legs (viewer-left = robot right leg, viewer-right = robot left leg)
        r_hip = (cx - 11, hip_y)
        l_hip = (cx + 11, hip_y)
        r_knee_pt = (
            cx - 13 - round(r_hip_r_deg * 0.15),
            (hip_y + foot_y) // 2 + round(r_knee_deg * 0.12),
        )
        l_knee_pt = (
            cx + 13 + round(l_hip_r_deg * 0.15),
            (hip_y + foot_y) // 2 + round(l_knee_deg * 0.12),
        )
        r_foot = (cx - 14, foot_y)
        l_foot = (cx + 14, foot_y)

        limb_color = (226, 232, 240)
        joint_color = (56, 189, 248)
        draw.line([r_hip, r_knee_pt, r_foot], fill=limb_color, width=4)
        draw.line([l_hip, l_knee_pt, l_foot], fill=limb_color, width=4)
        # Feet pads
        draw.line([(r_foot[0] - 8, foot_y), (r_foot[0] + 6, foot_y)], fill=(148, 163, 184), width=4)
        draw.line([(l_foot[0] - 6, foot_y), (l_foot[0] + 8, foot_y)], fill=(148, 163, 184), width=4)

        # Torso & shoulders & hips
        draw.line([(cx, hip_y), (sh_cx, sh_cy)], fill=(56, 189, 248), width=6)
        draw.line([(cx - 12, hip_y), (cx + 12, hip_y)], fill=limb_color, width=4)
        r_sh = (sh_cx - 16, sh_cy)
        l_sh = (sh_cx + 16, sh_cy)
        draw.line([r_sh, l_sh], fill=limb_color, width=4)

        # Head
        head_x = sh_cx + round(15 * math.sin(lean_rad))
        head_y = sh_cy - round(15 * math.cos(lean_rad))
        draw.ellipse([head_x - 10, head_y - 10, head_x + 10, head_y + 10], outline=(56, 189, 248), width=3)

        # Arms
        r_ang1 = math.radians(110.0 + r_sh_r_deg * 0.9 - r_sh_p_deg * 0.6)
        r_elb = (
            r_sh[0] + round(24 * math.cos(r_ang1)),
            r_sh[1] + round(24 * math.sin(r_ang1)),
        )
        r_ang2 = r_ang1 - math.radians(r_elb_deg * 0.7)
        r_hand = (
            r_elb[0] + round(22 * math.cos(r_ang2)),
            r_elb[1] + round(22 * math.sin(r_ang2)),
        )

        l_ang1 = math.radians(70.0 - l_sh_r_deg * 0.9 + l_sh_p_deg * 0.6)
        l_elb = (
            l_sh[0] + round(24 * math.cos(l_ang1)),
            l_sh[1] + round(24 * math.sin(l_ang1)),
        )
        l_ang2 = l_ang1 + math.radians(l_elb_deg * 0.7)
        l_hand = (
            l_elb[0] + round(22 * math.cos(l_ang2)),
            l_elb[1] + round(22 * math.sin(l_ang2)),
        )

        draw.line([r_sh, r_elb, r_hand], fill=limb_color, width=4)
        draw.line([l_sh, l_elb, l_hand], fill=limb_color, width=4)

        # Joint markers
        for pt in (r_sh, l_sh, r_elb, l_elb, r_hip, l_hip, r_knee_pt, l_knee_pt):
            draw.ellipse([pt[0] - 3, pt[1] - 3, pt[0] + 3, pt[1] + 3], fill=joint_color)

        # Compact deterministic HUD overlay
        tilt = math.hypot(self.torso_pitch_deg, self.torso_roll_deg)
        edge_tag = "EDGE_RISK" if self.edge_of_table else "CENTERED"
        hud = f"BAT {self.battery_v:.1f}V | TMP {self.servo_temp_c:.1f}C | TILT {tilt:.1f}d | {edge_tag}"
        draw.text((8, 8), hud, fill=(148, 163, 184))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=False)
        return buf.getvalue()

    def _dispatch_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "get_status":
            payload = self._status_payload()
            return {
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "structuredContent": payload,
            }

        if name == "perceive":
            jpeg_bytes = self.render_camera_jpeg()
            b64 = base64.b64encode(jpeg_bytes).decode("ascii")
            return {
                "content": [
                    {
                        "type": "image",
                        "data": b64,
                        "mimeType": "image/jpeg",
                    }
                ]
            }

        if name == "say":
            if not self.expose_say:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": "say tool not enabled"}],
                }
            text = str(arguments.get("text", ""))
            self.spoken_texts.append(text)
            payload = {"ok": True, "spoken": text}
            return {
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "structuredContent": payload,
            }

        if name == "run_keyframes":
            if self.mode != "batch":
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": f"run_keyframes unavailable in mode={self.mode}"}],
                }
            keyframes = arguments.get("keyframes", [])
            executed = 0
            peak_tilt = math.hypot(self.torso_pitch_deg, self.torso_roll_deg)
            aborted = False
            abort_reason: str | None = None
            for idx, kf in enumerate(keyframes):
                servos = kf.get("servos", {}) if isinstance(kf, dict) else {}
                tilt = self._apply_frame(servos)
                executed += 1
                peak_tilt = max(peak_tilt, tilt)
                if tilt >= MAX_TORSO_TILT_DEG:
                    aborted = True
                    abort_reason = (
                        f"Simulated table slip / tilt {tilt:.1f} deg >= {MAX_TORSO_TILT_DEG:.1f} deg "
                        f"at frame {idx}"
                    )
                    break
            payload = {
                "ok": not aborted,
                "frames_executed": executed,
                "peak_tilt_deg": round(peak_tilt, 2),
                "aborted": aborted,
                "abort_reason": abort_reason,
            }
            return {
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "structuredContent": payload,
            }

        if name == "set_servos":
            if self.mode != "frame_only":
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": f"set_servos unavailable in mode={self.mode}"}],
                }
            tilt = self._apply_frame(arguments.get("servos", {}))
            payload = {
                "ok": True,
                "tilt_deg": round(tilt, 2),
                "servo_positions": dict(self.servo_positions),
            }
            return {
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "structuredContent": payload,
            }

        if name == "save_action":
            if self.mode != "action_only":
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": f"save_action unavailable in mode={self.mode}"}],
                }
            action_name = str(arguments.get("name", "unnamed"))
            keyframes = list(arguments.get("keyframes", []))
            self.saved_actions[action_name] = keyframes
            payload = {"ok": True, "name": action_name, "frame_count": len(keyframes)}
            return {
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "structuredContent": payload,
            }

        if name == "run_action":
            if self.mode != "action_only":
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": f"run_action unavailable in mode={self.mode}"}],
                }
            action_name = str(arguments.get("name", ""))
            if action_name not in self.saved_actions:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Action '{action_name}' not found"}],
                }
            keyframes = self.saved_actions[action_name]
            executed = 0
            peak_tilt = math.hypot(self.torso_pitch_deg, self.torso_roll_deg)
            aborted = False
            abort_reason = None
            for idx, kf in enumerate(keyframes):
                servos = kf.get("servos", {}) if isinstance(kf, dict) else {}
                tilt = self._apply_frame(servos)
                executed += 1
                peak_tilt = max(peak_tilt, tilt)
                if tilt >= MAX_TORSO_TILT_DEG:
                    aborted = True
                    abort_reason = (
                        f"Simulated table slip / tilt {tilt:.1f} deg >= {MAX_TORSO_TILT_DEG:.1f} deg "
                        f"at frame {idx}"
                    )
                    break
            payload = {
                "ok": not aborted,
                "name": action_name,
                "frames_executed": executed,
                "peak_tilt_deg": round(peak_tilt, 2),
                "aborted": aborted,
                "abort_reason": abort_reason,
            }
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
        """Process a single MCP Streamable HTTP POST body and return `(status, headers, body)`."""
        req_headers = {k.lower(): v for k, v in (headers or {}).items()}
        try:
            msg = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            err_body = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
            ).encode("utf-8")
            return 400, {"Content-Type": "application/json"}, err_body

        method = str(msg.get("method", ""))
        req_id = msg.get("id")
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}

        with self._lock:
            log_entry: dict[str, Any] = {
                "method": method,
                "session_id": req_headers.get("mcp-session-id"),
            }
            if method == "tools/call":
                log_entry["tool"] = params.get("name")
                log_entry["arguments"] = params.get("arguments", {})
            self.call_log.append(log_entry)

            if method == "notifications/initialized":
                return 202, {"Mcp-Session-Id": self.session_id}, b""

            if method == "initialize":
                rpc_response: dict[str, Any] = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": "mock-tonypi-mcp", "version": "0.1.0"},
                    },
                }
            elif method == "ping":
                rpc_response = {"jsonrpc": "2.0", "id": req_id, "result": {}}
            elif method == "tools/list":
                rpc_response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"tools": self._build_tools()},
                }
            elif method == "tools/call":
                tool_name = str(params.get("name", ""))
                tool_args = params.get("arguments")
                if not isinstance(tool_args, dict):
                    tool_args = {}
                tool_res = self._dispatch_tool(tool_name, tool_args)
                rpc_response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": tool_res,
                }
            else:
                rpc_response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

        resp_headers = {"Mcp-Session-Id": self.session_id}
        if self.sse_responses:
            sse_payload = f"event: message\ndata: {json.dumps(rpc_response)}\n\n".encode()
            resp_headers["Content-Type"] = "text/event-stream"
            return 200, resp_headers, sse_payload

        resp_headers["Content-Type"] = "application/json"
        return 200, resp_headers, json.dumps(rpc_response).encode("utf-8")

    def as_httpx_transport(self) -> httpx.MockTransport:
        """Return an in-process `httpx.MockTransport` bound to this mock server."""

        def _handler(request: httpx.Request) -> httpx.Response:
            status, headers, content = self.handle_jsonrpc_bytes(
                request.content,
                dict(request.headers.items()),
            )
            return httpx.Response(status_code=status, headers=headers, content=content)

        return httpx.MockTransport(_handler)

    def start_http(self, host: str = "127.0.0.1", port: int = 0) -> str:
        """Start a lightweight local ThreadingHTTPServer exposing `/mcp` and return its URL."""
        if self._httpd is not None and self._url is not None:
            return self._url

        server_ref = self

        class _MCPRequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_POST(self) -> None:
                if self.path.rstrip("/") not in ("", "/mcp"):
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
                if self.path == "/health":
                    body = json.dumps({"ok": True, "mode": server_ref.mode}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(405)
                self.end_headers()

        self._httpd = _QuietThreadingHTTPServer((host, port), _MCPRequestHandler)
        bound_port = self._httpd.server_address[1]
        self._url = f"http://{host}:{bound_port}/mcp"
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
            name="mock-tonypi-mcp",
        )
        self._thread.start()
        return self._url

    def stop(self) -> None:
        """Stop the local HTTP server if running."""
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
