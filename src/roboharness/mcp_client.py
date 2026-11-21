"""Streamable HTTP MCP client for the Hiwonder TonyPi hardware driver boundary.

All hardware interaction crosses strictly through JSON-RPC 2.0 over Streamable HTTP MCP
(`TONYPI_MCP_URL`). Local deterministic safety latches from `roboharness.safety` are
enforced before any actuation command is dispatched and between frames during streaming.
"""

from __future__ import annotations

import base64
import json
import math
import re
import time
from collections.abc import Callable, Sequence
from types import TracebackType
from typing import Any, Literal, Self

import httpx

from roboharness.config import (
    MAX_TORSO_TILT_DEG,
    MIN_KEYFRAME_DURATION_MS,
    RoboHarnessSettings,
)
from roboharness.safety import check_robot_telemetry, check_skill_keyframes
from roboharness.schemas import (
    JOINTS_BY_ID,
    JOINTS_BY_NAME,
    NEUTRAL_POSE,
    ExecutionTrace,
    RobotStatus,
    ServoKeyframe,
    SkillArtifact,
)

MCP_PROTOCOL_VERSION = "2025-03-26"

_STATUS_TOOL_CANDIDATES: tuple[str, ...] = (
    "get_status",
    "status",
    "robot_status",
    "get_robot_status",
    "telemetry",
    "get_telemetry",
    "read_status",
)
_PERCEIVE_TOOL_CANDIDATES: tuple[str, ...] = (
    "perceive",
    "capture_frame",
    "get_camera_frame",
    "camera_frame",
    "get_frame",
    "snapshot",
    "capture_image",
)
_SAY_TOOL_CANDIDATES: tuple[str, ...] = (
    "say",
    "speak",
    "tts",
    "voice_say",
    "announce",
)
_BATCH_TOOL_CANDIDATES: tuple[str, ...] = (
    "play_frames",
    "run_keyframes",
    "play_keyframes",
    "execute_keyframes",
    "play_sequence",
    "run_sequence",
    "execute_sequence",
)
_FRAME_TOOL_CANDIDATES: tuple[str, ...] = (
    "set_servos",
    "move_servos",
    "set_joints",
    "set_servo_positions",
    "move_joints",
    "set_pose",
)
_SAVE_ACTION_CANDIDATES: tuple[str, ...] = (
    "save_action",
    "upload_action",
    "create_action",
    "save_action_group",
)
_RUN_ACTION_CANDIDATES: tuple[str, ...] = (
    "run_action",
    "play_action",
    "execute_action",
    "run_action_group",
)
_RESET_TOOL_CANDIDATES: tuple[str, ...] = (
    "reset",
    "reset_sim",
    "sim_reset",
)
_SIMULATE_TOOL_CANDIDATES: tuple[str, ...] = (
    "simulate_motion",
    "simulate_frames",
    "sim_motion",
)

_SERVO_ID_RE = re.compile(r"^(?:servo[_-]?|id[_-]?)?(\d+)$", re.IGNORECASE)
_POS_NUMERIC_KEY_RE = re.compile(r"""["']pos(?:itions)?["']\s*:\s*\{\s*["']\d+["']""")

# Physical TonyPi bus-servo calibration map (IDs 1..8 left, 9..16 right) applied as
# relative deltas around live `get_joints()` standing pulses on real `tonypi-mcp`.
_PHYSICAL_TONYPI_SERVO_MAP: tuple[tuple[str, str, float], ...] = (
    ("l_ankle_roll", "1", 0.25),
    ("l_ankle_pitch", "2", -0.25),
    ("l_knee", "3", -0.25),
    ("l_hip_pitch", "4", 0.25),
    ("l_hip_roll", "5", 0.25),
    ("l_elbow", "6", 1.0),
    ("l_shoulder_roll", "7", -0.85),
    ("l_shoulder_pitch", "8", -1.25),
    ("r_ankle_roll", "9", 0.25),
    ("r_ankle_pitch", "10", 0.25),
    ("r_knee", "11", 0.25),
    ("r_hip_pitch", "12", -0.25),
    ("r_hip_roll", "13", 0.25),
    ("r_elbow", "14", -1.0),
    ("r_shoulder_roll", "15", 0.85),
    ("r_shoulder_pitch", "16", 1.25),
)


class MCPClientError(RuntimeError):
    """Raised when an MCP transport, protocol, or remote tool invocation fails."""


class SafetyViolationError(ValueError):
    """Raised when a deterministic hardware or kinematics safety latch blocks execution."""

    def __init__(self, message: str, *, reasons: list[str] | None = None) -> None:
        super().__init__(message)
        self.reasons: list[str] = list(reasons or [message])


def _parse_sse_json_messages(body: str) -> list[dict[str, Any]]:
    """Parse JSON-RPC objects framed as `text/event-stream` SSE events."""
    messages: list[dict[str, Any]] = []
    data_lines: list[str] = []

    def _flush() -> None:
        if not data_lines:
            return
        payload = "\n".join(data_lines).strip()
        data_lines.clear()
        if not payload or payload == "[DONE]":
            return
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise MCPClientError(f"Invalid JSON inside SSE data frame: {payload[:200]}") from exc
        if isinstance(obj, dict):
            messages.append(obj)
        elif isinstance(obj, list):
            messages.extend(item for item in obj if isinstance(item, dict))

    for raw_line in body.splitlines():
        line = raw_line.rstrip("\r")
        if line == "":
            _flush()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            value = line[5:].removeprefix(" ")
            data_lines.append(value)
    _flush()
    return messages


def _extract_tool_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Extract structured JSON dict from an MCP `tools/call` result envelope."""
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured

    content = result.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"].strip()
                if text.startswith(("{", "[")):
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict):
                        return parsed
                    return {"items": parsed}
    return result if isinstance(result, dict) else {}


def _extract_error_text(result: dict[str, Any]) -> str:
    """Extract human-readable error text from an MCP `tools/call` error result."""
    content = result.get("content")
    if isinstance(content, list):
        parts = [
            str(block.get("text", "")).strip()
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
        ]
        if parts:
            return "; ".join(parts)
    return json.dumps(result)[:300]


def _normalize_servo_positions(raw_positions: Any) -> dict[str, int]:
    """Normalize canonical joint names or integer servo IDs into canonical TonyPi positions."""
    positions = dict(NEUTRAL_POSE)
    if isinstance(raw_positions, dict):
        for raw_key, raw_val in raw_positions.items():
            try:
                pulse = round(float(raw_val))
            except (TypeError, ValueError):
                continue
            key_str = str(raw_key).strip()
            if key_str in JOINTS_BY_NAME:
                positions[key_str] = pulse
                continue
            match = _SERVO_ID_RE.match(key_str)
            if match:
                sid = int(match.group(1))
                spec = JOINTS_BY_ID.get(sid)
                if spec is not None:
                    positions[spec.name] = pulse
    elif isinstance(raw_positions, list):
        for item in raw_positions:
            if not isinstance(item, dict):
                continue
            val = item.get("pulse", item.get("position", item.get("value")))
            if val is None:
                continue
            try:
                pulse = round(float(val))
            except (TypeError, ValueError):
                continue
            name = item.get("name") or item.get("joint")
            if isinstance(name, str) and name in JOINTS_BY_NAME:
                positions[name] = pulse
                continue
            sid = item.get("servo_id", item.get("id"))
            if sid is not None:
                try:
                    spec = JOINTS_BY_ID.get(int(sid))
                except (TypeError, ValueError):
                    spec = None
                if spec is not None:
                    positions[spec.name] = pulse
    return positions


def _first_float(mapping: dict[str, Any], keys: Sequence[str], default: float | None = None) -> float | None:
    """Return the first numeric value found for any of `keys` in `mapping`."""
    for key in keys:
        if key in mapping and mapping[key] is not None:
            val = mapping[key]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return float(val)
            if isinstance(val, str):
                try:
                    return float(val)
                except ValueError:
                    continue
    return default


def _normalize_robot_status(payload: dict[str, Any]) -> RobotStatus:
    """Normalize heterogeneous MCP status responses (flat or nested `state`) into `RobotStatus`."""
    nested_state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    # Nested `state` (real tonypi-mcp) takes precedence for telemetry fields while preserving top-level fallbacks.
    merged: dict[str, Any] = {**payload, **nested_state}

    imu = merged.get("imu") if isinstance(merged.get("imu"), dict) else {}
    orientation = merged.get("orientation") if isinstance(merged.get("orientation"), dict) else {}
    battery_obj = merged.get("battery") if isinstance(merged.get("battery"), dict) else {}
    power_obj = merged.get("power") if isinstance(merged.get("power"), dict) else {}

    battery_v = _first_float(
        merged, ("battery_v", "battery_voltage", "voltage_v", "voltage", "battery")
    )
    if battery_v is None and "battery_mv" in merged:
        mv = _first_float(merged, ("battery_mv",))
        if mv is not None:
            battery_v = mv / 1000.0
    if battery_v is None:
        battery_v = _first_float(battery_obj, ("voltage_v", "voltage", "v", "volts"))
    if battery_v is None and "mv" in battery_obj:
        mv = _first_float(battery_obj, ("mv",))
        if mv is not None:
            battery_v = mv / 1000.0
    if battery_v is None:
        battery_v = _first_float(power_obj, ("voltage_v", "voltage", "battery_v"))
    if battery_v is None:
        # Fail-safe: missing voltage reports 0.0V so the hard battery latch protects hardware.
        battery_v = 0.0
    elif battery_v > 100.0:
        # Handle millivolt telemetry from Hiwonder controllers.
        battery_v = battery_v / 1000.0

    max_temp = _first_float(
        merged,
        (
            "max_servo_temp_c",
            "servo_temp_c",
            "max_temp_c",
            "temperature_c",
            "temp_c",
            "max_temperature",
        ),
    )
    if max_temp is None:
        hottest = merged.get("hottest")
        if (
            isinstance(hottest, (list, tuple))
            and len(hottest) >= 3
            and isinstance(hottest[2], (int, float))
            and not isinstance(hottest[2], bool)
        ):
            max_temp = float(hottest[2])
    if max_temp is None:
        temps_raw = merged.get("servo_temps") or merged.get("temperatures") or merged.get("temps")
        if isinstance(temps_raw, dict) and temps_raw:
            numeric = [
                float(v)
                for v in temps_raw.values()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            max_temp = max(numeric) if numeric else 0.0
        elif isinstance(temps_raw, list) and temps_raw:
            numeric = [
                float(v) for v in temps_raw if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            max_temp = max(numeric) if numeric else 0.0
        else:
            max_temp = 0.0

    pitch = _first_float(merged, ("torso_pitch_deg", "pitch_deg", "pitch"))
    if pitch is None:
        pitch = _first_float(imu, ("pitch_deg", "pitch"))
    if pitch is None:
        pitch = _first_float(orientation, ("pitch_deg", "pitch"))

    roll = _first_float(merged, ("torso_roll_deg", "roll_deg", "roll"))
    if roll is None:
        roll = _first_float(imu, ("roll_deg", "roll"))
    if roll is None:
        roll = _first_float(orientation, ("roll_deg", "roll"))

    if pitch is None and roll is None:
        scalar_tilt = _first_float(merged, ("tilt_deg", "torso_tilt_deg"))
        if scalar_tilt is not None:
            pitch = scalar_tilt
            roll = 0.0

    pitch = 0.0 if pitch is None else pitch
    roll = 0.0 if roll is None else roll

    standing_val = merged.get("standing", merged.get("is_standing", merged.get("upright")))
    if isinstance(standing_val, bool):
        standing = standing_val
    elif any(key in merged for key in ("lying", "asleep", "stop_latched")):
        standing = (
            (not bool(merged.get("lying", False)))
            and (not bool(merged.get("asleep", False)))
            and (not bool(merged.get("stop_latched", False)))
        )
    elif isinstance(merged.get("posture"), str):
        standing = merged["posture"].lower() not in {"fallen", "prone", "supine", "sitting"}
    else:
        standing = True

    raw_servos = (
        merged.get("servo_positions")
        or merged.get("servos")
        or merged.get("joints")
        or merged.get("joint_positions")
        or merged.get("positions")
    )
    servo_positions = _normalize_servo_positions(raw_servos)

    return RobotStatus(
        battery_v=round(float(battery_v), 3),
        max_servo_temp_c=round(float(max_temp), 2),
        torso_pitch_deg=round(float(pitch), 2),
        torso_roll_deg=round(float(roll), 2),
        standing=standing,
        servo_positions=servo_positions,
        raw=dict(payload),
    )


def _decode_base64_image(raw_b64: str, default_mime: str = "image/jpeg") -> tuple[bytes, str]:
    """Decode raw base64 or `data:<mime>;base64,...` image string."""
    mime = default_mime
    data_str = raw_b64.strip()
    if data_str.startswith("data:") and "," in data_str:
        header, data_str = data_str.split(",", 1)
        mime_part = header[5:].split(";", 1)[0].strip()
        if mime_part:
            mime = mime_part
    try:
        return base64.b64decode(data_str), mime
    except Exception as exc:
        raise MCPClientError(f"Failed to base64-decode camera frame: {exc}") from exc


def _schema_prefers_servo_ids(tool_def: dict[str, Any]) -> bool:
    """Detect whether a tool's description/inputSchema expects string servo IDs ('1'..'16')."""
    schema = tool_def.get("inputSchema")
    desc = str(tool_def.get("description") or "")
    combined = (desc + " " + (json.dumps(schema) if isinstance(schema, dict) else "")).lower()
    if '"servo_ids"' in combined or '"servo_id"' in combined or '"by_id"' in combined:
        return True
    if "patternproperties" in combined and ("^[0-9]" in combined or "^\\d" in combined):
        return True
    if _POS_NUMERIC_KEY_RE.search(combined):
        return True
    return (
        "integer servo id" in combined
        or "servo id (1" in combined
        or "servo ids (1" in combined
        or "'1'..'16'" in combined
        or '"1".."16"' in combined
    )


def _format_servos_for_schema(
    servos: dict[str, int],
    *,
    use_servo_ids: bool,
    as_list: bool = False,
) -> dict[str, int] | list[dict[str, int]]:
    """Format canonical joint pulse mapping as joint names, integer ID keys, or item list."""
    if as_list:
        items: list[dict[str, int]] = []
        for name, pulse in servos.items():
            spec = JOINTS_BY_NAME.get(name)
            if spec is not None:
                items.append({"id": spec.servo_id, "pulse": int(pulse)})
        return items
    if use_servo_ids:
        return {
            str(JOINTS_BY_NAME[name].servo_id): int(pulse)
            for name, pulse in servos.items()
            if name in JOINTS_BY_NAME
        }
    return {name: int(pulse) for name, pulse in servos.items()}


class TonypiMCPClient:
    """Synchronous Streamable HTTP MCP client for Hiwonder TonyPi hardware & simulators."""

    def __init__(
        self,
        url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 15.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.url: str = url or RoboHarnessSettings.from_env().tonypi_mcp_url
        self._timeout: float = timeout
        self._extra_headers: dict[str, str] = dict(headers or {})
        self._client = httpx.Client(transport=transport, timeout=timeout)
        self._req_id: int = 0
        self.session_id: str | None = None
        self._connected: bool = False
        self._server_info: dict[str, Any] = {}
        self._tools: list[dict[str, Any]] = []
        self._tools_by_name: dict[str, dict[str, Any]] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _build_headers(self) -> dict[str, str]:
        hdrs = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self._extra_headers,
        }
        if self.session_id:
            hdrs["Mcp-Session-Id"] = self.session_id
        return hdrs

    def _send_rpc(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        is_notification: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        req_id: int | None = None
        if not is_notification:
            req_id = self._next_id()
            payload["id"] = req_id
        if params is not None:
            payload["params"] = params

        try:
            resp = self._client.post(self.url, json=payload, headers=self._build_headers())
        except httpx.HTTPError as exc:
            raise MCPClientError(f"MCP transport error calling {method} at {self.url}: {exc}") from exc

        sid = resp.headers.get("mcp-session-id")
        if sid:
            self.session_id = sid

        if resp.status_code >= 400:
            raise MCPClientError(
                f"MCP HTTP {resp.status_code} calling {method}: {resp.text[:300]}"
            )

        if is_notification and (resp.status_code in (202, 204) or not resp.content.strip()):
            return {}

        return self._parse_rpc_response(resp, req_id=req_id)

    def _parse_rpc_response(
        self,
        resp: httpx.Response,
        *,
        req_id: int | None,
    ) -> dict[str, Any]:
        content_type = resp.headers.get("content-type", "").lower()
        text = resp.text

        if not text.strip():
            return {}

        message: dict[str, Any] | None = None
        if "text/event-stream" in content_type or text.lstrip().startswith(("event:", "data:")):
            messages = _parse_sse_json_messages(text)
            if not messages:
                raise MCPClientError("MCP SSE response contained no JSON-RPC payload")
            if req_id is not None:
                for msg in messages:
                    if msg.get("id") == req_id:
                        message = msg
                        break
            if message is None:
                for msg in reversed(messages):
                    if "result" in msg or "error" in msg:
                        message = msg
                        break
            if message is None:
                message = messages[-1]
        else:
            try:
                parsed = resp.json()
            except ValueError as exc:
                raise MCPClientError(f"Invalid JSON-RPC response: {text[:300]}") from exc
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and (req_id is None or item.get("id") == req_id):
                        message = item
                        break
                if message is None:
                    raise MCPClientError("JSON-RPC batch response missing matching id")
            elif isinstance(parsed, dict):
                message = parsed
            else:
                raise MCPClientError(f"Unexpected JSON-RPC response type: {type(parsed).__name__}")

        if "error" in message and message["error"] is not None:
            err = message["error"]
            if isinstance(err, dict):
                code = err.get("code", "unknown")
                msg = err.get("message", str(err))
                raise MCPClientError(f"MCP JSON-RPC error ({code}): {msg}")
            raise MCPClientError(f"MCP JSON-RPC error: {err}")

        result = message.get("result", {})
        return result if isinstance(result, dict) else {"value": result}

    def connect(self) -> list[dict[str, Any]]:
        """Perform Streamable HTTP MCP handshake and discover server tools."""
        init_res = self._send_rpc(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "roboharness", "version": "0.1.0"},
            },
        )
        self._server_info = init_res.get("serverInfo", {}) if isinstance(init_res, dict) else {}
        self._send_rpc("notifications/initialized", is_notification=True)

        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"cursor": cursor} if cursor else {}
            list_res = self._send_rpc("tools/list", params)
            page = list_res.get("tools", [])
            if isinstance(page, list):
                tools.extend(item for item in page if isinstance(item, dict) and "name" in item)
            cursor = list_res.get("nextCursor")
            if not cursor:
                break

        self._tools = tools
        self._tools_by_name = {str(t["name"]): t for t in tools}
        self._connected = True
        return list(self._tools)

    def _ensure_connected(self) -> None:
        if not self._connected:
            self.connect()

    def list_tools(self) -> list[dict[str, Any]]:
        """Return cached MCP tools discovered from `tools/list` (connecting if needed)."""
        self._ensure_connected()
        return list(self._tools)

    def _find_tool(self, candidates: Sequence[str]) -> dict[str, Any] | None:
        """Auto-discover a tool by exact name or namespace-suffixed match."""
        self._ensure_connected()
        lower_map = {name.lower(): tool for name, tool in self._tools_by_name.items()}
        for cand in candidates:
            hit = lower_map.get(cand.lower())
            if hit is not None:
                return hit
        # Namespace-prefixed fallback (e.g. tonypi_get_status, tonypi.status, robot/perceive)
        for cand in candidates:
            cand_l = cand.lower()
            suffixes = (f"_{cand_l}", f".{cand_l}", f"/{cand_l}", f"-{cand_l}")
            for name_l, tool in lower_map.items():
                if name_l.endswith(suffixes):
                    return tool
        return None

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Invoke a remote MCP tool via `tools/call`."""
        self._ensure_connected()
        result = self._send_rpc(
            "tools/call",
            {"name": name, "arguments": dict(arguments or {})},
        )
        if result.get("isError"):
            raise MCPClientError(f"MCP tool '{name}' failed: {_extract_error_text(result)}")
        return result

    def get_status(self) -> RobotStatus:
        """Fetch and normalize live hardware telemetry via the discovered status tool."""
        tool = self._find_tool(_STATUS_TOOL_CANDIDATES)
        if tool is None:
            available = sorted(self._tools_by_name)
            raise MCPClientError(
                f"No status tool found on MCP server {self.url}; available tools: {available}"
            )
        result = self.call_tool(str(tool["name"]), {})
        payload = _extract_tool_payload(result)
        return _normalize_robot_status(payload)

    def perceive(self) -> tuple[bytes, str]:
        """Capture a camera frame via MCP and return `(image_bytes, mime_type)`."""
        tool = self._find_tool(_PERCEIVE_TOOL_CANDIDATES)
        if tool is None:
            available = sorted(self._tools_by_name)
            raise MCPClientError(
                f"No perceive/camera tool found on MCP server {self.url}; available tools: {available}"
            )
        result = self.call_tool(str(tool["name"]), {})

        content = result.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "image" and isinstance(block.get("data"), str):
                    mime = str(block.get("mimeType") or block.get("mime_type") or "image/jpeg")
                    return _decode_base64_image(block["data"], default_mime=mime)
                if btype == "resource" and isinstance(block.get("resource"), dict):
                    res = block["resource"]
                    if isinstance(res.get("blob"), str):
                        mime = str(res.get("mimeType") or res.get("mime_type") or "image/jpeg")
                        return _decode_base64_image(res["blob"], default_mime=mime)

        payload = _extract_tool_payload(result)
        for key in ("image_b64", "frame_b64", "jpeg_b64", "image", "data", "frame"):
            raw = payload.get(key)
            if isinstance(raw, str) and raw.strip():
                mime = str(payload.get("mime_type") or payload.get("mimeType") or "image/jpeg")
                return _decode_base64_image(raw, default_mime=mime)

        raise MCPClientError(f"Tool '{tool['name']}' returned no decodable image frame")

    def _try_perceive_b64(self) -> str | None:
        """Best-effort camera capture encoded as base64 ASCII string."""
        try:
            img_bytes, _ = self.perceive()
            return base64.b64encode(img_bytes).decode("ascii")
        except (MCPClientError, httpx.HTTPError, ValueError):
            return None

    def say(self, text: str) -> bool:
        """Speak text on the robot if a TTS tool is advertised; return False gracefully if absent."""
        tool = self._find_tool(_SAY_TOOL_CANDIDATES)
        if tool is None:
            return False
        schema = tool.get("inputSchema")
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        arg_key = "text"
        for candidate in ("text", "message", "utterance", "phrase", "speech"):
            if candidate in props:
                arg_key = candidate
                break
        self.call_tool(str(tool["name"]), {arg_key: text})
        return True

    def reset_sim(self, friction: float = 1.0) -> dict[str, Any] | None:
        """Reset simulator state with optional floor friction scale if `reset` is advertised."""
        tool = self._find_tool(_RESET_TOOL_CANDIDATES)
        if tool is None:
            return None
        schema = tool.get("inputSchema")
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        args: dict[str, Any] = {}
        if not props or "friction" in props:
            args["friction"] = float(friction)
        result = self.call_tool(str(tool["name"]), args)
        return _extract_tool_payload(result)

    def simulate_raw_frames(
        self,
        keyframes: Sequence[ServoKeyframe],
        *,
        friction: float = 1.0,
        settle_s: float = 0.8,
        render: bool = True,
    ) -> dict[str, Any] | None:
        """Simulate unguarded raw keyframes via `simulate_motion` if advertised by simulator MCP."""
        tool = self._find_tool(_SIMULATE_TOOL_CANDIDATES)
        if tool is None:
            return None
        schema = tool.get("inputSchema")
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        is_sim_motion = "simulate_motion" in str(tool.get("name", "")).lower()
        use_ids = is_sim_motion or _schema_prefers_servo_ids(tool)

        frames_key = "frames"
        for cand in ("frames", "keyframes", "sequence"):
            if cand in props:
                frames_key = cand
                break

        item_props: dict[str, Any] = {}
        frames_prop = props.get(frames_key)
        if isinstance(frames_prop, dict) and isinstance(frames_prop.get("items"), dict):
            item_props = frames_prop["items"].get("properties", {}) or {}

        servo_field = "pos"
        for cand in ("pos", "servos", "positions", "joints"):
            if cand in item_props:
                servo_field = cand
                break

        dur_field = "ms"
        if "duration_ms" in item_props and "ms" not in item_props:
            dur_field = "duration_ms"

        formatted_frames = [
            {
                dur_field: int(kf.duration_ms),
                servo_field: _format_servos_for_schema(kf.servos, use_servo_ids=use_ids),
            }
            for kf in keyframes
        ]
        args: dict[str, Any] = {frames_key: formatted_frames}
        if "start" in props:
            args["start"] = "home"
        if "settle_s" in props:
            args["settle_s"] = float(settle_s)
        if "render" in props:
            args["render"] = bool(render)
        if "friction" in props:
            args["friction"] = float(friction)
        else:
            self.reset_sim(friction=friction)

        result = self.call_tool(str(tool["name"]), args)
        return _extract_tool_payload(result)

    def _select_execution_tools(
        self,
    ) -> tuple[
        Literal["batch_keyframes", "frame_by_frame", "action_group"] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        batch_tool = self._find_tool(_BATCH_TOOL_CANDIDATES)
        if batch_tool is not None:
            return "batch_keyframes", batch_tool, None
        frame_tool = self._find_tool(_FRAME_TOOL_CANDIDATES)
        if frame_tool is not None:
            return "frame_by_frame", frame_tool, None
        save_tool = self._find_tool(_SAVE_ACTION_CANDIDATES)
        run_tool = self._find_tool(_RUN_ACTION_CANDIDATES)
        if save_tool is not None and run_tool is not None:
            return "action_group", save_tool, run_tool
        return None, None, None

    def _read_physical_base_joints(self) -> dict[str, int]:
        """Read live standing pulse calibration via `get_joints()` on real `tonypi-mcp`."""
        result = self.call_tool("get_joints", {})
        payload = _extract_tool_payload(result)
        raw_map = (
            payload.get("pos")
            if isinstance(payload.get("pos"), dict)
            else payload.get("raw")
            if isinstance(payload.get("raw"), dict)
            else payload
        )
        base_pos: dict[str, int] = {str(sid): 500 for sid in range(1, 17)}
        if isinstance(raw_map, dict):
            for raw_key, raw_val in raw_map.items():
                key_s = str(raw_key).strip()
                if key_s in base_pos:
                    try:
                        base_pos[key_s] = round(float(raw_val))
                    except (TypeError, ValueError):
                        continue
        return base_pos

    def _build_batch_args(
        self,
        tool_def: dict[str, Any],
        skill: SkillArtifact,
        speed_scale: float,
    ) -> dict[str, Any]:
        schema = tool_def.get("inputSchema")
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        tool_name_l = str(tool_def.get("name", "")).lower()
        is_play_frames = tool_name_l == "play_frames" or tool_name_l.endswith("_play_frames")

        # Real physical tonypi-mcp exposes `get_joints`: project relative deltas onto live
        # calibrated standing pulses and append a smooth return-to-stand settle frame.
        if is_play_frames and "get_joints" in self._tools_by_name:
            base_pos = self._read_physical_base_joints()
            calibrated_frames: list[dict[str, Any]] = []
            for kf in skill.keyframes:
                scaled_ms = max(MIN_KEYFRAME_DURATION_MS, round(kf.duration_ms / speed_scale))
                pos = dict(base_pos)
                for joint_name, sid, sign_scale in _PHYSICAL_TONYPI_SERVO_MAP:
                    if joint_name in kf.servos:
                        pulse = kf.servos[joint_name]
                        pos[sid] = max(
                            140,
                            min(860, round(base_pos[sid] + sign_scale * (pulse - 500))),
                        )
                calibrated_frames.append({"ms": scaled_ms, "pos": pos})
            calibrated_frames.append({"ms": 500, "pos": dict(base_pos)})
            return {
                "frames": calibrated_frames,
                "hold_s": 0.5,
                "guard_tilt_deg": MAX_TORSO_TILT_DEG,
                "wait": True,
            }

        # Both real tonypi-mcp and MujocoTonypiMCPServer accept string servo IDs "1".."16" in play_frames.
        use_ids = is_play_frames or _schema_prefers_servo_ids(tool_def)

        frames_key = "frames" if is_play_frames else "keyframes"
        for cand in ("frames", "keyframes", "sequence", "steps"):
            if cand in props:
                frames_key = cand
                break

        item_props: dict[str, Any] = {}
        frames_prop = props.get(frames_key)
        if isinstance(frames_prop, dict) and isinstance(frames_prop.get("items"), dict):
            item_props = frames_prop["items"].get("properties", {}) or {}

        servo_field = "pos" if is_play_frames else "servos"
        for cand in ("pos", "servos", "positions", "joints", "servo_positions"):
            if cand in item_props:
                servo_field = cand
                break
        servo_as_list = (
            isinstance(item_props.get(servo_field), dict)
            and item_props[servo_field].get("type") == "array"
        )

        formatted_frames: list[dict[str, Any]] = []
        for kf in skill.keyframes:
            scaled_ms = max(MIN_KEYFRAME_DURATION_MS, round(kf.duration_ms / speed_scale))
            entry: dict[str, Any] = {
                servo_field: _format_servos_for_schema(
                    kf.servos, use_servo_ids=use_ids, as_list=servo_as_list
                ),
            }
            if "ms" in item_props or (is_play_frames and "duration_ms" not in item_props):
                entry["ms"] = scaled_ms
            elif "duration" in item_props and "duration_ms" not in item_props:
                entry["duration"] = round(scaled_ms / 1000.0, 4)
            else:
                entry["duration_ms"] = scaled_ms
            if kf.note and not is_play_frames:
                entry["note"] = kf.note
            formatted_frames.append(entry)

        args: dict[str, Any] = {frames_key: formatted_frames}
        if "hold_s" in props or is_play_frames:
            args["hold_s"] = 0.5
        if "guard_tilt_deg" in props or is_play_frames:
            args["guard_tilt_deg"] = MAX_TORSO_TILT_DEG
        if "wait" in props or is_play_frames:
            args["wait"] = True
        if "speed_scale" in props:
            args["speed_scale"] = speed_scale
        elif "speed" in props:
            args["speed"] = speed_scale
        if "name" in props:
            args["name"] = skill.name
        elif "skill_name" in props:
            args["skill_name"] = skill.name
        return args

    def _build_frame_args(
        self,
        tool_def: dict[str, Any],
        frame: ServoKeyframe,
        speed_scale: float,
    ) -> dict[str, Any]:
        schema = tool_def.get("inputSchema")
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        tool_name_l = str(tool_def.get("name", "")).lower()
        is_set_pose = tool_name_l == "set_pose" or tool_name_l.endswith("_set_pose")
        use_ids = is_set_pose or _schema_prefers_servo_ids(tool_def)

        servo_key = "positions" if is_set_pose else "servos"
        for cand in ("positions", "pos", "servos", "joints", "targets", "servo_positions"):
            if cand in props:
                servo_key = cand
                break
        as_list = isinstance(props.get(servo_key), dict) and props[servo_key].get("type") == "array"

        args: dict[str, Any] = {
            servo_key: _format_servos_for_schema(
                frame.servos, use_servo_ids=use_ids, as_list=as_list
            )
        }
        scaled_ms = max(1, round(frame.duration_ms / speed_scale))
        if "ms" in props or (is_set_pose and "duration_ms" not in props):
            args["ms"] = scaled_ms
        elif "duration_ms" in props:
            args["duration_ms"] = scaled_ms
        elif "time_ms" in props:
            args["time_ms"] = scaled_ms
        elif "duration" in props:
            args["duration"] = round(scaled_ms / 1000.0, 4)
        if "hold_s" in props or is_set_pose:
            args["hold_s"] = 0.2
        return args

    def execute_skill(
        self,
        skill: SkillArtifact,
        *,
        dry_run: bool = False,
        speed_scale: float = 1.0,
        on_frame: Callable[[int, int, ServoKeyframe, RobotStatus], None] | None = None,
        raise_on_unsafe: bool = False,
    ) -> ExecutionTrace:
        """Execute a verified skill via dynamic MCP tool dispatch with hard safety latches.

        Pre-flight safety checks (`check_robot_telemetry` and `check_skill_keyframes`)
        ALWAYS execute before any servo actuation command reaches the MCP boundary.
        """
        if speed_scale <= 0.0:
            raise ValueError(f"speed_scale must be positive, got {speed_scale}")

        self._ensure_connected()
        planned_mode, primary_tool, secondary_tool = self._select_execution_tools()

        # Hard deterministic pre-flight safety latches (never bypassed, even on dry_run)
        pre_status = self.get_status()
        pre_tilt = round(math.hypot(pre_status.torso_pitch_deg, pre_status.torso_roll_deg), 2)
        telemetry_verdict = check_robot_telemetry(pre_status)
        kinematics_report = check_skill_keyframes(skill.keyframes)

        safety_reasons: list[str] = []
        if not telemetry_verdict.safe:
            safety_reasons.extend(telemetry_verdict.reasons)
        if not kinematics_report.verified:
            safety_reasons.extend(kinematics_report.violations)

        if safety_reasons:
            abort_reason = "Pre-execution safety latch blocked actuation: " + "; ".join(
                safety_reasons
            )
            if raise_on_unsafe:
                raise SafetyViolationError(abort_reason, reasons=safety_reasons)
            return ExecutionTrace(
                skill_name=skill.name,
                transport_mode="dry_run" if dry_run or planned_mode is None else planned_mode,
                frames_executed=0,
                aborted=True,
                abort_reason=abort_reason,
                peak_observed_tilt_deg=pre_tilt,
                final_status=pre_status,
                post_run_frame_b64=None,
            )

        if dry_run:
            return ExecutionTrace(
                skill_name=skill.name,
                transport_mode="dry_run",
                frames_executed=0,
                aborted=False,
                abort_reason=None,
                peak_observed_tilt_deg=pre_tilt,
                final_status=pre_status,
                post_run_frame_b64=self._try_perceive_b64(),
            )

        if planned_mode is None or primary_tool is None:
            available = sorted(self._tools_by_name)
            raise MCPClientError(
                f"No supported actuation tools found on MCP server {self.url}; "
                f"available tools: {available}"
            )

        frames_executed = 0
        aborted = False
        abort_reason: str | None = None
        peak_tilt = pre_tilt
        final_status = pre_status
        total_frames = len(skill.keyframes)

        payload_frame_b64: str | None = None
        if planned_mode == "batch_keyframes":
            args = self._build_batch_args(primary_tool, skill, speed_scale)
            raw_res = self.call_tool(str(primary_tool["name"]), args)
            payload = _extract_tool_payload(raw_res)
            frames_executed = int(payload.get("frames_executed", total_frames))
            payload_peak = _first_float(
                payload, ("peak_tilt_deg", "max_tilt_deg", "peak_observed_tilt_deg"), 0.0
            )
            peak_tilt = max(peak_tilt, float(payload_peak or 0.0))
            if isinstance(payload.get("frame_b64"), str) and payload["frame_b64"]:
                payload_frame_b64 = payload["frame_b64"]

            final_status = self.get_status()
            final_tilt = math.hypot(final_status.torso_pitch_deg, final_status.torso_roll_deg)
            peak_tilt = max(peak_tilt, final_tilt)

            if payload.get("aborted") or payload.get("fell"):
                aborted = True
                abort_reason = str(
                    payload.get("abort_reason")
                    or (
                        "Robot fell during batch execution"
                        if payload.get("fell")
                        else "Hardware batch execution aborted"
                    )
                )
            else:
                post_verdict = check_robot_telemetry(final_status)
                if not post_verdict.safe:
                    aborted = True
                    abort_reason = "Post-batch hardware safety violation: " + "; ".join(
                        post_verdict.reasons
                    )

            if on_frame is not None and frames_executed > 0:
                last_idx = min(frames_executed, total_frames) - 1
                on_frame(last_idx, total_frames, skill.keyframes[last_idx], final_status)

        elif planned_mode == "frame_by_frame":
            for idx, frame in enumerate(skill.keyframes):
                args = self._build_frame_args(primary_tool, frame, speed_scale)
                self.call_tool(str(primary_tool["name"]), args)
                frames_executed += 1

                sleep_sec = (frame.duration_ms / 1000.0) / speed_scale
                if sleep_sec > 0.0:
                    time.sleep(sleep_sec)

                cur_status = self.get_status()
                cur_tilt = math.hypot(cur_status.torso_pitch_deg, cur_status.torso_roll_deg)
                peak_tilt = max(peak_tilt, cur_tilt)
                final_status = cur_status

                if on_frame is not None:
                    on_frame(idx, total_frames, frame, cur_status)

                verdict = check_robot_telemetry(cur_status)
                if not verdict.safe:
                    aborted = True
                    abort_reason = (
                        f"Mid-execution safety latch tripped after frame {idx}: "
                        + "; ".join(verdict.reasons)
                    )
                    break

        elif planned_mode == "action_group":
            assert secondary_tool is not None
            save_args = self._build_batch_args(primary_tool, skill, speed_scale=1.0)
            save_props = (
                primary_tool.get("inputSchema", {}).get("properties", {})
                if isinstance(primary_tool.get("inputSchema"), dict)
                else {}
            )
            name_key = "action_name" if "action_name" in save_props else "name"
            save_args[name_key] = skill.name
            self.call_tool(str(primary_tool["name"]), save_args)

            run_props = (
                secondary_tool.get("inputSchema", {}).get("properties", {})
                if isinstance(secondary_tool.get("inputSchema"), dict)
                else {}
            )
            run_name_key = "action_name" if "action_name" in run_props else "name"
            run_args: dict[str, Any] = {run_name_key: skill.name}
            if "speed_scale" in run_props:
                run_args["speed_scale"] = speed_scale
            raw_res = self.call_tool(str(secondary_tool["name"]), run_args)
            payload = _extract_tool_payload(raw_res)
            frames_executed = int(payload.get("frames_executed", total_frames))
            payload_peak = _first_float(payload, ("peak_tilt_deg", "peak_observed_tilt_deg"), 0.0)
            peak_tilt = max(peak_tilt, float(payload_peak or 0.0))

            final_status = self.get_status()
            final_tilt = math.hypot(final_status.torso_pitch_deg, final_status.torso_roll_deg)
            peak_tilt = max(peak_tilt, final_tilt)

            if payload.get("aborted"):
                aborted = True
                abort_reason = str(
                    payload.get("abort_reason") or "Hardware action execution aborted"
                )
            else:
                post_verdict = check_robot_telemetry(final_status)
                if not post_verdict.safe:
                    aborted = True
                    abort_reason = "Post-action hardware safety violation: " + "; ".join(
                        post_verdict.reasons
                    )

            if on_frame is not None and frames_executed > 0:
                last_idx = min(frames_executed, total_frames) - 1
                on_frame(last_idx, total_frames, skill.keyframes[last_idx], final_status)

        post_run_frame_b64 = self._try_perceive_b64() or payload_frame_b64
        return ExecutionTrace(
            skill_name=skill.name,
            transport_mode=planned_mode,
            frames_executed=frames_executed,
            aborted=aborted,
            abort_reason=abort_reason,
            peak_observed_tilt_deg=round(peak_tilt, 2),
            final_status=final_status,
            post_run_frame_b64=post_run_frame_b64,
        )
