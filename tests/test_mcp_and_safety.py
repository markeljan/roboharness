"""Offline tests for Streamable HTTP MCP client, MockTonypiMCPServer, safety latches, capture, and SSE strip."""

from __future__ import annotations

import base64
import io
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from PIL import Image

from roboharness.capture import (
    CaptureError,
    extract_preview_frames,
    make_synthetic_test_video,
    record_webcam_clip,
)
from roboharness.mcp_client import SafetyViolationError, TonypiMCPClient
from roboharness.mock_mcp import MockTonypiMCPServer
from roboharness.safety import check_skill_keyframes, sign_skill_artifact
from roboharness.schemas import NEUTRAL_POSE, RobotStatus, ServoKeyframe, SkillArtifact
from roboharness.telemetry_sse import TelemetryHub, TelemetrySSEServer


def _make_safe_skill(name: str = "wave_groove") -> SkillArtifact:
    """Create a verified 3-frame skill with modest hip/arm motion (~11 deg nominal tilt)."""
    frames = [
        ServoKeyframe(
            duration_ms=400,
            servos={
                **NEUTRAL_POSE,
                "r_hip_pitch": 585,
                "l_hip_pitch": 585,
                "r_shoulder_pitch": 640,
                "l_shoulder_pitch": 360,
            },
            note="beat_1_lean_and_wave",
        ),
        ServoKeyframe(
            duration_ms=400,
            servos={
                **NEUTRAL_POSE,
                "r_hip_pitch": 540,
                "l_hip_pitch": 540,
                "r_shoulder_pitch": 580,
                "l_shoulder_pitch": 620,
            },
            note="beat_2_switch",
        ),
        ServoKeyframe(
            duration_ms=400,
            servos=dict(NEUTRAL_POSE),
            note="beat_3_neutral",
        ),
    ]
    report = check_skill_keyframes(frames)
    assert report.verified, f"Expected safe skill, got violations: {report.violations}"
    return sign_skill_artifact(SkillArtifact(name=name, keyframes=frames, kinematics=report))


def _actuation_calls(server: MockTonypiMCPServer) -> list[dict[str, Any]]:
    actuation_tools = {"run_keyframes", "set_servos", "save_action", "run_action"}
    return [
        entry
        for entry in server.call_log
        if entry.get("method") == "tools/call" and entry.get("tool") in actuation_tools
    ]


def test_mcp_client_batch_mode_and_dry_run() -> None:
    server = MockTonypiMCPServer(mode="batch")
    skill = _make_safe_skill()

    with TonypiMCPClient("http://mock.local/mcp", transport=server.as_httpx_transport()) as client:
        tools = client.connect()
        tool_names = {t["name"] for t in tools}
        assert {"get_status", "perceive", "say", "run_keyframes"} <= tool_names
        assert client.session_id == server.session_id

        # Verify handshake included notifications/initialized with session id
        methods = [c["method"] for c in server.call_log]
        assert methods[:3] == ["initialize", "notifications/initialized", "tools/list"]
        assert server.call_log[1]["session_id"] == server.session_id

        assert client.say("Starting groove!") is True
        assert server.spoken_texts == ["Starting groove!"]

        # Dry run must skip actuation while capturing telemetry & camera frame
        dry_trace = client.execute_skill(skill, dry_run=True)
        assert dry_trace.transport_mode == "dry_run"
        assert dry_trace.aborted is False
        assert dry_trace.frames_executed == 0
        assert dry_trace.post_run_frame_b64 is not None
        assert _actuation_calls(server) == []

        # Live batch execution
        seen_callbacks: list[tuple[int, int]] = []
        trace = client.execute_skill(
            skill,
            speed_scale=1.0,
            on_frame=lambda idx, total, _kf, _st: seen_callbacks.append((idx, total)),
        )
        assert trace.transport_mode == "batch_keyframes"
        assert trace.aborted is False
        assert trace.frames_executed == len(skill.keyframes)
        assert trace.post_run_frame_b64 is not None
        assert seen_callbacks == [(2, 3)]
        assert [c["tool"] for c in _actuation_calls(server)] == ["run_keyframes"]

        # Decode post-run JPEG and verify 320x240 size
        img_bytes = base64.b64decode(trace.post_run_frame_b64)
        with Image.open(io.BytesIO(img_bytes)) as img:
            assert img.size == (320, 240)
            assert img.format == "JPEG"


def test_mcp_client_frame_only_mode() -> None:
    server = MockTonypiMCPServer(mode="frame_only", expose_say=False)
    skill = _make_safe_skill()

    with TonypiMCPClient("http://mock.local/mcp", transport=server.as_httpx_transport()) as client:
        assert client.say("ignored when absent") is False

        callbacks: list[tuple[int, int, RobotStatus]] = []
        trace = client.execute_skill(
            skill,
            speed_scale=200.0,
            on_frame=lambda i, total, _kf, st: callbacks.append((i, total, st)),
        )
        assert trace.transport_mode == "frame_by_frame"
        assert trace.aborted is False
        assert trace.frames_executed == 3
        assert len(callbacks) == 3
        assert [c["tool"] for c in _actuation_calls(server)] == [
            "set_servos",
            "set_servos",
            "set_servos",
        ]


def test_mcp_client_action_only_and_servo_id_adaptation() -> None:
    server = MockTonypiMCPServer(mode="action_only", use_servo_ids=True)
    skill = _make_safe_skill("id_adapted_dance")

    with TonypiMCPClient("http://mock.local/mcp", transport=server.as_httpx_transport()) as client:
        trace = client.execute_skill(skill)
        assert trace.transport_mode == "action_group"
        assert trace.aborted is False
        assert trace.frames_executed == 3

        act_calls = _actuation_calls(server)
        assert [c["tool"] for c in act_calls] == ["save_action", "run_action"]
        saved_kf0 = act_calls[0]["arguments"]["keyframes"][0]["servos"]
        # Schema requested integer servo IDs (1..16) -> keys must be numeric strings
        assert all(k.isdigit() for k in saved_kf0)
        assert "6" in saved_kf0  # r_shoulder_pitch == servo_id 6


def test_mcp_local_http_and_sse_jsonrpc_parsing() -> None:
    server = MockTonypiMCPServer(mode="batch", sse_responses=True)
    url = server.start_http(host="127.0.0.1", port=0)
    try:
        with TonypiMCPClient(url) as client:
            tools = client.connect()
            assert len(tools) >= 4
            status = client.get_status()
            assert status.battery_v == pytest.approx(11.6, abs=0.01)

            frame_bytes, mime = client.perceive()
            assert mime == "image/jpeg"
            assert frame_bytes.startswith(b"\xff\xd8")

            # Deterministic rendering check + edge_of_table visual shift
            frame_bytes_2, _ = client.perceive()
            assert frame_bytes == frame_bytes_2
            server.edge_of_table = True
            edge_frame, _ = client.perceive()
            assert edge_frame != frame_bytes
    finally:
        server.stop()


def test_safety_latch_low_battery_blocks_actuation() -> None:
    server = MockTonypiMCPServer(mode="batch", battery_v=10.5)
    skill = _make_safe_skill()

    with TonypiMCPClient("http://mock.local/mcp", transport=server.as_httpx_transport()) as client:
        trace = client.execute_skill(skill)
        assert trace.aborted is True
        assert trace.frames_executed == 0
        assert trace.abort_reason is not None and "Battery voltage" in trace.abort_reason
        assert _actuation_calls(server) == []

        with pytest.raises(SafetyViolationError) as exc_info:
            client.execute_skill(skill, raise_on_unsafe=True)
        assert isinstance(exc_info.value, ValueError)
        assert _actuation_calls(server) == []


def test_safety_latch_high_temp_blocks_actuation() -> None:
    server = MockTonypiMCPServer(mode="frame_only", servo_temp_c=72.0)
    skill = _make_safe_skill()

    with TonypiMCPClient("http://mock.local/mcp", transport=server.as_httpx_transport()) as client:
        trace = client.execute_skill(skill, speed_scale=200.0)
        assert trace.aborted is True
        assert trace.frames_executed == 0
        assert trace.abort_reason is not None and "72.0C" in trace.abort_reason
        assert _actuation_calls(server) == []


def test_safety_latch_clipped_keyframe_blocks_actuation() -> None:
    server = MockTonypiMCPServer(mode="batch")
    clipped_frame = ServoKeyframe(
        duration_ms=400,
        servos={**NEUTRAL_POSE, "r_shoulder_pitch": 950},  # max_pulse is 850
    )
    bad_report = check_skill_keyframes([clipped_frame])
    assert bad_report.joint_clipping_count == 1
    bad_skill = SkillArtifact(name="clipped_skill", keyframes=[clipped_frame], kinematics=bad_report)

    with TonypiMCPClient("http://mock.local/mcp", transport=server.as_httpx_transport()) as client:
        trace = client.execute_skill(bad_skill)
        assert trace.aborted is True
        assert trace.frames_executed == 0
        assert trace.abort_reason is not None and "outside [150, 850]" in trace.abort_reason
        assert _actuation_calls(server) == []


def test_safety_latch_table_slip_aborts_mid_run() -> None:
    skill = _make_safe_skill()

    # Frame-by-frame mode: frame 0 trips simulated tilt >= 30 deg -> frames 1 & 2 never sent
    frame_server = MockTonypiMCPServer(mode="frame_only", table_slip_multiplier=3.0)
    with TonypiMCPClient(
        "http://mock.local/mcp", transport=frame_server.as_httpx_transport()
    ) as client:
        trace = client.execute_skill(skill, speed_scale=200.0)
        assert trace.aborted is True
        assert trace.frames_executed == 1
        assert trace.peak_observed_tilt_deg >= 30.0
        assert trace.abort_reason is not None and "Mid-execution safety latch" in trace.abort_reason
        assert len(_actuation_calls(frame_server)) == 1

    # Batch mode: mock server simulates slip and aborts batch; client flags aborted trace
    batch_server = MockTonypiMCPServer(mode="batch", table_slip_multiplier=3.0)
    with TonypiMCPClient(
        "http://mock.local/mcp", transport=batch_server.as_httpx_transport()
    ) as client:
        trace = client.execute_skill(skill)
        assert trace.aborted is True
        assert trace.frames_executed == 1
        assert trace.peak_observed_tilt_deg >= 30.0


def test_telemetry_sse_server_endpoints_and_streaming() -> None:
    hub = TelemetryHub(max_history=50)
    for i in range(55):
        hub.emit("log", {"message": f"warm_{i}"})
    # Ring buffer caps at 50 events
    hist = hub.history()
    assert len(hist) == 50
    assert hist[0]["data"]["message"] == "warm_5"
    assert hist[-1]["data"]["message"] == "warm_54"

    hub_live = TelemetryHub(max_history=50)
    hub_live.emit("stage", {"stage": "FORGE", "message": "Forging 16-DOF skill"})

    server = TelemetrySSEServer(hub_live, host="127.0.0.1", port=0)
    base_url = server.start()
    try:
        health = httpx.get(f"{base_url}/health", timeout=2.0)
        assert health.status_code == 200
        assert health.json()["ok"] is True

        index = httpx.get(f"{base_url}/", timeout=2.0)
        assert index.status_code == 200
        assert "text/html" in index.headers["content-type"]
        assert "gemini-4-hydrogen" in index.text
        assert "gemini-decision-1" in index.text
        assert "EventSource" in index.text

        received: list[dict[str, Any]] = []

        def _emit_delayed() -> None:
            time.sleep(0.01)
            hub_live.emit(
                "decision",
                {
                    "model": "gemini-decision-1",
                    "min_confidence": 0.88,
                    "effective_action": "execute_and_save",
                },
            )

        emitter = threading.Thread(target=_emit_delayed, daemon=True)
        emitter.start()

        with httpx.stream("GET", f"{base_url}/events", timeout=3.0) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    received.append(json.loads(line[6:]))
                    if len(received) >= 2:
                        break
        emitter.join(timeout=1.0)

        assert [r["event"] for r in received] == ["stage", "decision"]
        assert received[0]["data"]["stage"] == "FORGE"
        assert received[1]["data"]["effective_action"] == "execute_and_save"
    finally:
        server.stop()


def test_capture_webcam_command_construction_and_error(tmp_path: Path) -> None:
    out_file = tmp_path / "clip.mp4"
    recorded_cmds: list[list[str]] = []

    def _fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        recorded_cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    with patch("roboharness.capture.subprocess.run", side_effect=_fake_run):
        result = record_webcam_clip(duration_sec=5, device="1", output_path=out_file)

    assert result == out_file
    assert len(recorded_cmds) == 1
    cmd = recorded_cmds[0]
    assert cmd[0] == "ffmpeg"
    assert ["-f", "avfoundation", "-framerate", "30", "-video_size", "640x480"] == cmd[2:8]
    assert ["-i", "1:none", "-t", "5"] == cmd[8:12]
    assert ["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"] == cmd[12:18]
    assert cmd[-1] == str(out_file)

    # Error path includes macOS permission & device listing instructions
    def _fail_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            cmd, returncode=1, stdout="", stderr="Input/output error"
        )

    with (
        patch("roboharness.capture.subprocess.run", side_effect=_fail_run),
        pytest.raises(CaptureError) as exc_info,
    ):
        record_webcam_clip(duration_sec=5, device="0", output_path=out_file)
    err_text = str(exc_info.value)
    assert "Privacy & Security -> Camera" in err_text
    assert "-list_devices true" in err_text


def test_safety_latch_high_preflight_tilt_and_non_standing() -> None:
    server = MockTonypiMCPServer(mode="batch", torso_pitch_deg=31.5)
    skill = _make_safe_skill()

    with TonypiMCPClient("http://mock.local/mcp", transport=server.as_httpx_transport()) as client:
        trace = client.execute_skill(skill)
        assert trace.aborted is True
        assert trace.frames_executed == 0
        assert trace.abort_reason is not None and "Torso tilt" in trace.abort_reason
        assert _actuation_calls(server) == []


def test_mcp_multi_event_sse_and_heterogeneous_status_normalization() -> None:
    """Verify SSE stream with intermediate notification events and alternative status schema."""
    sample_jpg = MockTonypiMCPServer().render_camera_jpeg()
    sample_b64 = base64.b64encode(sample_jpg).decode("ascii")

    def _custom_handler(request: httpx.Request) -> httpx.Response:
        msg = json.loads(request.content.decode("utf-8"))
        method = msg["method"]
        req_id = msg.get("id")
        if method == "notifications/initialized":
            return httpx.Response(202, headers={"Mcp-Session-Id": "custom-sess"})
        if method == "initialize":
            res = {"protocolVersion": "2025-03-26", "capabilities": {}, "serverInfo": {"name": "alt"}}
        elif method == "tools/list":
            res = {
                "tools": [
                    {"name": "robot_status", "inputSchema": {"type": "object", "properties": {}}},
                    {"name": "capture_frame", "inputSchema": {"type": "object", "properties": {}}},
                ]
            }
        elif method == "tools/call" and msg["params"]["name"] == "robot_status":
            payload = {
                "battery_mv": 11500,
                "servo_temps": {"1": 39.5, "6": 44.2},
                "imu": {"pitch_deg": 4.2, "roll_deg": -2.1},
                "standing": True,
                "servos": {"6": 610},
            }
            res = {"content": [{"type": "text", "text": json.dumps(payload)}]}
        else:
            payload = {"image_b64": f"data:image/jpeg;base64,{sample_b64}"}
            res = {"content": [{"type": "text", "text": json.dumps(payload)}]}

        # Emit an SSE progress notification followed by the matching JSON-RPC result
        sse = (
            'event: message\ndata: {"jsonrpc":"2.0","method":"notifications/progress","params":{}}\n\n'
            f'event: message\ndata: {json.dumps({"jsonrpc": "2.0", "id": req_id, "result": res})}\n\n'
        )
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream", "Mcp-Session-Id": "custom-sess"},
            content=sse.encode(),
        )

    with TonypiMCPClient(
        "http://alt.local/mcp", transport=httpx.MockTransport(_custom_handler)
    ) as client:
        status = client.get_status()
        assert status.battery_v == pytest.approx(11.5, abs=0.01)
        assert status.max_servo_temp_c == pytest.approx(44.2, abs=0.01)
        assert status.torso_pitch_deg == pytest.approx(4.2, abs=0.01)
        assert status.torso_roll_deg == pytest.approx(-2.1, abs=0.01)
        assert status.servo_positions["r_shoulder_pitch"] == 610

        raw_img, mime = client.perceive()
        assert mime == "image/jpeg"
        assert raw_img == sample_jpg


def test_capture_synthetic_video_and_extract_preview_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "synth.mp4"
    created = make_synthetic_test_video(video_path, duration_sec=0.3)
    assert created.exists() and created.stat().st_size > 0

    frames = extract_preview_frames(created, count=2)
    assert len(frames) == 2
    for frame in frames:
        assert frame.startswith(b"\xff\xd8")
        with Image.open(io.BytesIO(frame)) as img:
            assert img.width == 320
            assert img.format == "JPEG"

    with pytest.raises(FileNotFoundError):
        extract_preview_frames(tmp_path / "missing.mp4")

    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a real mp4 file")
    fail_proc = subprocess.CompletedProcess(["ffmpeg"], returncode=1, stdout="", stderr="invalid")
    with (
        patch("roboharness.capture.subprocess.run", return_value=fail_proc),
        pytest.raises(CaptureError),
    ):
        extract_preview_frames(corrupt, count=1)
