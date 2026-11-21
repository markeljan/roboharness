"""Clean-room MuJoCo Streamable HTTP MCP server & TonypiMCPClient integration tests."""

from __future__ import annotations

import base64
import importlib.util
import io
import math
import socket
import time
from collections.abc import Iterator

import httpx
import pytest
from PIL import Image

from roboharness.mcp_client import TonypiMCPClient
from roboharness.mujoco_mcp_server import MujocoTonypiMCPServer, resolve_mujoco_xml_path
from roboharness.safety import check_skill_keyframes, sign_skill_artifact
from roboharness.schemas import NEUTRAL_POSE, ServoKeyframe, SkillArtifact
from roboharness.studio_plugin import RoboHarnessStudioPlugin

_HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
_XML_PATH = resolve_mujoco_xml_path()

pytestmark = pytest.mark.skipif(
    not (_HAS_MUJOCO and _XML_PATH.is_file()),
    reason="MuJoCo (`uv sync --extra sim`) or TonyPi scene.xml not available",
)


@pytest.fixture(scope="module")
def mujoco_server() -> Iterator[MujocoTonypiMCPServer]:
    """Start one clean-room MuJoCo MCP server reused across tests for sub-second speed."""
    server = MujocoTonypiMCPServer(xml_path=_XML_PATH, friction=1.0)
    try:
        yield server
    finally:
        server.stop()


def _make_arm_wave_skill() -> SkillArtifact:
    frames = [
        ServoKeyframe(
            duration_ms=320,
            servos={**NEUTRAL_POSE, "r_shoulder_pitch": 650, "r_elbow": 600},
            note="wave_up",
        ),
        ServoKeyframe(
            duration_ms=320,
            servos={**NEUTRAL_POSE, "r_shoulder_pitch": 650, "r_elbow": 520},
            note="wave_left",
        ),
        ServoKeyframe(
            duration_ms=320,
            servos=dict(NEUTRAL_POSE),
            note="return_neutral",
        ),
    ]
    report = check_skill_keyframes(frames)
    assert report.verified, report.violations
    return sign_skill_artifact(SkillArtifact(name="mujoco_wave", keyframes=frames, kinematics=report))


def test_mujoco_mcp_connect_status_perceive_and_say(
    mujoco_server: MujocoTonypiMCPServer,
) -> None:
    with TonypiMCPClient(
        "http://mujoco.local/mcp", transport=mujoco_server.as_httpx_transport()
    ) as client:
        tools = client.connect()
        names = {t["name"] for t in tools}
        assert {
            "status",
            "perceive",
            "say",
            "play_frames",
            "set_pose",
            "simulate_motion",
            "reset",
        } <= names

        reset_res = client.reset_sim(friction=1.0)
        assert reset_res is not None
        assert reset_res["ok"] is True

        status = client.get_status()
        assert status.standing is True
        assert status.battery_v >= 11.0
        tilt = math.hypot(status.torso_pitch_deg, status.torso_roll_deg)
        assert tilt < 5.0
        assert status.raw["tilt_deg"] < 5.0
        assert status.raw["sim"] is True

        jpeg_bytes, mime = client.perceive()
        assert mime == "image/jpeg"
        assert jpeg_bytes.startswith(b"\xff\xd8")
        with Image.open(io.BytesIO(jpeg_bytes)) as img:
            assert img.size == (480, 360)
            assert img.format == "JPEG"

        assert client.say("MuJoCo online!") is True
        assert "MuJoCo online!" in mujoco_server.spoken_texts


def test_mujoco_mcp_play_frames_wave_stays_upright(
    mujoco_server: MujocoTonypiMCPServer,
) -> None:
    skill = _make_arm_wave_skill()
    with TonypiMCPClient(
        "http://mujoco.local/mcp", transport=mujoco_server.as_httpx_transport()
    ) as client:
        client.reset_sim(friction=1.0)
        trace = client.execute_skill(skill)

        assert trace.transport_mode == "batch_keyframes"
        assert trace.aborted is False
        assert trace.frames_executed == len(skill.keyframes)
        assert trace.peak_observed_tilt_deg < 10.0
        assert trace.final_status.standing is True
        assert trace.post_run_frame_b64 is not None
        decoded = base64.b64decode(trace.post_run_frame_b64)
        assert decoded.startswith(b"\xff\xd8")


def test_mujoco_mcp_reset_low_friction_and_extreme_lean_simulation(
    mujoco_server: MujocoTonypiMCPServer,
) -> None:
    extreme_frames = [
        ServoKeyframe(
            duration_ms=400,
            servos={
                **NEUTRAL_POSE,
                "r_ankle_pitch": 750,
                "l_ankle_pitch": 750,
                "r_hip_pitch": 750,
                "l_hip_pitch": 750,
            },
            note="extreme_forward_lean",
        ),
    ]
    with TonypiMCPClient(
        "http://mujoco.local/mcp", transport=mujoco_server.as_httpx_transport()
    ) as client:
        reset_payload = client.reset_sim(friction=0.25)
        assert reset_payload is not None
        assert reset_payload["friction"] == pytest.approx(0.25, abs=0.01)

        sim_res = client.simulate_raw_frames(extreme_frames, friction=0.25, settle_s=0.6)
        assert sim_res is not None
        assert sim_res["peak_tilt_deg"] > 25.0
        assert isinstance(sim_res.get("frame_b64"), str)
        assert base64.b64decode(sim_res["frame_b64"]).startswith(b"\xff\xd8")


def test_mujoco_mcp_http_endpoints_health_frame_and_mcp(
    mujoco_server: MujocoTonypiMCPServer,
) -> None:
    mcp_url = mujoco_server.start_http(host="127.0.0.1", port=0)
    try:
        base_url = mcp_url.removesuffix("/mcp")
        health = httpx.get(f"{base_url}/health", timeout=2.0)
        assert health.status_code == 200
        assert health.json()["ok"] is True

        snap = httpx.get(f"{base_url}/frame.jpg", timeout=2.0)
        assert snap.status_code == 200
        assert snap.headers["content-type"] == "image/jpeg"
        assert snap.content.startswith(b"\xff\xd8")

        with httpx.stream("GET", f"{base_url}/stream.mjpeg", timeout=2.0) as mjpeg:
            assert mjpeg.status_code == 200
            assert "multipart/x-mixed-replace" in mjpeg.headers["content-type"]
            chunk = next(mjpeg.iter_bytes(chunk_size=1024))
            assert b"--frame" in chunk

        with TonypiMCPClient(mcp_url) as client:
            client.reset_sim(friction=1.0)
            status = client.get_status()
            assert status.standing is True
    finally:
        mujoco_server.stop_http()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_mujoco_4_studio_web_viewer_sync_and_plugin(
    mujoco_server: MujocoTonypiMCPServer,
) -> None:
    import mujoco

    major, minor, *_ = (int(part) for part in mujoco.__version__.split(".")[:2])
    assert (major, minor) >= (4, 0), f"Expected MuJoCo >= 4.0, got {mujoco.__version__}"

    plugin_probe = RoboHarnessStudioPlugin()
    plugin_probe.update_telemetry(
        skill_name="probe_wave",
        tilt_deg=4.25,
        contacts=8,
        friction=0.75,
        status_label="FRAME 1/3",
        safety_ok=True,
    )
    snap = plugin_probe.snapshot()
    assert snap.skill_name == "probe_wave"
    assert snap.tilt_deg == pytest.approx(4.25, abs=0.01)
    assert snap.contacts == 8
    assert snap.friction == pytest.approx(0.75, abs=0.01)
    assert snap.safety_ok is True

    free_port = _find_free_port()
    studio_url = mujoco_server.start_studio(
        mode="web",
        port=free_port,
        open_browser=False,
        realtime=False,
    )
    mcp_url = mujoco_server.start_http(host="127.0.0.1", port=0)
    try:
        assert studio_url == f"http://localhost:{free_port}/"
        web_resp: httpx.Response | None = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                web_resp = httpx.get(f"http://127.0.0.1:{free_port}/", timeout=1.0)
                break
            except httpx.ConnectError:
                time.sleep(0.03)
        assert web_resp is not None
        assert web_resp.status_code == 200
        assert len(web_resp.content) > 1000

        base_url = mcp_url.removesuffix("/mcp")
        health = httpx.get(f"{base_url}/health", timeout=2.0).json()
        assert health["mujoco_version"] == mujoco.__version__
        assert health["studio_url"] == studio_url

        syncs_before = mujoco_server.studio_sync_count
        skill = _make_arm_wave_skill()
        with TonypiMCPClient(
            "http://mujoco.local/mcp", transport=mujoco_server.as_httpx_transport()
        ) as client:
            status = client.get_status()
            assert status.raw["mujoco_version"] == mujoco.__version__
            assert status.raw["studio_url"] == studio_url

            trace = client.execute_skill(skill)
            assert trace.aborted is False
            assert trace.peak_observed_tilt_deg < 10.0

        assert mujoco_server.studio_sync_count > syncs_before + 5
        assert mujoco_server._studio_plugin is not None
        final_snap = mujoco_server._studio_plugin.snapshot()
        assert final_snap.safety_ok is True
        assert final_snap.tilt_deg < 10.0
    finally:
        mujoco_server.stop_http()
        mujoco_server.stop_studio()
        assert mujoco_server.studio_url is None
