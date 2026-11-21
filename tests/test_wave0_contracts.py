"""Wave 0 contract and deterministic safety latch tests."""

from pathlib import Path

import pytest

from roboharness.config import (
    API_VERSION,
    BETA_API_VERSION,
    DECISION_CONFIDENCE_THRESHOLD,
    DECISION_SAFETY_PROB_THRESHOLD,
    FORBIDDEN_GEMINI4_PARAMS,
    FORGE_AGENT_ID,
    FORGE_DEEP_AGENT_ID,
    FORGE_DEEP_MODEL,
    REMOTE_MCP_NAME,
    RESERVED_AGENT_ID_PREFIXES,
    STABLE_API_VERSION,
    VALID_THINKING_LEVELS,
    RoboHarnessSettings,
    load_dotenv_if_present,
)
from roboharness.safety import (
    check_robot_telemetry,
    check_skill_keyframes,
    project_to_upper_body_safe,
    sign_skill_artifact,
    verify_artifact_signature,
)
from roboharness.schemas import (
    NEUTRAL_POSE,
    TONYPI_JOINTS,
    RobotStatus,
    ServoKeyframe,
    SkillArtifact,
)


def test_tonypi_has_16_unique_joints() -> None:
    assert len(TONYPI_JOINTS) == 16
    assert len({j.servo_id for j in TONYPI_JOINTS}) == 16
    assert len({j.name for j in TONYPI_JOINTS}) == 16


def test_config_invariants() -> None:
    assert STABLE_API_VERSION == "v1"
    assert BETA_API_VERSION == "v1beta"
    assert API_VERSION == "v1beta"
    assert "minimal" not in VALID_THINKING_LEVELS
    assert not FORGE_AGENT_ID.startswith(RESERVED_AGENT_ID_PREFIXES)
    assert not FORGE_DEEP_AGENT_ID.startswith(RESERVED_AGENT_ID_PREFIXES)
    assert FORGE_DEEP_MODEL == "gemini-4-carbon"
    assert FORGE_DEEP_AGENT_ID == "roboharness-skill-forge-carbon"
    assert DECISION_CONFIDENCE_THRESHOLD == 0.75
    assert DECISION_SAFETY_PROB_THRESHOLD == 0.75
    assert "max_total_tokens" in FORBIDDEN_GEMINI4_PARAMS
    assert "max_output_tokens" in FORBIDDEN_GEMINI4_PARAMS
    assert REMOTE_MCP_NAME == "tonypi_mcp"


def test_telemetry_safety_latches() -> None:
    ok = check_robot_telemetry(RobotStatus(battery_v=11.4, max_servo_temp_c=42.0))
    assert ok.safe

    low_v = check_robot_telemetry(RobotStatus(battery_v=10.6, max_servo_temp_c=42.0))
    assert not low_v.safe

    hot = check_robot_telemetry(RobotStatus(battery_v=11.4, max_servo_temp_c=71.2))
    assert not hot.safe

    tilted = check_robot_telemetry(
        RobotStatus(battery_v=11.4, max_servo_temp_c=42.0, torso_pitch_deg=25.0, torso_roll_deg=20.0)
    )
    assert not tilted.safe


def test_kinematics_clipping_and_projection() -> None:
    unsafe = [
        ServoKeyframe(
            duration_ms=400,
            servos={**NEUTRAL_POSE, "r_shoulder_pitch": 950, "r_hip_pitch": 710, "r_ankle_pitch": 670},
        )
    ]
    rep = check_skill_keyframes(unsafe)
    assert not rep.verified
    assert rep.joint_clipping_count == 1

    projected = project_to_upper_body_safe(unsafe)
    rep_safe = check_skill_keyframes(projected, lower_body_adapted=True)
    assert rep_safe.verified
    assert rep_safe.joint_clipping_count == 0
    assert rep_safe.peak_torso_tilt_deg < 30.0


def test_artifact_signing_and_tamper_detection() -> None:
    frames = [ServoKeyframe(duration_ms=400, servos=dict(NEUTRAL_POSE))]
    report = check_skill_keyframes(frames)
    artifact = sign_skill_artifact(SkillArtifact(name="wave", keyframes=frames, kinematics=report))
    assert verify_artifact_signature(artifact)

    tampered = artifact.model_copy(update={"name": "tampered"})
    assert not verify_artifact_signature(tampered)


def test_dotenv_loading_and_env_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Comment line\n"
        'GEMINI_API_KEY="from-dotenv-key"\n'
        "export TONYPI_MCP_URL='http://10.0.0.5:8765/mcp'\n"
        "GEMINI_ENVIRONMENT_ID=env_self_hosted_mujoco\n"
        "ROBOHARNESS_WEBCAM_DEVICE=2 # inline comment\n",
        encoding="utf-8",
    )

    # Register keys with monkeypatch first so teardown cleanly removes vars added by dotenv
    for env_key in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_ENVIRONMENT_ID",
        "TONYPI_MCP_URL",
        "ROBOHARNESS_WEBCAM_DEVICE",
    ):
        monkeypatch.setenv(env_key, "")
        monkeypatch.delenv(env_key, raising=False)
    monkeypatch.setenv("TONYPI_MCP_URL", "http://127.0.0.1:9999/mcp")

    load_dotenv_if_present(env_file)
    settings = RoboHarnessSettings.from_env(env_file)

    assert settings.gemini_api_key == "from-dotenv-key"
    assert settings.tonypi_mcp_url == "http://127.0.0.1:9999/mcp"
    assert settings.gemini_environment_id == "env_self_hosted_mujoco"
    assert settings.webcam_device == "2"
