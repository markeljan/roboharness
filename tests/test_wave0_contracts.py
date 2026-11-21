"""Wave 0 contract and deterministic safety latch tests."""

from roboharness.config import (
    API_VERSION,
    FORGE_AGENT_ID,
    REMOTE_MCP_NAME,
    RESERVED_AGENT_ID_PREFIXES,
    VALID_THINKING_LEVELS,
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
    assert API_VERSION == "v1beta"
    assert "minimal" not in VALID_THINKING_LEVELS
    assert not FORGE_AGENT_ID.startswith(RESERVED_AGENT_ID_PREFIXES)
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
