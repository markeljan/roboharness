"""Deterministic, non-LLM safety latches and 16-DOF TonyPi kinematics checks."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable

from roboharness.config import (
    MAX_SERVO_TEMP_C,
    MAX_SLEW_PULSE_PER_SEC,
    MAX_TORSO_TILT_DEG,
    MIN_BATTERY_V,
    MIN_KEYFRAME_DURATION_MS,
)
from roboharness.schemas import (
    JOINTS_BY_NAME,
    LOWER_BODY_JOINTS,
    NEUTRAL_POSE,
    KinematicsReport,
    RobotStatus,
    SafetyVerdict,
    ServoKeyframe,
    SkillArtifact,
)

# Approximate conversion for Hiwonder serial bus servo: 240 deg / 1000 pulses = 0.24 deg/pulse
DEG_PER_PULSE = 0.24


def check_robot_telemetry(status: RobotStatus) -> SafetyVerdict:
    """Enforce hard hardware safety latches on live robot telemetry."""
    reasons: list[str] = []
    tilt_deg = math.hypot(status.torso_pitch_deg, status.torso_roll_deg)

    if status.battery_v < MIN_BATTERY_V:
        reasons.append(
            f"Battery voltage {status.battery_v:.2f}V below hard minimum {MIN_BATTERY_V:.1f}V"
        )
    if status.max_servo_temp_c >= MAX_SERVO_TEMP_C:
        reasons.append(
            f"Max servo temp {status.max_servo_temp_c:.1f}C reached/exceeded {MAX_SERVO_TEMP_C:.1f}C limit"
        )
    if tilt_deg >= MAX_TORSO_TILT_DEG:
        reasons.append(
            f"Torso tilt {tilt_deg:.1f} deg reached/exceeded {MAX_TORSO_TILT_DEG:.1f} deg limit"
        )
    if not status.standing:
        reasons.append("Robot is not in a stable standing posture")

    return SafetyVerdict(
        safe=len(reasons) == 0,
        reasons=reasons,
        battery_v=status.battery_v,
        max_servo_temp_c=status.max_servo_temp_c,
        tilt_deg=round(tilt_deg, 2),
    )


def estimate_frame_tilt_and_com(servos: dict[str, int]) -> tuple[float, float]:
    """Estimate torso tilt (deg) and normalized CoM offset from 16-DOF servo pulses.

    Clean-room simplified 2D bipedal inverted-pendulum + upper-limb counterbalance model:
    - Ankle/hip pitch deviations drive sagittal tilt.
    - Ankle/hip roll asymmetry drives lateral tilt.
    - Asymmetric arm extensions shift CoM proportionally to arm mass fraction (~18%).
    """
    pose = {**NEUTRAL_POSE, **servos}

    # Pitch deviations from neutral (deg)
    r_hip_p = (pose["r_hip_pitch"] - 500) * DEG_PER_PULSE
    l_hip_p = (pose["l_hip_pitch"] - 500) * DEG_PER_PULSE
    r_ankle_p = (pose["r_ankle_pitch"] - 500) * DEG_PER_PULSE
    l_ankle_p = (pose["l_ankle_pitch"] - 500) * DEG_PER_PULSE
    r_knee = (pose["r_knee"] - 500) * DEG_PER_PULSE
    l_knee = (pose["l_knee"] - 500) * DEG_PER_PULSE

    # Coordinated knee bend + counter-ankle pitch keeps torso upright; uncompensated pitch leans torso
    avg_ankle_p = 0.5 * (r_ankle_p + l_ankle_p)
    avg_hip_p = 0.5 * (r_hip_p + l_hip_p)
    avg_knee = 0.5 * (r_knee + l_knee)
    sagittal_tilt = abs(avg_ankle_p + 0.55 * avg_hip_p - 0.35 * avg_knee)

    # Lateral roll from ankle/hip roll + leg length asymmetry (knee diff)
    r_ankle_r = (pose["r_ankle_roll"] - 500) * DEG_PER_PULSE
    l_ankle_r = (pose["l_ankle_roll"] - 500) * DEG_PER_PULSE
    r_hip_r = (pose["r_hip_roll"] - 500) * DEG_PER_PULSE
    l_hip_r = (pose["l_hip_roll"] - 500) * DEG_PER_PULSE
    knee_asym = abs(r_knee - l_knee)
    hip_asym = abs(r_hip_p - l_hip_p)

    lateral_tilt = abs(0.5 * (r_ankle_r + l_ankle_r) + 0.45 * (r_hip_r + l_hip_r)) + 0.22 * knee_asym + 0.12 * hip_asym

    # Arm leverage contribution (upper body mass ~18% of robot)
    r_sh_p = (pose["r_shoulder_pitch"] - 500) * DEG_PER_PULSE
    l_sh_p = (pose["l_shoulder_pitch"] - 500) * DEG_PER_PULSE
    r_sh_r = (pose["r_shoulder_roll"] - 500) * DEG_PER_PULSE
    l_sh_r = (pose["l_shoulder_roll"] - 500) * DEG_PER_PULSE
    arm_pitch_bias = 0.06 * abs(0.5 * (r_sh_p + l_sh_p))
    arm_roll_bias = 0.05 * abs(r_sh_r - l_sh_r)

    total_tilt_deg = math.hypot(sagittal_tilt + arm_pitch_bias, lateral_tilt + arm_roll_bias)
    # Support polygon margin normalizes 30 deg effective tilt to 1.0
    com_offset_norm = total_tilt_deg / MAX_TORSO_TILT_DEG
    return round(total_tilt_deg, 2), round(com_offset_norm, 3)


def check_skill_keyframes(
    keyframes: Iterable[ServoKeyframe],
    *,
    lower_body_adapted: bool = False,
    adaptation_notes: list[str] | None = None,
    sandbox_environment_id: str | None = None,
) -> KinematicsReport:
    """Verify 16-DOF joint limits, slew rates, CoM margin, and peak torso tilt."""
    frames = list(keyframes)
    if not frames:
        return KinematicsReport(
            verified=False,
            peak_torso_tilt_deg=0.0,
            max_com_offset_norm=0.0,
            joint_clipping_count=0,
            max_slew_rate_pps=0.0,
            lower_body_adapted=lower_body_adapted,
            adaptation_notes=list(adaptation_notes or []),
            violations=["Skill contains zero keyframes"],
            sandbox_environment_id=sandbox_environment_id,
        )

    clipping_count = 0
    peak_tilt = 0.0
    max_com = 0.0
    max_slew = 0.0
    violations: list[str] = []

    prev_pose = dict(NEUTRAL_POSE)

    for idx, frame in enumerate(frames):
        if frame.duration_ms < MIN_KEYFRAME_DURATION_MS:
            violations.append(
                f"frame[{idx}] duration_ms={frame.duration_ms} < {MIN_KEYFRAME_DURATION_MS}ms"
            )

        merged_pose = {**NEUTRAL_POSE, **frame.servos}
        dt_sec = max(frame.duration_ms / 1000.0, 0.001)

        for joint_name, pulse in frame.servos.items():
            spec = JOINTS_BY_NAME.get(joint_name)
            if spec is None:
                clipping_count += 1
                violations.append(f"frame[{idx}] unknown joint '{joint_name}'")
                continue
            if pulse < spec.min_pulse or pulse > spec.max_pulse:
                clipping_count += 1
                violations.append(
                    f"frame[{idx}] {joint_name}={pulse} outside [{spec.min_pulse}, {spec.max_pulse}]"
                )
            slew = abs(pulse - prev_pose[joint_name]) / dt_sec
            max_slew = max(max_slew, slew)
            if slew > MAX_SLEW_PULSE_PER_SEC:
                violations.append(
                    f"frame[{idx}] {joint_name} slew {slew:.1f} pps > {MAX_SLEW_PULSE_PER_SEC:.0f} pps"
                )

        tilt_deg, com_norm = estimate_frame_tilt_and_com(merged_pose)
        peak_tilt = max(peak_tilt, tilt_deg)
        max_com = max(max_com, com_norm)
        if tilt_deg >= MAX_TORSO_TILT_DEG:
            violations.append(
                f"frame[{idx}] estimated torso tilt {tilt_deg:.1f} deg >= {MAX_TORSO_TILT_DEG:.1f} deg"
            )

        prev_pose = merged_pose

    verified = (
        clipping_count == 0
        and peak_tilt < MAX_TORSO_TILT_DEG
        and max_slew <= MAX_SLEW_PULSE_PER_SEC
        and len(violations) == 0
    )

    return KinematicsReport(
        verified=verified,
        peak_torso_tilt_deg=round(peak_tilt, 2),
        max_com_offset_norm=round(max_com, 3),
        joint_clipping_count=clipping_count,
        max_slew_rate_pps=round(max_slew, 1),
        lower_body_adapted=lower_body_adapted,
        adaptation_notes=list(adaptation_notes or []),
        violations=violations,
        sandbox_environment_id=sandbox_environment_id,
    )


def project_to_upper_body_safe(keyframes: list[ServoKeyframe]) -> list[ServoKeyframe]:
    """Lock lower body to a grounded athletic bounce while preserving safe upper-body motion."""
    safe_frames: list[ServoKeyframe] = []
    prev_pose = dict(NEUTRAL_POSE)

    for idx, frame in enumerate(keyframes):
        # Gentle symmetric rhythm bounce on even/odd beats (well within CoM polygon)
        bounce = 24 if idx % 2 == 1 else 0
        new_servos: dict[str, int] = dict(NEUTRAL_POSE)
        new_servos["r_knee"] = 500 + bounce
        new_servos["l_knee"] = 500 + bounce
        new_servos["r_hip_pitch"] = 500 - bounce // 2
        new_servos["l_hip_pitch"] = 500 - bounce // 2
        new_servos["r_ankle_pitch"] = 500 + bounce // 2
        new_servos["l_ankle_pitch"] = 500 + bounce // 2

        # Preserve upper-body commands, strictly clamped to joint limits
        for name, pulse in frame.servos.items():
            if name in LOWER_BODY_JOINTS:
                continue
            spec = JOINTS_BY_NAME[name]
            new_servos[name] = max(spec.min_pulse, min(spec.max_pulse, int(pulse)))

        # Compute minimum safe duration so slew rate stays below MAX_SLEW_PULSE_PER_SEC
        max_delta = max(abs(new_servos[k] - prev_pose[k]) for k in new_servos)
        min_safe_ms = math.ceil((max_delta / (MAX_SLEW_PULSE_PER_SEC * 0.90)) * 1000.0)
        safe_duration = max(frame.duration_ms, min_safe_ms, MIN_KEYFRAME_DURATION_MS)

        safe_frames.append(
            ServoKeyframe(
                duration_ms=min(safe_duration, 5000),
                servos=new_servos,
                note=f"{frame.note} [upper-body safe stance]".strip(),
            )
        )
        prev_pose = new_servos

    return safe_frames


def compute_artifact_signature(artifact: SkillArtifact) -> str:
    """Compute canonical SHA-256 integrity hash over skill payload and kinematics."""
    payload = {
        "schema_version": artifact.schema_version,
        "name": artifact.name,
        "adaptation_mode": artifact.adaptation_mode,
        "keyframes": [kf.model_dump(mode="json") for kf in artifact.keyframes],
        "kinematics": {
            "verified": artifact.kinematics.verified,
            "peak_torso_tilt_deg": artifact.kinematics.peak_torso_tilt_deg,
            "joint_clipping_count": artifact.kinematics.joint_clipping_count,
            "lower_body_adapted": artifact.kinematics.lower_body_adapted,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def sign_skill_artifact(artifact: SkillArtifact) -> SkillArtifact:
    """Return a copy of SkillArtifact stamped with its canonical SHA-256 signature."""
    sig = compute_artifact_signature(artifact)
    return artifact.model_copy(update={"artifact_sha256": sig})


def verify_artifact_signature(artifact: SkillArtifact) -> bool:
    """Check whether artifact_sha256 matches the canonical skill payload."""
    if not artifact.artifact_sha256:
        return False
    return artifact.artifact_sha256 == compute_artifact_signature(artifact)
