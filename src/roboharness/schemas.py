"""Frozen Pydantic v2 schemas and Hiwonder TonyPi 16-DOF kinematic definitions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

JointRole = Literal["upper_body", "lower_body"]


class JointSpec(BaseModel):
    """Hardware specification for a single TonyPi serial bus servo."""

    model_config = ConfigDict(frozen=True)

    servo_id: int = Field(ge=1, le=16)
    name: str
    min_pulse: int = Field(ge=0, le=1000)
    max_pulse: int = Field(ge=0, le=1000)
    neutral_pulse: int = Field(default=500, ge=0, le=1000)
    role: JointRole


# Clean-room Hiwonder TonyPi 16-DOF body servo layout (pulse range 0..1000, ~4.167 pulse/deg)
TONYPI_JOINTS: Final[tuple[JointSpec, ...]] = (
    # Right leg (IDs 1-5)
    JointSpec(servo_id=1, name="r_ankle_roll", min_pulse=350, max_pulse=650, neutral_pulse=500, role="lower_body"),
    JointSpec(servo_id=2, name="r_ankle_pitch", min_pulse=320, max_pulse=680, neutral_pulse=500, role="lower_body"),
    JointSpec(servo_id=3, name="r_knee", min_pulse=260, max_pulse=740, neutral_pulse=500, role="lower_body"),
    JointSpec(servo_id=4, name="r_hip_pitch", min_pulse=280, max_pulse=720, neutral_pulse=500, role="lower_body"),
    JointSpec(servo_id=5, name="r_hip_roll", min_pulse=360, max_pulse=640, neutral_pulse=500, role="lower_body"),
    # Right arm (IDs 6-8)
    JointSpec(servo_id=6, name="r_shoulder_pitch", min_pulse=150, max_pulse=850, neutral_pulse=500, role="upper_body"),
    JointSpec(servo_id=7, name="r_shoulder_roll", min_pulse=180, max_pulse=820, neutral_pulse=500, role="upper_body"),
    JointSpec(servo_id=8, name="r_elbow", min_pulse=180, max_pulse=820, neutral_pulse=500, role="upper_body"),
    # Left leg (IDs 9-13)
    JointSpec(servo_id=9, name="l_ankle_roll", min_pulse=350, max_pulse=650, neutral_pulse=500, role="lower_body"),
    JointSpec(servo_id=10, name="l_ankle_pitch", min_pulse=320, max_pulse=680, neutral_pulse=500, role="lower_body"),
    JointSpec(servo_id=11, name="l_knee", min_pulse=260, max_pulse=740, neutral_pulse=500, role="lower_body"),
    JointSpec(servo_id=12, name="l_hip_pitch", min_pulse=280, max_pulse=720, neutral_pulse=500, role="lower_body"),
    JointSpec(servo_id=13, name="l_hip_roll", min_pulse=360, max_pulse=640, neutral_pulse=500, role="lower_body"),
    # Left arm (IDs 14-16)
    JointSpec(servo_id=14, name="l_shoulder_pitch", min_pulse=150, max_pulse=850, neutral_pulse=500, role="upper_body"),
    JointSpec(servo_id=15, name="l_shoulder_roll", min_pulse=180, max_pulse=820, neutral_pulse=500, role="upper_body"),
    JointSpec(servo_id=16, name="l_elbow", min_pulse=180, max_pulse=820, neutral_pulse=500, role="upper_body"),
)

JOINTS_BY_NAME: Final[dict[str, JointSpec]] = {j.name: j for j in TONYPI_JOINTS}
JOINTS_BY_ID: Final[dict[int, JointSpec]] = {j.servo_id: j for j in TONYPI_JOINTS}
LOWER_BODY_JOINTS: Final[frozenset[str]] = frozenset(
    j.name for j in TONYPI_JOINTS if j.role == "lower_body"
)
UPPER_BODY_JOINTS: Final[frozenset[str]] = frozenset(
    j.name for j in TONYPI_JOINTS if j.role == "upper_body"
)
NEUTRAL_POSE: Final[dict[str, int]] = {j.name: j.neutral_pulse for j in TONYPI_JOINTS}


class HumanPoseKeyframe(BaseModel):
    """Single beat-synced human pose extracted from video by gemini-4-hydrogen."""

    t_ms: int = Field(ge=0, description="Timestamp in milliseconds from start of video.")
    beat_index: int = Field(ge=0, description="Zero-based rhythmic beat index.")
    torso_pitch_deg: float = Field(ge=-360.0, le=360.0, description="Forward(+) / backward(-) lean or inversion in degrees.")
    torso_roll_deg: float = Field(ge=-360.0, le=360.0, description="Right(+) / left(-) lateral lean or roll in degrees.")
    l_shoulder_pitch_deg: float = Field(ge=-360.0, le=360.0)
    l_shoulder_roll_deg: float = Field(ge=-360.0, le=360.0)
    l_elbow_deg: float = Field(ge=-180.0, le=180.0)
    r_shoulder_pitch_deg: float = Field(ge=-360.0, le=360.0)
    r_shoulder_roll_deg: float = Field(ge=-360.0, le=360.0)
    r_elbow_deg: float = Field(ge=-180.0, le=180.0)
    l_hip_pitch_deg: float = Field(ge=-180.0, le=180.0)
    l_knee_deg: float = Field(ge=-30.0, le=180.0)
    r_hip_pitch_deg: float = Field(ge=-180.0, le=180.0)
    r_knee_deg: float = Field(ge=-30.0, le=180.0)
    foot_contact: Literal["both", "left", "right", "airborne"] = "both"
    label: str = Field(default="", description="Short beat label, e.g. 'arms_left_swing'.")


class ImpossibleElement(BaseModel):
    """Physically impossible human motion flagged for TonyPi bipedal hardware."""

    element: str = Field(description="E.g. 'jump', 'backflip', 'splits', 'airborne_spin'.")
    timestamp_ms: int = Field(ge=0)
    reason: str = Field(description="Why TonyPi cannot physically perform this verbatim.")
    suggested_adaptation: str = Field(
        description="Grounded adaptation preserving visual rhythm, or 'refuse' if unadaptable."
    )


class PoseSequence(BaseModel):
    """Structured output produced by gemini-4-hydrogen from input video."""

    source_description: str = Field(description="Concise description of the observed dance or gesture.")
    bpm: float = Field(default=110.0, ge=30.0, le=240.0)
    duration_ms: int = Field(ge=200, le=60000)
    keyframes: list[HumanPoseKeyframe] = Field(min_length=2, max_length=32)
    impossible_elements: list[ImpossibleElement] = Field(default_factory=list)
    style_tags: list[str] = Field(default_factory=list)


class ServoKeyframe(BaseModel):
    """Retargeted 16-DOF TonyPi servo keyframe."""

    duration_ms: int = Field(ge=80, le=5000, description="Transition duration in milliseconds.")
    servos: dict[str, int] = Field(description="Mapping from canonical TonyPi joint name to pulse 0..1000.")
    note: str = ""

    @field_validator("servos")
    @classmethod
    def _validate_servo_keys(cls, v: dict[str, int]) -> dict[str, int]:
        unknown = [k for k in v if k not in JOINTS_BY_NAME]
        if unknown:
            raise ValueError(f"Unknown TonyPi joint names: {unknown}")
        return v


class KinematicsReport(BaseModel):
    """Deterministic kinematics & CoM verification summary."""

    verified: bool
    peak_torso_tilt_deg: float = Field(ge=0.0)
    max_com_offset_norm: float = Field(ge=0.0, description="CoM lateral/sagittal offset normalized to foot polygon (<=1.0 safe).")
    joint_clipping_count: int = Field(ge=0)
    max_slew_rate_pps: float = Field(ge=0.0, description="Peak servo speed in pulses/sec.")
    lower_body_adapted: bool = False
    adaptation_notes: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    sandbox_environment_id: str | None = None


DecisionRoute = Literal[
    "execute_and_save",
    "adapt_upper_body_only",
    "repair_in_forge",
    "refuse_impossible",
    "other",
]
EffectiveAction = Literal[
    "execute_and_save",
    "adapt_upper_body_only",
    "repair_in_forge",
    "refuse_impossible",
    "escalate_human",
]


class DecisionGateResult(BaseModel):
    """Output of a 0-output-token gemini-decision-1 multimodal gate."""

    is_physically_safe_prob: float = Field(ge=0.0, le=1.0)
    is_physically_safe_conf: float = Field(ge=0.0, le=1.0)
    needs_lower_body_adaptation_prob: float = Field(ge=0.0, le=1.0)
    needs_lower_body_adaptation_conf: float = Field(ge=0.0, le=1.0)
    move_fidelity_score: float = Field(ge=0.0, le=3.0)
    move_fidelity_legend: dict[str, str] = Field(default_factory=dict)
    move_fidelity_conf: float = Field(ge=0.0, le=1.0)
    route: DecisionRoute
    route_probabilities: dict[str, float] = Field(default_factory=dict)
    route_conf: float = Field(ge=0.0, le=1.0)
    min_confidence: float = Field(ge=0.0, le=1.0)
    auto_branch_allowed: bool
    effective_action: EffectiveAction
    raw_answers: dict[str, Any] = Field(default_factory=dict)


class SpatialFramingResult(BaseModel):
    """Workspace framing check from gemini-robotics-er-2-preview."""

    centered_on_table: bool
    workspace_clear: bool
    robot_bbox_norm: list[int] | None = Field(
        default=None,
        description="Optional [ymin, xmin, ymax, xmax] normalized 0..1000.",
    )
    clearance_margin: Literal["safe", "tight", "edge_risk"]
    summary: str


class RobotStatus(BaseModel):
    """Live hardware telemetry returned via Streamable HTTP MCP."""

    battery_v: float
    max_servo_temp_c: float
    torso_pitch_deg: float = 0.0
    torso_roll_deg: float = 0.0
    standing: bool = True
    servo_positions: dict[str, int] = Field(default_factory=lambda: dict(NEUTRAL_POSE))
    raw: dict[str, Any] = Field(default_factory=dict)


class SafetyVerdict(BaseModel):
    """Outcome of deterministic non-LLM safety latch evaluation."""

    safe: bool
    reasons: list[str] = Field(default_factory=list)
    battery_v: float | None = None
    max_servo_temp_c: float | None = None
    tilt_deg: float | None = None


class ExecutionTrace(BaseModel):
    """Summary of executing a verified skill on TonyPi MCP."""

    skill_name: str
    transport_mode: Literal["batch_keyframes", "frame_by_frame", "action_group", "dry_run"]
    frames_executed: int = Field(ge=0)
    aborted: bool = False
    abort_reason: str | None = None
    peak_observed_tilt_deg: float = Field(ge=0.0, default=0.0)
    final_status: RobotStatus
    post_run_frame_b64: str | None = None


class SkillArtifact(BaseModel):
    """Signed, verified, reusable TonyPi motion skill stored in skills/<name>.json."""

    schema_version: str = "1.0"
    name: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    source_video: str = ""
    description: str = ""
    adaptation_mode: Literal["full_body", "upper_body_adapted"] = "full_body"
    keyframes: list[ServoKeyframe] = Field(min_length=1)
    kinematics: KinematicsReport
    decision_gate: DecisionGateResult | None = None
    post_run_fidelity_score: float | None = Field(default=None, ge=0.0, le=3.0)
    artifact_sha256: str = ""
