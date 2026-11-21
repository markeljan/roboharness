"""Motion extraction, agentic video debugging, baseline generation, and reflex speech via gemini-4-hydrogen."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, Field

from roboharness.config import MOTION_MODEL
from roboharness.models import (
    build_gemini4_generation_config,
    compact_json_schema_for_gemini,
    extract_interaction_text,
    extract_json_from_text,
    upload_video_agentic,
)
from roboharness.schemas import (
    TONYPI_JOINTS,
    PoseSequence,
    RobotStatus,
    ServoKeyframe,
)

MAX_ANNOUNCEMENT_WORDS: Final[int] = 18

_JOINT_SPEC_SUMMARY: Final[str] = "\n".join(
    f"- {j.name} (ID {j.servo_id}, {j.role}): neutral={j.neutral_pulse}, "
    f"range=[{j.min_pulse}, {j.max_pulse}]"
    for j in TONYPI_JOINTS
)


class BaselineKeyframesPayload(BaseModel):
    """Structured wrapper returned by single-shot Baseline A servo generation."""

    keyframes: list[ServoKeyframe] = Field(min_length=1, max_length=32)


class TrialVideoObservation(BaseModel):
    """Timestamped limb/table observation produced by Agentic Video Processing."""

    timestamp_ms: int = Field(ge=0, description="Timestamp in milliseconds of observed event.")
    limb_or_region: str = Field(
        description="Body part or workspace region zoomed into (e.g. 'r_ankle', 'torso', 'table_edge')."
    )
    observation: str = Field(description="Visual finding at this timestamp.")
    severity: Literal["info", "warning", "failure"] = "info"


class TrialVideoDebugReport(BaseModel):
    """Structured post-trial failure & slip diagnosis from Agentic Video Processing."""

    skill_name: str
    succeeded: bool = Field(
        description="Whether the robot visually completed the expected move stably."
    )
    root_cause: str = Field(
        description="Concise diagnosis of any slip, tilt, limb clipping, or tracking error."
    )
    slip_detected: bool = False
    suspected_joints: list[str] = Field(default_factory=list)
    observations: list[TrialVideoObservation] = Field(default_factory=list)
    recommended_fixes: list[str] = Field(default_factory=list)


def _pose_extraction_prompt(*, skill_hint: str = "", description: str | None = None) -> str:
    source_clause = (
        "Analyze the attached human dance or gesture video using agentic timestamp scrubbing "
        "and limb inspection."
        if description is None
        else f"Synthesize a realistic human motion sequence for this requested dance/gesture:\n{description.strip()}"
    )
    hint_clause = (
        f"\nOptional skill hint from operator: {skill_hint.strip()}"
        if skill_hint.strip()
        else ""
    )
    return (
        f"{source_clause}{hint_clause}\n\n"
        "Output ONLY valid JSON matching the PoseSequence schema.\n"
        "Requirements:\n"
        "1. Extract 6 to 14 beat-synced HumanPoseKeyframe entries ordered by increasing `t_ms`, "
        "with zero-based `beat_index`.\n"
        "2. For every keyframe, capture realistic joint angles in degrees: "
        "`torso_pitch_deg` (+ forward, - backward), `torso_roll_deg` (+ right, - left), "
        "`l_shoulder_pitch_deg`, `l_shoulder_roll_deg`, `l_elbow_deg`, "
        "`r_shoulder_pitch_deg`, `r_shoulder_roll_deg`, `r_elbow_deg`, "
        "`l_hip_pitch_deg`, `l_knee_deg` (0..150), `r_hip_pitch_deg`, `r_knee_deg` (0..150), "
        "`foot_contact` ('both', 'left', 'right', or 'airborne'), and a concise `label`.\n"
        "3. Estimate `bpm`, total `duration_ms`, concise `source_description`, and `style_tags`.\n"
        "4. Explicitly populate `impossible_elements` for ANY motion that is physically impossible "
        "or unsafe verbatim for a 16-DOF tabletop servo humanoid robot (Hiwonder TonyPi, ~37cm tall, "
        "no jumping actuators): jumps/hops (airborne), backflips, frontflips, full splits, "
        "airborne spins, cartwheels, handstands, or high-impact leaps. For each entry provide "
        "`element`, `timestamp_ms`, `reason`, and `suggested_adaptation` (a grounded rhythmic "
        "adaptation preserving upper-body visual energy, or 'refuse' if inherently unadaptable)."
    )


def extract_motion_from_video(
    client: Any,
    video_path: Path,
    *,
    skill_hint: str = "",
    thinking_level: str = "medium",
) -> PoseSequence:
    """Extract a beat-synced ``PoseSequence`` from a reference video via Agentic Video Processing."""
    video_part = upload_video_agentic(client, video_path)
    prompt = _pose_extraction_prompt(skill_hint=skill_hint)

    interaction = client.interactions.create(
        model=MOTION_MODEL,
        input=[video_part, {"type": "text", "text": prompt}],
        generation_config=build_gemini4_generation_config(thinking_level),
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": compact_json_schema_for_gemini(PoseSequence),
        },
    )
    raw_text = extract_interaction_text(interaction)
    return PoseSequence.model_validate_json(extract_json_from_text(raw_text))


def debug_trial_video(
    client: Any,
    video_path: Path,
    *,
    skill_name: str,
    expected_behavior: str = "",
    thinking_level: str = "high",
) -> dict[str, Any]:
    """Diagnose a robot trial recording via Agentic Video Processing (``processing="agentic"``).

    Uploads the recorded trial video via the Files API with ``processing="agentic"`` and
    ``thinking_level="high"`` so Gemini can scrub timestamps and zoom into individual limbs,
    feet, and table edges to pinpoint slips, tipping, or joint-limit deviations.
    """
    video_part = upload_video_agentic(client, video_path)
    expected_clause = (
        f"Expected robot behavior: {expected_behavior.strip()}\n"
        if expected_behavior.strip()
        else ""
    )
    prompt = (
        "Inspect this Hiwonder TonyPi tabletop robot trial recording using agentic video "
        "timestamp scrubbing and close-up limb inspection.\n"
        f"Skill name: {skill_name}\n"
        f"{expected_clause}\n"
        "Output ONLY valid JSON matching the TrialVideoDebugReport schema:\n"
        "- Identify exact timestamps (ms) where feet slip, torso tilts excessively, arms clip "
        "or collide, or motion lags behind the intended choreography.\n"
        "- Populate `suspected_joints` using canonical TonyPi joint names when relevant.\n"
        "- Provide concrete, actionable `recommended_fixes` for the next Managed Agent forge pass."
    )

    interaction = client.interactions.create(
        model=MOTION_MODEL,
        input=[video_part, {"type": "text", "text": prompt}],
        generation_config=build_gemini4_generation_config(thinking_level),
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": compact_json_schema_for_gemini(TrialVideoDebugReport),
        },
    )
    raw_text = extract_interaction_text(interaction)
    report = TrialVideoDebugReport.model_validate_json(extract_json_from_text(raw_text))
    return report.model_dump(mode="json")


def extract_motion_from_description(
    client: Any,
    description: str,
) -> PoseSequence:
    """Extract a structured ``PoseSequence`` from a text motion description."""
    if not description.strip():
        raise ValueError("description must be non-empty")

    prompt = _pose_extraction_prompt(description=description)
    interaction = client.interactions.create(
        model=MOTION_MODEL,
        input=[{"type": "text", "text": prompt}],
        generation_config=build_gemini4_generation_config("medium"),
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": compact_json_schema_for_gemini(PoseSequence),
        },
    )
    raw_text = extract_interaction_text(interaction)
    return PoseSequence.model_validate_json(extract_json_from_text(raw_text))


def generate_baseline_raw_keyframes(
    client: Any,
    description_or_pose: str,
    video_path: Path | None = None,
) -> list[ServoKeyframe]:
    """Generate raw 16-DOF ``ServoKeyframe`` entries in a single LLM pass (Baseline A).

    Intentionally uses a single ``gemini-4-hydrogen`` call with NO Managed Agent sandbox
    verification and NO Decisions API gate so benchmarks can compare single-shot prompting
    directly against the full RoboHarness pipeline.
    """
    prompt = (
        "Directly generate a 16-DOF Hiwonder TonyPi servo keyframe sequence for the following "
        "motion request without external simulation.\n\n"
        f"Motion request / pose context:\n{description_or_pose.strip()}\n\n"
        "Use ONLY these 16 canonical TonyPi joint names (integer servo pulses 0..1000, "
        "500 = neutral, duration_ms 80..5000):\n"
        f"{_JOINT_SPEC_SUMMARY}\n\n"
        "Output ONLY valid JSON matching the BaselineKeyframesPayload schema with 4 to 12 "
        "expressive keyframes."
    )

    input_parts: list[dict[str, Any]] = []
    if video_path is not None:
        input_parts.append(upload_video_agentic(client, video_path))
    input_parts.append({"type": "text", "text": prompt})

    interaction = client.interactions.create(
        model=MOTION_MODEL,
        input=input_parts,
        generation_config=build_gemini4_generation_config("medium"),
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": compact_json_schema_for_gemini(BaselineKeyframesPayload),
        },
    )
    raw_text = extract_interaction_text(interaction)
    cleaned_json = extract_json_from_text(raw_text)
    parsed = json.loads(cleaned_json)
    if isinstance(parsed, list):
        return [ServoKeyframe.model_validate(item) for item in parsed]
    return BaselineKeyframesPayload.model_validate(parsed).keyframes


def _sanitize_announcement(raw_text: str) -> str:
    cleaned = re.sub(r"\s+", " ", raw_text).strip().strip("\"'`")
    if not cleaned:
        raise ValueError("Reflex announcement output was empty")

    words = cleaned.split()
    if len(words) > MAX_ANNOUNCEMENT_WORDS:
        trimmed = " ".join(words[:MAX_ANNOUNCEMENT_WORDS]).rstrip(",;:-")
        if trimmed and trimmed[-1] not in ".!?":
            trimmed += "."
        return trimmed
    return cleaned


def run_live_reflex_announcement(
    client: Any,
    skill_name: str,
    description: str,
    status: RobotStatus,
    frame_jpeg: bytes | None = None,
) -> str:
    """Generate a crisp <=18-word spoken robot announcement for ``mcp.say(...)``."""
    telemetry_payload = status.model_dump(mode="json", exclude={"raw", "servo_positions"})
    prompt = (
        "You are TonyPi, a friendly tabletop humanoid robot about to perform a verified skill.\n"
        f"Skill name: {skill_name}\n"
        f"Skill description: {description}\n"
        f"Live robot status JSON: {json.dumps(telemetry_payload, sort_keys=True)}\n\n"
        "Reply with ONE crisp, upbeat spoken sentence of at most 18 words announcing the move "
        "for text-to-speech. Output plain text only: no markdown, no bullet points, no quotes."
    )

    input_parts: list[dict[str, Any]] = []
    if frame_jpeg is not None:
        input_parts.append(
            {
                "type": "image",
                "data": base64.b64encode(frame_jpeg).decode("ascii"),
                "mime_type": "image/jpeg",
            }
        )
    input_parts.append({"type": "text", "text": prompt})

    interaction = client.interactions.create(
        model=MOTION_MODEL,
        input=input_parts,
        generation_config=build_gemini4_generation_config("medium"),
    )
    raw_text = extract_interaction_text(interaction)
    return _sanitize_announcement(raw_text)
