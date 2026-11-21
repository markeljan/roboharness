"""Multimodal 0-output-token safety, feasibility, and fidelity gate via gemini-decision-1."""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Mapping
from typing import Any, Final, cast

from roboharness.config import (
    DECISION_CONFIDENCE_THRESHOLD,
    DECISION_MODEL,
    MAX_SERVO_TEMP_C,
    MAX_TORSO_TILT_DEG,
    MIN_BATTERY_V,
)
from roboharness.safety import check_robot_telemetry
from roboharness.schemas import (
    NEUTRAL_POSE,
    DecisionGateResult,
    DecisionRoute,
    EffectiveAction,
    PoseSequence,
    RobotStatus,
    SkillArtifact,
)

MAX_DECISION_FRAMES: Final[int] = 6

VALID_ROUTES: Final[frozenset[str]] = frozenset(
    {
        "execute_and_save",
        "adapt_upper_body_only",
        "repair_in_forge",
        "refuse_impossible",
        "other",
    }
)

PRE_EXECUTION_QUESTIONS: Final[dict[str, dict[str, Any]]] = {
    "is_physically_safe": {
        "type": "noul",
        "instructions": (
            "Evaluate whether executing the candidate 16-DOF Hiwonder TonyPi skill on the "
            "physical bipedal humanoid robot is physically safe given the reference frames, "
            "live robot telemetry (battery >= 10.9V, servo temp < 70C, torso tilt < 30 deg, "
            "standing posture), workspace context, and kinematics verification report."
        ),
        "criteria": {
            "true_description": (
                "Physically safe to actuate: hardware latches pass, kinematics report is verified "
                "with zero joint clipping and peak torso tilt < 30 deg, and no tip-over, collision, "
                "or impossible airborne/inversion hazard is present."
            ),
            "false_description": (
                "Unsafe to actuate as proposed: risks tipping over, excessive tilt, joint-limit "
                "clipping, thermal/battery violation, table-edge hazard, or an ungrounded/impossible "
                "maneuver."
            ),
        },
    },
    "needs_lower_body_adaptation": {
        "type": "noul",
        "instructions": (
            "Determine whether the target human motion or proposed skill contains lower-body "
            "dynamics (jumps, airborne spins, splits, deep asymmetric squats, high kicks, single-leg "
            "balance, or destabilizing hip/ankle excursions) that should be adapted into a grounded "
            "athletic bounce stance while preserving expressive upper-body choreography."
        ),
        "criteria": {
            "true_description": (
                "Lower-body dynamics are unsafe or infeasible verbatim on TonyPi and should be "
                "locked/adapted to a grounded rhythmic bounce while keeping expressive arm motion."
            ),
            "false_description": (
                "The skill is already safely grounded or already adapted for bipedal stability "
                "without requiring additional lower-body stabilization."
            ),
        },
    },
    "move_fidelity": {
        "type": "score",
        "instructions": (
            "Score how faithfully the candidate TonyPi 16-DOF keyframe sequence preserves the "
            "visual rhythm, silhouette, and recognizable essence of the target human motion "
            "within TonyPi's physical bipedal limits."
        ),
        "criteria": [
            "0: Unrecognizable, broken/unsafe, or completely mismatched to the target motion.",
            "1: Minimal resemblance; core rhythm, timing, or upper-body silhouette is mostly lost.",
            "2: Recognizable, rhythmically aligned adaptation preserving the main motion essence.",
            "3: High-fidelity, beat-synced retargeting capturing the signature style and timing.",
        ],
    },
    "route": {
        "type": "choice",
        "instructions": (
            "Select the single best execution route for RoboHarness based on physical safety, "
            "live robot telemetry, kinematics verification, lower-body feasibility, and move "
            "fidelity."
        ),
        "criteria": {
            "execute_and_save": (
                "Skill is physically safe, kinematically verified, and faithful; execute on "
                "TonyPi and save to the signed skill library."
            ),
            "adapt_upper_body_only": (
                "Lower-body motion is ungrounded or risky, but the gesture/dance can be faithfully "
                "performed by locking legs to a grounded athletic bounce and animating the upper "
                "body."
            ),
            "repair_in_forge": (
                "Kinematics, balance, slew-rate, clipping, or fidelity defects exist that can be "
                "repaired by iterating in the Managed Agent skill forge."
            ),
            "refuse_impossible": (
                "Target motion or hardware/workspace state is fundamentally unsafe or impossible "
                "on TonyPi (e.g., backflip, full splits, critical battery/thermal/edge risk) and "
                "must be refused without moving servos."
            ),
            "other": (
                "Ambiguous, incomplete, or out-of-distribution situation not cleanly covered by "
                "the above routes; escalate for human review."
            ),
        },
    },
}

POST_RUN_QUESTIONS: Final[dict[str, dict[str, Any]]] = {
    "is_physically_safe": {
        "type": "noul",
        "instructions": (
            "Evaluate whether the observed post-run robot state and final IMU/thermal/battery "
            "telemetry confirm that executing this TonyPi skill was physically safe and left "
            "the robot upright within hard limits (battery >= 10.9V, temp < 70C, tilt < 30 deg)."
        ),
        "criteria": {
            "true_description": (
                "Observed execution remained safe and upright with healthy battery, thermal, "
                "and torso-tilt telemetry."
            ),
            "false_description": (
                "Observed execution showed instability, tip-over, excessive tilt, unsafe "
                "workspace drift, or hardware limit violation."
            ),
        },
    },
    "needs_lower_body_adaptation": {
        "type": "noul",
        "instructions": (
            "Based on the post-run observation and final IMU tilt telemetry, determine whether "
            "this skill requires lower-body stabilization/adaptation before being replayed again."
        ),
        "criteria": {
            "true_description": (
                "Observed sway, drift, or posture tilt indicates lower-body joints should be "
                "adapted to a safer grounded stance before replay."
            ),
            "false_description": (
                "Observed lower-body stability was solid and does not require further adaptation."
            ),
        },
    },
    "move_fidelity": {
        "type": "score",
        "instructions": (
            "Compare the reference motion frames against the observed post-run robot camera frame "
            "(when present) and final telemetry to score the observed execution fidelity."
        ),
        "criteria": [
            "0: Failed run, tipped/stalled robot, or zero resemblance to the reference motion.",
            "1: Poor observed fidelity with noticeable posture error or degraded execution.",
            "2: Good observed execution that remained upright and captured the core move.",
            "3: Crisp, balanced, high-fidelity observed execution matching the reference move.",
        ],
    },
    "route": {
        "type": "choice",
        "instructions": (
            "Select the single best post-run disposition for this skill based on observed safety, "
            "stability, and execution fidelity."
        ),
        "criteria": {
            "execute_and_save": (
                "Observed execution was safe, stable, and faithful; approve and keep/save in the "
                "signed skill library."
            ),
            "adapt_upper_body_only": (
                "Observed execution showed lower-body wobble; re-adapt to grounded upper-body-only "
                "mode before saving or replaying."
            ),
            "repair_in_forge": (
                "Observed execution had correctable balance or fidelity defects; send back to the "
                "skill forge for repair."
            ),
            "refuse_impossible": (
                "Observed execution proved unsafe or infeasible on this hardware; refuse and do "
                "not save."
            ),
            "other": (
                "Observed outcome is ambiguous or lacks sufficient evidence; escalate for human "
                "review."
            ),
        },
    },
}


def _select_frames(frame_jpegs: list[bytes], max_frames: int = MAX_DECISION_FRAMES) -> list[bytes]:
    """Return up to max_frames non-empty JPEG payloads evenly sampled across the list."""
    valid = [frame for frame in frame_jpegs if frame]
    if max_frames <= 0 or not valid:
        return []
    if len(valid) <= max_frames:
        return valid
    if max_frames == 1:
        return [valid[0]]
    last_idx = len(valid) - 1
    indices = [round(i * last_idx / (max_frames - 1)) for i in range(max_frames)]
    return [valid[idx] for idx in indices]


def _encode_image_block(jpeg_bytes: bytes) -> dict[str, Any]:
    """Encode raw JPEG bytes as a Decisions API low-resolution image input block."""
    return {
        "type": "image",
        "data": base64.b64encode(jpeg_bytes).decode("ascii"),
        "mime_type": "image/jpeg",
        "resolution": "low",
    }


def _summarize_keyframes(skill: SkillArtifact) -> list[dict[str, Any]]:
    """Build compact per-keyframe deltas from neutral pose for decision context."""
    summary: list[dict[str, Any]] = []
    for idx, kf in enumerate(skill.keyframes):
        active_deltas = {
            joint: pulse - NEUTRAL_POSE[joint]
            for joint, pulse in kf.servos.items()
            if pulse != NEUTRAL_POSE[joint]
        }
        summary.append(
            {
                "index": idx,
                "duration_ms": kf.duration_ms,
                "note": kf.note,
                "active_joint_deltas": active_deltas,
            }
        )
    return summary


def _summarize_pose_sequence(pose_seq: PoseSequence) -> dict[str, Any]:
    """Extract key stability and choreography signals from PoseSequence."""
    peak_pitch = max((abs(kf.torso_pitch_deg) for kf in pose_seq.keyframes), default=0.0)
    peak_roll = max((abs(kf.torso_roll_deg) for kf in pose_seq.keyframes), default=0.0)
    foot_contacts = [kf.foot_contact for kf in pose_seq.keyframes]
    return {
        "source_description": pose_seq.source_description,
        "bpm": pose_seq.bpm,
        "duration_ms": pose_seq.duration_ms,
        "keyframe_count": len(pose_seq.keyframes),
        "style_tags": list(pose_seq.style_tags),
        "foot_contacts": foot_contacts,
        "peak_human_torso_pitch_deg": round(peak_pitch, 2),
        "peak_human_torso_roll_deg": round(peak_roll, 2),
        "impossible_elements": [
            ie.model_dump(mode="json") for ie in pose_seq.impossible_elements
        ],
    }


def build_decision_input(
    skill: SkillArtifact,
    pose_seq: PoseSequence | None,
    frame_jpegs: list[bytes],
    robot_status: RobotStatus,
    *,
    extra_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build Decisions API input blocks: capped low-res JPEG images + JSON telemetry payload."""
    blocks: list[dict[str, Any]] = [
        _encode_image_block(jpg) for jpg in _select_frames(frame_jpegs, MAX_DECISION_FRAMES)
    ]

    tilt_deg = round(math.hypot(robot_status.torso_pitch_deg, robot_status.torso_roll_deg), 2)
    safety_verdict = check_robot_telemetry(robot_status)

    payload: dict[str, Any] = {
        "robot_status": {
            "battery_v": robot_status.battery_v,
            "max_servo_temp_c": robot_status.max_servo_temp_c,
            "torso_pitch_deg": robot_status.torso_pitch_deg,
            "torso_roll_deg": robot_status.torso_roll_deg,
            "torso_tilt_deg": tilt_deg,
            "standing": robot_status.standing,
        },
        "hardware_safety_verdict": safety_verdict.model_dump(mode="json"),
        "safety_limits": {
            "min_battery_v": MIN_BATTERY_V,
            "max_servo_temp_c": MAX_SERVO_TEMP_C,
            "max_torso_tilt_deg": MAX_TORSO_TILT_DEG,
        },
        "skill": {
            "name": skill.name,
            "description": skill.description,
            "adaptation_mode": skill.adaptation_mode,
            "source_video": skill.source_video,
            "keyframe_count": len(skill.keyframes),
            "total_duration_ms": sum(kf.duration_ms for kf in skill.keyframes),
            "kinematics": skill.kinematics.model_dump(mode="json"),
            "keyframes_summary": _summarize_keyframes(skill),
        },
    }

    if pose_seq is not None:
        payload["pose_sequence"] = _summarize_pose_sequence(pose_seq)

    if extra_context is not None:
        payload["extra_context"] = extra_context

    blocks.append({"type": "text", "text": json.dumps(payload, sort_keys=True)})
    return blocks


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a mapping or an object attribute."""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_jsonable_answer(answer: Any) -> dict[str, Any]:
    """Normalize an SDK Pydantic answer object or dict into a JSON-serializable dict."""
    if hasattr(answer, "model_dump"):
        return cast(dict[str, Any], answer.model_dump(mode="json"))
    if isinstance(answer, Mapping):
        return dict(answer)
    if hasattr(answer, "__dict__"):
        return {k: v for k, v in vars(answer).items() if not k.startswith("_")}
    return {"value": str(answer)}


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp numeric value into [low, high]."""
    return max(low, min(high, float(value)))


def _choose_safe_downgrade(
    is_physically_safe_prob: float,
    route_probabilities: dict[str, float],
) -> EffectiveAction:
    """Choose between repair_in_forge and refuse_impossible when safety overrides execution."""
    refuse_p = float(route_probabilities.get("refuse_impossible", 0.0))
    repair_p = float(route_probabilities.get("repair_in_forge", 0.0))
    if refuse_p > repair_p:
        return "refuse_impossible"
    if repair_p > refuse_p:
        return "repair_in_forge"
    if is_physically_safe_prob < 0.25:
        return "refuse_impossible"
    return "repair_in_forge"


def parse_decision_response(
    response: Any,
    *,
    confidence_threshold: float = DECISION_CONFIDENCE_THRESHOLD,
) -> DecisionGateResult:
    """Parse a CreateDecisionResponse (SDK model or dict) and enforce confidence/safety gates."""
    answers = _get_attr_or_key(response, "answers")
    if answers is None or not isinstance(answers, Mapping):
        raise ValueError("Decisions API response missing 'answers' mapping")

    required_keys = (
        "is_physically_safe",
        "needs_lower_body_adaptation",
        "move_fidelity",
        "route",
    )
    for key in required_keys:
        if key not in answers or answers[key] is None:
            raise ValueError(f"Decisions API response missing required answer '{key}'")

    safe_ans = answers["is_physically_safe"]
    adapt_ans = answers["needs_lower_body_adaptation"]
    fidelity_ans = answers["move_fidelity"]
    route_ans = answers["route"]

    safe_prob = _clamp(float(_get_attr_or_key(safe_ans, "noul")), 0.0, 1.0)
    safe_conf = _clamp(float(_get_attr_or_key(safe_ans, "confidence")), 0.0, 1.0)

    adapt_prob = _clamp(float(_get_attr_or_key(adapt_ans, "noul")), 0.0, 1.0)
    adapt_conf = _clamp(float(_get_attr_or_key(adapt_ans, "confidence")), 0.0, 1.0)

    fidelity_score = _clamp(float(_get_attr_or_key(fidelity_ans, "score")), 0.0, 3.0)
    fidelity_conf = _clamp(float(_get_attr_or_key(fidelity_ans, "confidence")), 0.0, 1.0)
    raw_legend = _get_attr_or_key(fidelity_ans, "legend", {}) or {}
    fidelity_legend = {str(k): str(v) for k, v in dict(raw_legend).items()}

    raw_choice = str(_get_attr_or_key(route_ans, "choice", "other"))
    route: DecisionRoute = (
        cast(DecisionRoute, raw_choice) if raw_choice in VALID_ROUTES else "other"
    )
    route_conf = _clamp(float(_get_attr_or_key(route_ans, "confidence")), 0.0, 1.0)
    raw_probs = _get_attr_or_key(route_ans, "probabilities", {}) or {}
    route_probabilities = {str(k): float(v) for k, v in dict(raw_probs).items()}

    min_confidence = min(route_conf, safe_conf)
    meets_confidence = (
        route_conf >= confidence_threshold and safe_conf >= confidence_threshold
    )

    if not meets_confidence or route == "other":
        auto_branch_allowed = False
        effective_action: EffectiveAction = "escalate_human"
    else:
        auto_branch_allowed = True
        needs_safety_downgrade = safe_prob < 0.50 and (
            route == "execute_and_save"
            or (route == "adapt_upper_body_only" and adapt_prob < 0.50)
        )
        if needs_safety_downgrade:
            effective_action = _choose_safe_downgrade(safe_prob, route_probabilities)
        else:
            effective_action = cast(EffectiveAction, route)

    raw_answers = {key: _to_jsonable_answer(answers[key]) for key in answers}

    return DecisionGateResult(
        is_physically_safe_prob=safe_prob,
        is_physically_safe_conf=safe_conf,
        needs_lower_body_adaptation_prob=adapt_prob,
        needs_lower_body_adaptation_conf=adapt_conf,
        move_fidelity_score=fidelity_score,
        move_fidelity_legend=fidelity_legend,
        move_fidelity_conf=fidelity_conf,
        route=route,
        route_probabilities=route_probabilities,
        route_conf=route_conf,
        min_confidence=min_confidence,
        auto_branch_allowed=auto_branch_allowed,
        effective_action=effective_action,
        raw_answers=raw_answers,
    )


def evaluate_pre_execution(
    client: Any,
    skill: SkillArtifact,
    pose_seq: PoseSequence | None,
    frame_jpegs: list[bytes],
    robot_status: RobotStatus,
    *,
    confidence_threshold: float = DECISION_CONFIDENCE_THRESHOLD,
    extra_context: dict[str, Any] | None = None,
) -> DecisionGateResult:
    """Evaluate candidate skill before actuation in one 0-output-token Decisions API call."""
    decision_input = build_decision_input(
        skill=skill,
        pose_seq=pose_seq,
        frame_jpegs=frame_jpegs,
        robot_status=robot_status,
        extra_context=extra_context,
    )
    response = client.decisions.create(
        model=DECISION_MODEL,
        input=decision_input,
        questions=PRE_EXECUTION_QUESTIONS,
    )
    return parse_decision_response(response, confidence_threshold=confidence_threshold)


def evaluate_post_run(
    client: Any,
    skill: SkillArtifact,
    reference_jpegs: list[bytes],
    post_run_jpeg: bytes | None,
    final_status: RobotStatus,
    *,
    confidence_threshold: float = DECISION_CONFIDENCE_THRESHOLD,
) -> DecisionGateResult:
    """Evaluate observed post-run camera frame and final IMU telemetry via gemini-decision-1."""
    has_post_run = bool(post_run_jpeg)
    if has_post_run and post_run_jpeg is not None:
        selected_refs = _select_frames(reference_jpegs, MAX_DECISION_FRAMES - 1)
        combined_frames = [*selected_refs, post_run_jpeg]
        post_run_context: dict[str, Any] = {
            "evaluation_stage": "post_run",
            "reference_frame_count": len(selected_refs),
            "has_post_run_frame": True,
            "post_run_frame_index": len(selected_refs),
        }
    else:
        selected_refs = _select_frames(reference_jpegs, MAX_DECISION_FRAMES)
        combined_frames = selected_refs
        post_run_context = {
            "evaluation_stage": "post_run",
            "reference_frame_count": len(selected_refs),
            "has_post_run_frame": False,
        }

    decision_input = build_decision_input(
        skill=skill,
        pose_seq=None,
        frame_jpegs=combined_frames,
        robot_status=final_status,
        extra_context=post_run_context,
    )
    response = client.decisions.create(
        model=DECISION_MODEL,
        input=decision_input,
        questions=POST_RUN_QUESTIONS,
    )
    return parse_decision_response(response, confidence_threshold=confidence_threshold)


def evaluate_baseline_skill(
    client: Any,
    skill: SkillArtifact,
    scenario_description: str,
    robot_status: RobotStatus,
    *,
    frame_jpegs: list[bytes] | None = None,
    confidence_threshold: float = DECISION_CONFIDENCE_THRESHOLD,
) -> DecisionGateResult:
    """Score a benchmark skill (Baseline A or Full RoboHarness) on the exact same rubric."""
    decision_input = build_decision_input(
        skill=skill,
        pose_seq=None,
        frame_jpegs=list(frame_jpegs or []),
        robot_status=robot_status,
        extra_context={
            "evaluation_stage": "benchmark_rubric",
            "scenario_description": scenario_description,
        },
    )
    response = client.decisions.create(
        model=DECISION_MODEL,
        input=decision_input,
        questions=PRE_EXECUTION_QUESTIONS,
    )
    return parse_decision_response(response, confidence_threshold=confidence_threshold)
