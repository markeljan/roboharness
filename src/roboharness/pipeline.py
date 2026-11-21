"""End-to-end 4-model RoboHarness MimicPipeline and preflight verification."""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from roboharness.capture import CaptureError, extract_preview_frames
from roboharness.config import RoboHarnessSettings
from roboharness.decision_gate import evaluate_post_run, evaluate_pre_execution
from roboharness.genai_client import build_gemini4_generation_config, extract_interaction_text
from roboharness.managed_forge import (
    ForgeResult,
    deterministic_retarget_fallback,
    ensure_forge_agent,
    forge_skill,
)
from roboharness.mcp_client import MCPClientError, TonypiMCPClient
from roboharness.mock_mcp import MockTonypiMCPServer
from roboharness.motion_extractor import extract_motion_from_video, run_live_reflex_announcement
from roboharness.safety import (
    check_robot_telemetry,
    check_skill_keyframes,
    project_to_upper_body_safe,
    sign_skill_artifact,
)
from roboharness.schemas import (
    DecisionGateResult,
    ExecutionTrace,
    HumanPoseKeyframe,
    PoseSequence,
    RobotStatus,
    ServoKeyframe,
    SkillArtifact,
    SpatialFramingResult,
)
from roboharness.skill_store import load_skill, save_skill, slugify_skill_name
from roboharness.spatial import check_workspace_framing
from roboharness.telemetry_sse import TelemetryHub

logger = logging.getLogger(__name__)

MimicStatus = Literal[
    "executed_and_saved",
    "dry_run_saved",
    "refused",
    "escalated_human",
    "safety_aborted",
]


class MimicResult(BaseModel):
    """Structured outcome of a RoboHarness mimic or saved-skill replay run."""

    skill_name: str
    status: MimicStatus
    reason: str = ""
    saved_path: str | None = None
    pose_sequence: PoseSequence | None = None
    spatial: SpatialFramingResult | None = None
    artifact: SkillArtifact | None = None
    pre_decision: DecisionGateResult | None = None
    post_decision: DecisionGateResult | None = None
    execution: ExecutionTrace | None = None
    announcement: str = ""
    forge_attempts: int = 0


class CheckItem(BaseModel):
    """Result of a single preflight component probe."""

    component: str
    ok: bool
    detail: str
    latency_ms: float = 0.0


class PreflightReport(BaseModel):
    """Aggregate preflight status across all 4 Gemini models and TonyPi MCP."""

    all_ok: bool
    items: list[CheckItem] = Field(default_factory=list)


def _offline_synthetic_wave_pose(skill_name: str) -> PoseSequence:
    """Deterministic grounded greeting-wave PoseSequence used in offline mode without a client."""
    return PoseSequence(
        source_description=f"Offline synthetic greeting wave for '{skill_name}'",
        bpm=112.0,
        duration_ms=1600,
        keyframes=[
            HumanPoseKeyframe(
                t_ms=0,
                beat_index=0,
                torso_pitch_deg=2.0,
                torso_roll_deg=0.0,
                l_shoulder_pitch_deg=5.0,
                l_shoulder_roll_deg=8.0,
                l_elbow_deg=10.0,
                r_shoulder_pitch_deg=55.0,
                r_shoulder_roll_deg=25.0,
                r_elbow_deg=35.0,
                l_hip_pitch_deg=4.0,
                l_knee_deg=8.0,
                r_hip_pitch_deg=4.0,
                r_knee_deg=8.0,
                foot_contact="both",
                label="wave_raise",
            ),
            HumanPoseKeyframe(
                t_ms=450,
                beat_index=1,
                torso_pitch_deg=2.0,
                torso_roll_deg=3.0,
                l_shoulder_pitch_deg=5.0,
                l_shoulder_roll_deg=8.0,
                l_elbow_deg=10.0,
                r_shoulder_pitch_deg=65.0,
                r_shoulder_roll_deg=45.0,
                r_elbow_deg=50.0,
                l_hip_pitch_deg=4.0,
                l_knee_deg=8.0,
                r_hip_pitch_deg=4.0,
                r_knee_deg=8.0,
                foot_contact="both",
                label="wave_out",
            ),
            HumanPoseKeyframe(
                t_ms=900,
                beat_index=2,
                torso_pitch_deg=2.0,
                torso_roll_deg=-2.0,
                l_shoulder_pitch_deg=5.0,
                l_shoulder_roll_deg=8.0,
                l_elbow_deg=10.0,
                r_shoulder_pitch_deg=60.0,
                r_shoulder_roll_deg=15.0,
                r_elbow_deg=25.0,
                l_hip_pitch_deg=4.0,
                l_knee_deg=8.0,
                r_hip_pitch_deg=4.0,
                r_knee_deg=8.0,
                foot_contact="both",
                label="wave_in",
            ),
            HumanPoseKeyframe(
                t_ms=1400,
                beat_index=3,
                torso_pitch_deg=0.0,
                torso_roll_deg=0.0,
                l_shoulder_pitch_deg=0.0,
                l_shoulder_roll_deg=0.0,
                l_elbow_deg=0.0,
                r_shoulder_pitch_deg=0.0,
                r_shoulder_roll_deg=0.0,
                r_elbow_deg=0.0,
                l_hip_pitch_deg=0.0,
                l_knee_deg=0.0,
                r_hip_pitch_deg=0.0,
                r_knee_deg=0.0,
                foot_contact="both",
                label="neutral_return",
            ),
        ],
        impossible_elements=[],
        style_tags=["gesture", "greeting", "offline"],
    )


def _offline_decision_gate(
    artifact: SkillArtifact,
    pose_seq: PoseSequence,
    status: RobotStatus,
    *,
    confidence_threshold: float,
) -> DecisionGateResult:
    """Deterministic local decision gate when running with offline_mode=True and client=None."""
    telemetry_ok = check_robot_telemetry(status).safe
    has_hard_refusal = any(
        el.suggested_adaptation.strip().lower() == "refuse"
        for el in pose_seq.impossible_elements
    )

    if not telemetry_ok or has_hard_refusal:
        return DecisionGateResult(
            is_physically_safe_prob=0.10,
            is_physically_safe_conf=0.95,
            needs_lower_body_adaptation_prob=0.90,
            needs_lower_body_adaptation_conf=0.90,
            move_fidelity_score=0.5,
            move_fidelity_legend={"0": "unsafe", "1": "poor", "2": "good", "3": "high"},
            move_fidelity_conf=0.90,
            route="refuse_impossible",
            route_probabilities={"refuse_impossible": 0.92, "other": 0.08},
            route_conf=0.95,
            min_confidence=0.95,
            auto_branch_allowed=0.95 >= confidence_threshold,
            effective_action="refuse_impossible",
        )

    if not artifact.kinematics.verified:
        return DecisionGateResult(
            is_physically_safe_prob=0.25,
            is_physically_safe_conf=0.92,
            needs_lower_body_adaptation_prob=0.80,
            needs_lower_body_adaptation_conf=0.88,
            move_fidelity_score=1.2,
            move_fidelity_legend={"0": "unsafe", "1": "poor", "2": "good", "3": "high"},
            move_fidelity_conf=0.85,
            route="repair_in_forge",
            route_probabilities={"repair_in_forge": 0.88, "other": 0.12},
            route_conf=0.92,
            min_confidence=0.92,
            auto_branch_allowed=0.92 >= confidence_threshold,
            effective_action="repair_in_forge",
        )

    fidelity = 2.6 if artifact.adaptation_mode == "full_body" else 2.2
    return DecisionGateResult(
        is_physically_safe_prob=0.95,
        is_physically_safe_conf=0.94,
        needs_lower_body_adaptation_prob=0.15 if artifact.adaptation_mode == "full_body" else 0.75,
        needs_lower_body_adaptation_conf=0.90,
        move_fidelity_score=fidelity,
        move_fidelity_legend={"0": "unsafe", "1": "poor", "2": "good", "3": "high"},
        move_fidelity_conf=0.91,
        route="execute_and_save",
        route_probabilities={"execute_and_save": 0.93, "other": 0.07},
        route_conf=0.94,
        min_confidence=0.94,
        auto_branch_allowed=0.94 >= confidence_threshold,
        effective_action="execute_and_save",
    )


class MimicPipeline:
    """Orchestrates the 4-model RoboHarness mimicry & execution workflow."""

    def __init__(
        self,
        *,
        client: Any | None,
        mcp: TonypiMCPClient,
        settings: RoboHarnessSettings,
        hub: TelemetryHub | None = None,
        offline_mode: bool = False,
    ) -> None:
        self.client = client
        self.mcp = mcp
        self.settings = settings
        self.hub = hub
        self.offline_mode = offline_mode

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        if self.hub is not None:
            self.hub.emit(event_type, data)

    def _use_live_models(self) -> bool:
        return self.client is not None and not self.offline_mode

    def _check_hardware_preflight(self, skill_name: str) -> tuple[RobotStatus, str | None]:
        self._emit("stage", {"stage": "preflight", "skill_name": skill_name, "message": "Checking TonyPi telemetry"})
        status = self.mcp.get_status()
        verdict = check_robot_telemetry(status)
        self._emit(
            "telemetry",
            {
                "battery_v": status.battery_v,
                "max_servo_temp_c": status.max_servo_temp_c,
                "tilt_deg": verdict.tilt_deg if verdict.tilt_deg is not None else 0.0,
                "standing": status.standing,
                "safe": verdict.safe,
                "message": (
                    f"Telemetry OK (bat={status.battery_v:.2f}V, temp={status.max_servo_temp_c:.1f}C, "
                    f"tilt={verdict.tilt_deg:.1f} deg)"
                    if verdict.safe
                    else "; ".join(verdict.reasons)
                ),
            },
        )
        if not verdict.safe:
            return status, "; ".join(verdict.reasons)
        return status, None

    def _run_spatial_check(
        self,
        *,
        run_spatial: bool,
    ) -> tuple[SpatialFramingResult | None, bytes | None]:
        if not run_spatial or not self._use_live_models():
            return None, None

        self._emit(
            "stage",
            {
                "stage": "spatial",
                "model": self.settings.spatial_model,
                "status": "active",
                "message": "Checking tabletop workspace framing",
            },
        )
        try:
            frame_jpeg, _ = self.mcp.perceive()
        except (MCPClientError, OSError, ValueError) as exc:
            logger.warning("Skipping spatial check because mcp.perceive() failed: %s", exc)
            self._emit(
                "spatial",
                {
                    "model": self.settings.spatial_model,
                    "status": "done",
                    "summary": f"Skipped spatial framing (camera unavailable: {exc})",
                },
            )
            return None, None

        spatial_res = check_workspace_framing(self.client, frame_jpeg)
        self._emit(
            "spatial",
            {
                "model": self.settings.spatial_model,
                "status": "done",
                "centered_on_table": spatial_res.centered_on_table,
                "workspace_clear": spatial_res.workspace_clear,
                "clearance_margin": spatial_res.clearance_margin,
                "summary": spatial_res.summary,
            },
        )
        return spatial_res, frame_jpeg

    def _invoke_forge(
        self,
        pose_seq: PoseSequence,
        skill_name: str,
        *,
        source_video: str,
        adaptation_hint: str | None,
        repair_feedback: str | None,
        use_remote_mcp: bool,
    ) -> ForgeResult:
        if self._use_live_models():
            return forge_skill(
                self.client,
                self.settings,
                pose_seq,
                skill_name,
                source_video=source_video,
                adaptation_hint=adaptation_hint,
                repair_feedback=repair_feedback,
                use_remote_mcp=use_remote_mcp,
            )

        refuse_elements = [
            el for el in pose_seq.impossible_elements
            if el.suggested_adaptation.strip().lower() == "refuse"
        ]
        if refuse_elements and not adaptation_hint:
            names = ", ".join(el.element for el in refuse_elements)
            return ForgeResult(
                artifact=None,
                refused=True,
                refusal_reason=f"Impossible motion refused in offline forge: {names}",
                environment_id=None,
                interaction_id=None,
                raw_output="",
            )

        artifact = deterministic_retarget_fallback(
            pose_seq,
            skill_name,
            source_video=source_video,
            force_upper_body_adaptation=bool(adaptation_hint or repair_feedback),
        )
        return ForgeResult(
            artifact=artifact,
            refused=False,
            refusal_reason="",
            environment_id=None,
            interaction_id=None,
            raw_output="",
        )

    def _invoke_pre_decision(
        self,
        artifact: SkillArtifact,
        pose_seq: PoseSequence,
        frame_jpegs: list[bytes],
        status: RobotStatus,
        extra_context: dict[str, Any],
    ) -> DecisionGateResult:
        if self._use_live_models():
            return evaluate_pre_execution(
                self.client,
                artifact,
                pose_seq,
                frame_jpegs,
                status,
                confidence_threshold=self.settings.decision_confidence_threshold,
                extra_context=extra_context,
            )
        return _offline_decision_gate(
            artifact,
            pose_seq,
            status,
            confidence_threshold=self.settings.decision_confidence_threshold,
        )

    def _make_announcement(
        self,
        skill_name: str,
        description: str,
        status: RobotStatus,
        frame_jpeg: bytes | None,
    ) -> str:
        fallback = f"Performing verified skill {skill_name} now."
        if not self._use_live_models():
            return fallback
        try:
            return run_live_reflex_announcement(
                self.client,
                skill_name,
                description,
                status,
                frame_jpeg=frame_jpeg,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reflex announcement fallback after error: %s", exc)
            return fallback

    def _safe_say(self, text: str) -> None:
        if not text:
            return
        try:
            self.mcp.say(text)
        except (MCPClientError, OSError, ValueError) as exc:
            logger.debug("Best-effort mcp.say skipped: %s", exc)

    def _project_upper_body(
        self,
        artifact: SkillArtifact,
        *,
        note: str,
        pre_decision: DecisionGateResult | None = None,
    ) -> SkillArtifact:
        projected_frames = project_to_upper_body_safe(artifact.keyframes)
        notes = [*artifact.kinematics.adaptation_notes, note]
        recomputed = check_skill_keyframes(
            projected_frames,
            lower_body_adapted=True,
            adaptation_notes=notes,
            sandbox_environment_id=artifact.kinematics.sandbox_environment_id,
        )
        updates: dict[str, Any] = {
            "adaptation_mode": "upper_body_adapted",
            "keyframes": projected_frames,
            "kinematics": recomputed,
        }
        if pre_decision is not None:
            updates["decision_gate"] = pre_decision
        return sign_skill_artifact(artifact.model_copy(update=updates))

    def mimic_video(
        self,
        video_path: Path,
        skill_name: str,
        *,
        dry_run: bool = False,
        run_spatial: bool = True,
        use_remote_mcp: bool = False,
        speed_scale: float = 1.0,
        force_execute_on_escalate: bool = False,
    ) -> MimicResult:
        """Watch a short video, extract poses, forge, gate, save, and execute on TonyPi."""
        # Fast telemetry latch before spending API time on video extraction
        _, preflight_err = self._check_hardware_preflight(skill_name)
        if preflight_err is not None:
            self._emit("done", {"stage": "safety_aborted", "status": "safety_aborted", "message": preflight_err})
            return MimicResult(
                skill_name=skill_name,
                status="safety_aborted",
                reason=preflight_err,
            )

        self._emit(
            "stage",
            {
                "stage": "extract",
                "model": self.settings.motion_model,
                "status": "active",
                "message": f"Extracting motion from {video_path.name}",
            },
        )

        try:
            preview_jpegs = extract_preview_frames(video_path, count=4)
        except (CaptureError, OSError, ValueError) as exc:
            logger.debug("Preview frame extraction skipped for %s: %s", video_path, exc)
            preview_jpegs = []

        if self._use_live_models():
            pose_seq = extract_motion_from_video(self.client, video_path, skill_hint=skill_name)
        else:
            pose_seq = _offline_synthetic_wave_pose(skill_name)

        self._emit(
            "extract",
            {
                "model": self.settings.motion_model,
                "status": "done",
                "keyframes": len(pose_seq.keyframes),
                "impossible_elements": len(pose_seq.impossible_elements),
                "preview_frames": len(preview_jpegs),
                "message": f"Extracted {len(pose_seq.keyframes)} beats ({pose_seq.bpm:.0f} BPM)",
            },
        )

        return self.mimic_pose(
            pose_seq,
            skill_name,
            source_video=str(video_path),
            preview_jpegs=preview_jpegs,
            dry_run=dry_run,
            run_spatial=run_spatial,
            use_remote_mcp=use_remote_mcp,
            speed_scale=speed_scale,
            force_execute_on_escalate=force_execute_on_escalate,
        )

    def mimic_pose(
        self,
        pose_seq: PoseSequence,
        skill_name: str,
        *,
        source_video: str = "",
        preview_jpegs: list[bytes] | None = None,
        dry_run: bool = False,
        run_spatial: bool = True,
        use_remote_mcp: bool = False,
        speed_scale: float = 1.0,
        force_execute_on_escalate: bool = False,
    ) -> MimicResult:
        """Retarget a PoseSequence via Skill Forge + Decision Gate, save, and optionally execute."""
        # 1. Deterministic hardware telemetry pre-flight
        status, preflight_err = self._check_hardware_preflight(skill_name)
        if preflight_err is not None:
            self._emit("done", {"stage": "safety_aborted", "status": "safety_aborted", "message": preflight_err})
            return MimicResult(
                skill_name=skill_name,
                status="safety_aborted",
                reason=preflight_err,
                pose_sequence=pose_seq,
            )

        # 2. Optional live spatial workspace framing check
        spatial_res, robot_frame_jpeg = self._run_spatial_check(run_spatial=run_spatial)
        adaptation_hint: str | None = None
        if spatial_res is not None:
            if not spatial_res.workspace_clear:
                reason = f"Workspace not clear: {spatial_res.summary}"
                self._emit("done", {"stage": "safety_aborted", "status": "safety_aborted", "message": reason})
                return MimicResult(
                    skill_name=skill_name,
                    status="safety_aborted",
                    reason=reason,
                    pose_sequence=pose_seq,
                    spatial=spatial_res,
                )
            if spatial_res.clearance_margin == "edge_risk":
                adaptation_hint = "edge_risk_upper_body_only"
                self._emit(
                    "spatial",
                    {
                        "stage": "spatial",
                        "clearance_margin": "edge_risk",
                        "message": f"Edge risk detected ({spatial_res.summary}); enforcing upper-body adaptation.",
                    },
                )

        # 3. Managed Agent Skill Forge (Attempt 1)
        self._emit(
            "stage",
            {
                "stage": "forge",
                "model": self.settings.managed_agent_id,
                "status": "active",
                "attempt": 1,
                "message": f"Forging 16-DOF skill '{skill_name}'",
            },
        )
        forge_attempts = 1
        forge_res = self._invoke_forge(
            pose_seq,
            skill_name,
            source_video=source_video,
            adaptation_hint=adaptation_hint,
            repair_feedback=None,
            use_remote_mcp=use_remote_mcp,
        )
        if forge_res.refused or forge_res.artifact is None:
            reason = forge_res.refusal_reason or "Motion refused by Managed Agent Skill Forge."
            self._emit(
                "forge",
                {
                    "model": self.settings.managed_agent_id,
                    "status": "done",
                    "attempt": forge_attempts,
                    "refused": True,
                    "message": reason,
                },
            )
            self._emit("done", {"stage": "refused", "status": "refused", "message": reason})
            return MimicResult(
                skill_name=skill_name,
                status="refused",
                reason=reason,
                pose_sequence=pose_seq,
                spatial=spatial_res,
                forge_attempts=forge_attempts,
            )

        artifact = forge_res.artifact
        self._emit(
            "forge",
            {
                "model": self.settings.managed_agent_id,
                "status": "done",
                "attempt": forge_attempts,
                "verified": artifact.kinematics.verified,
                "peak_torso_tilt_deg": artifact.kinematics.peak_torso_tilt_deg,
                "joint_clipping_count": artifact.kinematics.joint_clipping_count,
                "adaptation_mode": artifact.adaptation_mode,
                "message": (
                    f"Forged {len(artifact.keyframes)} keyframes "
                    f"(tilt={artifact.kinematics.peak_torso_tilt_deg:.1f} deg, "
                    f"clip={artifact.kinematics.joint_clipping_count})"
                ),
            },
        )

        # 4. Pre-execution Decisions API gate (0 output tokens)
        decision_frames: list[bytes] = list(preview_jpegs or [])
        if robot_frame_jpeg is not None:
            decision_frames.append(robot_frame_jpeg)

        extra_context: dict[str, Any] = {
            "forge_attempts": forge_attempts,
            "adaptation_hint": adaptation_hint,
        }
        if spatial_res is not None:
            extra_context["spatial_framing"] = spatial_res.model_dump(mode="json")

        self._emit(
            "stage",
            {
                "stage": "decision",
                "model": self.settings.decision_model,
                "status": "active",
                "message": "Evaluating safety, feasibility, and fidelity gate",
            },
        )
        pre_decision = self._invoke_pre_decision(
            artifact, pose_seq, decision_frames, status, extra_context
        )
        self._emit(
            "decision",
            {
                "model": self.settings.decision_model,
                "status": "done",
                "route": pre_decision.route,
                "effective_action": pre_decision.effective_action,
                "min_confidence": pre_decision.min_confidence,
                "is_physically_safe_prob": pre_decision.is_physically_safe_prob,
                "move_fidelity_score": pre_decision.move_fidelity_score,
                "message": (
                    f"Gate={pre_decision.effective_action} "
                    f"(conf={pre_decision.min_confidence:.2f}, safe={pre_decision.is_physically_safe_prob:.2f}, "
                    f"fidelity={pre_decision.move_fidelity_score:.2f}/3)"
                ),
            },
        )

        # 5. Bounded 1-shot repair loop when Decision Gate requests repair_in_forge
        if pre_decision.effective_action == "repair_in_forge":
            forge_attempts = 2
            violations_text = "; ".join(artifact.kinematics.violations) or "Decision gate requested balance/fidelity repair."
            repair_feedback = (
                f"Previous attempt failed gate (safe_prob={pre_decision.is_physically_safe_prob:.2f}, "
                f"needs_lower_body_adaptation={pre_decision.needs_lower_body_adaptation_prob:.2f}, "
                f"fidelity={pre_decision.move_fidelity_score:.2f}/3.0). "
                f"Violations: {violations_text}. Repair bounds/slew and project lower body if needed."
            )
            self._emit(
                "stage",
                {
                    "stage": "forge",
                    "model": self.settings.managed_agent_id,
                    "status": "active",
                    "attempt": forge_attempts,
                    "message": "Running bounded forge repair pass (attempt 2)",
                },
            )
            repair_res = self._invoke_forge(
                pose_seq,
                skill_name,
                source_video=source_video,
                adaptation_hint=adaptation_hint,
                repair_feedback=repair_feedback,
                use_remote_mcp=use_remote_mcp,
            )
            if repair_res.refused or repair_res.artifact is None:
                reason = repair_res.refusal_reason or "Skill refused during forge repair pass."
                self._emit("done", {"stage": "refused", "status": "refused", "message": reason})
                return MimicResult(
                    skill_name=skill_name,
                    status="refused",
                    reason=reason,
                    pose_sequence=pose_seq,
                    spatial=spatial_res,
                    artifact=artifact,
                    pre_decision=pre_decision,
                    forge_attempts=forge_attempts,
                )

            artifact = repair_res.artifact
            self._emit(
                "forge",
                {
                    "model": self.settings.managed_agent_id,
                    "status": "done",
                    "attempt": forge_attempts,
                    "verified": artifact.kinematics.verified,
                    "peak_torso_tilt_deg": artifact.kinematics.peak_torso_tilt_deg,
                    "joint_clipping_count": artifact.kinematics.joint_clipping_count,
                    "message": f"Repair attempt 2 verified={artifact.kinematics.verified}",
                },
            )
            extra_context["forge_attempts"] = forge_attempts
            pre_decision = self._invoke_pre_decision(
                artifact, pose_seq, decision_frames, status, extra_context
            )
            self._emit(
                "decision",
                {
                    "model": self.settings.decision_model,
                    "status": "done",
                    "route": pre_decision.route,
                    "effective_action": pre_decision.effective_action,
                    "min_confidence": pre_decision.min_confidence,
                    "is_physically_safe_prob": pre_decision.is_physically_safe_prob,
                    "move_fidelity_score": pre_decision.move_fidelity_score,
                    "message": f"Post-repair gate={pre_decision.effective_action} (conf={pre_decision.min_confidence:.2f})",
                },
            )

        # 6. Branch on final pre-execution decision
        if pre_decision.effective_action == "adapt_upper_body_only":
            artifact = self._project_upper_body(
                artifact,
                note="Projected lower body to grounded athletic bounce per Decision Gate (adapt_upper_body_only).",
                pre_decision=pre_decision,
            )
        elif pre_decision.effective_action == "refuse_impossible":
            reason = (
                f"Decision Gate refused impossible/unsafe motion "
                f"(safe_prob={pre_decision.is_physically_safe_prob:.2f}, route={pre_decision.route})."
            )
            self._emit("done", {"stage": "refused", "status": "refused", "message": reason})
            return MimicResult(
                skill_name=skill_name,
                status="refused",
                reason=reason,
                pose_sequence=pose_seq,
                spatial=spatial_res,
                artifact=artifact,
                pre_decision=pre_decision,
                forge_attempts=forge_attempts,
            )
        elif pre_decision.effective_action == "escalate_human" and not force_execute_on_escalate:
            reason = (
                f"Escalated to human review: confidence {pre_decision.min_confidence:.2f} "
                f"< {self.settings.decision_confidence_threshold:.2f} (route={pre_decision.route})."
            )
            self._emit("done", {"stage": "escalated_human", "status": "escalated_human", "message": reason})
            return MimicResult(
                skill_name=skill_name,
                status="escalated_human",
                reason=reason,
                pose_sequence=pose_seq,
                spatial=spatial_res,
                artifact=artifact,
                pre_decision=pre_decision,
                forge_attempts=forge_attempts,
            )
        elif pre_decision.effective_action == "repair_in_forge":
            reason = (
                "Skill still failed Decision Gate (repair_in_forge) after bounded repair retry; "
                "aborting without moving servos."
            )
            self._emit("done", {"stage": "safety_aborted", "status": "safety_aborted", "message": reason})
            return MimicResult(
                skill_name=skill_name,
                status="safety_aborted",
                reason=reason,
                pose_sequence=pose_seq,
                spatial=spatial_res,
                artifact=artifact,
                pre_decision=pre_decision,
                forge_attempts=forge_attempts,
            )

        # Enforce edge_risk upper-body safety projection if not already adapted
        if (
            spatial_res is not None
            and spatial_res.clearance_margin == "edge_risk"
            and artifact.adaptation_mode != "upper_body_adapted"
        ):
            artifact = self._project_upper_body(
                artifact,
                note="Enforced upper-body-only stance due to spatial edge_risk.",
                pre_decision=pre_decision,
            )

        # 7. Final deterministic kinematics safety latch before saving or actuating
        final_kin = check_skill_keyframes(
            artifact.keyframes,
            lower_body_adapted=artifact.kinematics.lower_body_adapted,
            adaptation_notes=artifact.kinematics.adaptation_notes,
            sandbox_environment_id=artifact.kinematics.sandbox_environment_id,
        )
        artifact = sign_skill_artifact(
            artifact.model_copy(update={"kinematics": final_kin, "decision_gate": pre_decision})
        )
        if not final_kin.verified:
            reason = "Final deterministic kinematics latch failed: " + "; ".join(final_kin.violations)
            self._emit("done", {"stage": "safety_aborted", "status": "safety_aborted", "message": reason})
            return MimicResult(
                skill_name=skill_name,
                status="safety_aborted",
                reason=reason,
                pose_sequence=pose_seq,
                spatial=spatial_res,
                artifact=artifact,
                pre_decision=pre_decision,
                forge_attempts=forge_attempts,
            )

        # 8. Save verified, signed skill artifact
        saved_path = save_skill(artifact, self.settings.skills_dir)

        if dry_run:
            reason = "Dry run complete: skill verified and saved without actuating servos."
            self._emit(
                "done",
                {
                    "stage": "dry_run_saved",
                    "status": "dry_run_saved",
                    "saved_path": str(saved_path),
                    "message": reason,
                },
            )
            return MimicResult(
                skill_name=skill_name,
                status="dry_run_saved",
                reason=reason,
                saved_path=str(saved_path),
                pose_sequence=pose_seq,
                spatial=spatial_res,
                artifact=artifact,
                pre_decision=pre_decision,
                forge_attempts=forge_attempts,
            )

        # 9. Announce + Execute on TonyPi MCP + Post-run fidelity audit
        announcement = self._make_announcement(
            skill_name, artifact.description, status, robot_frame_jpeg
        )
        self._safe_say(announcement)
        self._emit("stage", {"stage": "execute", "message": announcement})

        def _on_frame(idx: int, total: int, _frame: ServoKeyframe, f_status: RobotStatus) -> None:
            f_verdict = check_robot_telemetry(f_status)
            self._emit(
                "telemetry",
                {
                    "battery_v": f_status.battery_v,
                    "max_servo_temp_c": f_status.max_servo_temp_c,
                    "tilt_deg": f_verdict.tilt_deg if f_verdict.tilt_deg is not None else 0.0,
                    "frame_index": idx,
                    "total_frames": total,
                    "message": f"Frame {idx + 1}/{total}",
                },
            )

        execution = self.mcp.execute_skill(
            artifact,
            dry_run=False,
            speed_scale=speed_scale,
            on_frame=_on_frame,
        )
        self._emit(
            "execute",
            {
                "transport_mode": execution.transport_mode,
                "frames_executed": execution.frames_executed,
                "aborted": execution.aborted,
                "abort_reason": execution.abort_reason,
                "peak_observed_tilt_deg": execution.peak_observed_tilt_deg,
                "message": (
                    f"Executed {execution.frames_executed} frames ({execution.transport_mode})"
                    if not execution.aborted
                    else f"Execution aborted: {execution.abort_reason}"
                ),
            },
        )

        if execution.aborted:
            reason = execution.abort_reason or "Execution aborted by TonyPi safety watchdog."
            self._emit("done", {"stage": "safety_aborted", "status": "safety_aborted", "message": reason})
            return MimicResult(
                skill_name=skill_name,
                status="safety_aborted",
                reason=reason,
                saved_path=str(saved_path),
                pose_sequence=pose_seq,
                spatial=spatial_res,
                artifact=artifact,
                pre_decision=pre_decision,
                execution=execution,
                announcement=announcement,
                forge_attempts=forge_attempts,
            )

        post_decision: DecisionGateResult | None = None
        if self._use_live_models():
            post_run_jpeg = (
                base64.b64decode(execution.post_run_frame_b64)
                if execution.post_run_frame_b64
                else None
            )
            post_decision = evaluate_post_run(
                self.client,
                artifact,
                decision_frames,
                post_run_jpeg,
                execution.final_status,
                confidence_threshold=self.settings.decision_confidence_threshold,
            )
            artifact = sign_skill_artifact(
                artifact.model_copy(
                    update={"post_run_fidelity_score": post_decision.move_fidelity_score}
                )
            )
            saved_path = save_skill(artifact, self.settings.skills_dir)
            self._emit(
                "decision",
                {
                    "stage": "post_run",
                    "model": self.settings.decision_model,
                    "status": "done",
                    "route": post_decision.route,
                    "effective_action": post_decision.effective_action,
                    "min_confidence": post_decision.min_confidence,
                    "move_fidelity_score": post_decision.move_fidelity_score,
                    "message": f"Post-run fidelity={post_decision.move_fidelity_score:.2f}/3",
                },
            )

        self._emit(
            "done",
            {
                "stage": "executed_and_saved",
                "status": "executed_and_saved",
                "saved_path": str(saved_path),
                "message": f"Executed and saved skill '{skill_name}'",
            },
        )
        return MimicResult(
            skill_name=skill_name,
            status="executed_and_saved",
            reason="Skill verified, executed, and saved.",
            saved_path=str(saved_path),
            pose_sequence=pose_seq,
            spatial=spatial_res,
            artifact=artifact,
            pre_decision=pre_decision,
            post_decision=post_decision,
            execution=execution,
            announcement=announcement,
            forge_attempts=forge_attempts,
        )

    def run_saved_skill(
        self,
        skill_name: str,
        *,
        dry_run: bool = False,
        run_spatial: bool = True,
        speed_scale: float = 1.0,
    ) -> MimicResult:
        """Load a signed skill from skills/, verify safety latches, and replay on TonyPi."""
        artifact = load_skill(skill_name, self.settings.skills_dir, verify_signature=True)
        skill_file = self.settings.skills_dir / f"{slugify_skill_name(skill_name)}.json"

        status, preflight_err = self._check_hardware_preflight(artifact.name)
        if preflight_err is not None:
            self._emit("done", {"stage": "safety_aborted", "status": "safety_aborted", "message": preflight_err})
            return MimicResult(
                skill_name=artifact.name,
                status="safety_aborted",
                reason=preflight_err,
                saved_path=str(skill_file),
                artifact=artifact,
            )

        spatial_res, robot_frame_jpeg = self._run_spatial_check(run_spatial=run_spatial)
        if spatial_res is not None:
            if not spatial_res.workspace_clear:
                reason = f"Workspace not clear: {spatial_res.summary}"
                self._emit("done", {"stage": "safety_aborted", "status": "safety_aborted", "message": reason})
                return MimicResult(
                    skill_name=artifact.name,
                    status="safety_aborted",
                    reason=reason,
                    saved_path=str(skill_file),
                    spatial=spatial_res,
                    artifact=artifact,
                )
            if (
                spatial_res.clearance_margin == "edge_risk"
                and artifact.adaptation_mode != "upper_body_adapted"
            ):
                artifact = self._project_upper_body(
                    artifact,
                    note="Projected saved skill to upper-body safe stance due to edge_risk.",
                )

        kin = check_skill_keyframes(
            artifact.keyframes,
            lower_body_adapted=artifact.kinematics.lower_body_adapted,
            adaptation_notes=artifact.kinematics.adaptation_notes,
            sandbox_environment_id=artifact.kinematics.sandbox_environment_id,
        )
        if not kin.verified:
            reason = "Saved skill failed kinematics safety latch: " + "; ".join(kin.violations)
            self._emit("done", {"stage": "safety_aborted", "status": "safety_aborted", "message": reason})
            return MimicResult(
                skill_name=artifact.name,
                status="safety_aborted",
                reason=reason,
                saved_path=str(skill_file),
                spatial=spatial_res,
                artifact=artifact,
            )

        if dry_run:
            execution = self.mcp.execute_skill(artifact, dry_run=True, speed_scale=speed_scale)
            self._emit("done", {"stage": "dry_run_saved", "status": "dry_run_saved", "message": "Dry-run OK"})
            return MimicResult(
                skill_name=artifact.name,
                status="dry_run_saved",
                reason="Dry run complete on saved skill.",
                saved_path=str(skill_file),
                spatial=spatial_res,
                artifact=artifact,
                execution=execution,
            )

        announcement = self._make_announcement(
            artifact.name, artifact.description, status, robot_frame_jpeg
        )
        self._safe_say(announcement)
        self._emit("stage", {"stage": "execute", "message": announcement})
        execution = self.mcp.execute_skill(artifact, dry_run=False, speed_scale=speed_scale)
        self._emit(
            "execute",
            {
                "transport_mode": execution.transport_mode,
                "frames_executed": execution.frames_executed,
                "aborted": execution.aborted,
                "abort_reason": execution.abort_reason,
                "peak_observed_tilt_deg": execution.peak_observed_tilt_deg,
            },
        )
        if execution.aborted:
            reason = execution.abort_reason or "Saved skill replay aborted by safety watchdog."
            self._emit("done", {"stage": "safety_aborted", "status": "safety_aborted", "message": reason})
            return MimicResult(
                skill_name=artifact.name,
                status="safety_aborted",
                reason=reason,
                saved_path=str(skill_file),
                spatial=spatial_res,
                artifact=artifact,
                execution=execution,
                announcement=announcement,
            )

        post_decision: DecisionGateResult | None = None
        if self._use_live_models():
            post_run_jpeg = (
                base64.b64decode(execution.post_run_frame_b64)
                if execution.post_run_frame_b64
                else None
            )
            post_decision = evaluate_post_run(
                self.client,
                artifact,
                [robot_frame_jpeg] if robot_frame_jpeg else [],
                post_run_jpeg,
                execution.final_status,
                confidence_threshold=self.settings.decision_confidence_threshold,
            )
            artifact = sign_skill_artifact(
                artifact.model_copy(
                    update={"post_run_fidelity_score": post_decision.move_fidelity_score}
                )
            )
            save_skill(artifact, self.settings.skills_dir)

        self._emit(
            "done",
            {
                "stage": "executed_and_saved",
                "status": "executed_and_saved",
                "saved_path": str(skill_file),
                "message": f"Replayed saved skill '{artifact.name}'",
            },
        )
        return MimicResult(
            skill_name=artifact.name,
            status="executed_and_saved",
            reason="Saved skill verified and executed.",
            saved_path=str(skill_file),
            spatial=spatial_res,
            artifact=artifact,
            post_decision=post_decision,
            execution=execution,
            announcement=announcement,
        )


def run_preflight_check(
    client: Any | None,
    mcp: TonypiMCPClient | None,
    settings: RoboHarnessSettings,
    *,
    skip_forge_create: bool = False,
) -> PreflightReport:
    """Verify all 4 Gemini 4 models and the TonyPi Streamable HTTP MCP boundary."""
    items: list[CheckItem] = []

    # 1. gemini-4-hydrogen ping
    t0 = time.perf_counter()
    if client is None:
        items.append(
            CheckItem(
                component=settings.motion_model,
                ok=False,
                detail="Gemini client unavailable (set GEMINI_API_KEY).",
            )
        )
    else:
        try:
            resp = client.interactions.create(
                model=settings.motion_model,
                input=[{"type": "text", "text": "Reply with OK."}],
                generation_config=build_gemini4_generation_config("low"),
            )
            text = extract_interaction_text(resp).strip()
            items.append(
                CheckItem(
                    component=settings.motion_model,
                    ok=bool(text),
                    detail=f"Responded: {text[:60]!r}" if text else "Empty response from model.",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                )
            )
        except Exception as exc:  # noqa: BLE001
            items.append(
                CheckItem(
                    component=settings.motion_model,
                    ok=False,
                    detail=f"Ping failed: {exc}",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                )
            )

    # 2. roboharness-skill-forge Managed Agent check/create
    t0 = time.perf_counter()
    if client is None:
        items.append(
            CheckItem(
                component=settings.managed_agent_id,
                ok=False,
                detail="Gemini client unavailable (set GEMINI_API_KEY).",
            )
        )
    else:
        try:
            if skip_forge_create:
                existing = client.agents.get(id=settings.managed_agent_id)
                ok = existing is not None
                detail = f"Found agent '{settings.managed_agent_id}'." if ok else "Agent not found."
            else:
                state = ensure_forge_agent(client, settings)
                ok = True
                detail = (
                    f"Ready (model={state.model}, env={state.base_environment_id or 'inline'}, "
                    f"created={state.created_now})"
                )
            items.append(
                CheckItem(
                    component=settings.managed_agent_id,
                    ok=ok,
                    detail=detail,
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                )
            )
        except Exception as exc:  # noqa: BLE001
            items.append(
                CheckItem(
                    component=settings.managed_agent_id,
                    ok=False,
                    detail=f"Forge agent check failed: {exc}",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                )
            )

    # 3. gemini-decision-1 tiny noul ping
    t0 = time.perf_counter()
    if client is None:
        items.append(
            CheckItem(
                component=settings.decision_model,
                ok=False,
                detail="Gemini client unavailable (set GEMINI_API_KEY).",
            )
        )
    else:
        try:
            dec_resp = client.decisions.create(
                model=settings.decision_model,
                input=[{"type": "text", "text": json.dumps({"battery_v": 11.6, "standing": True})}],
                questions={
                    "health_check": {
                        "type": "noul",
                        "instructions": "Is the robot standing with battery >= 10.9V?",
                    }
                },
            )
            answers = dec_resp.get("answers") if isinstance(dec_resp, dict) else getattr(dec_resp, "answers", None)
            hc = answers.get("health_check") if isinstance(answers, dict) else None
            noul = hc.get("noul") if isinstance(hc, dict) else getattr(hc, "noul", None)
            conf = hc.get("confidence") if isinstance(hc, dict) else getattr(hc, "confidence", None)
            items.append(
                CheckItem(
                    component=settings.decision_model,
                    ok=hc is not None,
                    detail=f"0-token noul={noul}, conf={conf}" if hc is not None else "Missing health_check answer.",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                )
            )
        except Exception as exc:  # noqa: BLE001
            items.append(
                CheckItem(
                    component=settings.decision_model,
                    ok=False,
                    detail=f"Decisions API ping failed: {exc}",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                )
            )

    # 4. gemini-robotics-er-2-preview tiny spatial image ping
    t0 = time.perf_counter()
    if client is None:
        items.append(
            CheckItem(
                component=settings.spatial_model,
                ok=False,
                detail="Gemini client unavailable (set GEMINI_API_KEY).",
            )
        )
    else:
        try:
            probe_jpeg = MockTonypiMCPServer().render_camera_jpeg()
            sp = check_workspace_framing(client, probe_jpeg)
            items.append(
                CheckItem(
                    component=settings.spatial_model,
                    ok=True,
                    detail=f"margin={sp.clearance_margin}, clear={sp.workspace_clear}",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                )
            )
        except Exception as exc:  # noqa: BLE001
            items.append(
                CheckItem(
                    component=settings.spatial_model,
                    ok=False,
                    detail=f"Spatial ping failed: {exc}",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                )
            )

    # 5. TonyPi Streamable HTTP MCP connect + tools/list + get_status
    t0 = time.perf_counter()
    if mcp is None:
        items.append(
            CheckItem(
                component="tonypi_mcp",
                ok=False,
                detail=f"MCP client not configured ({settings.tonypi_mcp_url}).",
            )
        )
    else:
        try:
            tools = mcp.connect()
            st = mcp.get_status()
            verdict = check_robot_telemetry(st)
            items.append(
                CheckItem(
                    component="tonypi_mcp",
                    ok=verdict.safe,
                    detail=(
                        f"{len(tools)} tools, battery={st.battery_v:.2f}V, "
                        f"temp={st.max_servo_temp_c:.1f}C, safe={verdict.safe}"
                        + ("" if verdict.safe else f" ({'; '.join(verdict.reasons)})")
                    ),
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                )
            )
        except Exception as exc:  # noqa: BLE001
            items.append(
                CheckItem(
                    component="tonypi_mcp",
                    ok=False,
                    detail=f"MCP check failed ({settings.tonypi_mcp_url}): {exc}",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                )
            )

    return PreflightReport(all_ok=all(item.ok for item in items), items=items)
