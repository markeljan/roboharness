"""End-to-end Dual-Client 4-Model (+ Carbon Escalation) RoboHarness MimicPipeline.

Dual-client routing (Gemini 4 Hackathon Builder Guide):
- Stable ``v1`` client (`clients.client`): GA Gemini 4 Interactions API calls —
  ``gemini-4-hydrogen`` motion extraction (Agentic Video Processing) & live
  reflex speech (plus ``gemini-4-neon`` / ``gemini-4-carbon`` Interactions).
- Beta ``v1beta`` client (`clients.beta_client`): Managed Agent Skill Forge
  (``roboharness-skill-forge`` on ``gemini-4-neon``, escalating to
  ``roboharness-skill-forge-carbon`` on ``gemini-4-carbon``), Environments,
  0-output-token ``gemini-decision-1`` pre/post-run gates (``confidence >= 0.75``
  AND ``is_physically_safe >= 0.75``), and preview robotics model
  ``gemini-robotics-er-2-preview`` spatial framing.

Zero mocked demo fallbacks in live paths: when ``offline_mode=False``, missing
``DualGenAIClients`` raises ``RuntimeError`` immediately.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from roboharness.capture import CaptureError, extract_preview_frames
from roboharness.config import (
    BETA_API_VERSION,
    STABLE_API_VERSION,
    RoboHarnessSettings,
)
from roboharness.decision_gate import evaluate_post_run, evaluate_pre_execution
from roboharness.managed_forge import (
    ForgeResult,
    deterministic_retarget_fallback,
    ensure_forge_agent,
    forge_skill_with_escalation,
    resolve_forge_environment,
)
from roboharness.mcp_client import MCPClientError, TonypiMCPClient
from roboharness.mock_mcp import MockTonypiMCPServer
from roboharness.models import (
    DualGenAIClients,
    build_gemini4_generation_config,
    extract_interaction_text,
)
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

_LIVE_CLIENTS_ERROR = (
    "Live pipeline requires DualGenAIClients (set GEMINI_API_KEY or pass --offline explicitly)."
)

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
    forge_model_used: str = ""
    escalated_to_carbon: bool = False


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


def _resolve_dual_clients(
    *,
    clients: DualGenAIClients | None,
    client: Any | None,
    beta_client: Any | None,
) -> tuple[Any | None, Any | None]:
    if clients is not None:
        return clients.client, clients.beta_client
    if isinstance(client, DualGenAIClients):
        return client.client, client.beta_client
    resolved_v1 = client
    resolved_beta = beta_client if beta_client is not None else client
    return resolved_v1, resolved_beta


def _offline_synthetic_wave_pose(skill_name: str) -> PoseSequence:
    """Deterministic grounded greeting-wave PoseSequence used ONLY when offline_mode=True."""
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
    safety_prob_threshold: float,
) -> DecisionGateResult:
    """Deterministic local decision gate used ONLY when offline_mode=True."""
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

    safe_prob = 0.95
    conf = 0.94
    can_auto = conf >= confidence_threshold and safe_prob >= safety_prob_threshold
    fidelity = 2.6 if artifact.adaptation_mode == "full_body" else 2.2
    return DecisionGateResult(
        is_physically_safe_prob=safe_prob,
        is_physically_safe_conf=conf,
        needs_lower_body_adaptation_prob=0.15 if artifact.adaptation_mode == "full_body" else 0.75,
        needs_lower_body_adaptation_conf=0.90,
        move_fidelity_score=fidelity,
        move_fidelity_legend={"0": "unsafe", "1": "poor", "2": "good", "3": "high"},
        move_fidelity_conf=0.91,
        route="execute_and_save",
        route_probabilities={"execute_and_save": 0.93, "other": 0.07},
        route_conf=conf,
        min_confidence=conf,
        auto_branch_allowed=can_auto,
        effective_action="execute_and_save" if can_auto else "escalate_human",
    )


class MimicPipeline:
    """Orchestrates the dual-client 4-model (+ Carbon escalation) RoboHarness workflow."""

    def __init__(
        self,
        *,
        clients: DualGenAIClients | None = None,
        client: Any | None = None,
        beta_client: Any | None = None,
        mcp: TonypiMCPClient,
        settings: RoboHarnessSettings,
        hub: TelemetryHub | None = None,
        offline_mode: bool = False,
    ) -> None:
        v1_client, resolved_beta = _resolve_dual_clients(
            clients=clients,
            client=client,
            beta_client=beta_client,
        )
        if not offline_mode and (v1_client is None or resolved_beta is None):
            raise RuntimeError(_LIVE_CLIENTS_ERROR)

        self.v1_client = v1_client
        self.beta_client = resolved_beta
        self.client = v1_client
        self.clients = (
            DualGenAIClients(client=v1_client, beta_client=resolved_beta)
            if (v1_client is not None and resolved_beta is not None)
            else None
        )
        self.mcp = mcp
        self.settings = settings
        self.hub = hub
        self.offline_mode = offline_mode

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        if self.hub is not None:
            self.hub.emit(event_type, data)

    def _emit_frame(self, frame_jpeg: bytes | None, label: str) -> None:
        """Stream a JPEG camera/MuJoCo frame to the SSE telemetry viewport."""
        if self.hub is not None and frame_jpeg:
            self.hub.emit(
                "frame",
                {
                    "b64": base64.b64encode(frame_jpeg).decode("ascii"),
                    "label": label,
                },
            )

    def _emit_frame_b64(self, frame_b64: str | None, label: str) -> None:
        """Stream an already base64-encoded JPEG frame to the SSE telemetry viewport."""
        if self.hub is not None and frame_b64:
            self.hub.emit("frame", {"b64": frame_b64, "label": label})

    def _capture_preflight_frame(self, skill_name: str) -> bytes | None:
        """Best-effort camera snapshot to seed the live SSE viewport before execution."""
        if self.hub is None:
            return None
        try:
            frame_jpeg, _ = self.mcp.perceive()
        except (MCPClientError, OSError, ValueError) as exc:
            logger.debug("Best-effort preflight perceive skipped: %s", exc)
            return None
        self._emit_frame(frame_jpeg, f"pre-flight:{skill_name}")
        return frame_jpeg

    def _make_on_frame_callback(
        self, skill_name: str
    ) -> Callable[[int, int, ServoKeyframe, RobotStatus], None]:
        def _on_frame(idx: int, total: int, _frame: ServoKeyframe, f_status: RobotStatus) -> None:
            f_verdict = check_robot_telemetry(f_status)
            self._emit(
                "telemetry",
                {
                    "stage": "execute",
                    "battery_v": f_status.battery_v,
                    "max_servo_temp_c": f_status.max_servo_temp_c,
                    "tilt_deg": f_verdict.tilt_deg if f_verdict.tilt_deg is not None else 0.0,
                    "frame_index": idx,
                    "total_frames": total,
                    "message": f"Frame {idx + 1}/{total}",
                },
            )
            raw_b64 = f_status.raw.get("frame_b64") if isinstance(f_status.raw, dict) else None
            if isinstance(raw_b64, str) and raw_b64:
                self._emit_frame_b64(raw_b64, f"frame:{skill_name}:{idx + 1}/{total}")

        return _on_frame

    def _require_live_clients(self) -> tuple[Any, Any]:
        if self.v1_client is None or self.beta_client is None:
            raise RuntimeError(_LIVE_CLIENTS_ERROR)
        return self.v1_client, self.beta_client

    def _check_hardware_preflight(self, skill_name: str) -> tuple[RobotStatus, str | None]:
        self._emit(
            "stage",
            {
                "stage": "preflight",
                "skill_name": skill_name,
                "message": "Checking TonyPi telemetry",
            },
        )
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
        skill_name: str = "",
    ) -> tuple[SpatialFramingResult | None, bytes | None]:
        if not run_spatial or self.offline_mode:
            return None, self._capture_preflight_frame(skill_name)

        _, beta_client = self._require_live_clients()
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

        label = f"pre-flight:{skill_name}" if skill_name else "pre-flight"
        self._emit_frame(frame_jpeg, label)

        spatial_res = check_workspace_framing(beta_client, frame_jpeg)
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
        deep: bool,
    ) -> ForgeResult:
        if not self.offline_mode:
            _, beta_client = self._require_live_clients()
            return forge_skill_with_escalation(
                beta_client,
                self.settings,
                pose_seq,
                skill_name,
                source_video=source_video,
                adaptation_hint=adaptation_hint,
                repair_feedback=repair_feedback,
                use_remote_mcp=use_remote_mcp,
                deep=deep,
            )

        # Explicit offline_mode=True path ONLY
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
                model_used="offline-deterministic",
                escalated_to_carbon=False,
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
            model_used="offline-deterministic",
            escalated_to_carbon=False,
        )

    def _invoke_pre_decision(
        self,
        artifact: SkillArtifact,
        pose_seq: PoseSequence,
        frame_jpegs: list[bytes],
        status: RobotStatus,
        extra_context: dict[str, Any],
    ) -> DecisionGateResult:
        if not self.offline_mode:
            _, beta_client = self._require_live_clients()
            return evaluate_pre_execution(
                beta_client,
                artifact,
                pose_seq,
                frame_jpegs,
                status,
                confidence_threshold=self.settings.decision_confidence_threshold,
                safety_prob_threshold=self.settings.decision_safety_prob_threshold,
                extra_context=extra_context,
            )
        return _offline_decision_gate(
            artifact,
            pose_seq,
            status,
            confidence_threshold=self.settings.decision_confidence_threshold,
            safety_prob_threshold=self.settings.decision_safety_prob_threshold,
        )

    @staticmethod
    def _audio_muted() -> bool:
        return os.environ.get("ROBOHARNESS_MUTE_AUDIO", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _make_announcement(
        self,
        skill_name: str,
        description: str,
        status: RobotStatus,
        frame_jpeg: bytes | None,
    ) -> str:
        fallback = f"Performing verified skill {skill_name} now."
        if self.offline_mode or self._audio_muted():
            return f"[muted] {fallback}" if self._audio_muted() else fallback
        v1_client, _ = self._require_live_clients()
        try:
            return run_live_reflex_announcement(
                v1_client,
                skill_name,
                description,
                status,
                frame_jpeg=frame_jpeg,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reflex announcement fallback after error: %s", exc)
            return fallback

    def _safe_say(self, text: str) -> None:
        if not text or self._audio_muted():
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
        deep: bool = False,
    ) -> MimicResult:
        """Watch a short video, extract poses (Agentic Video), forge, gate, save, and execute."""
        if not self.offline_mode:
            self._require_live_clients()

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

        if not self.offline_mode:
            v1_client, _ = self._require_live_clients()
            pose_seq = extract_motion_from_video(v1_client, video_path, skill_hint=skill_name)
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
            deep=deep,
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
        deep: bool = False,
    ) -> MimicResult:
        """Retarget a PoseSequence via Skill Forge (Neon -> Carbon) + Decision Gate, save & execute."""
        if not self.offline_mode:
            self._require_live_clients()

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

        # 2. Optional live spatial workspace framing check (v1beta preview model)
        spatial_res, robot_frame_jpeg = self._run_spatial_check(
            run_spatial=run_spatial, skill_name=skill_name
        )
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

        # 3. Managed Agent Skill Forge on v1beta (Neon with automatic Carbon escalation, or --deep)
        target_agent_label = (
            self.settings.forge_deep_agent_id if deep else self.settings.managed_agent_id
        )
        self._emit(
            "stage",
            {
                "stage": "forge",
                "model": target_agent_label,
                "status": "active",
                "attempt": 1,
                "deep": deep,
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
            deep=deep,
        )
        if forge_res.refused or forge_res.artifact is None:
            reason = forge_res.refusal_reason or "Motion refused by Managed Agent Skill Forge."
            self._emit(
                "forge",
                {
                    "model": target_agent_label,
                    "status": "done",
                    "attempt": forge_attempts,
                    "refused": True,
                    "model_used": forge_res.model_used,
                    "escalated_to_carbon": forge_res.escalated_to_carbon,
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
                forge_model_used=forge_res.model_used,
                escalated_to_carbon=forge_res.escalated_to_carbon,
            )

        artifact = forge_res.artifact
        forge_model_used = forge_res.model_used
        escalated_to_carbon = forge_res.escalated_to_carbon
        self._emit(
            "forge",
            {
                "model": target_agent_label,
                "status": "done",
                "attempt": forge_attempts,
                "verified": artifact.kinematics.verified,
                "peak_torso_tilt_deg": artifact.kinematics.peak_torso_tilt_deg,
                "joint_clipping_count": artifact.kinematics.joint_clipping_count,
                "adaptation_mode": artifact.adaptation_mode,
                "model_used": forge_model_used,
                "escalated_to_carbon": escalated_to_carbon,
                "message": (
                    f"Forged {len(artifact.keyframes)} keyframes via {forge_model_used} "
                    f"(tilt={artifact.kinematics.peak_torso_tilt_deg:.1f} deg, "
                    f"clip={artifact.kinematics.joint_clipping_count})"
                ),
            },
        )

        # 4. Pre-execution Decisions API gate on v1beta (0 output tokens, conf>=0.75 & safe>=0.75)
        decision_frames: list[bytes] = list(preview_jpegs or [])
        if robot_frame_jpeg is not None:
            decision_frames.append(robot_frame_jpeg)

        extra_context: dict[str, Any] = {
            "forge_attempts": forge_attempts,
            "adaptation_hint": adaptation_hint,
            "forge_model_used": forge_model_used,
            "escalated_to_carbon": escalated_to_carbon,
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
            violations_text = (
                "; ".join(artifact.kinematics.violations)
                or "Decision gate requested balance/fidelity repair."
            )
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
                    "model": target_agent_label,
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
                deep=deep,
            )
            forge_model_used = repair_res.model_used
            escalated_to_carbon = escalated_to_carbon or repair_res.escalated_to_carbon
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
                    forge_model_used=forge_model_used,
                    escalated_to_carbon=escalated_to_carbon,
                )

            artifact = repair_res.artifact
            self._emit(
                "forge",
                {
                    "model": target_agent_label,
                    "status": "done",
                    "attempt": forge_attempts,
                    "verified": artifact.kinematics.verified,
                    "peak_torso_tilt_deg": artifact.kinematics.peak_torso_tilt_deg,
                    "joint_clipping_count": artifact.kinematics.joint_clipping_count,
                    "model_used": forge_model_used,
                    "escalated_to_carbon": escalated_to_carbon,
                    "message": f"Repair attempt 2 verified={artifact.kinematics.verified}",
                },
            )
            extra_context["forge_attempts"] = forge_attempts
            extra_context["forge_model_used"] = forge_model_used
            extra_context["escalated_to_carbon"] = escalated_to_carbon
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
                forge_model_used=forge_model_used,
                escalated_to_carbon=escalated_to_carbon,
            )
        elif pre_decision.effective_action == "escalate_human" and not force_execute_on_escalate:
            reason = (
                f"Escalated to human review: confidence={pre_decision.min_confidence:.2f} "
                f"(min {self.settings.decision_confidence_threshold:.2f}), "
                f"safe_prob={pre_decision.is_physically_safe_prob:.2f} "
                f"(min {self.settings.decision_safety_prob_threshold:.2f}), "
                f"route={pre_decision.route}."
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
                forge_model_used=forge_model_used,
                escalated_to_carbon=escalated_to_carbon,
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
                forge_model_used=forge_model_used,
                escalated_to_carbon=escalated_to_carbon,
            )

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
                forge_model_used=forge_model_used,
                escalated_to_carbon=escalated_to_carbon,
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
                forge_model_used=forge_model_used,
                escalated_to_carbon=escalated_to_carbon,
            )

        # 9. Announce (v1) + Execute on TonyPi MCP + Post-run fidelity audit (v1beta)
        announcement = self._make_announcement(
            skill_name, artifact.description, status, robot_frame_jpeg
        )
        self._safe_say(announcement)
        self._emit("stage", {"stage": "execute", "message": announcement})

        execution = self.mcp.execute_skill(
            artifact,
            dry_run=False,
            speed_scale=speed_scale,
            on_frame=self._make_on_frame_callback(skill_name),
        )
        self._emit_frame_b64(execution.post_run_frame_b64, f"post-run:{skill_name}")
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
                forge_model_used=forge_model_used,
                escalated_to_carbon=escalated_to_carbon,
            )

        post_decision: DecisionGateResult | None = None
        if not self.offline_mode:
            _, beta_client = self._require_live_clients()
            post_run_jpeg = (
                base64.b64decode(execution.post_run_frame_b64)
                if execution.post_run_frame_b64
                else None
            )
            post_decision = evaluate_post_run(
                beta_client,
                artifact,
                decision_frames,
                post_run_jpeg,
                execution.final_status,
                confidence_threshold=self.settings.decision_confidence_threshold,
                safety_prob_threshold=self.settings.decision_safety_prob_threshold,
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
            forge_model_used=forge_model_used,
            escalated_to_carbon=escalated_to_carbon,
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
        if not self.offline_mode:
            self._require_live_clients()

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

        spatial_res, robot_frame_jpeg = self._run_spatial_check(
            run_spatial=run_spatial, skill_name=artifact.name
        )
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
        execution = self.mcp.execute_skill(
            artifact,
            dry_run=False,
            speed_scale=speed_scale,
            on_frame=self._make_on_frame_callback(artifact.name),
        )
        self._emit_frame_b64(execution.post_run_frame_b64, f"post-run:{artifact.name}")
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
        if not self.offline_mode:
            _, beta_client = self._require_live_clients()
            post_run_jpeg = (
                base64.b64decode(execution.post_run_frame_b64)
                if execution.post_run_frame_b64
                else None
            )
            post_decision = evaluate_post_run(
                beta_client,
                artifact,
                [robot_frame_jpeg] if robot_frame_jpeg else [],
                post_run_jpeg,
                execution.final_status,
                confidence_threshold=self.settings.decision_confidence_threshold,
                safety_prob_threshold=self.settings.decision_safety_prob_threshold,
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
    client: Any | None = None,
    mcp: TonypiMCPClient | None = None,
    settings: RoboHarnessSettings | None = None,
    *,
    clients: DualGenAIClients | None = None,
    beta_client: Any | None = None,
    skip_forge_create: bool = False,
) -> PreflightReport:
    """Verify stable v1 + v1beta Gemini 4 models and TonyPi Streamable HTTP MCP."""
    resolved_settings = settings if settings is not None else RoboHarnessSettings.from_env()
    v1_client, resolved_beta = _resolve_dual_clients(
        clients=clients,
        client=client,
        beta_client=beta_client,
    )
    items: list[CheckItem] = []

    # 1. gemini-4-hydrogen ping (stable v1)
    t0 = time.perf_counter()
    if v1_client is None:
        items.append(
            CheckItem(
                component=resolved_settings.motion_model,
                ok=False,
                detail=f"[{STABLE_API_VERSION}] Gemini v1 client unavailable (set GEMINI_API_KEY).",
            )
        )
    else:
        try:
            resp = v1_client.interactions.create(
                model=resolved_settings.motion_model,
                input=[{"type": "text", "text": "Reply with OK."}],
                generation_config=build_gemini4_generation_config("medium"),
            )
            text = extract_interaction_text(resp).strip()
            items.append(
                CheckItem(
                    component=resolved_settings.motion_model,
                    ok=bool(text),
                    detail=(
                        f"[{STABLE_API_VERSION}] Responded: {text[:60]!r}"
                        if text
                        else f"[{STABLE_API_VERSION}] Empty response from model."
                    ),
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                )
            )
        except Exception as exc:  # noqa: BLE001
            items.append(
                CheckItem(
                    component=resolved_settings.motion_model,
                    ok=False,
                    detail=f"[{STABLE_API_VERSION}] Ping failed: {exc}",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                )
            )

    # 2. roboharness-skill-forge Managed Agent check/create (v1beta)
    t0 = time.perf_counter()
    env_label = resolve_forge_environment(resolved_settings)
    env_note = (
        f"GEMINI_ENVIRONMENT_ID={env_label}"
        if resolved_settings.gemini_environment_id
        else 'env="remote"'
    )
    if resolved_beta is None:
        items.append(
            CheckItem(
                component=resolved_settings.managed_agent_id,
                ok=False,
                detail=f"[{BETA_API_VERSION}] Gemini v1beta client unavailable ({env_note}).",
            )
        )
    else:
        try:
            if skip_forge_create:
                existing = resolved_beta.agents.get(id=resolved_settings.managed_agent_id)
                ok = existing is not None
                detail = (
                    f"[{BETA_API_VERSION}] Found agent '{resolved_settings.managed_agent_id}' ({env_note})."
                    if ok
                    else f"[{BETA_API_VERSION}] Agent not found ({env_note})."
                )
            else:
                state = ensure_forge_agent(resolved_beta, resolved_settings)
                ok = True
                detail = (
                    f"[{BETA_API_VERSION}] Ready (model={state.model}, "
                    f"env={state.base_environment_id or env_label}, {env_note}, "
                    f"created={state.created_now})"
                )
            items.append(
                CheckItem(
                    component=resolved_settings.managed_agent_id,
                    ok=ok,
                    detail=detail,
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                )
            )
        except Exception as exc:  # noqa: BLE001
            items.append(
                CheckItem(
                    component=resolved_settings.managed_agent_id,
                    ok=False,
                    detail=f"[{BETA_API_VERSION}] Forge agent check failed ({env_note}): {exc}",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                )
            )

    # 3. gemini-decision-1 tiny noul ping (v1beta)
    t0 = time.perf_counter()
    if resolved_beta is None:
        items.append(
            CheckItem(
                component=resolved_settings.decision_model,
                ok=False,
                detail=f"[{BETA_API_VERSION}] Gemini v1beta client unavailable (set GEMINI_API_KEY).",
            )
        )
    else:
        try:
            dec_resp = resolved_beta.decisions.create(
                model=resolved_settings.decision_model,
                input=[{"type": "text", "text": json.dumps({"battery_v": 11.6, "standing": True})}],
                questions={
                    "health_check": {
                        "type": "noul",
                        "instructions": "Is the robot standing with battery >= 10.9V?",
                    }
                },
            )
            answers = (
                dec_resp.get("answers")
                if isinstance(dec_resp, dict)
                else getattr(dec_resp, "answers", None)
            )
            hc = answers.get("health_check") if isinstance(answers, dict) else None
            noul = hc.get("noul") if isinstance(hc, dict) else getattr(hc, "noul", None)
            conf = hc.get("confidence") if isinstance(hc, dict) else getattr(hc, "confidence", None)
            items.append(
                CheckItem(
                    component=resolved_settings.decision_model,
                    ok=hc is not None,
                    detail=(
                        f"[{BETA_API_VERSION}] 0-token noul={noul}, conf={conf}"
                        if hc is not None
                        else f"[{BETA_API_VERSION}] Missing health_check answer."
                    ),
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                )
            )
        except Exception as exc:  # noqa: BLE001
            items.append(
                CheckItem(
                    component=resolved_settings.decision_model,
                    ok=False,
                    detail=f"[{BETA_API_VERSION}] Decisions API ping failed: {exc}",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                )
            )

    # 4. gemini-robotics-er-2-preview spatial image ping (v1beta preview model)
    t0 = time.perf_counter()
    if resolved_beta is None:
        items.append(
            CheckItem(
                component=resolved_settings.spatial_model,
                ok=False,
                detail=f"[{BETA_API_VERSION}] Gemini v1beta client unavailable (set GEMINI_API_KEY).",
            )
        )
    else:
        try:
            probe_jpeg = MockTonypiMCPServer().render_camera_jpeg()
            sp = check_workspace_framing(resolved_beta, probe_jpeg)
            items.append(
                CheckItem(
                    component=resolved_settings.spatial_model,
                    ok=True,
                    detail=f"[{BETA_API_VERSION}] margin={sp.clearance_margin}, clear={sp.workspace_clear}",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                )
            )
        except Exception as exc:  # noqa: BLE001
            items.append(
                CheckItem(
                    component=resolved_settings.spatial_model,
                    ok=False,
                    detail=f"[{BETA_API_VERSION}] Spatial ping failed: {exc}",
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
                detail=f"MCP client not configured ({resolved_settings.tonypi_mcp_url}).",
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
                    detail=f"MCP check failed ({resolved_settings.tonypi_mcp_url}): {exc}",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                )
            )

    return PreflightReport(all_ok=all(item.ok for item in items), items=items)
