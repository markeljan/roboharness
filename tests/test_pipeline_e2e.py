"""End-to-end offline tests for RoboHarness MimicPipeline and run_preflight_check."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from roboharness.config import (
    FORGE_AGENT_ID,
    FORGE_MODEL,
    MOTION_MODEL,
    SPATIAL_MODEL,
    RoboHarnessSettings,
)
from roboharness.mcp_client import TonypiMCPClient
from roboharness.mock_mcp import MockTonypiMCPServer
from roboharness.pipeline import MimicPipeline, PreflightReport, run_preflight_check
from roboharness.safety import verify_artifact_signature
from roboharness.schemas import (
    NEUTRAL_POSE,
    HumanPoseKeyframe,
    PoseSequence,
    SpatialFramingResult,
)
from roboharness.skill_store import load_skill
from roboharness.telemetry_sse import TelemetryHub


def _pose_seq() -> PoseSequence:
    return PoseSequence(
        source_description="Two-beat joyful arm wave",
        bpm=120.0,
        duration_ms=1000,
        keyframes=[
            HumanPoseKeyframe(
                t_ms=0,
                beat_index=0,
                torso_pitch_deg=3.0,
                torso_roll_deg=1.0,
                l_shoulder_pitch_deg=20.0,
                l_shoulder_roll_deg=15.0,
                l_elbow_deg=25.0,
                r_shoulder_pitch_deg=60.0,
                r_shoulder_roll_deg=30.0,
                r_elbow_deg=40.0,
                l_hip_pitch_deg=5.0,
                l_knee_deg=10.0,
                r_hip_pitch_deg=5.0,
                r_knee_deg=10.0,
                foot_contact="both",
                label="wave_up",
            ),
            HumanPoseKeyframe(
                t_ms=500,
                beat_index=1,
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
                label="neutral",
            ),
        ],
        impossible_elements=[],
        style_tags=["wave"],
    )


def _safe_forge_json(name: str = "wave") -> str:
    return json.dumps(
        {
            "refused": False,
            "refusal_reason": "",
            "name": name,
            "description": "Verified grounded wave",
            "adaptation_mode": "full_body",
            "lower_body_adapted": False,
            "adaptation_notes": [],
            "keyframes": [
                {
                    "duration_ms": 450,
                    "servos": {**NEUTRAL_POSE, "r_shoulder_pitch": 680, "r_elbow": 600},
                    "note": "wave_up",
                },
                {
                    "duration_ms": 450,
                    "servos": dict(NEUTRAL_POSE),
                    "note": "neutral",
                },
            ],
        }
    )


def _clipped_forge_json(name: str = "bad_clip") -> str:
    return json.dumps(
        {
            "refused": False,
            "name": name,
            "description": "Clipped candidate",
            "adaptation_mode": "full_body",
            "keyframes": [
                {
                    "duration_ms": 450,
                    "servos": {**NEUTRAL_POSE, "r_shoulder_pitch": 950},
                    "note": "clipped",
                }
            ],
        }
    )


def _decision_response(
    *,
    route: str = "execute_and_save",
    safe_prob: float = 0.94,
    safe_conf: float = 0.92,
    adapt_prob: float = 0.15,
    adapt_conf: float = 0.90,
    fidelity: float = 2.7,
    fidelity_conf: float = 0.89,
    route_conf: float = 0.91,
) -> dict[str, Any]:
    return {
        "answers": {
            "is_physically_safe": {"noul": safe_prob, "confidence": safe_conf},
            "needs_lower_body_adaptation": {"noul": adapt_prob, "confidence": adapt_conf},
            "move_fidelity": {
                "score": fidelity,
                "legend": {"0": "bad", "1": "low", "2": "good", "3": "great"},
                "confidence": fidelity_conf,
            },
            "route": {
                "choice": route,
                "probabilities": {route: 0.9, "other": 0.1},
                "confidence": route_conf,
            },
        }
    }


class _PipelineStubClient:
    """Multi-model stub routing Interactions, Agents, and Decisions API calls cleanly."""

    def __init__(
        self,
        *,
        spatial_result: SpatialFramingResult | None = None,
        forge_outputs: list[str] | None = None,
        decision_responses: list[dict[str, Any]] | None = None,
        pose_seq: PoseSequence | None = None,
        fail_hydrogen: bool = False,
    ) -> None:
        self.spatial_result = spatial_result or SpatialFramingResult(
            centered_on_table=True,
            workspace_clear=True,
            robot_bbox_norm=[120, 220, 880, 780],
            clearance_margin="safe",
            summary="Centered on table with ample clearance.",
        )
        self.forge_outputs = list(forge_outputs or [_safe_forge_json()])
        self.decision_responses = list(decision_responses or [_decision_response(), _decision_response(fidelity=2.85)])
        self.pose_seq = pose_seq or _pose_seq()
        self.fail_hydrogen = fail_hydrogen

        self.forge_prompts: list[str] = []
        self.decision_calls: list[dict[str, Any]] = []
        self.interaction_calls: list[dict[str, Any]] = []

        self.agents = self._Agents(self)
        self.interactions = self._Interactions(self)
        self.decisions = self._Decisions(self)

    class _Agents:
        def __init__(self, parent: _PipelineStubClient) -> None:
            self.parent = parent

        def get(self, *, id: str) -> Any:
            return SimpleNamespace(
                id=id,
                base_environment="env_fork_ok",
                agent_config={"type": "antigravity", "model": FORGE_MODEL},
            )

        def create(self, **kwargs: Any) -> Any:
            return SimpleNamespace(**kwargs)

        def delete(self, *, id: str) -> dict[str, Any]:
            return {}

    class _Interactions:
        def __init__(self, parent: _PipelineStubClient) -> None:
            self.parent = parent

        def create(self, **kwargs: Any) -> Any:
            self.parent.interaction_calls.append(kwargs)
            model = kwargs.get("model")
            agent = kwargs.get("agent")

            if agent == FORGE_AGENT_ID:
                self.parent.forge_prompts.append(str(kwargs.get("input", "")))
                out = (
                    self.parent.forge_outputs.pop(0)
                    if len(self.parent.forge_outputs) > 1
                    else self.parent.forge_outputs[0]
                )
                return SimpleNamespace(
                    id=f"inter_forge_{len(self.parent.forge_prompts)}",
                    status="completed",
                    environment_id="env_sandbox_1",
                    output_text=out,
                )

            if model == SPATIAL_MODEL:
                return SimpleNamespace(
                    id="inter_spatial",
                    status="completed",
                    output_text=self.parent.spatial_result.model_dump_json(),
                )

            if model == MOTION_MODEL:
                if self.parent.fail_hydrogen:
                    raise RuntimeError("Simulated hydrogen outage")
                resp_fmt = kwargs.get("response_format")
                if resp_fmt is not None:
                    return SimpleNamespace(
                        id="inter_motion",
                        status="completed",
                        output_text=self.parent.pose_seq.model_dump_json(),
                    )
                return SimpleNamespace(
                    id="inter_reflex",
                    status="completed",
                    output_text="Watch me perform the verified wave skill now!",
                )

            return SimpleNamespace(id="inter_generic", status="completed", output_text="OK")

    class _Decisions:
        def __init__(self, parent: _PipelineStubClient) -> None:
            self.parent = parent

        def create(self, **kwargs: Any) -> dict[str, Any]:
            self.parent.decision_calls.append(kwargs)
            if "health_check" in kwargs.get("questions", {}):
                return {"answers": {"health_check": {"noul": 0.96, "confidence": 0.95}}}
            if len(self.parent.decision_responses) > 1:
                return self.parent.decision_responses.pop(0)
            return self.parent.decision_responses[0]


def _make_mcp(server: MockTonypiMCPServer) -> TonypiMCPClient:
    return TonypiMCPClient(url="http://mock.local/mcp", transport=server.as_httpx_transport())


def _actuation_calls(server: MockTonypiMCPServer) -> list[dict[str, Any]]:
    return [
        c for c in server.call_log
        if c.get("method") == "tools/call"
        and c.get("tool") in {"run_keyframes", "set_servos", "run_action"}
    ]


# ---------------------------------------------------------------------------
# E2E route coverage
# ---------------------------------------------------------------------------


def test_pipeline_execute_and_save_full_4model_flow(tmp_path: Path) -> None:
    server = MockTonypiMCPServer(mode="batch")
    client = _PipelineStubClient()
    hub = TelemetryHub()
    settings = RoboHarnessSettings(
        skills_dir=tmp_path / "skills",
        forge_state_path=tmp_path / "forge_state.json",
    )

    video_path = tmp_path / "fake.mp4"
    video_path.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    with _make_mcp(server) as mcp:
        pipe = MimicPipeline(client=client, mcp=mcp, settings=settings, hub=hub)
        res = pipe.mimic_video(video_path, "judge_wave", run_spatial=True)

    assert res.status == "executed_and_saved"
    assert res.saved_path is not None
    assert Path(res.saved_path).is_file()
    assert res.artifact is not None
    assert verify_artifact_signature(res.artifact) is True
    assert res.artifact.post_run_fidelity_score == 2.85
    assert res.pre_decision is not None
    assert res.post_decision is not None
    assert res.execution is not None and not res.execution.aborted
    assert res.forge_attempts == 1
    assert server.spoken_texts == ["Watch me perform the verified wave skill now!"]
    assert len(_actuation_calls(server)) == 1

    # Verify reloaded skill from disk keeps valid SHA-256 signature + post-run fidelity
    reloaded = load_skill("judge_wave", settings.skills_dir)
    assert reloaded.post_run_fidelity_score == 2.85

    # Verify telemetry hub received full pipeline event lifecycle
    event_types = {ev["event"] for ev in hub.history()}
    assert {"stage", "telemetry", "spatial", "extract", "forge", "decision", "execute", "done"} <= event_types


def test_pipeline_adapt_upper_body_only_route(tmp_path: Path) -> None:
    server = MockTonypiMCPServer(mode="batch")
    client = _PipelineStubClient(
        decision_responses=[
            _decision_response(route="adapt_upper_body_only", adapt_prob=0.88),
            _decision_response(route="execute_and_save", fidelity=2.4),
        ]
    )
    settings = RoboHarnessSettings(
        skills_dir=tmp_path / "skills",
        forge_state_path=tmp_path / "forge_state.json",
    )

    with _make_mcp(server) as mcp:
        pipe = MimicPipeline(client=client, mcp=mcp, settings=settings)
        res = pipe.mimic_pose(_pose_seq(), "jump_tribute", dry_run=True)

    assert res.status == "dry_run_saved"
    assert res.artifact is not None
    assert res.artifact.adaptation_mode == "upper_body_adapted"
    assert res.artifact.kinematics.lower_body_adapted is True
    assert res.artifact.kinematics.verified is True
    assert verify_artifact_signature(res.artifact) is True
    assert len(_actuation_calls(server)) == 0


def test_pipeline_repair_in_forge_bounded_retry_succeeds(tmp_path: Path) -> None:
    server = MockTonypiMCPServer(mode="batch")
    client = _PipelineStubClient(
        forge_outputs=[_clipped_forge_json("repaired_wave"), _safe_forge_json("repaired_wave")],
        decision_responses=[
            _decision_response(route="repair_in_forge", safe_prob=0.25),
            _decision_response(route="execute_and_save", safe_prob=0.95, fidelity=2.6),
            _decision_response(route="execute_and_save", safe_prob=0.96, fidelity=2.7),
        ],
    )
    settings = RoboHarnessSettings(
        skills_dir=tmp_path / "skills",
        forge_state_path=tmp_path / "forge_state.json",
    )

    with _make_mcp(server) as mcp:
        pipe = MimicPipeline(client=client, mcp=mcp, settings=settings)
        res = pipe.mimic_pose(_pose_seq(), "repaired_wave", run_spatial=False)

    assert res.status == "executed_and_saved"
    assert res.forge_attempts == 2
    assert len(client.forge_prompts) == 2
    assert "Repair feedback" in client.forge_prompts[1]
    assert res.artifact is not None and res.artifact.kinematics.verified is True
    assert len(_actuation_calls(server)) == 1


def test_pipeline_refuse_impossible_via_decision_and_forge(tmp_path: Path) -> None:
    server = MockTonypiMCPServer(mode="batch")
    settings = RoboHarnessSettings(
        skills_dir=tmp_path / "skills",
        forge_state_path=tmp_path / "forge_state.json",
    )

    # Case 1: Forge refuses directly
    client_forge_refuse = _PipelineStubClient(
        forge_outputs=[json.dumps({"refused": True, "refusal_reason": "Backflip impossible.", "keyframes": []})]
    )
    with _make_mcp(server) as mcp:
        res1 = MimicPipeline(client=client_forge_refuse, mcp=mcp, settings=settings).mimic_pose(
            _pose_seq(), "backflip_1", run_spatial=False
        )
    assert res1.status == "refused"
    assert "backflip" in res1.reason.lower()
    assert len(_actuation_calls(server)) == 0

    # Case 2: Decision Gate refuses
    client_dec_refuse = _PipelineStubClient(
        decision_responses=[_decision_response(route="refuse_impossible", safe_prob=0.08)]
    )
    with _make_mcp(server) as mcp:
        res2 = MimicPipeline(client=client_dec_refuse, mcp=mcp, settings=settings).mimic_pose(
            _pose_seq(), "backflip_2", run_spatial=False
        )
    assert res2.status == "refused"
    assert len(_actuation_calls(server)) == 0


def test_pipeline_escalate_human_low_confidence(tmp_path: Path) -> None:
    server = MockTonypiMCPServer(mode="batch")
    client = _PipelineStubClient(
        decision_responses=[_decision_response(route="execute_and_save", route_conf=0.61)]
    )
    settings = RoboHarnessSettings(
        skills_dir=tmp_path / "skills",
        forge_state_path=tmp_path / "forge_state.json",
    )

    with _make_mcp(server) as mcp:
        pipe = MimicPipeline(client=client, mcp=mcp, settings=settings)
        res = pipe.mimic_pose(_pose_seq(), "ambiguous_move", run_spatial=False)
    assert res.status == "escalated_human"
    assert len(_actuation_calls(server)) == 0


def test_pipeline_safety_aborts_low_battery_obstructed_and_clipped_latch(tmp_path: Path) -> None:
    settings = RoboHarnessSettings(
        skills_dir=tmp_path / "skills",
        forge_state_path=tmp_path / "forge_state.json",
    )

    # 1. Low battery hardware latch
    low_bat_server = MockTonypiMCPServer(battery_v=10.4)
    with _make_mcp(low_bat_server) as mcp:
        res_bat = MimicPipeline(client=None, mcp=mcp, settings=settings, offline_mode=True).mimic_pose(
            _pose_seq(), "low_bat_test"
        )
    assert res_bat.status == "safety_aborted"
    assert "10.40V" in res_bat.reason
    assert len(_actuation_calls(low_bat_server)) == 0

    # 2. Spatial workspace obstructed
    ok_server = MockTonypiMCPServer()
    blocked_client = _PipelineStubClient(
        spatial_result=SpatialFramingResult(
            centered_on_table=True,
            workspace_clear=False,
            clearance_margin="edge_risk",
            summary="Coffee mug directly inside arm swing arc.",
        )
    )
    with _make_mcp(ok_server) as mcp:
        res_obs = MimicPipeline(client=blocked_client, mcp=mcp, settings=settings).mimic_pose(
            _pose_seq(), "blocked_test", run_spatial=True
        )
    assert res_obs.status == "safety_aborted"
    assert "Coffee mug" in res_obs.reason
    assert len(_actuation_calls(ok_server)) == 0

    # 3. Spatial edge_risk forces upper-body adaptation hint
    edge_client = _PipelineStubClient(
        spatial_result=SpatialFramingResult(
            centered_on_table=False,
            workspace_clear=True,
            clearance_margin="edge_risk",
            summary="Near front table edge.",
        )
    )
    with _make_mcp(ok_server) as mcp:
        res_edge = MimicPipeline(client=edge_client, mcp=mcp, settings=settings).mimic_pose(
            _pose_seq(), "edge_safe", dry_run=True, run_spatial=True
        )
    assert res_edge.status == "dry_run_saved"
    assert res_edge.artifact is not None
    assert res_edge.artifact.adaptation_mode == "upper_body_adapted"
    assert "edge_risk_upper_body_only" in edge_client.forge_prompts[0]

    # 4. Final deterministic kinematics latch blocks clipped skill even if Decision Gate approves
    dishonest_client = _PipelineStubClient(
        forge_outputs=[_clipped_forge_json("slip_past_gate")],
        decision_responses=[_decision_response(route="execute_and_save", safe_prob=0.95)],
    )
    with _make_mcp(ok_server) as mcp:
        res_latch = MimicPipeline(client=dishonest_client, mcp=mcp, settings=settings).mimic_pose(
            _pose_seq(), "slip_past_gate", run_spatial=False
        )
    assert res_latch.status == "safety_aborted"
    assert "kinematics latch" in res_latch.reason.lower()
    assert len(_actuation_calls(ok_server)) == 0


def test_pipeline_offline_video_and_run_saved_skill(tmp_path: Path) -> None:
    server = MockTonypiMCPServer(mode="batch")
    settings = RoboHarnessSettings(
        skills_dir=tmp_path / "skills",
        forge_state_path=tmp_path / "forge_state.json",
    )
    dummy_video = tmp_path / "clip.mp4"
    dummy_video.write_bytes(b"not-a-real-mp4-fallback-ok")

    with _make_mcp(server) as mcp:
        pipe = MimicPipeline(client=None, mcp=mcp, settings=settings, offline_mode=True)
        mimic_res = pipe.mimic_video(dummy_video, "offline_wave", dry_run=True)
        assert mimic_res.status == "dry_run_saved"
        assert mimic_res.artifact is not None
        assert verify_artifact_signature(mimic_res.artifact) is True

        replay_res = pipe.run_saved_skill("offline_wave", dry_run=False, run_spatial=False)
        assert replay_res.status == "executed_and_saved"
        assert replay_res.execution is not None
        assert replay_res.execution.frames_executed == len(mimic_res.artifact.keyframes)
        assert len(_actuation_calls(server)) == 1


def test_run_preflight_check_all_ok_and_failure_capture(tmp_path: Path) -> None:
    server = MockTonypiMCPServer()
    client = _PipelineStubClient()
    settings = RoboHarnessSettings(
        skills_dir=tmp_path / "skills",
        forge_state_path=tmp_path / "forge_state.json",
    )

    with _make_mcp(server) as mcp:
        report = run_preflight_check(client, mcp, settings)
    assert isinstance(report, PreflightReport)
    assert report.all_ok is True
    assert len(report.items) == 5
    assert [i.component for i in report.items] == [
        MOTION_MODEL,
        FORGE_AGENT_ID,
        "gemini-decision-1",
        SPATIAL_MODEL,
        "tonypi_mcp",
    ]

    # Failure capture: hydrogen error + low battery MCP
    bad_client = _PipelineStubClient(fail_hydrogen=True)
    bad_server = MockTonypiMCPServer(battery_v=10.2)
    with _make_mcp(bad_server) as bad_mcp:
        bad_report = run_preflight_check(bad_client, bad_mcp, settings, skip_forge_create=True)
    assert bad_report.all_ok is False
    by_comp = {i.component: i for i in bad_report.items}
    assert by_comp[MOTION_MODEL].ok is False
    assert by_comp["tonypi_mcp"].ok is False
    assert by_comp[FORGE_AGENT_ID].ok is True
