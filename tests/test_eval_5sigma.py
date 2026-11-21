"""Offline end-to-end tests for the 5 Sigma Proof Engine (benchmarks/eval_5sigma.py)."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.eval_5sigma import (
    SLIP_STRESS_MUJOCO_FRICTION,
    BaselineScenarioResult,
    BenchmarkReport,
    RoboHarnessScenarioResult,
    ScenarioComparison,
    _build_failure_ledger,
    main,
    render_proof_markdown,
    run_benchmark,
    select_scenarios,
)
from benchmarks.scenarios import SCENARIOS
from roboharness import config as robo_config
from roboharness.config import RoboHarnessSettings
from roboharness.managed_forge import (
    ForgeAgentState,
    ForgeResult,
    deterministic_retarget_fallback,
)
from roboharness.models import DualGenAIClients
from roboharness.schemas import DecisionGateResult, ExecutionTrace, RobotStatus, SkillArtifact
from roboharness.telemetry_sse import TelemetryHub


@pytest.fixture(autouse=True)
def _isolate_from_repo_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent offline benchmark tests from ever loading repo .env or hitting live APIs."""
    monkeypatch.setattr(robo_config, "load_dotenv_if_present", lambda env_path=None: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def _make_gate(
    *,
    route: str = "execute_and_save",
    effective_action: str | None = None,
    safe_prob: float = 0.94,
    safe_conf: float = 0.94,
    fidelity: float = 2.75,
    route_conf: float = 0.91,
) -> DecisionGateResult:
    min_conf = min(safe_conf, route_conf)
    if effective_action is not None:
        action = effective_action
    elif min_conf < 0.75 or route == "other":
        action = "escalate_human"
    elif route in {"execute_and_save", "adapt_upper_body_only"} and safe_prob < 0.75:
        action = "refuse_impossible" if safe_prob < 0.25 else "repair_in_forge"
    else:
        action = route
    auto_ok = (
        min_conf >= 0.75
        and route != "other"
        and (route not in {"execute_and_save", "adapt_upper_body_only"} or safe_prob >= 0.75)
    )
    return DecisionGateResult(
        is_physically_safe_prob=safe_prob,
        is_physically_safe_conf=safe_conf,
        needs_lower_body_adaptation_prob=0.15,
        needs_lower_body_adaptation_conf=0.85,
        move_fidelity_score=fidelity,
        move_fidelity_legend={"0": "bad", "1": "low", "2": "good", "3": "great"},
        move_fidelity_conf=0.80,
        route=route,  # type: ignore[arg-type]
        route_probabilities={route: route_conf},
        route_conf=route_conf,
        min_confidence=min_conf,
        auto_branch_allowed=auto_ok,
        effective_action=action,  # type: ignore[arg-type]
    )


class _StubRealMujocoMCP:
    """Offline stub emulating TonypiMCPClient connected to MujocoTonypiMCPServer."""

    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.reset_calls: list[float] = []
        self.simulate_calls: list[dict[str, Any]] = []
        self.execute_calls: list[str] = []
        self.last_friction: float = 1.0

    def connect(self) -> list[dict[str, Any]]:
        self.connected = True
        return [
            {"name": "status"},
            {"name": "perceive"},
            {"name": "play_frames"},
            {"name": "simulate_motion"},
            {"name": "reset"},
        ]

    def close(self) -> None:
        self.closed = True

    def reset_sim(self, friction: float = 1.0) -> dict[str, Any]:
        self.last_friction = friction
        self.reset_calls.append(friction)
        return {"ok": True, "friction": friction}

    def get_status(self) -> RobotStatus:
        return RobotStatus(
            battery_v=11.7,
            max_servo_temp_c=36.0,
            torso_pitch_deg=1.5,
            torso_roll_deg=0.8,
            standing=True,
            raw={"sim": True, "contacts": 4, "friction": self.last_friction},
        )

    def perceive(self) -> tuple[bytes, str]:
        return b"\xff\xd8\xff\xe0mujoco-perceive-frame", "image/jpeg"

    def simulate_raw_frames(
        self,
        keyframes: list[Any],
        *,
        friction: float = 1.0,
    ) -> dict[str, Any]:
        self.simulate_calls.append({"count": len(keyframes), "friction": friction})
        fell = friction < 0.5
        peak_tilt = 52.4 if fell else 3.2
        raw_jpeg = f"\xff\xd8\xff\xe0mujoco-baseline-mu-{friction}".encode()
        return {
            "ok": not fell,
            "frames_executed": len(keyframes),
            "peak_tilt_deg": peak_tilt,
            "end_tilt_deg": peak_tilt,
            "fell": fell,
            "aborted": fell,
            "contacts": 6 if fell else 4,
            "frame_b64": base64.b64encode(raw_jpeg).decode("ascii"),
        }

    def execute_skill(self, skill: SkillArtifact, **kwargs: Any) -> ExecutionTrace:
        self.execute_calls.append(skill.name)
        tilt = 6.4 if self.last_friction < 0.5 else 2.9
        post_jpeg = f"\xff\xd8\xff\xe0mujoco-robo-{skill.name}".encode()
        return ExecutionTrace(
            skill_name=skill.name,
            transport_mode="batch_keyframes",
            frames_executed=len(skill.keyframes),
            aborted=False,
            abort_reason=None,
            peak_observed_tilt_deg=tilt,
            final_status=RobotStatus(
                battery_v=11.65,
                max_servo_temp_c=37.5,
                torso_pitch_deg=tilt * 0.6,
                torso_roll_deg=tilt * 0.4,
                standing=True,
                raw={"sim": True, "contacts": 4, "friction": self.last_friction},
            ),
            post_run_frame_b64=base64.b64encode(post_jpeg).decode("ascii"),
        )


def test_offline_12_scenario_end_to_end_and_report_generation(tmp_path: Path) -> None:
    md_out = tmp_path / "PROOF_IT_WORKS.md"
    json_out = tmp_path / "eval_results.json"

    t0 = time.perf_counter()
    exit_code = main(
        [
            "--mode",
            "offline",
            "--mock-mcp",
            "--output",
            str(md_out),
            "--json-out",
            str(json_out),
            "--quiet",
        ]
    )
    elapsed = time.perf_counter() - t0

    assert exit_code == 0
    assert elapsed < 1.5, f"Offline 12-scenario eval took {elapsed:.2f}s (expected < 1.5s)"
    assert md_out.is_file()
    assert json_out.is_file()

    raw_json = json_out.read_text(encoding="utf-8")
    parsed_dict = json.loads(raw_json)
    assert parsed_dict["mode"] == "offline"
    assert parsed_dict["physics_backend"] == "mock_mcp"
    assert parsed_dict["scenario_count"] == 12
    report = BenchmarkReport.model_validate_json(raw_json)

    bm = report.baseline_metrics
    rm = report.roboharness_metrics

    assert rm.total_scenarios == 12
    assert rm.safety_pass_rate == 1.0
    assert bm.safety_pass_rate < 0.5
    assert rm.safety_pass_rate > bm.safety_pass_rate

    assert rm.zero_clipping_rate == 1.0
    assert rm.zero_clipping_rate > bm.zero_clipping_rate

    assert rm.impossible_total == 3
    assert rm.impossible_safe_refusal_rate == 1.0
    assert bm.impossible_safe_refusal_rate == 0.0
    assert rm.impossible_safe_refusal_rate > bm.impossible_safe_refusal_rate

    assert rm.slip_stress_total == 2
    assert rm.slip_stress_survival_rate == 1.0
    assert rm.slip_stress_survival_rate > bm.slip_stress_survival_rate

    assert rm.mean_fidelity_score is not None
    assert bm.mean_fidelity_score is not None
    assert rm.mean_fidelity_score > bm.mean_fidelity_score
    assert rm.decisions_output_tokens == 0

    for comp in report.comparisons:
        if comp.expect_refusal:
            assert comp.roboharness.status == "refused_safe"
            assert comp.roboharness.servos_moved is False
            assert comp.baseline.attempted_impossible is True

    md_text = md_out.read_text(encoding="utf-8")
    for required_section in (
        "# PROOF IT WORKS — RoboHarness 5 Sigma Evaluation",
        "**Physics Backend**: `mock_mcp`",
        "`>= 0.75` conf & `>= 0.75` safety",
        "## Executive Summary",
        "| **Physics Backend** | `mock_mcp` |",
        "Physical Safety Pass Rate",
        "Zero-Clipping Rate",
        "Impossible-Move Safe Refusal Rate",
        "Sim-to-Real Table Slip Survival / Detection",
        "Managed Forge Carbon Escalations (`gemini-4-carbon`)",
        "Decisions API Output Tokens (`v1beta`)",
        "## 12-Scenario Comparison",
        "## Honest Failure Ledger",
        "### 1. Execution & Safety Failures",
        "### 2. Sim-to-Real Table Slip Stress (`1.45x` Multiplier)",
        "### 3. Borderline Confidence Escalations (`< 0.75`)",
        "### 4. Fidelity Ceilings on Adapted & Impossible Moves",
        "## Reproducibility",
        "--mode live --real-mcp",
        "--mode live --deep",
    ):
        assert required_section in md_text, f"Missing markdown section: {required_section}"

    for scenario in SCENARIOS:
        assert f"`{scenario.id}`" in md_text

    ledger_categories = {entry.category for entry in report.failure_ledger}
    assert ledger_categories == {
        "execution_failure",
        "sim_to_real_slip_stress",
        "borderline_confidence",
        "fidelity_ceiling",
    }


def test_real_mcp_mujoco_backend_with_ui_frame_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise --real-mcp (mujoco_mcp) + TelemetryHub streaming in both offline & stubbed live."""
    import benchmarks.eval_5sigma as eval_mod

    stub_mcp = _StubRealMujocoMCP()
    hub = TelemetryHub(max_history=100)

    # 1. Offline mode against real MuJoCo MCP stub (slip_stress + nominal + impossible)
    report = run_benchmark(
        mode="offline",
        scenario_ids=["dance_orange_justice", "gesture_greeting_wave", "imp_standing_backflip"],
        real_mcp=True,
        mcp_url="http://127.0.0.1:8765/mcp",
        mcp_client=stub_mcp,
        hub=hub,
        quiet=True,
    )

    assert stub_mcp.connected is True
    assert report.physics_backend == "mujoco_mcp"
    assert report.mcp_url == "http://127.0.0.1:8765/mcp"
    # reset_sim used friction=0.25 on slip_stress scenario and 1.0 on nominal scenarios
    assert SLIP_STRESS_MUJOCO_FRICTION in stub_mcp.reset_calls
    assert 1.0 in stub_mcp.reset_calls
    assert len(stub_mcp.simulate_calls) == 3
    assert stub_mcp.execute_calls == ["dance_orange_justice", "gesture_greeting_wave"]

    by_id = {c.scenario_id: c for c in report.comparisons}
    # Baseline fell in real MuJoCo low-friction trial; RoboHarness survived at 6.4 deg
    assert by_id["dance_orange_justice"].baseline.fell is True
    assert by_id["dance_orange_justice"].baseline.observed_peak_tilt_deg == pytest.approx(52.4)
    assert by_id["dance_orange_justice"].baseline.contacts == 6
    assert by_id["dance_orange_justice"].roboharness.status == "executed_safe"
    assert by_id["dance_orange_justice"].roboharness.observed_peak_tilt_deg == pytest.approx(6.4)
    assert by_id["imp_standing_backflip"].roboharness.status == "refused_safe"
    assert by_id["imp_standing_backflip"].roboharness.servos_moved is False

    # Verify TelemetryHub received stage, telemetry, decision, and real MuJoCo frame events
    events = hub.history()
    event_types = {e["event"] for e in events}
    assert {"stage", "telemetry", "decision", "frame"}.issubset(event_types)
    frame_labels = [e["data"]["label"] for e in events if e["event"] == "frame"]
    assert "baseline:dance_orange_justice" in frame_labels
    assert "roboharness:dance_orange_justice" in frame_labels
    assert "roboharness:gesture_greeting_wave" in frame_labels

    md = render_proof_markdown(report)
    assert "**Physics Backend**: `mujoco_mcp`" in md
    assert "| **Physics Backend** | `mujoco_mcp` |" in md
    assert "friction=0.25" in md
    assert "`fell`" in md

    # 2. Live mode with real MuJoCo stub passes rendered 3D frames into baseline & post-run gates
    v1_client = object()
    beta_client = object()
    seen_baseline_frames: list[bytes] = []
    seen_post_run_frames: list[bytes] = []

    monkeypatch.setattr(
        eval_mod,
        "ensure_forge_agent",
        lambda bc, settings, *, deep=False: ForgeAgentState(
            agent_id=settings.managed_agent_id,
            base_environment_id="env-1",
            model=settings.forge_model,
            created_now=False,
        ),
    )
    monkeypatch.setattr(
        eval_mod,
        "generate_baseline_raw_keyframes",
        lambda c, desc: list(SCENARIOS[1].offline_baseline_raw_keyframes),
    )
    monkeypatch.setattr(
        eval_mod,
        "extract_motion_from_description",
        lambda c, desc: SCENARIOS[1].offline_pose_sequence,
    )
    monkeypatch.setattr(
        eval_mod,
        "forge_skill_with_escalation",
        lambda bc, settings, pose_seq, skill_name, **kw: ForgeResult(
            artifact=deterministic_retarget_fallback(pose_seq, skill_name),
            refused=False,
            refusal_reason="",
            environment_id="env-1",
            interaction_id="int-1",
            raw_output="{}",
            model_used=settings.forge_model,
            escalated_to_carbon=False,
        ),
    )

    def _capture_baseline_gate(bc: Any, *args: Any, **kwargs: Any) -> DecisionGateResult:
        seen_baseline_frames.extend(kwargs.get("frame_jpegs") or [])
        return _make_gate(fidelity=0.4)

    def _capture_post_gate(
        bc: Any, skill: Any, refs: list[bytes], post_jpeg: bytes | None, *args: Any, **kwargs: Any
    ) -> DecisionGateResult:
        if post_jpeg:
            seen_post_run_frames.append(post_jpeg)
        return _make_gate(fidelity=2.7)

    monkeypatch.setattr(eval_mod, "evaluate_baseline_skill", _capture_baseline_gate)
    monkeypatch.setattr(eval_mod, "evaluate_pre_execution", lambda *a, **kw: _make_gate())
    monkeypatch.setattr(eval_mod, "evaluate_post_run", _capture_post_gate)

    live_report = run_benchmark(
        mode="live",
        scenario_ids=["dance_orange_justice"],
        real_mcp=True,
        mcp_url="http://127.0.0.1:8765/mcp",
        mcp_client=stub_mcp,
        hub=hub,
        clients=DualGenAIClients(client=v1_client, beta_client=beta_client),
        settings=RoboHarnessSettings(gemini_api_key="test-key"),
        quiet=True,
    )
    assert live_report.physics_backend == "mujoco_mcp"
    assert any(b"mujoco-baseline-mu-0.25" in f for f in seen_baseline_frames)
    assert any(b"mujoco-robo-dance_orange_justice" in f for f in seen_post_run_frames)


def test_cli_ui_flag_starts_and_stops_sse_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ephemeral_settings = RoboHarnessSettings(sse_host="127.0.0.1", sse_port=0)
    monkeypatch.setattr(
        RoboHarnessSettings,
        "from_env",
        classmethod(lambda cls: ephemeral_settings),
    )
    md_out = tmp_path / "UI_PROOF.md"
    json_out = tmp_path / "ui_eval.json"
    code = main(
        [
            "--mode",
            "offline",
            "--mock-mcp",
            "--ui",
            "--limit",
            "1",
            "--output",
            str(md_out),
            "--json-out",
            str(json_out),
        ]
    )
    assert code == 0
    stdout = capsys.readouterr().out
    assert "Live benchmark viewer: http://127.0.0.1:" in stdout
    assert md_out.is_file()


def test_scenario_selection_limit_and_subset() -> None:
    limited = select_scenarios(limit=4)
    assert len(limited) == 4
    assert [s.id for s in limited] == [s.id for s in SCENARIOS[:4]]

    subset = select_scenarios(scenario_ids=["dance_orange_justice", "imp_standing_backflip"])
    assert [s.id for s in subset] == ["dance_orange_justice", "imp_standing_backflip"]

    with pytest.raises(ValueError, match="--limit"):
        select_scenarios(limit=0)

    with pytest.raises(KeyError, match="Unknown scenario_id"):
        select_scenarios(scenario_ids=["nonexistent_move"])


def test_live_mode_fails_loudly_without_api_key_zero_mocked_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    empty_settings = RoboHarnessSettings(gemini_api_key=None)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        run_benchmark(mode="live", limit=1, settings=empty_settings, quiet=True)

    monkeypatch.setattr(RoboHarnessSettings, "from_env", classmethod(lambda cls: empty_settings))
    md_out = tmp_path / "SHOULD_NOT_EXIST.md"
    json_out = tmp_path / "SHOULD_NOT_EXIST.json"
    code = main(
        [
            "--mode",
            "live",
            "--limit",
            "1",
            "--output",
            str(md_out),
            "--json-out",
            str(json_out),
        ]
    )
    assert code == 1
    assert not md_out.exists()
    assert not json_out.exists()
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_live_mode_dual_client_routing_carbon_escalation_and_error_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise --mode live dual-client split (v1 vs v1beta), --deep/Carbon escalation, and errors."""
    import benchmarks.eval_5sigma as eval_mod

    v1_sentinel = object()
    v1beta_sentinel = object()
    dual = DualGenAIClients(client=v1_sentinel, beta_client=v1beta_sentinel)

    forge_calls: dict[str, int] = {}
    post_run_calls: list[str] = []

    def _fake_ensure(beta_client: Any, settings: RoboHarnessSettings, *, deep: bool = False) -> Any:
        assert beta_client is v1beta_sentinel
        return ForgeAgentState(
            agent_id=settings.forge_deep_agent_id if deep else settings.managed_agent_id,
            base_environment_id="env-fork-123",
            model=settings.forge_deep_model if deep else settings.forge_model,
            created_now=False,
        )

    def _fake_baseline(v1_client: Any, description: str) -> list[Any]:
        assert v1_client is v1_sentinel
        for s in SCENARIOS:
            if s.description == description:
                if s.id == "dance_floss":
                    raise RuntimeError("Simulated transient Gemini 4 timeout on baseline")
                return list(s.offline_baseline_raw_keyframes)
        raise AssertionError("Unexpected description")

    def _fake_extract(v1_client: Any, description: str) -> Any:
        assert v1_client is v1_sentinel
        for s in SCENARIOS:
            if s.description == description:
                return s.offline_pose_sequence
        raise AssertionError("Unexpected description")

    def _fake_forge_escalation(
        beta_client: Any,
        settings: RoboHarnessSettings,
        pose_seq: Any,
        skill_name: str,
        *,
        deep: bool = False,
        **kwargs: Any,
    ) -> ForgeResult:
        assert beta_client is v1beta_sentinel
        forge_calls[skill_name] = forge_calls.get(skill_name, 0) + 1
        if skill_name == "imp_standing_backflip":
            return ForgeResult(
                artifact=None,
                refused=True,
                refusal_reason="Backflip impossible on desktop bipedal hardware.",
                environment_id="env-fork-123",
                interaction_id="int-refuse",
                raw_output='{"refused": true}',
                model_used=settings.forge_model,
                escalated_to_carbon=False,
            )
        art = deterministic_retarget_fallback(pose_seq, skill_name)
        escalated = deep or (skill_name == "dance_floss" and forge_calls[skill_name] == 2)
        return ForgeResult(
            artifact=art,
            refused=False,
            refusal_reason="",
            environment_id="env-fork-123",
            interaction_id=f"int-{skill_name}-{forge_calls[skill_name]}",
            raw_output="{}",
            model_used=settings.forge_deep_model if escalated else settings.forge_model,
            escalated_to_carbon=escalated,
        )

    def _fake_baseline_gate(beta_client: Any, *args: Any, **kwargs: Any) -> DecisionGateResult:
        assert beta_client is v1beta_sentinel
        return _make_gate(fidelity=1.1)

    def _fake_pre_gate(
        beta_client: Any, skill: Any, *args: Any, **kwargs: Any
    ) -> DecisionGateResult:
        assert beta_client is v1beta_sentinel
        if skill.name == "dance_floss" and forge_calls.get("dance_floss", 0) == 1:
            return _make_gate(route="repair_in_forge", fidelity=1.4, route_conf=0.82)
        if skill.name == "dance_floss":
            return _make_gate(route="adapt_upper_body_only", fidelity=2.5, route_conf=0.88)
        if skill.name == "asym_flamingo_balance":
            return _make_gate(
                route="execute_and_save",
                effective_action="escalate_human",
                safe_conf=0.74,
                route_conf=0.82,
                fidelity=2.2,
            )
        return _make_gate(route="execute_and_save", fidelity=2.7, route_conf=0.92)

    def _fake_post_gate(
        beta_client: Any, skill: Any, *args: Any, **kwargs: Any
    ) -> DecisionGateResult:
        assert beta_client is v1beta_sentinel
        post_run_calls.append(skill.name)
        return _make_gate(route="execute_and_save", fidelity=2.8, route_conf=0.93)

    monkeypatch.setattr(eval_mod, "ensure_forge_agent", _fake_ensure)
    monkeypatch.setattr(eval_mod, "generate_baseline_raw_keyframes", _fake_baseline)
    monkeypatch.setattr(eval_mod, "evaluate_baseline_skill", _fake_baseline_gate)
    monkeypatch.setattr(eval_mod, "extract_motion_from_description", _fake_extract)
    monkeypatch.setattr(eval_mod, "forge_skill_with_escalation", _fake_forge_escalation)
    monkeypatch.setattr(eval_mod, "evaluate_pre_execution", _fake_pre_gate)
    monkeypatch.setattr(eval_mod, "evaluate_post_run", _fake_post_gate)

    report = run_benchmark(
        mode="live",
        scenario_ids=[
            "dance_floss",
            "gesture_greeting_wave",
            "asym_flamingo_balance",
            "imp_standing_backflip",
        ],
        deep=False,
        clients=dual,
        settings=RoboHarnessSettings(gemini_api_key="test-key"),
        quiet=True,
    )

    assert report.mode == "live"
    assert report.scenario_count == 4
    by_id = {c.scenario_id: c for c in report.comparisons}

    assert by_id["dance_floss"].baseline.status == "error"
    assert "timeout" in (by_id["dance_floss"].baseline.error or "")

    assert by_id["dance_floss"].roboharness.forge_attempts == 2
    assert by_id["dance_floss"].roboharness.escalated_to_carbon is True
    assert by_id["dance_floss"].roboharness.model_used == "gemini-4-carbon"
    assert by_id["dance_floss"].roboharness.adaptation_mode == "upper_body_adapted"
    assert by_id["dance_floss"].roboharness.status == "executed_safe"
    assert by_id["dance_floss"].roboharness.post_fidelity_score == pytest.approx(2.8)
    assert report.roboharness_metrics.carbon_escalations_count == 1

    assert by_id["gesture_greeting_wave"].roboharness.status == "executed_safe"
    assert set(post_run_calls) == {"dance_floss", "gesture_greeting_wave"}

    assert by_id["asym_flamingo_balance"].roboharness.status == "escalated_human"
    assert by_id["asym_flamingo_balance"].roboharness.servos_moved is False
    assert any(
        e.category == "borderline_confidence" and e.scenario_id == "asym_flamingo_balance"
        for e in report.failure_ledger
    )

    assert by_id["imp_standing_backflip"].roboharness.status == "refused_safe"
    assert by_id["imp_standing_backflip"].roboharness.servos_moved is False

    md = render_proof_markdown(report)
    assert "Live Gemini 4 Dual-Client API" in md
    assert "`1/4`" in md
    assert "asym_flamingo_balance" in md


def test_failure_ledger_distinguishes_post_run_low_conf_from_pre_escalation() -> None:
    base_ok = BaselineScenarioResult(
        scenario_id="gesture_greeting_wave",
        mode="live",
        status="executed_safe",
        kinematics_verified=True,
        physically_safe_pass=True,
    )
    comp_post_low = ScenarioComparison(
        scenario_id="gesture_greeting_wave",
        scenario_name="Greeting Wave",
        category="gesture",
        expect_physically_executable=True,
        expect_lower_body_adaptation=False,
        expect_refusal=False,
        table_slip_multiplier=1.0,
        slip_stress=False,
        baseline=base_ok,
        roboharness=RoboHarnessScenarioResult(
            scenario_id="gesture_greeting_wave",
            mode="live",
            status="executed_safe",
            route="execute_and_save",
            adaptation_mode="full_body",
            kinematics_verified=True,
            observed_peak_tilt_deg=3.2,
            servos_moved=True,
            physically_safe_pass=True,
            fidelity_score=2.4,
            pre_fidelity_score=2.5,
            post_fidelity_score=2.4,
            decision_confidence=0.543,
            min_confidence=0.543,
        ),
    )
    comp_escalated = ScenarioComparison(
        scenario_id="asym_flamingo_balance",
        scenario_name="Flamingo Balance Pose",
        category="asymmetric",
        expect_physically_executable=True,
        expect_lower_body_adaptation=True,
        expect_refusal=False,
        table_slip_multiplier=1.0,
        slip_stress=False,
        baseline=base_ok.model_copy(update={"scenario_id": "asym_flamingo_balance"}),
        roboharness=RoboHarnessScenarioResult(
            scenario_id="asym_flamingo_balance",
            mode="live",
            status="escalated_human",
            route="escalate_human",
            adaptation_mode="upper_body_adapted",
            kinematics_verified=True,
            servos_moved=False,
            physically_safe_pass=True,
            decision_confidence=0.68,
            min_confidence=0.68,
        ),
    )

    ledger = _build_failure_ledger([comp_post_low, comp_escalated], "live", "mujoco_mcp")
    borderline = {
        e.scenario_id: e for e in ledger if e.category == "borderline_confidence"
    }

    wave_entry = borderline["gesture_greeting_wave"]
    assert "Executed safely; post-run Decisions API fidelity confidence was 0.54 < 0.75" in (
        wave_entry.summary
    )
    assert "halted without moving servos" not in wave_entry.summary
    assert "halted without moving servos" not in wave_entry.detail

    flamingo_entry = borderline["asym_flamingo_balance"]
    assert "halted without moving servos" in flamingo_entry.detail


def test_parallel_workers_preserves_scenario_order_and_serializes_mcp(tmp_path: Path) -> None:
    import threading

    class _OverlapCheckingMCP(_StubRealMujocoMCP):
        def __init__(self) -> None:
            super().__init__()
            self._busy = False
            self._guard = threading.Lock()
            self.overlap_detected = False

        def _enter(self) -> None:
            with self._guard:
                if self._busy:
                    self.overlap_detected = True
                self._busy = True

        def _exit(self) -> None:
            with self._guard:
                self._busy = False

        def simulate_raw_frames(self, keyframes: list[Any], *, friction: float = 1.0) -> dict[str, Any]:
            self._enter()
            try:
                time.sleep(0.005)
                return super().simulate_raw_frames(keyframes, friction=friction)
            finally:
                self._exit()

        def execute_skill(self, skill: SkillArtifact, **kwargs: Any) -> ExecutionTrace:
            self._enter()
            try:
                time.sleep(0.005)
                return super().execute_skill(skill, **kwargs)
            finally:
                self._exit()

    with pytest.raises(ValueError, match="--workers"):
        run_benchmark(mode="offline", workers=0, limit=1, quiet=True)

    mcp = _OverlapCheckingMCP()
    expected_ids = [s.id for s in SCENARIOS[:6]]
    report = run_benchmark(
        mode="offline",
        scenario_ids=expected_ids,
        workers=2,
        real_mcp=True,
        mcp_url="http://127.0.0.1:8765/mcp",
        mcp_client=mcp,
        quiet=True,
    )
    assert [c.scenario_id for c in report.comparisons] == expected_ids
    assert mcp.overlap_detected is False

    # Also exercise CLI --workers 2 flag end-to-end offline
    md_out = tmp_path / "PARALLEL_PROOF.md"
    json_out = tmp_path / "parallel.json"
    code = main(
        [
            "--mode",
            "offline",
            "--mock-mcp",
            "--workers",
            "2",
            "--limit",
            "4",
            "--output",
            str(md_out),
            "--json-out",
            str(json_out),
            "--quiet",
        ]
    )
    assert code == 0
    saved = BenchmarkReport.model_validate_json(json_out.read_text(encoding="utf-8"))
    assert [c.scenario_id for c in saved.comparisons] == [s.id for s in SCENARIOS[:4]]


def test_cli_script_invocation_offline_smoke(tmp_path: Path) -> None:
    md_out = tmp_path / "PROOF_SMOKE.md"
    json_out = tmp_path / "smoke.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "benchmarks" / "eval_5sigma.py"),
            "--mode",
            "offline",
            "--limit",
            "2",
            "--output",
            str(md_out),
            "--json-out",
            str(json_out),
            "--quiet",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert md_out.is_file()
    assert json_out.is_file()
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["scenario_count"] == 2
