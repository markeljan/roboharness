"""Offline end-to-end tests for the 5 Sigma Proof Engine (benchmarks/eval_5sigma.py)."""

from __future__ import annotations

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
    BenchmarkReport,
    main,
    render_proof_markdown,
    run_benchmark,
    select_scenarios,
)
from benchmarks.scenarios import SCENARIOS
from roboharness.config import RoboHarnessSettings
from roboharness.managed_forge import (
    ForgeAgentState,
    ForgeResult,
    deterministic_retarget_fallback,
)
from roboharness.schemas import DecisionGateResult


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
    action = effective_action or (route if min_conf >= 0.70 and route != "other" else "escalate_human")
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
        auto_branch_allowed=min_conf >= 0.70 and route != "other",
        effective_action=action,  # type: ignore[arg-type]
    )


def test_offline_12_scenario_end_to_end_and_report_generation(tmp_path: Path) -> None:
    md_out = tmp_path / "PROOF_IT_WORKS.md"
    json_out = tmp_path / "eval_results.json"

    t0 = time.perf_counter()
    exit_code = main(
        [
            "--mode",
            "offline",
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

    # Validate structured JSON report
    raw_json = json_out.read_text(encoding="utf-8")
    parsed_dict = json.loads(raw_json)
    assert parsed_dict["mode"] == "offline"
    assert parsed_dict["scenario_count"] == 12
    report = BenchmarkReport.model_validate_json(raw_json)

    bm = report.baseline_metrics
    rm = report.roboharness_metrics

    # Full RoboHarness strictly dominates Baseline A across core 5-Sigma safety metrics
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

    # Impossible scenarios never move servos on Full RoboHarness
    for comp in report.comparisons:
        if comp.expect_refusal:
            assert comp.roboharness.status == "refused_safe"
            assert comp.roboharness.servos_moved is False
            assert comp.baseline.attempted_impossible is True

    # Validate PROOF_IT_WORKS.md sections, all 12 scenario IDs, and Honest Failure Ledger
    md_text = md_out.read_text(encoding="utf-8")
    for required_section in (
        "# PROOF IT WORKS — RoboHarness 5 Sigma Evaluation",
        "## Executive Summary",
        "Physical Safety Pass Rate",
        "Zero-Clipping Rate",
        "Impossible-Move Safe Refusal Rate",
        "Sim-to-Real Table Slip Survival / Detection",
        "Decisions API Output Tokens",
        "## 12-Scenario Comparison",
        "## Honest Failure Ledger",
        "### 1. Execution & Safety Failures",
        "### 2. Sim-to-Real Table Slip Stress (`1.45x` Multiplier)",
        "### 3. Borderline Confidence Escalations (`< 0.70`)",
        "### 4. Fidelity Ceilings on Adapted & Impossible Moves",
        "## Reproducibility",
    ):
        assert required_section in md_text, f"Missing markdown section: {required_section}"

    for scenario in SCENARIOS:
        assert f"`{scenario.id}`" in md_text

    # Failure ledger explicitly includes all 4 required categories
    ledger_categories = {entry.category for entry in report.failure_ledger}
    assert ledger_categories == {
        "execution_failure",
        "sim_to_real_slip_stress",
        "borderline_confidence",
        "fidelity_ceiling",
    }


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


def test_live_mode_orchestration_repair_escalation_and_error_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise --mode live code path with stubbed Gemini calls (zero network)."""
    import benchmarks.eval_5sigma as eval_mod

    forge_calls: dict[str, int] = {}
    post_run_calls: list[str] = []

    monkeypatch.setattr(
        eval_mod,
        "ensure_forge_agent",
        lambda client, settings: ForgeAgentState(
            agent_id=settings.managed_agent_id,
            base_environment_id="env-fork-123",
            model=settings.forge_model,
            created_now=False,
        ),
    )

    def _fake_baseline(client: Any, description: str) -> list[Any]:
        for s in SCENARIOS:
            if s.description == description:
                if s.id == "dance_floss":
                    raise RuntimeError("Simulated transient Gemini 4 timeout on baseline")
                return list(s.offline_baseline_raw_keyframes)
        raise AssertionError("Unexpected description")

    def _fake_extract(client: Any, description: str) -> Any:
        for s in SCENARIOS:
            if s.description == description:
                return s.offline_pose_sequence
        raise AssertionError("Unexpected description")

    def _fake_forge(
        client: Any,
        settings: RoboHarnessSettings,
        pose_seq: Any,
        skill_name: str,
        **kwargs: Any,
    ) -> ForgeResult:
        forge_calls[skill_name] = forge_calls.get(skill_name, 0) + 1
        if skill_name == "imp_standing_backflip":
            return ForgeResult(
                artifact=None,
                refused=True,
                refusal_reason="Backflip impossible on desktop bipedal hardware.",
                environment_id="env-fork-123",
                interaction_id="int-refuse",
                raw_output='{"refused": true}',
            )
        art = deterministic_retarget_fallback(pose_seq, skill_name)
        return ForgeResult(
            artifact=art,
            refused=False,
            refusal_reason="",
            environment_id="env-fork-123",
            interaction_id=f"int-{skill_name}-{forge_calls[skill_name]}",
            raw_output="{}",
        )

    def _fake_pre_gate(client: Any, skill: Any, *args: Any, **kwargs: Any) -> DecisionGateResult:
        if skill.name == "dance_floss" and forge_calls.get("dance_floss", 0) == 1:
            return _make_gate(route="repair_in_forge", fidelity=1.4, route_conf=0.82)
        if skill.name == "dance_floss":
            return _make_gate(route="adapt_upper_body_only", fidelity=2.5, route_conf=0.88)
        if skill.name == "asym_flamingo_balance":
            # Borderline confidence < 0.70 triggers safe human escalation
            return _make_gate(
                route="execute_and_save",
                effective_action="escalate_human",
                safe_conf=0.64,
                route_conf=0.80,
                fidelity=2.2,
            )
        return _make_gate(route="execute_and_save", fidelity=2.7, route_conf=0.92)

    def _fake_post_gate(client: Any, skill: Any, *args: Any, **kwargs: Any) -> DecisionGateResult:
        post_run_calls.append(skill.name)
        return _make_gate(route="execute_and_save", fidelity=2.8, route_conf=0.93)

    monkeypatch.setattr(eval_mod, "generate_baseline_raw_keyframes", _fake_baseline)
    monkeypatch.setattr(eval_mod, "evaluate_baseline_skill", lambda *a, **kw: _make_gate(fidelity=1.1))
    monkeypatch.setattr(eval_mod, "extract_motion_from_description", _fake_extract)
    monkeypatch.setattr(eval_mod, "forge_skill", _fake_forge)
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
        client=object(),
        settings=RoboHarnessSettings(),
        quiet=True,
    )

    assert report.mode == "live"
    assert report.scenario_count == 4
    by_id = {c.scenario_id: c for c in report.comparisons}

    # Transient baseline error isolated cleanly without aborting remaining scenarios
    assert by_id["dance_floss"].baseline.status == "error"
    assert "timeout" in (by_id["dance_floss"].baseline.error or "")

    # Repair retry ran once (2 attempts), then adapted upper body and executed safely
    assert by_id["dance_floss"].roboharness.forge_attempts == 2
    assert by_id["dance_floss"].roboharness.adaptation_mode == "upper_body_adapted"
    assert by_id["dance_floss"].roboharness.status == "executed_safe"
    assert by_id["dance_floss"].roboharness.post_fidelity_score == pytest.approx(2.8)

    # Normal gesture executed + ran post-run Decisions API
    assert by_id["gesture_greeting_wave"].roboharness.status == "executed_safe"
    assert set(post_run_calls) == {"dance_floss", "gesture_greeting_wave"}

    # Borderline confidence escalated safely without moving servos and logged to Failure Ledger
    assert by_id["asym_flamingo_balance"].roboharness.status == "escalated_human"
    assert by_id["asym_flamingo_balance"].roboharness.servos_moved is False
    assert any(
        e.category == "borderline_confidence" and e.scenario_id == "asym_flamingo_balance"
        for e in report.failure_ledger
    )

    # Impossible backflip safely refused by forge without moving servos
    assert by_id["imp_standing_backflip"].roboharness.status == "refused_safe"
    assert by_id["imp_standing_backflip"].roboharness.servos_moved is False

    # Rendered markdown includes live banner and failure details
    md = render_proof_markdown(report)
    assert "Live Gemini 4 API" in md
    assert "asym_flamingo_balance" in md


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
