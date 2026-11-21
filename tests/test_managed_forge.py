"""Tests for Google Managed Agent Skill Forge and standalone kinematics_verify.py."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from roboharness.config import (
    FORBIDDEN_GEMINI4_PARAMS,
    FORGE_AGENT_ID,
    FORGE_BASE_AGENT,
    FORGE_DEEP_AGENT_ID,
    FORGE_DEEP_MODEL,
    FORGE_MODEL,
    REMOTE_MCP_NAME,
    RESERVED_AGENT_ID_PREFIXES,
    RoboHarnessSettings,
)
from roboharness.managed_forge import (
    BUNDLE_DIR,
    ForgeAgentState,
    build_forge_tools,
    deterministic_retarget_fallback,
    ensure_forge_agent,
    extract_json_object,
    forge_skill,
    forge_skill_with_escalation,
    load_bundle_inline_sources,
    validate_agent_id,
    validate_mcp_tool_name,
)
from roboharness.safety import check_skill_keyframes, verify_artifact_signature
from roboharness.schemas import (
    NEUTRAL_POSE,
    TONYPI_JOINTS,
    HumanPoseKeyframe,
    ImpossibleElement,
    PoseSequence,
    ServoKeyframe,
)

VERIFY_SCRIPT = BUNDLE_DIR / "kinematics_verify.py"


def _sample_pose_sequence(*, with_jump: bool = False) -> PoseSequence:
    kfs = [
        HumanPoseKeyframe(
            t_ms=0,
            beat_index=0,
            torso_pitch_deg=4.0,
            torso_roll_deg=2.0,
            l_shoulder_pitch_deg=45.0,
            l_shoulder_roll_deg=25.0,
            l_elbow_deg=35.0,
            r_shoulder_pitch_deg=-30.0,
            r_shoulder_roll_deg=20.0,
            r_elbow_deg=25.0,
            l_hip_pitch_deg=10.0,
            l_knee_deg=15.0,
            r_hip_pitch_deg=10.0,
            r_knee_deg=15.0,
            foot_contact="both",
            label="arms_left_swing",
        ),
        HumanPoseKeyframe(
            t_ms=500,
            beat_index=1,
            torso_pitch_deg=6.0,
            torso_roll_deg=-3.0,
            l_shoulder_pitch_deg=-30.0,
            l_shoulder_roll_deg=15.0,
            l_elbow_deg=20.0,
            r_shoulder_pitch_deg=60.0,
            r_shoulder_roll_deg=30.0,
            r_elbow_deg=40.0,
            l_hip_pitch_deg=12.0,
            l_knee_deg=20.0,
            r_hip_pitch_deg=12.0,
            r_knee_deg=20.0,
            foot_contact="airborne" if with_jump else "both",
            label="jump_clap" if with_jump else "arms_right_swing",
        ),
    ]
    impossible = (
        [
            ImpossibleElement(
                element="jump",
                timestamp_ms=500,
                reason="TonyPi cannot jump safely on a table.",
                suggested_adaptation="Grounded knee bounce + expressive arms.",
            )
        ]
        if with_jump
        else []
    )
    return PoseSequence(
        source_description="Upbeat two-beat arm wave dance",
        bpm=120.0,
        duration_ms=1000,
        keyframes=kfs,
        impossible_elements=impossible,
        style_tags=["dance", "upbeat"],
    )


class _FakeAgentsAPI:
    def __init__(self, existing: Any = None, *, existing_by_id: dict[str, Any] | None = None) -> None:
        self.existing_by_id: dict[str, Any] = dict(existing_by_id or {})
        if existing is not None:
            key = getattr(existing, "id", FORGE_AGENT_ID)
            self.existing_by_id[key] = existing
        self.get_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []

    def get(self, *, id: str) -> Any:
        self.get_calls.append({"id": id})
        if id not in self.existing_by_id:
            raise KeyError(f"Agent '{id}' not found")
        return self.existing_by_id[id]

    def delete(self, *, id: str) -> dict[str, Any]:
        self.delete_calls.append({"id": id})
        self.existing_by_id.pop(id, None)
        return {}

    def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        agent = SimpleNamespace(**kwargs)
        self.existing_by_id[kwargs["id"]] = agent
        return agent


class _FakeInteractionsAPI:
    def __init__(self, responses: list[Any], *, poll_responses: dict[str, list[Any]] | None = None) -> None:
        self._responses = list(responses)
        self._poll_responses = {k: list(v) for k, v in (poll_responses or {}).items()}
        self.create_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    def get(self, *, id: str) -> Any:
        self.get_calls.append({"id": id})
        return self._poll_responses[id].pop(0)


class _FakeGenAIClient:
    def __init__(
        self,
        *,
        existing_agent: Any = None,
        existing_by_id: dict[str, Any] | None = None,
        interaction_responses: list[Any] | None = None,
        poll_responses: dict[str, list[Any]] | None = None,
    ) -> None:
        self.agents = _FakeAgentsAPI(existing=existing_agent, existing_by_id=existing_by_id)
        self.interactions = _FakeInteractionsAPI(
            interaction_responses or [],
            poll_responses=poll_responses,
        )


def _run_verifier(*args: str, stdin_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), *args],
        input=stdin_text,
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# Subprocess tests for agent_bundle/kinematics_verify.py
# ---------------------------------------------------------------------------


def test_kinematics_verify_self_test_subprocess() -> None:
    proc = _run_verifier("--self-test")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["self_test"] == "ok"
    assert payload["joints"] == 16


def test_kinematics_verify_safe_skill_file_and_stdin(tmp_path: Path) -> None:
    safe_skill = {
        "name": "safe_wave",
        "keyframes": [
            {
                "duration_ms": 450,
                "servos": {**NEUTRAL_POSE, "r_shoulder_pitch": 680, "r_elbow": 600},
                "note": "raise",
            },
            {
                "duration_ms": 450,
                "servos": {**NEUTRAL_POSE, "r_shoulder_pitch": 500, "r_elbow": 500},
                "note": "neutral",
            },
        ],
    }
    path = tmp_path / "safe.json"
    path.write_text(json.dumps(safe_skill), encoding="utf-8")

    proc = _run_verifier(str(path))
    assert proc.returncode == 0, proc.stdout
    report = json.loads(proc.stdout)
    assert report["verified"] is True
    assert report["joint_clipping_count"] == 0
    assert report["peak_torso_tilt_deg"] < 30.0

    # Also verify stdin '-' mode and parity with roboharness.safety
    proc_stdin = _run_verifier("-", stdin_text=json.dumps(safe_skill))
    assert proc_stdin.returncode == 0
    stdin_report = json.loads(proc_stdin.stdout)
    local_report = check_skill_keyframes([ServoKeyframe(**kf) for kf in safe_skill["keyframes"]])
    assert stdin_report["peak_torso_tilt_deg"] == local_report.peak_torso_tilt_deg
    assert stdin_report["max_com_offset_norm"] == local_report.max_com_offset_norm
    assert stdin_report["max_slew_rate_pps"] == local_report.max_slew_rate_pps


def test_kinematics_verify_detects_clipped_servo(tmp_path: Path) -> None:
    clipped = {
        "keyframes": [
            {
                "duration_ms": 500,
                "servos": {**NEUTRAL_POSE, "r_shoulder_pitch": 950},
                "note": "over_limit",
            }
        ]
    }
    path = tmp_path / "clipped.json"
    path.write_text(json.dumps(clipped), encoding="utf-8")

    proc = _run_verifier(str(path))
    assert proc.returncode == 1
    report = json.loads(proc.stdout)
    assert report["verified"] is False
    assert report["joint_clipping_count"] == 1
    assert any("r_shoulder_pitch" in v for v in report["violations"])
    assert report["suggested_fixes"]


def test_kinematics_verify_detects_high_tilt(tmp_path: Path) -> None:
    tilted = {
        "keyframes": [
            {
                "duration_ms": 600,
                "servos": {
                    **NEUTRAL_POSE,
                    "r_ankle_pitch": 680,
                    "l_ankle_pitch": 680,
                    "r_hip_pitch": 720,
                    "l_hip_pitch": 720,
                },
                "note": "lean_over",
            }
        ]
    }
    path = tmp_path / "tilted.json"
    path.write_text(json.dumps(tilted), encoding="utf-8")

    proc = _run_verifier(str(path))
    assert proc.returncode == 1
    report = json.loads(proc.stdout)
    assert report["verified"] is False
    assert report["peak_torso_tilt_deg"] >= 30.0
    assert report["max_com_offset_norm"] >= 1.0


def test_kinematics_verify_auto_fix_repairs_to_verified(tmp_path: Path) -> None:
    unsafe = {
        "name": "wild_move",
        "keyframes": [
            {
                "duration_ms": 100,
                "servos": {
                    **NEUTRAL_POSE,
                    "r_shoulder_pitch": 960,
                    "r_ankle_pitch": 680,
                    "l_ankle_pitch": 680,
                    "r_hip_pitch": 720,
                    "l_hip_pitch": 720,
                },
                "note": "unsafe_0",
            },
            {
                "duration_ms": 100,
                "servos": {
                    **NEUTRAL_POSE,
                    "l_shoulder_pitch": 100,
                    "r_ankle_pitch": 680,
                    "l_ankle_pitch": 680,
                    "r_hip_pitch": 720,
                    "l_hip_pitch": 720,
                },
                "note": "unsafe_1",
            },
        ],
    }
    in_path = tmp_path / "unsafe.json"
    out_path = tmp_path / "fixed.json"
    in_path.write_text(json.dumps(unsafe), encoding="utf-8")

    fix_proc = _run_verifier("--auto-fix", str(in_path), "--out", str(out_path))
    assert fix_proc.returncode == 0, fix_proc.stdout
    fix_report = json.loads(fix_proc.stdout)
    assert fix_report["verified"] is True
    assert fix_report["joint_clipping_count"] == 0
    assert fix_report["peak_torso_tilt_deg"] < 30.0
    assert fix_report["lower_body_adapted"] is True
    assert out_path.is_file()

    # Re-verify the written file via standard CLI mode
    recheck = _run_verifier(str(out_path))
    assert recheck.returncode == 0, recheck.stdout
    recheck_report = json.loads(recheck.stdout)
    assert recheck_report["verified"] is True


# ---------------------------------------------------------------------------
# Unit tests for bundle loading and agent lifecycle
# ---------------------------------------------------------------------------


def test_load_bundle_inline_sources_contents_and_targets() -> None:
    sources = load_bundle_inline_sources()
    assert len(sources) == 3
    by_target = {s["target"]: s for s in sources}
    assert set(by_target.keys()) == {
        ".agents/AGENTS.md",
        ".agents/skills/kinematics-verifier/SKILL.md",
        "/workspace/kinematics_verify.py",
    }
    for src in sources:
        assert src["type"] == "inline"
        assert src["content"].strip()

    skill_md = by_target[".agents/skills/kinematics-verifier/SKILL.md"]["content"]
    assert "name: kinematics-verifier" in skill_md
    for joint in TONYPI_JOINTS:
        assert joint.name in skill_md

    agents_md = by_target[".agents/AGENTS.md"]["content"]
    assert "/workspace/kinematics_verify.py" in agents_md
    assert "SKILL.md" in agents_md


def test_ensure_forge_agent_bootstraps_and_forks_environment(tmp_path: Path) -> None:
    bootstrap_resp = SimpleNamespace(
        id="inter_boot_1",
        status="completed",
        environment_id="env_pre_staged_abc123",
        output_text='{"self_test": "ok"}',
    )
    client = _FakeGenAIClient(existing_agent=None, interaction_responses=[bootstrap_resp])
    state_file = tmp_path / "forge_state.json"
    settings = RoboHarnessSettings(forge_state_path=state_file)

    state = ensure_forge_agent(client, settings, poll_interval_sec=0.0)
    assert isinstance(state, ForgeAgentState)
    assert state.agent_id == FORGE_AGENT_ID == "roboharness-skill-forge"
    assert not state.agent_id.startswith(RESERVED_AGENT_ID_PREFIXES)
    assert state.base_environment_id == "env_pre_staged_abc123"
    assert state.model == FORGE_MODEL == "gemini-4-neon"
    assert state.created_now is True

    # Bootstrap call checked
    assert len(client.interactions.create_calls) == 1
    boot_call = client.interactions.create_calls[0]
    assert boot_call["agent"] == FORGE_BASE_AGENT == "antigravity-preview-09-2026"
    assert boot_call["environment"]["type"] == "remote"
    assert len(boot_call["environment"]["sources"]) == 3

    # Custom agent creation invariants checked
    assert len(client.agents.create_calls) == 1
    create_call = client.agents.create_calls[0]
    assert create_call["id"] == "roboharness-skill-forge"
    assert create_call["base_agent"] == "antigravity-preview-09-2026"
    assert create_call["agent_config"] == {"type": "antigravity", "model": "gemini-4-neon"}
    assert create_call["base_environment"] == "env_pre_staged_abc123"

    # Saved state persisted cleanly
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved["agent_id"] == "roboharness-skill-forge"
    assert saved["base_environment_id"] == "env_pre_staged_abc123"

    # Second call reuses existing agent without recreating
    state_again = ensure_forge_agent(client, settings, poll_interval_sec=0.0)
    assert state_again.created_now is False
    assert state_again.base_environment_id == "env_pre_staged_abc123"
    assert len(client.agents.create_calls) == 1


def test_ensure_forge_agent_force_recreate_and_inline_fallback(tmp_path: Path) -> None:
    existing = SimpleNamespace(
        id=FORGE_AGENT_ID,
        base_environment="env_old",
        agent_config={"type": "antigravity", "model": FORGE_MODEL},
    )
    client = _FakeGenAIClient(
        existing_agent=existing,
        interaction_responses=[RuntimeError("transient bootstrap timeout")],
    )
    settings = RoboHarnessSettings(forge_state_path=tmp_path / "state.json")

    state = ensure_forge_agent(client, settings, force_recreate=True, poll_interval_sec=0.0)
    assert state.created_now is True
    assert state.base_environment_id is None
    assert client.agents.delete_calls == [{"id": FORGE_AGENT_ID}]

    # Fell back cleanly to direct inline base_environment
    create_call = client.agents.create_calls[0]
    assert isinstance(create_call["base_environment"], dict)
    assert create_call["base_environment"]["type"] == "remote"
    assert len(create_call["base_environment"]["sources"]) == 3


@pytest.mark.parametrize("bad_id", ["gemini-forge", "google-skill", "antigravity-custom", ""])
def test_validate_agent_id_rejects_reserved_prefixes(bad_id: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        validate_agent_id(bad_id)

    client = _FakeGenAIClient()
    bad_settings = RoboHarnessSettings(
        managed_agent_id=bad_id,
        forge_state_path=tmp_path / "state.json",
    )
    with pytest.raises(ValueError):
        ensure_forge_agent(client, bad_settings)
    assert not client.agents.create_calls


# ---------------------------------------------------------------------------
# Unit tests for forge_skill, remote MCP tool validation, and signing
# ---------------------------------------------------------------------------


def test_forge_skill_success_polling_mcp_and_sha256_signature(tmp_path: Path) -> None:
    existing = SimpleNamespace(
        id=FORGE_AGENT_ID,
        base_environment="env_base_1",
        agent_config={"type": "antigravity", "model": FORGE_MODEL},
    )
    forge_payload = {
        "refused": False,
        "refusal_reason": "",
        "name": "disco_wave",
        "description": "Grounded two-beat disco wave",
        "adaptation_mode": "full_body",
        "lower_body_adapted": False,
        "adaptation_notes": [],
        "keyframes": [
            {
                "duration_ms": 500,
                "servos": {**NEUTRAL_POSE, "r_shoulder_pitch": 660, "l_shoulder_pitch": 340},
                "note": "beat 0",
            },
            {
                "duration_ms": 500,
                "servos": {**NEUTRAL_POSE, "r_shoulder_pitch": 500, "l_shoulder_pitch": 500},
                "note": "beat 1",
            },
        ],
    }
    # Simulate background polling + markdown fenced JSON output
    initial_inter = SimpleNamespace(
        id="inter_forge_42",
        status="in_progress",
        environment_id="env_run_42",
        output_text=None,
    )
    done_inter = SimpleNamespace(
        id="inter_forge_42",
        status="completed",
        environment_id="env_run_42",
        output_text=f"```json\n{json.dumps(forge_payload)}\n```",
    )
    client = _FakeGenAIClient(
        existing_agent=existing,
        interaction_responses=[initial_inter],
        poll_responses={"inter_forge_42": [done_inter]},
    )
    settings = RoboHarnessSettings(
        remote_tonypi_mcp_url="https://demo.example.com/mcp",
        forge_state_path=tmp_path / "state.json",
    )

    result = forge_skill(
        client,
        settings,
        _sample_pose_sequence(),
        "disco_wave",
        source_video="clip.mp4",
        use_remote_mcp=True,
        poll_interval_sec=0.0,
    )

    assert not result.refused
    assert result.artifact is not None
    assert result.environment_id == "env_run_42"
    assert result.interaction_id == "inter_forge_42"
    assert result.artifact.kinematics.verified is True
    assert result.artifact.kinematics.joint_clipping_count == 0
    assert result.artifact.kinematics.peak_torso_tilt_deg < 30.0
    assert result.artifact.kinematics.sandbox_environment_id == "env_run_42"
    assert verify_artifact_signature(result.artifact) is True

    # Verify SDK call invariants
    assert len(client.interactions.create_calls) == 1
    call = client.interactions.create_calls[0]
    assert call["agent"] == "roboharness-skill-forge"
    assert call["environment"] == "remote"
    assert "agent_config" not in call
    for forbidden in FORBIDDEN_GEMINI4_PARAMS | {"max_output_tokens"}:
        assert forbidden not in call

    mcp_tools = [t for t in call["tools"] if t["type"] == "mcp_server"]
    assert len(mcp_tools) == 1
    assert mcp_tools[0]["name"] == REMOTE_MCP_NAME == "tonypi_mcp"
    assert re.match(r"^[a-z0-9_-]+$", mcp_tools[0]["name"])
    assert mcp_tools[0]["url"] == "https://demo.example.com/mcp"
    assert len(client.interactions.get_calls) == 1


@pytest.mark.parametrize("bad_name", ["TonyPi", "tonypi.mcp", "tonypi mcp", "mcp/1", ""])
def test_remote_mcp_tool_name_validation(bad_name: str) -> None:
    with pytest.raises(ValueError):
        validate_mcp_tool_name(bad_name)

    settings = RoboHarnessSettings(remote_tonypi_mcp_url="https://demo.example.com/mcp")
    with pytest.raises(ValueError):
        build_forge_tools(settings, use_remote_mcp=True, mcp_tool_name=bad_name)


def test_forge_skill_defense_in_depth_catches_sandbox_clipping_and_optional_repair(tmp_path: Path) -> None:
    existing = SimpleNamespace(
        id=FORGE_AGENT_ID,
        base_environment="env_base",
        agent_config={"type": "antigravity", "model": FORGE_MODEL},
    )
    # Sandbox erroneously claims verified=true despite clipped servo pulse
    dishonest_payload = {
        "refused": False,
        "name": "bad_arm",
        "keyframes": [
            {
                "duration_ms": 500,
                "servos": {**NEUTRAL_POSE, "r_shoulder_pitch": 940},
                "note": "clipped",
            }
        ],
        "kinematics": {"verified": True},
    }
    resp = SimpleNamespace(
        id="inter_bad",
        status="completed",
        environment_id="env_bad",
        output_text=json.dumps(dishonest_payload),
    )
    settings = RoboHarnessSettings(forge_state_path=tmp_path / "state.json")

    # Default mode: defense-in-depth marks verified=False accurately
    client1 = _FakeGenAIClient(existing_agent=existing, interaction_responses=[resp])
    res_strict = forge_skill(client1, settings, _sample_pose_sequence(), "bad_arm", poll_interval_sec=0.0)
    assert res_strict.artifact is not None
    assert res_strict.artifact.kinematics.verified is False
    assert res_strict.artifact.kinematics.joint_clipping_count == 1
    assert verify_artifact_signature(res_strict.artifact) is True

    # auto_repair_locally=True: repairs into verified + signed artifact
    client2 = _FakeGenAIClient(existing_agent=existing, interaction_responses=[resp])
    res_fixed = forge_skill(
        client2,
        settings,
        _sample_pose_sequence(),
        "bad_arm",
        auto_repair_locally=True,
        poll_interval_sec=0.0,
    )
    assert res_fixed.artifact is not None
    assert res_fixed.artifact.kinematics.verified is True
    assert res_fixed.artifact.kinematics.joint_clipping_count == 0
    assert verify_artifact_signature(res_fixed.artifact) is True


def test_forge_skill_handles_explicit_refusal(tmp_path: Path) -> None:
    existing = SimpleNamespace(
        id=FORGE_AGENT_ID,
        base_environment="env_base",
        agent_config={"type": "antigravity", "model": FORGE_MODEL},
    )
    refusal_payload = {
        "refused": True,
        "refusal_reason": "Standing backflip cannot be executed safely on TonyPi.",
        "keyframes": [],
    }
    resp = SimpleNamespace(
        id="inter_refuse",
        status="completed",
        environment_id="env_refuse",
        output_text=json.dumps(refusal_payload),
    )
    client = _FakeGenAIClient(existing_agent=existing, interaction_responses=[resp])
    settings = RoboHarnessSettings(forge_state_path=tmp_path / "state.json")

    result = forge_skill(client, settings, _sample_pose_sequence(with_jump=True), "backflip")
    assert result.refused is True
    assert result.artifact is None
    assert "backflip" in result.refusal_reason.lower()


def test_extract_json_object_handles_prose_wrapper() -> None:
    raw = 'Here is the verified skill:\n{"refused": false, "keyframes": []}\nDone.'
    obj = extract_json_object(raw)
    assert obj == {"refused": False, "keyframes": []}


def test_deterministic_retarget_fallback_grounded_and_jump_adaptation() -> None:
    grounded = deterministic_retarget_fallback(_sample_pose_sequence(with_jump=False), "wave_ci")
    assert grounded.adaptation_mode == "full_body"
    assert grounded.kinematics.verified is True
    assert grounded.kinematics.joint_clipping_count == 0
    assert grounded.kinematics.peak_torso_tilt_deg < 30.0
    assert verify_artifact_signature(grounded) is True

    adapted = deterministic_retarget_fallback(_sample_pose_sequence(with_jump=True), "jump_ci")
    assert adapted.adaptation_mode == "upper_body_adapted"
    assert adapted.kinematics.lower_body_adapted is True
    assert adapted.kinematics.verified is True
    assert adapted.kinematics.joint_clipping_count == 0
    assert adapted.kinematics.peak_torso_tilt_deg < 30.0
    assert verify_artifact_signature(adapted) is True


# ---------------------------------------------------------------------------
# Builder Guide Adjustment #3: GEMINI_ENVIRONMENT_ID & Neon -> Carbon escalation
# ---------------------------------------------------------------------------


def test_self_hosted_gemini_environment_id_omits_sources_and_allowlist(tmp_path: Path) -> None:
    forge_payload = {
        "refused": False,
        "name": "mujoco_wave",
        "keyframes": [
            {
                "duration_ms": 450,
                "servos": {**NEUTRAL_POSE, "r_shoulder_pitch": 620},
                "note": "beat 0",
            }
        ],
    }
    resp = SimpleNamespace(
        id="inter_self_hosted",
        status="completed",
        environment_id="env_docker_mujoco_99",
        output_text=json.dumps(forge_payload),
    )
    client = _FakeGenAIClient(existing_agent=None, interaction_responses=[resp])
    settings = RoboHarnessSettings(
        gemini_environment_id="env_docker_mujoco_99",
        forge_state_path=tmp_path / "state.json",
    )

    state = ensure_forge_agent(client, settings, poll_interval_sec=0.0)
    assert state.base_environment_id == "env_docker_mujoco_99"
    assert state.created_now is True
    # Must NOT run hosted bootstrap with inline sources when GEMINI_ENVIRONMENT_ID is set
    assert len(client.interactions.create_calls) == 0
    create_call = client.agents.create_calls[0]
    assert create_call["base_environment"] == "env_docker_mujoco_99"
    assert "sources" not in json.dumps(create_call)
    assert "allowlist" not in json.dumps(create_call)
    assert "max_total_tokens" not in create_call["agent_config"]

    result = forge_skill(client, settings, _sample_pose_sequence(), "mujoco_wave", poll_interval_sec=0.0)
    assert result.artifact is not None and result.artifact.kinematics.verified is True
    inter_call = client.interactions.create_calls[0]
    assert inter_call["environment"] == "env_docker_mujoco_99"
    assert "sources" not in json.dumps(inter_call)
    assert "allowlist" not in json.dumps(inter_call)


def test_forge_escalation_from_neon_to_carbon_after_two_failures(tmp_path: Path) -> None:
    clipped_payload = {
        "refused": False,
        "name": "hard_lunge",
        "keyframes": [
            {
                "duration_ms": 450,
                "servos": {**NEUTRAL_POSE, "r_shoulder_pitch": 950},
                "note": "clipped",
            }
        ],
    }
    clean_payload = {
        "refused": False,
        "name": "hard_lunge",
        "adaptation_mode": "upper_body_adapted",
        "lower_body_adapted": True,
        "keyframes": [
            {
                "duration_ms": 450,
                "servos": {**NEUTRAL_POSE, "r_shoulder_pitch": 680},
                "note": "carbon_fixed",
            }
        ],
    }
    neon_agent = SimpleNamespace(
        id=FORGE_AGENT_ID,
        base_environment="env_neon",
        agent_config={"type": "antigravity", "model": FORGE_MODEL},
    )
    carbon_agent = SimpleNamespace(
        id=FORGE_DEEP_AGENT_ID,
        base_environment="env_carbon",
        agent_config={"type": "antigravity", "model": FORGE_DEEP_MODEL},
    )
    client = _FakeGenAIClient(
        existing_by_id={FORGE_AGENT_ID: neon_agent, FORGE_DEEP_AGENT_ID: carbon_agent},
        interaction_responses=[
            SimpleNamespace(id="i1", status="completed", environment_id="e1", output_text=json.dumps(clipped_payload)),
            SimpleNamespace(id="i2", status="completed", environment_id="e2", output_text=json.dumps(clipped_payload)),
            SimpleNamespace(id="i3", status="completed", environment_id="e3", output_text=json.dumps(clean_payload)),
        ],
    )
    settings = RoboHarnessSettings(forge_state_path=tmp_path / "state.json")

    res = forge_skill_with_escalation(
        client,
        settings,
        _sample_pose_sequence(),
        "hard_lunge",
        poll_interval_sec=0.0,
    )
    assert res.escalated_to_carbon is True
    assert res.model_used == FORGE_DEEP_MODEL == "gemini-4-carbon"
    assert res.artifact is not None
    assert res.artifact.kinematics.verified is True
    assert verify_artifact_signature(res.artifact) is True

    # Two Neon calls followed by Carbon escalation carrying violation diagnostics
    agents_called = [c["agent"] for c in client.interactions.create_calls]
    assert agents_called == [FORGE_AGENT_ID, FORGE_AGENT_ID, FORGE_DEEP_AGENT_ID]
    assert "ESCALATED TO gemini-4-carbon" in client.interactions.create_calls[2]["input"]
    assert "r_shoulder_pitch=950" in client.interactions.create_calls[2]["input"]


def test_forge_deep_flag_routes_directly_to_carbon(tmp_path: Path) -> None:
    clean_payload = {
        "refused": False,
        "name": "deep_move",
        "keyframes": [
            {
                "duration_ms": 450,
                "servos": {**NEUTRAL_POSE, "r_shoulder_pitch": 640},
                "note": "carbon_direct",
            }
        ],
    }
    bootstrap_resp = SimpleNamespace(
        id="boot_carbon",
        status="completed",
        environment_id="env_carbon_fork",
        output_text='{"self_test": "ok"}',
    )
    forge_resp = SimpleNamespace(
        id="inter_carbon_direct",
        status="completed",
        environment_id="env_carbon_run",
        output_text=json.dumps(clean_payload),
    )
    client = _FakeGenAIClient(
        existing_agent=None,
        interaction_responses=[bootstrap_resp, forge_resp],
    )
    settings = RoboHarnessSettings(forge_state_path=tmp_path / "state.json")

    res = forge_skill_with_escalation(
        client,
        settings,
        _sample_pose_sequence(),
        "deep_move",
        deep=True,
        poll_interval_sec=0.0,
    )
    assert res.escalated_to_carbon is True
    assert res.model_used == "gemini-4-carbon"
    assert res.artifact is not None and res.artifact.kinematics.verified is True

    # Created Carbon agent locked to gemini-4-carbon without reserved prefix or token cap
    create_call = client.agents.create_calls[0]
    assert create_call["id"] == "roboharness-skill-forge-carbon"
    assert not create_call["id"].startswith(RESERVED_AGENT_ID_PREFIXES)
    assert create_call["agent_config"] == {"type": "antigravity", "model": "gemini-4-carbon"}
    assert client.interactions.create_calls[1]["agent"] == "roboharness-skill-forge-carbon"
