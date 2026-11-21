"""Offline tests for gemini-decision-1 multimodal gate and signed skill store."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
from google.genai._gaos.types.decisions import (
    ChoiceAnswer,
    CreateDecisionRequest,
    CreateDecisionResponse,
    NoulAnswer,
    ScoreAnswer,
)

from roboharness.config import DECISION_MODEL
from roboharness.decision_gate import (
    MAX_DECISION_FRAMES,
    POST_RUN_QUESTIONS,
    PRE_EXECUTION_QUESTIONS,
    build_decision_input,
    evaluate_baseline_skill,
    evaluate_post_run,
    evaluate_pre_execution,
    parse_decision_response,
)
from roboharness.safety import check_skill_keyframes, sign_skill_artifact, verify_artifact_signature
from roboharness.schemas import (
    NEUTRAL_POSE,
    HumanPoseKeyframe,
    ImpossibleElement,
    PoseSequence,
    RobotStatus,
    ServoKeyframe,
    SkillArtifact,
)
from roboharness.skill_store import (
    SKILL_SLUG_RE,
    delete_skill,
    list_skills,
    load_skill,
    save_skill,
    slugify_skill_name,
)


def _make_sample_skill(
    name: str = "disco-wave",
    *,
    created_at: str = "2026-11-21T12:00:00+00:00",
    signed: bool = False,
) -> SkillArtifact:
    frames = [
        ServoKeyframe(
            duration_ms=350,
            servos={**NEUTRAL_POSE, "r_shoulder_pitch": 620, "l_shoulder_pitch": 420},
            note="beat_0_reach",
        ),
        ServoKeyframe(
            duration_ms=350,
            servos={**NEUTRAL_POSE, "r_shoulder_pitch": 440, "l_shoulder_pitch": 620},
            note="beat_1_switch",
        ),
    ]
    report = check_skill_keyframes(frames)
    artifact = SkillArtifact(
        name=name,
        created_at=created_at,
        source_video="clips/disco.mp4",
        description="Alternating rhythmic disco arm wave.",
        keyframes=frames,
        kinematics=report,
    )
    return sign_skill_artifact(artifact) if signed else artifact


def _make_sample_pose_seq() -> PoseSequence:
    return PoseSequence(
        source_description="Rhythmic arm wave with small jump at the end.",
        bpm=120.0,
        duration_ms=2000,
        keyframes=[
            HumanPoseKeyframe(
                t_ms=0,
                beat_index=0,
                torso_pitch_deg=4.0,
                torso_roll_deg=2.0,
                l_shoulder_pitch_deg=-20.0,
                l_shoulder_roll_deg=10.0,
                l_elbow_deg=35.0,
                r_shoulder_pitch_deg=45.0,
                r_shoulder_roll_deg=15.0,
                r_elbow_deg=40.0,
                l_hip_pitch_deg=5.0,
                l_knee_deg=10.0,
                r_hip_pitch_deg=5.0,
                r_knee_deg=10.0,
                foot_contact="both",
                label="wave_right",
            ),
            HumanPoseKeyframe(
                t_ms=1000,
                beat_index=1,
                torso_pitch_deg=6.0,
                torso_roll_deg=-3.0,
                l_shoulder_pitch_deg=50.0,
                l_shoulder_roll_deg=12.0,
                l_elbow_deg=30.0,
                r_shoulder_pitch_deg=-15.0,
                r_shoulder_roll_deg=10.0,
                r_elbow_deg=35.0,
                l_hip_pitch_deg=10.0,
                l_knee_deg=20.0,
                r_hip_pitch_deg=10.0,
                r_knee_deg=20.0,
                foot_contact="airborne",
                label="jump_pop",
            ),
        ],
        impossible_elements=[
            ImpossibleElement(
                element="jump",
                timestamp_ms=1000,
                reason="TonyPi cannot jump airborne safely.",
                suggested_adaptation="Grounded rhythmic knee bounce",
            )
        ],
        style_tags=["disco", "upbeat"],
    )


def _make_robot_status(**overrides: Any) -> RobotStatus:
    defaults: dict[str, Any] = {
        "battery_v": 11.6,
        "max_servo_temp_c": 39.5,
        "torso_pitch_deg": 2.0,
        "torso_roll_deg": -1.5,
        "standing": True,
    }
    defaults.update(overrides)
    return RobotStatus(**defaults)


def _make_sdk_response(
    *,
    safe_prob: float = 0.94,
    safe_conf: float = 0.94,
    adapt_prob: float = 0.12,
    adapt_conf: float = 0.88,
    fidelity_score: float = 2.65,
    fidelity_conf: float = 0.72,
    route: str = "execute_and_save",
    route_conf: float = 0.91,
    route_probabilities: dict[str, float] | None = None,
) -> CreateDecisionResponse:
    probs = route_probabilities or {
        "execute_and_save": route_conf if route == "execute_and_save" else 0.03,
        "adapt_upper_body_only": route_conf if route == "adapt_upper_body_only" else 0.02,
        "repair_in_forge": route_conf if route == "repair_in_forge" else 0.02,
        "refuse_impossible": route_conf if route == "refuse_impossible" else 0.01,
        "other": route_conf if route == "other" else 0.01,
    }
    return CreateDecisionResponse(
        answers={
            "is_physically_safe": NoulAnswer(noul=safe_prob, confidence=safe_conf),
            "needs_lower_body_adaptation": NoulAnswer(noul=adapt_prob, confidence=adapt_conf),
            "move_fidelity": ScoreAnswer(
                score=fidelity_score,
                confidence=fidelity_conf,
                legend={
                    "0": "Unrecognizable",
                    "1": "Minimal resemblance",
                    "2": "Recognizable adaptation",
                    "3": "High-fidelity retargeting",
                },
                probabilities={"0": 0.01, "1": 0.04, "2": 0.23, "3": 0.72},
            ),
            "route": ChoiceAnswer(
                choice=route,
                confidence=route_conf,
                probabilities=probs,
            ),
        }
    )


class _StubDecisionsAPI:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, *, model: str, input: Any, questions: Any, **kwargs: Any) -> Any:
        self.calls.append({"model": model, "input": input, "questions": questions, **kwargs})
        return self.response


class _StubClient:
    def __init__(self, response: Any) -> None:
        self.decisions = _StubDecisionsAPI(response)


# ---------------------------------------------------------------------------
# Payload & schema invariant tests
# ---------------------------------------------------------------------------


def test_question_schemas_invariants_and_sdk_validation() -> None:
    for qset in (PRE_EXECUTION_QUESTIONS, POST_RUN_QUESTIONS):
        serialized = json.dumps(qset)
        assert "none_of_the_above" not in serialized

        assert set(qset.keys()) == {
            "is_physically_safe",
            "needs_lower_body_adaptation",
            "move_fidelity",
            "route",
        }
        assert qset["is_physically_safe"]["type"] == "noul"
        assert qset["needs_lower_body_adaptation"]["type"] == "noul"
        assert qset["move_fidelity"]["type"] == "score"
        assert isinstance(qset["move_fidelity"]["criteria"], list)
        assert len(qset["move_fidelity"]["criteria"]) == 4

        assert qset["route"]["type"] == "choice"
        assert set(qset["route"]["criteria"].keys()) == {
            "execute_and_save",
            "adapt_upper_body_only",
            "repair_in_forge",
            "refuse_impossible",
            "other",
        }

        # Validate directly against real google-genai CreateDecisionRequest Pydantic schema
        decision_input = build_decision_input(
            _make_sample_skill(),
            _make_sample_pose_seq(),
            [b"\xff\xd8\xff\xe0fakejpeg"],
            _make_robot_status(),
        )
        req = CreateDecisionRequest.model_validate(
            {
                "model": DECISION_MODEL,
                "input": decision_input,
                "questions": qset,
            }
        )
        assert req.model == "gemini-decision-1"


def test_build_decision_input_format_and_frame_cap() -> None:
    skill = _make_sample_skill()
    pose_seq = _make_sample_pose_seq()
    status = _make_robot_status()
    frames = [f"jpeg-frame-{i}".encode() for i in range(10)]

    blocks = build_decision_input(
        skill,
        pose_seq,
        frames,
        status,
        extra_context={"spatial_margin": "safe"},
    )

    image_blocks = [b for b in blocks if b["type"] == "image"]
    text_blocks = [b for b in blocks if b["type"] == "text"]

    assert len(image_blocks) == MAX_DECISION_FRAMES == 6
    assert len(text_blocks) == 1

    # Every image block enforces low resolution and JPEG mime type
    for img in image_blocks:
        assert img["mime_type"] == "image/jpeg"
        assert img["resolution"] == "low"
        decoded = base64.b64decode(img["data"])
        assert decoded.startswith(b"jpeg-frame-")

    # Even sampling keeps first and last frame
    assert base64.b64decode(image_blocks[0]["data"]) == b"jpeg-frame-0"
    assert base64.b64decode(image_blocks[-1]["data"]) == b"jpeg-frame-9"

    payload = json.loads(text_blocks[0]["text"])
    assert payload["robot_status"]["battery_v"] == 11.6
    assert payload["hardware_safety_verdict"]["safe"] is True
    assert payload["skill"]["name"] == "disco-wave"
    assert payload["skill"]["kinematics"]["verified"] is True
    assert payload["pose_sequence"]["bpm"] == 120.0
    assert len(payload["pose_sequence"]["impossible_elements"]) == 1
    assert payload["extra_context"] == {"spatial_margin": "safe"}


# ---------------------------------------------------------------------------
# Response parsing, confidence gating, and safety overrides
# ---------------------------------------------------------------------------


def test_parse_sdk_response_and_dict_stub() -> None:
    sdk_resp = _make_sdk_response(
        safe_prob=0.92,
        safe_conf=0.92,
        adapt_prob=0.15,
        adapt_conf=0.85,
        fidelity_score=2.7,
        fidelity_conf=0.78,
        route="execute_and_save",
        route_conf=0.89,
    )
    parsed_sdk = parse_decision_response(sdk_resp)
    assert parsed_sdk.is_physically_safe_prob == pytest.approx(0.92)
    assert parsed_sdk.is_physically_safe_conf == pytest.approx(0.92)
    assert parsed_sdk.needs_lower_body_adaptation_prob == pytest.approx(0.15)
    assert parsed_sdk.move_fidelity_score == pytest.approx(2.7)
    assert parsed_sdk.move_fidelity_legend["3"] == "High-fidelity retargeting"
    assert parsed_sdk.route == "execute_and_save"
    assert parsed_sdk.route_probabilities["execute_and_save"] == pytest.approx(0.89)
    assert parsed_sdk.min_confidence == pytest.approx(0.89)
    assert parsed_sdk.auto_branch_allowed is True
    assert parsed_sdk.effective_action == "execute_and_save"

    # Dict-style stub response parses identically
    dict_resp = {
        "answers": {
            "is_physically_safe": {"noul": 0.92, "confidence": 0.92},
            "needs_lower_body_adaptation": {"noul": 0.15, "confidence": 0.85},
            "move_fidelity": {
                "score": 2.7,
                "confidence": 0.78,
                "legend": {"0": "bad", "1": "weak", "2": "good", "3": "great"},
            },
            "route": {
                "choice": "execute_and_save",
                "confidence": 0.89,
                "probabilities": {"execute_and_save": 0.89, "other": 0.11},
            },
        }
    }
    parsed_dict = parse_decision_response(dict_resp)
    assert parsed_dict.effective_action == "execute_and_save"
    assert parsed_dict.move_fidelity_legend["3"] == "great"
    assert parsed_dict.auto_branch_allowed is True


@pytest.mark.parametrize(
    "route",
    [
        "execute_and_save",
        "adapt_upper_body_only",
        "repair_in_forge",
        "refuse_impossible",
    ],
)
def test_auto_branching_high_confidence_routes(route: str) -> None:
    client = _StubClient(
        _make_sdk_response(
            safe_prob=0.88,
            safe_conf=0.88,
            adapt_prob=0.82 if route == "adapt_upper_body_only" else 0.15,
            adapt_conf=0.82,
            route=route,
            route_conf=0.84,
        )
    )
    res = evaluate_pre_execution(
        client,
        _make_sample_skill(),
        _make_sample_pose_seq(),
        [b"jpg1"],
        _make_robot_status(),
    )
    assert len(client.decisions.calls) == 1
    assert client.decisions.calls[0]["model"] == "gemini-decision-1"
    assert res.auto_branch_allowed is True
    assert res.route == route
    assert res.effective_action == route


def test_route_other_escalates_even_at_high_confidence() -> None:
    client = _StubClient(
        _make_sdk_response(safe_prob=0.90, safe_conf=0.90, route="other", route_conf=0.85)
    )
    res = evaluate_pre_execution(
        client, _make_sample_skill(), None, [b"jpg1"], _make_robot_status()
    )
    assert res.route == "other"
    assert res.auto_branch_allowed is False
    assert res.effective_action == "escalate_human"


def test_low_route_confidence_escalates_safely() -> None:
    client = _StubClient(
        _make_sdk_response(
            safe_prob=0.92,
            safe_conf=0.92,
            route="execute_and_save",
            route_conf=0.65,
        )
    )
    res = evaluate_pre_execution(
        client, _make_sample_skill(), None, [b"jpg1"], _make_robot_status()
    )
    assert res.min_confidence == pytest.approx(0.65)
    assert res.auto_branch_allowed is False
    assert res.effective_action == "escalate_human"


def test_low_safety_confidence_escalates_safely() -> None:
    client = _StubClient(
        _make_sdk_response(
            safe_prob=0.62,
            safe_conf=0.62,
            route="execute_and_save",
            route_conf=0.91,
        )
    )
    res = evaluate_pre_execution(
        client, _make_sample_skill(), None, [b"jpg1"], _make_robot_status()
    )
    assert res.min_confidence == pytest.approx(0.62)
    assert res.auto_branch_allowed is False
    assert res.effective_action == "escalate_human"


def test_exact_threshold_confidence_allows_auto_branch() -> None:
    client = _StubClient(
        _make_sdk_response(
            safe_prob=0.70,
            safe_conf=0.70,
            route="execute_and_save",
            route_conf=0.70,
        )
    )
    res = evaluate_pre_execution(
        client, _make_sample_skill(), None, [b"jpg1"], _make_robot_status()
    )
    assert res.min_confidence == pytest.approx(0.70)
    assert res.auto_branch_allowed is True
    assert res.effective_action == "execute_and_save"


def test_safety_override_blocks_execution_when_unsafe() -> None:
    # Even if route chose execute_and_save with high conf, unsafe noul (0.15, conf 0.90) overrides.
    client_refuse = _StubClient(
        _make_sdk_response(
            safe_prob=0.15,
            safe_conf=0.90,
            route="execute_and_save",
            route_conf=0.82,
            route_probabilities={
                "execute_and_save": 0.82,
                "adapt_upper_body_only": 0.02,
                "repair_in_forge": 0.04,
                "refuse_impossible": 0.10,
                "other": 0.02,
            },
        )
    )
    res_refuse = evaluate_pre_execution(
        client_refuse, _make_sample_skill(), None, [b"jpg1"], _make_robot_status()
    )
    assert res_refuse.route == "execute_and_save"
    assert res_refuse.auto_branch_allowed is True
    assert res_refuse.effective_action == "refuse_impossible"

    # When repair_in_forge has higher runner-up probability, downgrades to repair_in_forge.
    client_repair = _StubClient(
        _make_sdk_response(
            safe_prob=0.15,
            safe_conf=0.90,
            route="execute_and_save",
            route_conf=0.80,
            route_probabilities={
                "execute_and_save": 0.80,
                "adapt_upper_body_only": 0.02,
                "repair_in_forge": 0.14,
                "refuse_impossible": 0.03,
                "other": 0.01,
            },
        )
    )
    res_repair = evaluate_pre_execution(
        client_repair, _make_sample_skill(), None, [b"jpg1"], _make_robot_status()
    )
    assert res_repair.effective_action == "repair_in_forge"


def test_parse_missing_answer_raises_value_error() -> None:
    with pytest.raises(ValueError, match="missing required answer"):
        parse_decision_response({"answers": {"is_physically_safe": {"noul": 0.9, "confidence": 0.9}}})


# ---------------------------------------------------------------------------
# Post-run and baseline evaluation tests
# ---------------------------------------------------------------------------


def test_evaluate_post_run_with_and_without_post_run_jpeg() -> None:
    skill = _make_sample_skill()
    final_status = _make_robot_status(torso_pitch_deg=3.2, torso_roll_deg=1.1)
    ref_frames = [f"ref-{i}".encode() for i in range(8)]
    post_frame = b"post-run-camera-frame"

    client_with = _StubClient(
        _make_sdk_response(
            safe_prob=0.95,
            safe_conf=0.95,
            fidelity_score=2.8,
            fidelity_conf=0.81,
            route="execute_and_save",
            route_conf=0.93,
        )
    )
    res_with = evaluate_post_run(client_with, skill, ref_frames, post_frame, final_status)
    assert res_with.move_fidelity_score == pytest.approx(2.8)
    assert res_with.effective_action == "execute_and_save"

    call_with = client_with.decisions.calls[0]
    assert call_with["model"] == "gemini-decision-1"
    assert call_with["questions"] is POST_RUN_QUESTIONS
    img_blocks = [b for b in call_with["input"] if b["type"] == "image"]
    assert len(img_blocks) == 6
    assert all(b["resolution"] == "low" for b in img_blocks)
    # Post-run frame is guaranteed to be preserved as the final image block
    assert base64.b64decode(img_blocks[-1]["data"]) == post_frame
    text_payload = json.loads(call_with["input"][-1]["text"])
    assert text_payload["extra_context"]["evaluation_stage"] == "post_run"
    assert text_payload["extra_context"]["has_post_run_frame"] is True
    assert text_payload["extra_context"]["post_run_frame_index"] == 5

    # Without post_run_jpeg
    client_without = _StubClient(
        _make_sdk_response(
            safe_prob=0.91,
            safe_conf=0.91,
            fidelity_score=2.1,
            fidelity_conf=0.74,
            route="execute_and_save",
            route_conf=0.85,
        )
    )
    res_without = evaluate_post_run(client_without, skill, ref_frames[:2], None, final_status)
    assert res_without.move_fidelity_score == pytest.approx(2.1)
    call_without = client_without.decisions.calls[0]
    img_blocks_without = [b for b in call_without["input"] if b["type"] == "image"]
    assert len(img_blocks_without) == 2
    payload_without = json.loads(call_without["input"][-1]["text"])
    assert payload_without["extra_context"]["has_post_run_frame"] is False


def test_evaluate_baseline_skill_uses_same_rubric() -> None:
    client = _StubClient(
        _make_sdk_response(
            safe_prob=0.22,
            safe_conf=0.88,
            fidelity_score=0.9,
            fidelity_conf=0.76,
            route="repair_in_forge",
            route_conf=0.84,
        )
    )
    res = evaluate_baseline_skill(
        client,
        _make_sample_skill("baseline-jump"),
        "Human performs a 360 jump spin.",
        _make_robot_status(),
    )
    assert res.effective_action == "repair_in_forge"
    assert res.move_fidelity_score == pytest.approx(0.9)

    call = client.decisions.calls[0]
    assert call["model"] == "gemini-decision-1"
    assert call["questions"] is PRE_EXECUTION_QUESTIONS
    payload = json.loads(call["input"][-1]["text"])
    assert payload["extra_context"]["scenario_description"] == "Human performs a 360 jump spin."


# ---------------------------------------------------------------------------
# Skill store tests
# ---------------------------------------------------------------------------


def test_slugify_skill_name_valid_and_rejects_traversal() -> None:
    assert slugify_skill_name("Disco Wave!") == "disco-wave"
    assert slugify_skill_name("  Victory_Dance 2026  ") == "victory_dance-2026"
    assert slugify_skill_name("---Wave---") == "wave"
    assert SKILL_SLUG_RE.match(slugify_skill_name("Robot Pop & Lock #1"))

    for bad in ("../evil", "../../etc/passwd", "/etc/passwd", "sub/dir", "a\\b", "..", "", "   ", "!!!"):
        with pytest.raises(ValueError):
            slugify_skill_name(bad)


def test_skill_store_atomic_save_load_list_delete_and_tamper(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"

    # Auto-signs unsigned skill and preserves DecisionGateResult cleanly through JSON round-trip
    gate = parse_decision_response(_make_sdk_response())
    skill_old = _make_sample_skill(
        "Disco Wave!",
        created_at="2026-11-20T10:00:00+00:00",
        signed=False,
    ).model_copy(update={"decision_gate": gate})
    path_old = save_skill(skill_old, skills_dir)
    assert path_old == skills_dir / "disco-wave.json"
    assert path_old.is_file()
    # No leftover temporary files
    assert list(skills_dir.glob("*.tmp")) == []

    loaded = load_skill("Disco Wave!", skills_dir)
    assert loaded.name == "Disco Wave!"
    assert verify_artifact_signature(loaded)
    assert loaded.decision_gate is not None
    assert loaded.decision_gate.effective_action == "execute_and_save"

    # Saving an already-tampered artifact is rejected before writing
    bad_sig = loaded.model_copy(update={"artifact_sha256": "0" * 64})
    with pytest.raises(ValueError, match="artifact_sha256"):
        save_skill(bad_sig, skills_dir)

    # Save a newer skill and verify list_skills descending order + graceful skipping
    skill_new = _make_sample_skill(
        "victory_bow",
        created_at="2026-11-21T15:30:00+00:00",
        signed=True,
    )
    save_skill(skill_new, skills_dir)
    (skills_dir / "README.txt").write_text("ignore me", encoding="utf-8")
    (skills_dir / "corrupt.json").write_text("{not-json", encoding="utf-8")

    listed = list_skills(skills_dir)
    assert [s.name for s in listed] == ["victory_bow", "Disco Wave!"]

    # Tamper with saved JSON on disk -> load_skill detects SHA-256 mismatch
    raw = json.loads(path_old.read_text(encoding="utf-8"))
    raw["keyframes"][0]["servos"]["r_shoulder_pitch"] = 625
    path_old.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="signature verification failed"):
        load_skill("disco-wave", skills_dir)

    # Disabling verify_signature allows forensic inspection
    tampered_loaded = load_skill("disco-wave", skills_dir, verify_signature=False)
    assert tampered_loaded.keyframes[0].servos["r_shoulder_pitch"] == 625

    # Tampered file is excluded from list_skills
    assert [s.name for s in list_skills(skills_dir)] == ["victory_bow"]

    # Delete behavior & missing skill FileNotFoundError
    assert delete_skill("victory_bow", skills_dir) is True
    assert delete_skill("victory_bow", skills_dir) is False
    with pytest.raises(FileNotFoundError):
        load_skill("victory_bow", skills_dir)
    with pytest.raises(ValueError):
        delete_skill("../evil", skills_dir)
    assert list_skills(tmp_path / "does_not_exist") == []
