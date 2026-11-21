"""Offline unit tests for models (Dual Clients + agentic video), motion_extractor, and spatial."""

from __future__ import annotations

import base64
import json
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import interactions as genai_interactions

from roboharness import genai_client, models
from roboharness.config import (
    API_VERSION,
    FORBIDDEN_GEMINI4_PARAMS,
    MOTION_MODEL,
    SPATIAL_MODEL,
    VALID_THINKING_LEVELS,
)
from roboharness.models import (
    DualGenAIClients,
    build_gemini4_generation_config,
    compact_json_schema_for_gemini,
    extract_interaction_text,
    extract_json_from_text,
    make_beta_client,
    make_dual_clients,
    make_v1_client,
    prepare_video_input,
    upload_video_agentic,
)
from roboharness.motion_extractor import (
    MAX_ANNOUNCEMENT_WORDS,
    BaselineKeyframesPayload,
    TrialVideoDebugReport,
    debug_trial_video,
    extract_motion_from_description,
    extract_motion_from_video,
    generate_baseline_raw_keyframes,
    run_live_reflex_announcement,
)
from roboharness.schemas import (
    NEUTRAL_POSE,
    PoseSequence,
    RobotStatus,
    ServoKeyframe,
    SpatialFramingResult,
)
from roboharness.spatial import check_workspace_framing

FORBIDDEN_KEYS = FORBIDDEN_GEMINI4_PARAMS | {
    "max_output_tokens",
    "maxOutputTokens",
    "max_total_tokens",
    "maxTotalTokens",
}


class _FakeFileState(Enum):
    PROCESSING = "PROCESSING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"


class _FakeFilesAPI:
    def __init__(
        self,
        states: list[Any],
        *,
        uri: str = "https://files.example/v1/files/vid1",
        mime_type: str | None = "video/mp4",
    ) -> None:
        self._states = list(states)
        self._uri = uri
        self._mime_type = mime_type
        self.uploads: list[str] = []
        self.gets: list[str] = []

    def upload(self, *, file: str) -> SimpleNamespace:
        self.uploads.append(file)
        state = self._states.pop(0) if self._states else _FakeFileState.ACTIVE
        return SimpleNamespace(
            name="files/vid1",
            state=state,
            uri=self._uri,
            mime_type=self._mime_type,
            error=None if str(state) != "FAILED" else "codec error",
        )

    def get(self, *, name: str) -> SimpleNamespace:
        self.gets.append(name)
        state = self._states.pop(0) if self._states else _FakeFileState.ACTIVE
        return SimpleNamespace(
            name=name,
            state=state,
            uri=self._uri,
            mime_type=self._mime_type,
            error="decode failure" if state in (_FakeFileState.FAILED, "FAILED") else None,
        )


class _FakeInteractionsAPI:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeGenAIClient:
    def __init__(
        self,
        responses: list[Any] | None = None,
        file_states: list[Any] | None = None,
        *,
        file_mime_type: str | None = "video/mp4",
    ) -> None:
        self.interactions = _FakeInteractionsAPI(responses or [])
        self.files = _FakeFilesAPI(
            file_states or [_FakeFileState.ACTIVE],
            mime_type=file_mime_type,
        )


def _sample_pose_sequence_dict() -> dict[str, Any]:
    return {
        "source_description": "Energetic arm wave dance with a mid-clip jump",
        "bpm": 120.0,
        "duration_ms": 3000,
        "keyframes": [
            {
                "t_ms": 0,
                "beat_index": 0,
                "torso_pitch_deg": 2.0,
                "torso_roll_deg": 0.0,
                "l_shoulder_pitch_deg": 25.0,
                "l_shoulder_roll_deg": 15.0,
                "l_elbow_deg": 40.0,
                "r_shoulder_pitch_deg": -20.0,
                "r_shoulder_roll_deg": -15.0,
                "r_elbow_deg": 35.0,
                "l_hip_pitch_deg": 5.0,
                "l_knee_deg": 10.0,
                "r_hip_pitch_deg": 5.0,
                "r_knee_deg": 10.0,
                "foot_contact": "both",
                "label": "ready_stance",
            },
            {
                "t_ms": 500,
                "beat_index": 1,
                "torso_pitch_deg": 4.0,
                "torso_roll_deg": 6.0,
                "l_shoulder_pitch_deg": 75.0,
                "l_shoulder_roll_deg": 35.0,
                "l_elbow_deg": 60.0,
                "r_shoulder_pitch_deg": 65.0,
                "r_shoulder_roll_deg": -30.0,
                "r_elbow_deg": 55.0,
                "l_hip_pitch_deg": 10.0,
                "l_knee_deg": 25.0,
                "r_hip_pitch_deg": 10.0,
                "r_knee_deg": 25.0,
                "foot_contact": "airborne",
                "label": "hop_arms_up",
            },
        ],
        "impossible_elements": [
            {
                "element": "jump",
                "timestamp_ms": 500,
                "reason": "TonyPi desktop servos cannot launch the robot airborne safely.",
                "suggested_adaptation": "Replace jump with grounded rhythmic knee bounce and full arm raise.",
            }
        ],
        "style_tags": ["dance", "upbeat", "arm_wave"],
    }


def _sample_debug_report_dict(skill_name: str = "dance_orange_justice") -> dict[str, Any]:
    return {
        "skill_name": skill_name,
        "succeeded": False,
        "root_cause": "Right ankle slid laterally on the glossy tabletop during beat 2 arm clap.",
        "slip_detected": True,
        "suspected_joints": ["r_ankle_roll", "r_hip_roll"],
        "observations": [
            {
                "timestamp_ms": 820,
                "limb_or_region": "r_ankle",
                "observation": "Right foot shifted ~1.5 cm outward as torso rolled right.",
                "severity": "failure",
            }
        ],
        "recommended_fixes": [
            "Reduce lateral hip roll excursion by 30% and widen keyframe transition duration."
        ],
    }


def _assert_gemini4_invariants(
    call: dict[str, Any],
    *,
    expected_model: str,
    expected_thinking: str,
) -> None:
    assert call["model"] == expected_model
    gen_cfg = call["generation_config"]
    assert gen_cfg["thinking_level"] == expected_thinking
    assert gen_cfg["thinking_level"] in VALID_THINKING_LEVELS
    assert gen_cfg["thinking_level"] != "minimal"

    for forbidden in FORBIDDEN_KEYS:
        assert forbidden not in gen_cfg, f"Forbidden key {forbidden} found in generation_config"
        assert forbidden not in call, f"Forbidden key {forbidden} found in interaction kwargs"


# ---------------------------------------------------------------------------
# Dual Clients (models.py & genai_client.py) tests
# ---------------------------------------------------------------------------


def test_make_v1_and_beta_clients_enforce_api_version_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    class _DummyClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            captured.append(kwargs)

    monkeypatch.setattr(models.genai, "Client", _DummyClient)
    monkeypatch.setenv("GEMINI_API_KEY", "from-gemini-env")
    monkeypatch.setenv("GOOGLE_API_KEY", "from-google-env")

    # Stable v1 client MUST NOT pass http_options={"api_version": "v1beta"}
    make_v1_client("explicit-v1-key")
    assert captured[-1] == {"api_key": "explicit-v1-key"}
    assert "http_options" not in captured[-1]

    # Falls back to GEMINI_API_KEY
    make_v1_client()
    assert captured[-1] == {"api_key": "from-gemini-env"}

    # Falls back to GOOGLE_API_KEY when GEMINI_API_KEY absent
    monkeypatch.delenv("GEMINI_API_KEY")
    make_v1_client()
    assert captured[-1] == {"api_key": "from-google-env"}

    # Beta client MUST explicitly pass v1beta
    make_beta_client("explicit-beta-key")
    assert captured[-1] == {
        "api_key": "explicit-beta-key",
        "http_options": {"api_version": API_VERSION},
    }

    # Dual clients instantiate paired v1 and v1beta clients
    dual = make_dual_clients("dual-key")
    assert isinstance(dual, DualGenAIClients)
    assert dual.client.kwargs == {"api_key": "dual-key"}
    assert dual.beta_client.kwargs == {
        "api_key": "dual-key",
        "http_options": {"api_version": "v1beta"},
    }

    # Legacy alias genai_client.make_client routes to make_beta_client
    assert genai_client.make_client is make_beta_client
    genai_client.make_client("legacy-key")
    assert captured[-1] == {
        "api_key": "legacy-key",
        "http_options": {"api_version": "v1beta"},
    }


def test_real_sdk_dual_clients_api_versions() -> None:
    """Confirm real google-genai Clients configure v1 (default) and v1beta respectively."""
    dual = make_dual_clients("dummy-test-key")
    assert dual.client._api_client._http_options.api_version == "v1"
    assert dual.beta_client._api_client._http_options.api_version == "v1beta"


@pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh"])
def test_build_gemini4_generation_config_accepts_valid_levels(level: str) -> None:
    cfg = build_gemini4_generation_config(level)
    assert cfg == {"thinking_level": level}
    assert "max_output_tokens" not in cfg
    assert "max_total_tokens" not in cfg


@pytest.mark.parametrize("bad_level", ["minimal", "none", "ultra", ""])
def test_build_gemini4_generation_config_rejects_invalid_levels(bad_level: str) -> None:
    with pytest.raises(ValueError, match="thinking_level"):
        build_gemini4_generation_config(bad_level)


@pytest.mark.parametrize(
    "forbidden_kwargs",
    [
        {"temperature": 0.5},
        {"top_p": 0.9},
        {"top_k": 40},
        {"cachedContents": "cachedContents/abc"},
        {"cached_content": "cachedContents/abc"},
        {"max_output_tokens": 1024},
        {"maxOutputTokens": 1024},
        {"max_total_tokens": 8192},
        {"maxTotalTokens": 8192},
        {"temperature": None},
    ],
)
def test_build_gemini4_generation_config_rejects_forbidden_params(
    forbidden_kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="Forbidden Gemini 4"):
        build_gemini4_generation_config("medium", **forbidden_kwargs)


# ---------------------------------------------------------------------------
# Agentic Video Processing upload tests
# ---------------------------------------------------------------------------


def test_upload_video_agentic_always_uploads_polls_and_sets_agentic(tmp_path: Path) -> None:
    # Even tiny 15-byte clip MUST upload via Files API with processing="agentic" (never inline).
    clip = tmp_path / "wave.mp4"
    clip.write_bytes(b"tiny-mp4-clip!!")

    client = _FakeGenAIClient(
        file_states=[
            _FakeFileState.PROCESSING,
            _FakeFileState.PROCESSING,
            _FakeFileState.ACTIVE,
        ],
        file_mime_type=None,
    )
    part = upload_video_agentic(client, clip, poll_interval_s=0.0, timeout_s=5.0)

    assert part == {
        "type": "video",
        "uri": "https://files.example/v1/files/vid1",
        "mime_type": "video/mp4",
        "processing": "agentic",
    }
    assert "data" not in part
    assert client.files.uploads == [str(clip)]
    assert client.files.gets == ["files/vid1", "files/vid1"]
    genai_interactions.VideoContent.model_validate(part)


def test_upload_video_agentic_mov_mime_and_prepare_video_input_delegate(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mov"
    clip.write_bytes(b"fake-mov-data")

    client = _FakeGenAIClient(file_states=["PROCESSING", "ACTIVE"], file_mime_type=None)
    part = prepare_video_input(client, clip, poll_interval_s=0.0)

    assert part == {
        "type": "video",
        "uri": "https://files.example/v1/files/vid1",
        "mime_type": "video/mov",
        "processing": "agentic",
    }
    assert client.files.uploads == [str(clip)]


def test_upload_video_agentic_failure_modes(tmp_path: Path) -> None:
    clip = tmp_path / "broken.mp4"
    clip.write_bytes(b"x" * 128)

    # FAILED state raises RuntimeError
    fail_client = _FakeGenAIClient(file_states=["PROCESSING", "FAILED"])
    with pytest.raises(RuntimeError, match="failed processing"):
        upload_video_agentic(fail_client, clip, poll_interval_s=0.0)

    # Timeout raises TimeoutError
    stuck_client = _FakeGenAIClient(file_states=["PROCESSING"] * 100)
    with pytest.raises(TimeoutError, match="Timed out"):
        upload_video_agentic(stuck_client, clip, poll_interval_s=0.0, timeout_s=0.0)

    # Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        upload_video_agentic(fail_client, tmp_path / "does_not_exist.mp4")


def test_extract_json_from_text_and_interaction_text() -> None:
    assert json.loads(extract_json_from_text('  {"ok": true}  ')) == {"ok": True}
    assert json.loads(extract_json_from_text('```json\n{"ok": 1}\n```')) == {"ok": 1}
    assert json.loads(extract_json_from_text("Here is the result:\n```\n[1, 2]\n```\nThanks!")) == [
        1,
        2,
    ]
    assert json.loads(extract_json_from_text('Prefix text {"nested": {"a": 2}} trailing')) == {
        "nested": {"a": 2}
    }

    with pytest.raises(ValueError):
        extract_json_from_text("   ")

    steps_resp = SimpleNamespace(
        output_text="",
        steps=[
            SimpleNamespace(
                type="model_output",
                content=[SimpleNamespace(type="text", text='{"hello": "world"}')],
            )
        ],
    )
    assert extract_interaction_text(steps_resp) == '{"hello": "world"}'


# ---------------------------------------------------------------------------
# motion_extractor.py tests
# ---------------------------------------------------------------------------


def test_extract_motion_from_video_uses_agentic_video_and_thinking_levels(tmp_path: Path) -> None:
    clip = tmp_path / "dance.mp4"
    clip.write_bytes(b"small-mp4-bytes")

    fenced_payload = f"```json\n{json.dumps(_sample_pose_sequence_dict())}\n```"
    client = _FakeGenAIClient(
        responses=[
            SimpleNamespace(output_text=fenced_payload),
            SimpleNamespace(output_text=json.dumps(_sample_pose_sequence_dict())),
        ],
        file_states=[_FakeFileState.ACTIVE, _FakeFileState.ACTIVE],
    )

    # Default thinking_level="medium"
    seq = extract_motion_from_video(client, clip, skill_hint="disco wave")
    assert isinstance(seq, PoseSequence)
    assert len(seq.keyframes) == 2
    assert seq.impossible_elements[0].element == "jump"

    call0 = client.interactions.calls[0]
    _assert_gemini4_invariants(call0, expected_model=MOTION_MODEL, expected_thinking="medium")
    assert call0["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": compact_json_schema_for_gemini(PoseSequence),
    }
    assert call0["input"][0] == {
        "type": "video",
        "uri": "https://files.example/v1/files/vid1",
        "mime_type": "video/mp4",
        "processing": "agentic",
    }
    assert "disco wave" in call0["input"][1]["text"]
    assert "impossible_elements" in call0["input"][1]["text"]
    assert client.files.uploads == [str(clip)]

    # Explicit thinking_level="high"
    extract_motion_from_video(client, clip, thinking_level="high")
    call1 = client.interactions.calls[1]
    _assert_gemini4_invariants(call1, expected_model=MOTION_MODEL, expected_thinking="high")
    assert call1["input"][0]["processing"] == "agentic"


def test_debug_trial_video_uses_agentic_video_and_high_thinking(tmp_path: Path) -> None:
    clip = tmp_path / "trial_orange_justice.mp4"
    clip.write_bytes(b"trial-recording-mp4")

    expected_dict = _sample_debug_report_dict("dance_orange_justice")
    client = _FakeGenAIClient(
        responses=[SimpleNamespace(output_text=f"```json\n{json.dumps(expected_dict)}\n```")],
        file_states=[_FakeFileState.ACTIVE],
    )

    report = debug_trial_video(
        client,
        clip,
        skill_name="dance_orange_justice",
        expected_behavior="Grounded rhythmic knee bounce with alternating diagonal arm claps.",
    )

    assert isinstance(report, dict)
    assert report["skill_name"] == "dance_orange_justice"
    assert report["succeeded"] is False
    assert report["slip_detected"] is True
    assert report["suspected_joints"] == ["r_ankle_roll", "r_hip_roll"]
    assert report["observations"][0]["timestamp_ms"] == 820
    assert len(report["recommended_fixes"]) == 1

    call = client.interactions.calls[0]
    _assert_gemini4_invariants(call, expected_model=MOTION_MODEL, expected_thinking="high")
    assert call["input"][0] == {
        "type": "video",
        "uri": "https://files.example/v1/files/vid1",
        "mime_type": "video/mp4",
        "processing": "agentic",
    }
    assert call["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": compact_json_schema_for_gemini(TrialVideoDebugReport),
    }
    assert "dance_orange_justice" in call["input"][1]["text"]
    assert "diagonal arm claps" in call["input"][1]["text"]
    assert client.files.uploads == [str(clip)]


def test_extract_motion_from_description_calls_hydrogen_with_invariants() -> None:
    payload = json.dumps(_sample_pose_sequence_dict())
    client = _FakeGenAIClient(responses=[SimpleNamespace(output_text=payload)])

    seq = extract_motion_from_description(client, "Robot disco arm wave with a hop")

    assert isinstance(seq, PoseSequence)
    assert seq.bpm == 120.0

    call = client.interactions.calls[0]
    _assert_gemini4_invariants(call, expected_model=MOTION_MODEL, expected_thinking="medium")
    assert call["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": compact_json_schema_for_gemini(PoseSequence),
    }
    assert len(call["input"]) == 1
    assert "Robot disco arm wave with a hop" in call["input"][0]["text"]

    with pytest.raises(ValueError, match="non-empty"):
        extract_motion_from_description(client, "   ")


def test_generate_baseline_raw_keyframes_object_and_agentic_video_modes(tmp_path: Path) -> None:
    clip = tmp_path / "ref.mp4"
    clip.write_bytes(b"mp4-content")

    raw_frames = [
        {"duration_ms": 350, "servos": {**NEUTRAL_POSE, "r_shoulder_pitch": 700}, "note": "raise"},
        {"duration_ms": 350, "servos": dict(NEUTRAL_POSE), "note": "reset"},
    ]
    client = _FakeGenAIClient(
        responses=[
            SimpleNamespace(output_text=json.dumps({"keyframes": raw_frames})),
            SimpleNamespace(output_text=json.dumps(raw_frames)),
        ],
        file_states=[_FakeFileState.ACTIVE],
    )

    frames_with_video = generate_baseline_raw_keyframes(client, "Wave hello", video_path=clip)
    assert len(frames_with_video) == 2
    assert all(isinstance(kf, ServoKeyframe) for kf in frames_with_video)
    assert frames_with_video[0].servos["r_shoulder_pitch"] == 700

    call0 = client.interactions.calls[0]
    _assert_gemini4_invariants(call0, expected_model=MOTION_MODEL, expected_thinking="medium")
    assert call0["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": compact_json_schema_for_gemini(BaselineKeyframesPayload),
    }
    assert call0["input"][0] == {
        "type": "video",
        "uri": "https://files.example/v1/files/vid1",
        "mime_type": "video/mp4",
        "processing": "agentic",
    }
    assert client.files.uploads == [str(clip)]

    # Bare list response without video path also supported
    frames_text_only = generate_baseline_raw_keyframes(client, "Wave hello")
    assert len(frames_text_only) == 2
    call1 = client.interactions.calls[1]
    _assert_gemini4_invariants(call1, expected_model=MOTION_MODEL, expected_thinking="medium")
    assert len(call1["input"]) == 1


def test_run_live_reflex_announcement_uses_medium_thinking() -> None:
    status = RobotStatus(
        battery_v=11.5,
        max_servo_temp_c=40.2,
        torso_pitch_deg=1.2,
        torso_roll_deg=-0.5,
    )
    jpeg_bytes = b"\xff\xd8\xff\xe0fake-jpeg-bytes\xff\xd9"

    verbose_reply = (
        '"Hello everyone on this wonderful afternoon, I am now going to perform an extra long '
        'disco wave dance move right on top of this table for all of you!"'
    )
    client = _FakeGenAIClient(
        responses=[
            SimpleNamespace(output_text="Watch me nail this verified disco wave!"),
            SimpleNamespace(output_text=verbose_reply),
        ]
    )

    line1 = run_live_reflex_announcement(
        client,
        skill_name="disco_wave",
        description="Upbeat disco arm wave",
        status=status,
        frame_jpeg=jpeg_bytes,
    )
    assert line1 == "Watch me nail this verified disco wave!"
    assert len(line1.split()) <= MAX_ANNOUNCEMENT_WORDS

    call0 = client.interactions.calls[0]
    _assert_gemini4_invariants(call0, expected_model=MOTION_MODEL, expected_thinking="medium")
    assert call0["input"][0] == {
        "type": "image",
        "data": base64.b64encode(jpeg_bytes).decode("ascii"),
        "mime_type": "image/jpeg",
    }
    assert '"battery_v": 11.5' in call0["input"][1]["text"]

    line2 = run_live_reflex_announcement(
        client,
        skill_name="disco_wave",
        description="Upbeat disco arm wave",
        status=status,
        frame_jpeg=None,
    )
    assert len(line2.split()) <= MAX_ANNOUNCEMENT_WORDS
    assert not line2.startswith('"')
    assert len(client.interactions.calls[1]["input"]) == 1


# ---------------------------------------------------------------------------
# spatial.py tests
# ---------------------------------------------------------------------------


def test_check_workspace_framing_calls_robotics_er2_and_parses_result() -> None:
    jpeg_bytes = b"\xff\xd8\xff\xe0table-frame\xff\xd9"
    safe_json = json.dumps(
        {
            "centered_on_table": True,
            "workspace_clear": True,
            "robot_bbox_norm": [180, 320, 820, 680],
            "clearance_margin": "safe",
            "summary": "TonyPi is centered on the table with generous clearance.",
        }
    )
    edge_json = (
        "```json\n"
        + json.dumps(
            {
                "centered_on_table": False,
                "workspace_clear": True,
                "robot_bbox_norm": [200, 740, 850, 970],
                "clearance_margin": "edge_risk",
                "summary": "Robot is dangerously close to the right table drop-off.",
            }
        )
        + "\n```"
    )
    client = _FakeGenAIClient(
        responses=[
            SimpleNamespace(output_text=safe_json),
            SimpleNamespace(output_text=edge_json),
        ]
    )

    res1 = check_workspace_framing(client, jpeg_bytes)
    assert isinstance(res1, SpatialFramingResult)
    assert res1.centered_on_table is True
    assert res1.workspace_clear is True
    assert res1.robot_bbox_norm == [180, 320, 820, 680]
    assert res1.clearance_margin == "safe"

    call0 = client.interactions.calls[0]
    _assert_gemini4_invariants(call0, expected_model=SPATIAL_MODEL, expected_thinking="medium")
    assert call0["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": compact_json_schema_for_gemini(SpatialFramingResult),
    }
    assert call0["input"][0] == {
        "type": "image",
        "data": base64.b64encode(jpeg_bytes).decode("ascii"),
        "mime_type": "image/jpeg",
    }

    res2 = check_workspace_framing(client, jpeg_bytes)
    assert res2.centered_on_table is False
    assert res2.clearance_margin == "edge_risk"

    with pytest.raises(ValueError, match="non-empty"):
        check_workspace_framing(client, b"")


def test_compact_json_schema_for_gemini_strips_heavy_fsm_keywords() -> None:
    """Verify response_format schemas strip FSM-bloating bounds while Pydantic stays strict."""
    stripped_keywords = {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "title",
        "description",
        "default",
        "examples",
    }

    def _collect_keys(node: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(node, dict):
            for k, v in node.items():
                found.add(k)
                found.update(_collect_keys(v))
        elif isinstance(node, list):
            for item in node:
                found.update(_collect_keys(item))
        return found

    for cls in (
        PoseSequence,
        TrialVideoDebugReport,
        BaselineKeyframesPayload,
        SpatialFramingResult,
    ):
        raw_keys = _collect_keys(cls.model_json_schema())
        compact = compact_json_schema_for_gemini(cls)
        compact_keys = _collect_keys(compact)

        assert raw_keys & stripped_keywords, f"Expected raw {cls.__name__} schema to have metadata/bounds"
        assert not (compact_keys & stripped_keywords), (
            f"Heavy FSM keywords remained in compact {cls.__name__} schema: "
            f"{sorted(compact_keys & stripped_keywords)}"
        )
        assert compact["type"] == "object"
        assert "properties" in compact
        assert "required" in compact
        genai_interactions.TextResponseFormat.model_validate(
            {"type": "text", "mime_type": "application/json", "schema": compact}
        )

    # Local Pydantic validation still enforces strict numeric bounds on PoseSequence.
    out_of_range = _sample_pose_sequence_dict()
    out_of_range["keyframes"][0]["torso_pitch_deg"] = 145.0
    with pytest.raises(ValueError, match="less than or equal to 90"):
        PoseSequence.model_validate_json(json.dumps(out_of_range))


def test_sdk_normalize_create_body_accepts_all_request_payloads(tmp_path: Path) -> None:
    """Verify every kwargs dict emitted by our modules is accepted by google-genai 3.6.1."""
    real_v1 = make_v1_client("dummy-key")
    normalize_fn = real_v1.interactions.create.__globals__["_normalize_create_body"]

    clip = tmp_path / "mini.mp4"
    clip.write_bytes(b"mini-mp4")
    jpeg = b"\xff\xd8\xff\xe0mini-jpeg\xff\xd9"
    status = RobotStatus(battery_v=11.4, max_servo_temp_c=40.0)

    client = _FakeGenAIClient(
        responses=[
            SimpleNamespace(output_text=json.dumps(_sample_pose_sequence_dict())),
            SimpleNamespace(output_text=json.dumps(_sample_debug_report_dict("mini"))),
            SimpleNamespace(output_text=json.dumps(_sample_pose_sequence_dict())),
            SimpleNamespace(
                output_text=json.dumps(
                    {"keyframes": [{"duration_ms": 300, "servos": dict(NEUTRAL_POSE)}]}
                )
            ),
            SimpleNamespace(output_text="Ready to perform the verified wave!"),
            SimpleNamespace(
                output_text=json.dumps(
                    {
                        "centered_on_table": True,
                        "workspace_clear": True,
                        "robot_bbox_norm": [150, 300, 850, 700],
                        "clearance_margin": "safe",
                        "summary": "Clear workspace.",
                    }
                )
            ),
        ],
        file_states=[_FakeFileState.ACTIVE, _FakeFileState.ACTIVE],
    )

    extract_motion_from_video(client, clip)
    debug_trial_video(client, clip, skill_name="mini", expected_behavior="Wave smoothly")
    extract_motion_from_description(client, "Wave both arms")
    generate_baseline_raw_keyframes(client, "Wave both arms")
    run_live_reflex_announcement(client, "wave", "Wave arms", status, frame_jpeg=jpeg)
    check_workspace_framing(client, jpeg)

    assert len(client.interactions.calls) == 6
    for call_kwargs in client.interactions.calls:
        normalized = normalize_fn(call_kwargs)
        assert normalized["model"] in {MOTION_MODEL, SPATIAL_MODEL}
        assert isinstance(normalized["input"], list)
        for item in call_kwargs["input"]:
            if item.get("type") == "video":
                assert item["processing"] == "agentic"
                genai_interactions.VideoContent.model_validate(item)
