"""Offline unit tests for genai_client, motion_extractor, and spatial framing."""

from __future__ import annotations

import base64
import json
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from roboharness import genai_client
from roboharness.config import (
    API_VERSION,
    FORBIDDEN_GEMINI4_PARAMS,
    MOTION_MODEL,
    SPATIAL_MODEL,
    VALID_THINKING_LEVELS,
)
from roboharness.genai_client import (
    build_gemini4_generation_config,
    extract_interaction_text,
    extract_json_from_text,
    make_client,
    prepare_video_input,
)
from roboharness.motion_extractor import (
    MAX_ANNOUNCEMENT_WORDS,
    BaselineKeyframesPayload,
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

FORBIDDEN_KEYS = FORBIDDEN_GEMINI4_PARAMS | {"max_output_tokens", "maxOutputTokens"}


class _FakeFileState(Enum):
    PROCESSING = "PROCESSING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"


class _FakeFilesAPI:
    def __init__(self, states: list[Any], *, uri: str = "https://files.example/v1beta/files/vid1", mime_type: str | None = "video/mp4") -> None:
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
        self.files = _FakeFilesAPI(file_states or [_FakeFileState.ACTIVE], mime_type=file_mime_type)


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


def _assert_gemini4_invariants(call: dict[str, Any], *, expected_model: str, expected_thinking: str) -> None:
    assert call["model"] == expected_model
    gen_cfg = call["generation_config"]
    assert gen_cfg["thinking_level"] == expected_thinking
    assert gen_cfg["thinking_level"] in VALID_THINKING_LEVELS
    assert gen_cfg["thinking_level"] != "minimal"

    for forbidden in FORBIDDEN_KEYS:
        assert forbidden not in gen_cfg, f"Forbidden key {forbidden} found in generation_config"
        assert forbidden not in call, f"Forbidden key {forbidden} found in interaction kwargs"


# ---------------------------------------------------------------------------
# genai_client.py tests
# ---------------------------------------------------------------------------


def test_make_client_locks_v1beta_and_resolves_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, Any]] = []

    class _DummyClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(genai_client.genai, "Client", _DummyClient)

    # Explicit API key wins over env vars.
    monkeypatch.setenv("GEMINI_API_KEY", "from-gemini-env")
    monkeypatch.setenv("GOOGLE_API_KEY", "from-google-env")
    make_client("explicit-key")
    assert captured[-1] == {
        "api_key": "explicit-key",
        "http_options": {"api_version": API_VERSION},
    }

    # Falls back to GEMINI_API_KEY.
    make_client()
    assert captured[-1] == {
        "api_key": "from-gemini-env",
        "http_options": {"api_version": "v1beta"},
    }

    # Falls back to GOOGLE_API_KEY if GEMINI_API_KEY absent.
    monkeypatch.delenv("GEMINI_API_KEY")
    make_client()
    assert captured[-1] == {
        "api_key": "from-google-env",
        "http_options": {"api_version": "v1beta"},
    }


@pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh"])
def test_build_gemini4_generation_config_accepts_valid_levels(level: str) -> None:
    cfg = build_gemini4_generation_config(level)
    assert cfg == {"thinking_level": level}
    assert "max_output_tokens" not in cfg


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
        {"temperature": None},
    ],
)
def test_build_gemini4_generation_config_rejects_forbidden_params(
    forbidden_kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="Forbidden Gemini 4"):
        build_gemini4_generation_config("medium", **forbidden_kwargs)


def test_prepare_video_input_inline_for_small_file(tmp_path: Path) -> None:
    clip = tmp_path / "wave.mp4"
    payload = b"\x00\x00\x00\x18ftypmp42fakevideo"
    clip.write_bytes(payload)

    client = _FakeGenAIClient()
    part = prepare_video_input(client, clip, inline_max_bytes=1024)

    assert part == {
        "type": "video",
        "data": base64.b64encode(payload).decode("ascii"),
        "mime_type": "video/mp4",
    }
    assert client.files.uploads == []


def test_prepare_video_input_mov_mime_normalization(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mov"
    clip.write_bytes(b"fake-mov-data")
    part = prepare_video_input(_FakeGenAIClient(), clip, inline_max_bytes=1024)
    assert part["mime_type"] == "video/mov"


def test_prepare_video_input_files_api_and_polling_enum_states(tmp_path: Path) -> None:
    clip = tmp_path / "large.mp4"
    clip.write_bytes(b"x" * 2048)

    client = _FakeGenAIClient(
        file_states=[
            _FakeFileState.PROCESSING,
            _FakeFileState.PROCESSING,
            _FakeFileState.ACTIVE,
        ],
        file_mime_type=None,
    )
    part = prepare_video_input(
        client,
        clip,
        inline_max_bytes=512,
        poll_interval_s=0.0,
        timeout_s=5.0,
    )

    assert part == {
        "type": "video",
        "uri": "https://files.example/v1beta/files/vid1",
        "mime_type": "video/mp4",
    }
    assert client.files.uploads == [str(clip)]
    assert client.files.gets == ["files/vid1", "files/vid1"]


def test_prepare_video_input_files_api_string_states_and_failure_modes(tmp_path: Path) -> None:
    clip = tmp_path / "large.mp4"
    clip.write_bytes(b"x" * 1024)

    # String states: PROCESSING -> ACTIVE
    ok_client = _FakeGenAIClient(file_states=["PROCESSING", "ACTIVE"])
    part = prepare_video_input(ok_client, clip, inline_max_bytes=100, poll_interval_s=0.0)
    assert part["uri"] == "https://files.example/v1beta/files/vid1"

    # FAILED state raises RuntimeError
    fail_client = _FakeGenAIClient(file_states=["PROCESSING", "FAILED"])
    with pytest.raises(RuntimeError, match="failed processing"):
        prepare_video_input(fail_client, clip, inline_max_bytes=100, poll_interval_s=0.0)

    # Timeout raises TimeoutError
    stuck_client = _FakeGenAIClient(file_states=["PROCESSING"] * 100)
    with pytest.raises(TimeoutError, match="Timed out"):
        prepare_video_input(
            stuck_client,
            clip,
            inline_max_bytes=100,
            poll_interval_s=0.0,
            timeout_s=0.0,
        )

    # Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        prepare_video_input(ok_client, tmp_path / "does_not_exist.mp4")


def test_extract_json_from_text_and_interaction_text() -> None:
    assert json.loads(extract_json_from_text('  {"ok": true}  ')) == {"ok": True}
    assert json.loads(extract_json_from_text('```json\n{"ok": 1}\n```')) == {"ok": 1}
    assert json.loads(extract_json_from_text('Here is the result:\n```\n[1, 2]\n```\nThanks!')) == [1, 2]
    assert json.loads(extract_json_from_text('Prefix text {"nested": {"a": 2}} trailing')) == {
        "nested": {"a": 2}
    }

    with pytest.raises(ValueError):
        extract_json_from_text("   ")

    # Supports google-genai Interaction.steps structure
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


def test_extract_motion_from_video_calls_hydrogen_with_invariants(tmp_path: Path) -> None:
    clip = tmp_path / "dance.mp4"
    clip.write_bytes(b"small-mp4-bytes")

    fenced_payload = f"```json\n{json.dumps(_sample_pose_sequence_dict())}\n```"
    client = _FakeGenAIClient(responses=[SimpleNamespace(output_text=fenced_payload)])

    seq = extract_motion_from_video(client, clip, skill_hint="disco wave")

    assert isinstance(seq, PoseSequence)
    assert len(seq.keyframes) == 2
    assert seq.impossible_elements[0].element == "jump"

    assert len(client.interactions.calls) == 1
    call = client.interactions.calls[0]
    _assert_gemini4_invariants(call, expected_model=MOTION_MODEL, expected_thinking="medium")
    assert call["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": PoseSequence.model_json_schema(),
    }
    assert call["input"][0]["type"] == "video"
    assert "data" in call["input"][0]
    assert call["input"][1]["type"] == "text"
    assert "disco wave" in call["input"][1]["text"]
    assert "impossible_elements" in call["input"][1]["text"]


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
        "schema": PoseSequence.model_json_schema(),
    }
    assert len(call["input"]) == 1
    assert "Robot disco arm wave with a hop" in call["input"][0]["text"]

    with pytest.raises(ValueError, match="non-empty"):
        extract_motion_from_description(client, "   ")


def test_generate_baseline_raw_keyframes_object_and_video_modes(tmp_path: Path) -> None:
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
        ]
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
        "schema": BaselineKeyframesPayload.model_json_schema(),
    }
    assert call0["input"][0]["type"] == "video"
    assert call0["input"][1]["type"] == "text"

    # Bare list response without video path also supported
    frames_text_only = generate_baseline_raw_keyframes(client, "Wave hello")
    assert len(frames_text_only) == 2
    call1 = client.interactions.calls[1]
    _assert_gemini4_invariants(call1, expected_model=MOTION_MODEL, expected_thinking="medium")
    assert len(call1["input"]) == 1


def test_run_live_reflex_announcement_with_and_without_camera_frame() -> None:
    status = RobotStatus(battery_v=11.5, max_servo_temp_c=40.2, torso_pitch_deg=1.2, torso_roll_deg=-0.5)
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
    _assert_gemini4_invariants(call0, expected_model=MOTION_MODEL, expected_thinking="low")
    assert call0["input"][0] == {
        "type": "image",
        "data": base64.b64encode(jpeg_bytes).decode("ascii"),
        "mime_type": "image/jpeg",
    }
    assert '"battery_v": 11.5' in call0["input"][1]["text"]

    # Without camera frame, truncates verbose replies to <= 18 words cleanly
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
        "schema": SpatialFramingResult.model_json_schema(),
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


def test_sdk_normalize_create_body_accepts_all_request_payloads(tmp_path: Path) -> None:
    """Verify every kwargs dict emitted by our modules is accepted by google-genai 3.6.1."""
    from google import genai as real_genai

    real_client = real_genai.Client(api_key="dummy-key", http_options={"api_version": "v1beta"})
    normalize_fn = real_client.interactions.create.__globals__["_normalize_create_body"]

    clip = tmp_path / "mini.mp4"
    clip.write_bytes(b"mini-mp4")
    jpeg = b"\xff\xd8\xff\xe0mini-jpeg\xff\xd9"
    status = RobotStatus(battery_v=11.4, max_servo_temp_c=40.0)

    client = _FakeGenAIClient(
        responses=[
            SimpleNamespace(output_text=json.dumps(_sample_pose_sequence_dict())),
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
        ]
    )

    extract_motion_from_video(client, clip)
    extract_motion_from_description(client, "Wave both arms")
    generate_baseline_raw_keyframes(client, "Wave both arms")
    run_live_reflex_announcement(client, "wave", "Wave arms", status, frame_jpeg=jpeg)
    check_workspace_framing(client, jpeg)

    assert len(client.interactions.calls) == 5
    for call_kwargs in client.interactions.calls:
        normalized = normalize_fn(call_kwargs)
        assert normalized["model"] in {MOTION_MODEL, SPATIAL_MODEL}
        assert isinstance(normalized["input"], list)
