"""Gemini 4 v1beta client factory, media helpers, and strict parameter guards."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Final

from google import genai

from roboharness.config import (
    API_VERSION,
    FORBIDDEN_GEMINI4_PARAMS,
    VALID_THINKING_LEVELS,
)

DEFAULT_INLINE_VIDEO_MAX_BYTES: Final[int] = 15 * 1024 * 1024

# Gemini 4 forbids sampling/caching overrides and leaves max_output_tokens unset (65536 default).
_FORBIDDEN_GENERATION_KEYS: Final[frozenset[str]] = FORBIDDEN_GEMINI4_PARAMS | frozenset(
    {"max_output_tokens", "maxOutputTokens"}
)

_VIDEO_MIME_BY_SUFFIX: Final[dict[str, str]] = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/mov",
    ".webm": "video/webm",
    ".avi": "video/avi",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpg",
    ".flv": "video/x-flv",
    ".wmv": "video/wmv",
    ".3gp": "video/3gpp",
    ".3gpp": "video/3gpp",
}

_FENCED_BLOCK_RE: Final[re.Pattern[str]] = re.compile(
    r"```(?:json|JSON)?\s*\n?(.*?)```",
    flags=re.DOTALL,
)


def make_client(api_key: str | None = None) -> genai.Client:
    """Create a ``google-genai`` client locked to the ``v1beta`` API version."""
    resolved_key = (
        api_key
        if api_key is not None and api_key != ""
        else (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    )
    return genai.Client(
        api_key=resolved_key,
        http_options={"api_version": API_VERSION},
    )


def build_gemini4_generation_config(
    thinking_level: str = "medium",
    **extra: Any,
) -> dict[str, Any]:
    """Build a Gemini 4 ``generation_config`` dict while enforcing SDK invariants.

    Validates that ``thinking_level`` is one of ``low``, ``medium``, ``high``, or
    ``xhigh`` (never ``minimal``) and rejects forbidden Gemini 4 parameters such as
    ``temperature``, ``top_p``, ``top_k``, ``cachedContents``, and ``max_output_tokens``.
    """
    if thinking_level not in VALID_THINKING_LEVELS:
        valid_str = ", ".join(sorted(VALID_THINKING_LEVELS))
        raise ValueError(
            f"Invalid thinking_level {thinking_level!r}; allowed values: {valid_str}"
        )

    forbidden_found = sorted(key for key in extra if key in _FORBIDDEN_GENERATION_KEYS)
    if forbidden_found:
        raise ValueError(
            "Forbidden Gemini 4 generation_config parameter(s): "
            + ", ".join(forbidden_found)
        )

    config: dict[str, Any] = {"thinking_level": thinking_level}
    for key, value in extra.items():
        if value is not None:
            config[key] = value
    return config


def _guess_video_mime_type(video_path: Path) -> str:
    suffix_mime = _VIDEO_MIME_BY_SUFFIX.get(video_path.suffix.lower())
    if suffix_mime is not None:
        return suffix_mime
    guessed, _ = mimetypes.guess_type(str(video_path))
    if guessed and guessed.startswith("video/"):
        return guessed
    return "video/mp4"


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_file_state(file_obj: Any) -> str:
    state = _get_field(file_obj, "state")
    if state is None:
        return ""
    if hasattr(state, "name"):
        raw = str(state.name)
    else:
        raw = str(state)
    normalized = raw.strip().upper()
    if "." in normalized:
        normalized = normalized.rsplit(".", maxsplit=1)[-1]
    return normalized


def prepare_video_input(
    client: Any,
    video_path: Path,
    *,
    inline_max_bytes: int = DEFAULT_INLINE_VIDEO_MAX_BYTES,
    poll_interval_s: float = 2.0,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Prepare a video file as an Interactions API content part.

    Small clips (``<= inline_max_bytes``) are encoded inline as base64 bytes for low
    latency. Larger clips are uploaded through the Files API and polled until their
    processing state becomes ``ACTIVE``.
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    mime_type = _guess_video_mime_type(path)
    size_bytes = path.stat().st_size

    if size_bytes <= inline_max_bytes:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {
            "type": "video",
            "data": encoded,
            "mime_type": mime_type,
        }

    current = client.files.upload(file=str(path))
    file_name = _get_field(current, "name")
    deadline = time.monotonic() + timeout_s

    while True:
        state = _extract_file_state(current)
        if state == "ACTIVE":
            break
        if state == "FAILED":
            err = _get_field(current, "error")
            raise RuntimeError(
                f"Video upload {file_name or path.name} failed processing: {err}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out after {timeout_s:.1f}s waiting for video file "
                f"{file_name or path.name} to become ACTIVE (last state: {state or 'UNKNOWN'})"
            )
        if not file_name:
            raise RuntimeError(
                f"Uploaded video object missing 'name' for polling: {current!r}"
            )
        if poll_interval_s > 0:
            time.sleep(min(poll_interval_s, remaining))
        current = client.files.get(name=file_name)

    uri = _get_field(current, "uri")
    if not uri:
        raise RuntimeError(
            f"Active video file {file_name or path.name} missing 'uri': {current!r}"
        )

    resolved_mime = _get_field(current, "mime_type") or mime_type or "video/mp4"
    return {
        "type": "video",
        "uri": str(uri),
        "mime_type": str(resolved_mime),
    }


def extract_json_from_text(text: str) -> str:
    """Strip optional markdown code fences or prose around a JSON payload."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty response text; expected JSON payload")

    # 1. Extract from fenced markdown code block if present.
    fence_match = _FENCED_BLOCK_RE.search(stripped)
    candidate = fence_match.group(1).strip() if fence_match else stripped

    # 2. Fast path: candidate is already valid JSON.
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        pass

    # 3. Fallback: scan for the first valid JSON object or array inside surrounding prose.
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(candidate):
        if ch in "{[":
            try:
                _, end_idx = decoder.raw_decode(candidate, idx)
                return candidate[idx:end_idx].strip()
            except json.JSONDecodeError:
                continue

    return candidate


def extract_interaction_text(interaction: Any) -> str:
    """Extract concatenated model text from a ``google-genai`` Interaction or stub."""
    output_text = _get_field(interaction, "output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    # Standard google-genai >= 3.2 Interaction.steps shape.
    steps = _get_field(interaction, "steps")
    if isinstance(steps, list):
        parts: list[str] = []
        for step in steps:
            if _get_field(step, "type") != "model_output":
                continue
            content = _get_field(step, "content")
            if not isinstance(content, list):
                continue
            for item in content:
                if _get_field(item, "type") == "text":
                    chunk = _get_field(item, "text")
                    if isinstance(chunk, str):
                        parts.append(chunk)
        joined = "".join(parts)
        if joined.strip():
            return joined

    # Legacy or lightweight stub Interaction.outputs shape.
    outputs = _get_field(interaction, "outputs")
    if isinstance(outputs, list):
        parts = []
        for item in outputs:
            item_type = _get_field(item, "type")
            if item_type is None or item_type == "text":
                chunk = _get_field(item, "text")
                if isinstance(chunk, str):
                    parts.append(chunk)
        joined = "".join(parts)
        if joined.strip():
            return joined

    # Minimal stub shape exposing .text directly.
    direct_text = _get_field(interaction, "text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text

    status = _get_field(interaction, "status")
    errors = _get_field(interaction, "errors")
    if status == "failed" or errors:
        raise RuntimeError(
            f"Gemini interaction failed (status={status!r}, errors={errors!r})"
        )
    raise ValueError("Gemini interaction response contained no text output")
