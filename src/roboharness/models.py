"""Dual Gemini 4 SDK clients (stable v1 + v1beta), agentic video upload, and strict guards.

Official Gemini 4 Hackathon Builder Guide split:
- ``make_v1_client()``: stable ``v1`` default for GA Gemini 4 Interactions API calls
  (``gemini-4-hydrogen``, ``gemini-4-neon``, ``gemini-4-carbon``).
- ``make_beta_client()``: explicit ``v1beta`` client for Managed Agents
  (``antigravity-preview-09-2026``), Environments, Decisions API
  (``gemini-decision-1``), and preview robotics model
  (``gemini-robotics-er-2-preview``).
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from google import genai
from google.genai import interactions as genai_interactions
from pydantic import BaseModel

from roboharness.config import (
    API_VERSION,
    FORBIDDEN_GEMINI4_PARAMS,
    VALID_THINKING_LEVELS,
)

# Gemini 4 forbids sampling/caching overrides and leaves output/total token caps unset.
_FORBIDDEN_GENERATION_KEYS: Final[frozenset[str]] = FORBIDDEN_GEMINI4_PARAMS | frozenset(
    {
        "max_output_tokens",
        "maxOutputTokens",
        "max_total_tokens",
        "maxTotalTokens",
    }
)

# Constrained-decoding keywords that blow up Gemini's serving-time JSON-schema FSM
# (numeric range automata, unrolled min/maxItems loops, and verbose metadata).
# Strict validation of all ranges/lengths still runs locally via Pydantic model_validate_json().
_COMPACT_SCHEMA_STRIP_KEYS: Final[frozenset[str]] = frozenset(
    {
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


@dataclass(frozen=True)
class DualGenAIClients:
    """Paired Google GenAI clients separating stable ``v1`` and ``v1beta`` surfaces."""

    client: Any
    beta_client: Any


def _resolve_api_key(api_key: str | None) -> str | None:
    if api_key is not None and api_key != "":
        return api_key
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def make_v1_client(api_key: str | None = None) -> genai.Client:
    """Create a stable ``v1`` ``google-genai`` client for standard Interactions API calls."""
    resolved_key = _resolve_api_key(api_key)
    return genai.Client(api_key=resolved_key)


def make_beta_client(api_key: str | None = None) -> genai.Client:
    """Create a ``v1beta`` client for Managed Agents, Environments, Decisions & preview models."""
    resolved_key = _resolve_api_key(api_key)
    return genai.Client(
        api_key=resolved_key,
        http_options={"api_version": API_VERSION},
    )


def make_dual_clients(api_key: str | None = None) -> DualGenAIClients:
    """Instantiate paired stable ``v1`` and ``v1beta`` Google GenAI clients."""
    return DualGenAIClients(
        client=make_v1_client(api_key),
        beta_client=make_beta_client(api_key),
    )


def build_gemini4_generation_config(
    thinking_level: str = "medium",
    **extra: Any,
) -> dict[str, Any]:
    """Build a Gemini 4 ``generation_config`` dict while enforcing SDK invariants.

    Validates that ``thinking_level`` is one of ``low``, ``medium``, ``high``, or
    ``xhigh`` (never ``minimal``) and strictly rejects forbidden Gemini 4 parameters
    including ``temperature``, ``top_p``, ``top_k``, ``cachedContents``,
    ``max_output_tokens``, and ``max_total_tokens``.
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


def _strip_heavy_schema_keywords(node: Any) -> Any:
    if isinstance(node, dict):
        cleaned: dict[str, Any] = {}
        for key, value in node.items():
            if key in _COMPACT_SCHEMA_STRIP_KEYS:
                continue
            cleaned[key] = _strip_heavy_schema_keywords(value)
        return cleaned
    if isinstance(node, list):
        return [_strip_heavy_schema_keywords(item) for item in node]
    return node


def compact_json_schema_for_gemini(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Return a Gemini-serving-safe JSON Schema stripped of heavy FSM constraint keywords.

    Gemini's structured-output compiler builds finite-state machines for numeric
    ``minimum``/``maximum`` bounds and ``minItems``/``maxItems`` array limits, which
    can exceed serving state limits on multi-DOF schemas like :class:`PoseSequence`.
    Stripping those keywords keeps JSON structure, property names, enums, and required
    keys strictly enforced on the wire while local Pydantic ``model_validate_json()``
    enforces every numeric and length invariant afterward.
    """
    raw_schema = model_cls.model_json_schema()
    compacted = _strip_heavy_schema_keywords(raw_schema)
    return compacted if isinstance(compacted, dict) else {}


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
    raw = str(state.name) if hasattr(state, "name") else str(state)
    normalized = raw.strip().upper()
    if "." in normalized:
        normalized = normalized.rsplit(".", maxsplit=1)[-1]
    return normalized


def upload_video_agentic(
    client: Any,
    video_path: Path,
    *,
    poll_interval_s: float = 2.0,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Upload a video via the Files API and emit an Agentic Video Processing content block.

    Always uploads via ``client.files.upload(file=str(video_path))``, polls
    ``client.files.get(name=...)`` until state is ``ACTIVE``, validates the resulting
    dict against ``google.genai.interactions.VideoContent``, and returns::

        {"type": "video", "uri": clip.uri, "mime_type": mime, "processing": "agentic"}
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    mime_type = _guess_video_mime_type(path)
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
    video_part: dict[str, Any] = {
        "type": "video",
        "uri": str(uri),
        "mime_type": str(resolved_mime),
        "processing": "agentic",
    }
    # Validate shape against google-genai VideoContent so schema drift fails immediately.
    genai_interactions.VideoContent.model_validate(video_part)
    return video_part


def prepare_video_input(
    client: Any,
    video_path: Path,
    *,
    inline_max_bytes: int | None = None,
    poll_interval_s: float = 2.0,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Prepare a video file as an Agentic Video Processing Interactions content part.

    Delegates directly to :func:`upload_video_agentic` so every video call uses
    Files API upload + ``processing="agentic"``.
    """
    _ = inline_max_bytes
    return upload_video_agentic(
        client,
        video_path,
        poll_interval_s=poll_interval_s,
        timeout_s=timeout_s,
    )


def extract_json_from_text(text: str) -> str:
    """Strip optional markdown code fences or surrounding prose from a JSON payload."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty response text; expected JSON payload")

    fence_match = _FENCED_BLOCK_RE.search(stripped)
    candidate = fence_match.group(1).strip() if fence_match else stripped

    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        pass

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
