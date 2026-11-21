"""Live tabletop spatial workspace framing check via gemini-robotics-er-2-preview (v1beta beta_client)."""

from __future__ import annotations

import base64
from typing import Any, Final

from roboharness.config import SPATIAL_MODEL
from roboharness.models import (
    build_gemini4_generation_config,
    compact_json_schema_for_gemini,
    extract_interaction_text,
    extract_json_from_text,
)
from roboharness.schemas import SpatialFramingResult

_SPATIAL_FRAMING_PROMPT: Final[str] = (
    "Inspect this live camera frame of a tabletop Hiwonder TonyPi humanoid robot workspace.\n"
    "Output ONLY valid JSON matching the SpatialFramingResult schema:\n"
    "- `centered_on_table`: true if the robot is well inside the table boundary away from drop-off edges.\n"
    "- `workspace_clear`: true if the immediate motion envelope around the robot's arms and legs "
    "is free of obstacles, cups, cables, laptops, or hands.\n"
    "- `robot_bbox_norm`: [ymin, xmin, ymax, xmax] normalized to 0..1000 integers, or null if not visible.\n"
    "- `clearance_margin`: 'safe' (plenty of room), 'tight' (near an obstacle or edge, reduce motion), "
    "or 'edge_risk' (dangerously close to table drop-off or collision).\n"
    "- `summary`: concise one-sentence spatial assessment."
)


def check_workspace_framing(
    client: Any,
    frame_jpeg: bytes,
) -> SpatialFramingResult:
    """Assess robot centering, table edge risk, and workspace clearance on a camera frame."""
    if not frame_jpeg:
        raise ValueError("frame_jpeg must be non-empty JPEG bytes")

    image_part = {
        "type": "image",
        "data": base64.b64encode(frame_jpeg).decode("ascii"),
        "mime_type": "image/jpeg",
    }

    interaction = client.interactions.create(
        model=SPATIAL_MODEL,
        input=[image_part, {"type": "text", "text": _SPATIAL_FRAMING_PROMPT}],
        generation_config=build_gemini4_generation_config("medium"),
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": compact_json_schema_for_gemini(SpatialFramingResult),
        },
    )
    raw_text = extract_interaction_text(interaction)
    return SpatialFramingResult.model_validate_json(extract_json_from_text(raw_text))
