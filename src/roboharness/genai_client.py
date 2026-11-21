"""Compatibility re-exports for Gemini 4 Dual Clients and media helpers.

Prefer importing directly from :mod:`roboharness.models`.
"""

from __future__ import annotations

from google import genai

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

# Legacy alias retained for callers expecting the original v1beta factory name.
make_client = make_beta_client

__all__ = [
    "DualGenAIClients",
    "build_gemini4_generation_config",
    "compact_json_schema_for_gemini",
    "extract_interaction_text",
    "extract_json_from_text",
    "genai",
    "make_beta_client",
    "make_client",
    "make_dual_clients",
    "make_v1_client",
    "prepare_video_input",
    "upload_video_agentic",
]
