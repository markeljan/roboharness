"""Central configuration and safety constants for RoboHarness."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# Deterministic hardware safety latches (non-LLM, hard-enforced)
MIN_BATTERY_V: Final[float] = 10.9
MAX_SERVO_TEMP_C: Final[float] = 70.0
MAX_TORSO_TILT_DEG: Final[float] = 30.0
MAX_SLEW_PULSE_PER_SEC: Final[float] = 900.0
MIN_KEYFRAME_DURATION_MS: Final[int] = 80
DECISION_CONFIDENCE_THRESHOLD: Final[float] = 0.70

# Gemini 4 v1beta SDK constants
API_VERSION: Final[str] = "v1beta"
MOTION_MODEL: Final[str] = "gemini-4-hydrogen"
FORGE_BASE_AGENT: Final[str] = "antigravity-preview-09-2026"
FORGE_MODEL: Final[str] = "gemini-4-neon"
FORGE_AGENT_ID: Final[str] = "roboharness-skill-forge"
DECISION_MODEL: Final[str] = "gemini-decision-1"
SPATIAL_MODEL: Final[str] = "gemini-robotics-er-2-preview"
REMOTE_MCP_NAME: Final[str] = "tonypi_mcp"

VALID_THINKING_LEVELS: Final[frozenset[str]] = frozenset({"low", "medium", "high", "xhigh"})
FORBIDDEN_GEMINI4_PARAMS: Final[frozenset[str]] = frozenset(
    {"temperature", "top_p", "top_k", "cachedContents", "cached_content"}
)
RESERVED_AGENT_ID_PREFIXES: Final[tuple[str, ...]] = (
    "gemini-",
    "google-",
    "antigravity-",
    "deep-research-",
    "gemma-",
)


@dataclass(frozen=True)
class RoboHarnessSettings:
    """Runtime settings loaded from environment variables with safe defaults."""

    gemini_api_key: str | None = field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    tonypi_mcp_url: str = field(
        default_factory=lambda: os.environ.get("TONYPI_MCP_URL", "http://127.0.0.1:8765/mcp")
    )
    remote_tonypi_mcp_url: str | None = field(
        default_factory=lambda: os.environ.get("TONYPI_REMOTE_MCP_URL") or None
    )
    managed_agent_id: str = FORGE_AGENT_ID
    base_agent: str = FORGE_BASE_AGENT
    forge_model: str = FORGE_MODEL
    motion_model: str = MOTION_MODEL
    decision_model: str = DECISION_MODEL
    spatial_model: str = SPATIAL_MODEL
    webcam_device: str = field(
        default_factory=lambda: os.environ.get("ROBOHARNESS_WEBCAM_DEVICE", "0")
    )
    skills_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("ROBOHARNESS_SKILLS_DIR", "skills"))
    )
    forge_state_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("ROBOHARNESS_FORGE_STATE", ".roboharness_forge_state.json")
        )
    )
    decision_confidence_threshold: float = DECISION_CONFIDENCE_THRESHOLD
    sse_host: str = "127.0.0.1"
    sse_port: int = 8770

    @classmethod
    def from_env(cls) -> RoboHarnessSettings:
        """Build settings freshly from current environment variables."""
        return cls()
