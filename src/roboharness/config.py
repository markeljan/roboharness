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
DECISION_CONFIDENCE_THRESHOLD: Final[float] = 0.75
DECISION_SAFETY_PROB_THRESHOLD: Final[float] = 0.75

# Gemini 4 Dual-Client SDK constants (v1 stable + v1beta for Managed Agents & Decisions)
STABLE_API_VERSION: Final[str] = "v1"
BETA_API_VERSION: Final[str] = "v1beta"
API_VERSION: Final[str] = BETA_API_VERSION
MOTION_MODEL: Final[str] = "gemini-4-hydrogen"
FORGE_BASE_AGENT: Final[str] = "antigravity-preview-09-2026"
FORGE_MODEL: Final[str] = "gemini-4-neon"
FORGE_AGENT_ID: Final[str] = "roboharness-skill-forge"
FORGE_DEEP_MODEL: Final[str] = "gemini-4-carbon"
FORGE_DEEP_AGENT_ID: Final[str] = "roboharness-skill-forge-carbon"
DECISION_MODEL: Final[str] = "gemini-decision-1"
SPATIAL_MODEL: Final[str] = "gemini-robotics-er-2-preview"
REMOTE_MCP_NAME: Final[str] = "tonypi_mcp"

VALID_THINKING_LEVELS: Final[frozenset[str]] = frozenset({"low", "medium", "high", "xhigh"})
FORBIDDEN_GEMINI4_PARAMS: Final[frozenset[str]] = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "cachedContents",
        "cached_content",
        "max_output_tokens",
        "max_total_tokens",
    }
)
RESERVED_AGENT_ID_PREFIXES: Final[tuple[str, ...]] = (
    "gemini-",
    "google-",
    "antigravity-",
    "deep-research-",
    "gemma-",
)


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[7:].strip()
    if "=" not in stripped:
        return None
    key, raw_value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    elif " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return key, value


def load_dotenv_if_present(env_path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from a local .env file without overriding existing env vars."""
    candidates: list[Path]
    if env_path is not None:
        candidates = [Path(env_path)]
    else:
        cwd_env = Path.cwd() / ".env"
        repo_env = Path(__file__).resolve().parents[2] / ".env"
        candidates = [cwd_env] if cwd_env.resolve() == repo_env.resolve() else [cwd_env, repo_env]

    for path in candidates:
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in content.splitlines():
            parsed = _parse_dotenv_line(line)
            if parsed is not None:
                key, value = parsed
                os.environ.setdefault(key, value)


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
    forge_deep_agent_id: str = FORGE_DEEP_AGENT_ID
    base_agent: str = FORGE_BASE_AGENT
    forge_model: str = FORGE_MODEL
    forge_deep_model: str = FORGE_DEEP_MODEL
    motion_model: str = MOTION_MODEL
    decision_model: str = DECISION_MODEL
    spatial_model: str = SPATIAL_MODEL
    gemini_environment_id: str | None = field(
        default_factory=lambda: os.environ.get("GEMINI_ENVIRONMENT_ID") or None
    )
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
    decision_safety_prob_threshold: float = DECISION_SAFETY_PROB_THRESHOLD
    sse_host: str = "127.0.0.1"
    sse_port: int = 8770

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> RoboHarnessSettings:
        """Build settings freshly from current environment variables (auto-loading .env)."""
        load_dotenv_if_present(env_path)
        return cls()
