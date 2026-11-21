"""Signed, atomic local skill library store for RoboHarness (skills/<slug>.json)."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from roboharness.safety import sign_skill_artifact, verify_artifact_signature
from roboharness.schemas import SkillArtifact

logger = logging.getLogger(__name__)

SKILL_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_UNSAFE_SLUG_CHARS_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9_-]+")
_MULTI_DASH_RE: Final[re.Pattern[str]] = re.compile(r"-{2,}")
_MAX_SLUG_LENGTH: Final[int] = 64


def slugify_skill_name(name: str) -> str:
    """Convert a skill name into a strict safe filename slug matching ^[a-z0-9][a-z0-9_-]*$.

    Rejects path traversal segments ('..'), path separators ('/', '\\'), null bytes,
    and empty/un-slugifiable names with ValueError.
    """
    if not isinstance(name, str):
        raise TypeError("Skill name must be a string")

    raw = name.strip()
    if not raw:
        raise ValueError("Skill name cannot be empty")
    if "\x00" in raw or "/" in raw or "\\" in raw or ".." in raw:
        raise ValueError(f"Unsafe skill name '{name}': path separators or traversal not allowed")

    slug = raw.lower()
    slug = _UNSAFE_SLUG_CHARS_RE.sub("-", slug)
    slug = _MULTI_DASH_RE.sub("-", slug)
    slug = slug.strip("-_")[:_MAX_SLUG_LENGTH].rstrip("-_")

    if not slug or not SKILL_SLUG_RE.match(slug):
        raise ValueError(f"Skill name '{name}' cannot be converted to a valid slug")
    return slug


def _skill_path(name: str, skills_dir: Path) -> Path:
    """Resolve and verify target JSON path strictly inside skills_dir."""
    slug = slugify_skill_name(name)
    base_dir = skills_dir.resolve()
    target = (base_dir / f"{slug}.json").resolve()
    if target.parent != base_dir:
        raise ValueError(f"Resolved skill path escapes skills directory: {target}")
    return skills_dir / f"{slug}.json"


def save_skill(artifact: SkillArtifact, skills_dir: Path) -> Path:
    """Validate/sign a SkillArtifact and write it atomically to skills_dir/<slug>.json."""
    target = _skill_path(artifact.name, skills_dir)

    if artifact.artifact_sha256:
        if not verify_artifact_signature(artifact):
            raise ValueError(
                f"Refusing to save skill '{artifact.name}': existing artifact_sha256 "
                "does not match canonical payload"
            )
        signed = artifact
    else:
        signed = sign_skill_artifact(artifact)

    skills_dir.mkdir(parents=True, exist_ok=True)
    payload_text = signed.model_dump_json(indent=2) + "\n"

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=skills_dir,
            prefix=f".{target.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(payload_text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    return target


def load_skill(
    name: str,
    skills_dir: Path,
    *,
    verify_signature: bool = True,
) -> SkillArtifact:
    """Load and validate a SkillArtifact from skills_dir/<slug>.json."""
    path = _skill_path(name, skills_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Skill '{name}' not found at {path}")

    raw_text = path.read_text(encoding="utf-8")
    artifact = SkillArtifact.model_validate_json(raw_text)

    if verify_signature and not verify_artifact_signature(artifact):
        raise ValueError(
            f"SHA-256 signature verification failed for skill '{artifact.name}' ({path})"
        )

    return artifact


def _created_at_sort_key(artifact: SkillArtifact) -> tuple[float, str]:
    """Parse ISO-8601 created_at to an epoch float for accurate descending ordering."""
    raw = artifact.created_at
    try:
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return (parsed.timestamp(), artifact.name)
    except ValueError:
        return (0.0, raw)


def list_skills(skills_dir: Path) -> list[SkillArtifact]:
    """Return all valid, signature-verified skills sorted by created_at descending."""
    if not skills_dir.exists() or not skills_dir.is_dir():
        return []

    skills: list[SkillArtifact] = []
    for entry in sorted(skills_dir.iterdir()):
        if not entry.is_file() or entry.name.startswith(".") or entry.suffix != ".json":
            continue
        try:
            artifact = SkillArtifact.model_validate_json(entry.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as exc:
            logger.debug("Skipping unreadable/invalid skill file %s: %s", entry, exc)
            continue
        if not verify_artifact_signature(artifact):
            logger.debug("Skipping skill with invalid signature: %s", entry)
            continue
        skills.append(artifact)

    skills.sort(key=_created_at_sort_key, reverse=True)
    return skills


def delete_skill(name: str, skills_dir: Path) -> bool:
    """Delete skills_dir/<slug>.json if present; return True if deleted else False."""
    path = _skill_path(name, skills_dir)
    if not path.is_file():
        return False
    path.unlink()
    return True
