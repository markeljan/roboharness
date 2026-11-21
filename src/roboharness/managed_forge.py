"""Google Managed Agent Skill Forge (antigravity-preview-09-2026 + gemini-4-neon).

Stages `.agents/AGENTS.md`, `.agents/skills/kinematics-verifier/SKILL.md`, and
`/workspace/kinematics_verify.py` into a forked remote sandbox environment,
creates/reuses the custom Managed Agent `roboharness-skill-forge`, parses its
retargeted 16-DOF TonyPi skill output, runs deterministic local defense-in-depth
verification, and signs the resulting `SkillArtifact` with SHA-256.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from roboharness.config import (
    MAX_SLEW_PULSE_PER_SEC,
    MIN_KEYFRAME_DURATION_MS,
    REMOTE_MCP_NAME,
    RESERVED_AGENT_ID_PREFIXES,
    RoboHarnessSettings,
)
from roboharness.safety import (
    check_skill_keyframes,
    estimate_frame_tilt_and_com,
    sign_skill_artifact,
)
from roboharness.schemas import (
    JOINTS_BY_NAME,
    LOWER_BODY_JOINTS,
    NEUTRAL_POSE,
    PoseSequence,
    ServoKeyframe,
    SkillArtifact,
)

logger = logging.getLogger(__name__)

# Repository-relative bundle directory containing .agents/ and kinematics_verify.py
BUNDLE_DIR: Path = Path(__file__).resolve().parents[2] / "agent_bundle"

MCP_TOOL_NAME_RE: re.Pattern[str] = re.compile(r"^[a-z0-9_-]+$")
AUTO_FIX_TILT_TRIGGER_DEG: float = 28.0
PULSES_PER_DEG: float = 3.2
_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"completed", "failed", "cancelled", "incomplete", "budget_exceeded", "idle"}
)
_POLL_STATUSES: frozenset[str] = frozenset({"in_progress", "queued"})

FORGE_SYSTEM_INSTRUCTION: str = (
    "You are roboharness-skill-forge for the Hiwonder TonyPi 16-DOF humanoid robot. "
    "Always follow .agents/AGENTS.md and .agents/skills/kinematics-verifier/SKILL.md. "
    "Write candidate keyframes to /workspace/candidate_skill.json, verify them with "
    "`python3 /workspace/kinematics_verify.py /workspace/candidate_skill.json`, "
    "repair any joint clipping, slew-rate, or tilt >= 30 deg violation (using lower-body "
    "grounded knee-bounce adaptation when needed), and return ONLY a single valid JSON "
    "object matching the ForgeResponse schema."
)


@dataclass(frozen=True)
class ForgeAgentState:
    """Resolved state for the persisted custom Managed Agent."""

    agent_id: str
    base_environment_id: str | None
    model: str
    created_now: bool


@dataclass(frozen=True)
class ForgeResult:
    """Outcome of invoking the Managed Agent Skill Forge on a PoseSequence."""

    artifact: SkillArtifact | None
    refused: bool
    refusal_reason: str
    environment_id: str | None
    interaction_id: str | None
    raw_output: str


def validate_agent_id(agent_id: str) -> str:
    """Enforce Gemini 4 Managed Agent custom ID naming invariants."""
    cleaned = agent_id.strip()
    if not cleaned:
        raise ValueError("Managed agent ID must be non-empty.")
    lowered = cleaned.lower()
    for prefix in RESERVED_AGENT_ID_PREFIXES:
        if lowered.startswith(prefix):
            raise ValueError(
                f"Managed agent ID '{cleaned}' must not start with reserved prefix '{prefix}'."
            )
    return cleaned


def validate_mcp_tool_name(name: str) -> str:
    """Enforce remote MCP tool name regex ^[a-z0-9_-]+$."""
    if not MCP_TOOL_NAME_RE.match(name):
        raise ValueError(
            f"Remote MCP tool name '{name}' violates required pattern '^[a-z0-9_-]+$'."
        )
    return name


def load_bundle_inline_sources(bundle_dir: Path | None = None) -> list[dict[str, str]]:
    """Read the agent bundle files into Interactions API inline environment sources."""
    root = bundle_dir if bundle_dir is not None else BUNDLE_DIR
    file_map: tuple[tuple[Path, str], ...] = (
        (root / ".agents" / "AGENTS.md", ".agents/AGENTS.md"),
        (
            root / ".agents" / "skills" / "kinematics-verifier" / "SKILL.md",
            ".agents/skills/kinematics-verifier/SKILL.md",
        ),
        (root / "kinematics_verify.py", "/workspace/kinematics_verify.py"),
    )
    sources: list[dict[str, str]] = []
    for local_path, remote_target in file_map:
        if not local_path.is_file():
            raise FileNotFoundError(f"Required agent bundle file missing: {local_path}")
        sources.append(
            {
                "type": "inline",
                "target": remote_target,
                "content": local_path.read_text(encoding="utf-8"),
            }
        )
    return sources


def build_forge_tools(
    settings: RoboHarnessSettings,
    *,
    use_remote_mcp: bool = False,
    mcp_tool_name: str = REMOTE_MCP_NAME,
) -> list[dict[str, Any]]:
    """Construct tool declarations for a Managed Agent Skill Forge interaction."""
    tools: list[dict[str, Any]] = [{"type": "code_execution"}]
    if use_remote_mcp and settings.remote_tonypi_mcp_url:
        validated_name = validate_mcp_tool_name(mcp_tool_name)
        tools.append(
            {
                "type": "mcp_server",
                "name": validated_name,
                "url": settings.remote_tonypi_mcp_url,
            }
        )
    return tools


def _get_field(obj: Any, field: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)


def _extract_interaction_text(interaction: Any) -> str:
    """Extract final textual output from a google-genai Interaction object or stub."""
    output_text = _get_field(interaction, "output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    # Fallback: check steps (ModelOutputStep.content) or outputs list
    steps = _get_field(interaction, "steps")
    if isinstance(steps, list):
        chunks: list[str] = []
        for step in steps:
            if _get_field(step, "type") != "model_output":
                continue
            content_items = _get_field(step, "content") or []
            for item in content_items:
                text = _get_field(item, "text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text)
        if chunks:
            return "\n".join(chunks)

    outputs = _get_field(interaction, "outputs")
    if isinstance(outputs, list):
        chunks = []
        for item in outputs:
            text = _get_field(item, "text")
            if isinstance(text, str) and text.strip():
                chunks.append(text)
        if chunks:
            return "\n".join(chunks)

    return ""


def _wait_for_interaction(
    client: Any,
    interaction: Any,
    *,
    poll_interval_sec: float = 1.0,
    max_wait_sec: float = 180.0,
) -> Any:
    """Poll an in-progress or queued Interaction until it reaches a terminal state."""
    status = _get_field(interaction, "status")
    if status not in _POLL_STATUSES:
        return interaction

    interaction_id = _get_field(interaction, "id")
    if not interaction_id:
        return interaction

    deadline = time.monotonic() + max_wait_sec
    current = interaction
    while status in _POLL_STATUSES:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Interaction '{interaction_id}' timed out in status '{status}' after {max_wait_sec:.0f}s."
            )
        if poll_interval_sec > 0:
            time.sleep(poll_interval_sec)
        current = client.interactions.get(id=interaction_id)
        status = _get_field(current, "status")
        if status in _TERMINAL_STATUSES:
            break
    return current


def _read_saved_state(state_path: Path) -> dict[str, Any]:
    if not state_path.is_file():
        return {}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _write_saved_state(state_path: Path, state: ForgeAgentState, base_agent: str) -> None:
    payload = {
        "agent_id": state.agent_id,
        "base_agent": base_agent,
        "model": state.model,
        "base_environment_id": state.base_environment_id,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(state_path)


def ensure_forge_agent(
    client: Any,
    settings: RoboHarnessSettings,
    *,
    force_recreate: bool = False,
    poll_interval_sec: float = 1.0,
    bundle_dir: Path | None = None,
) -> ForgeAgentState:
    """Ensure the custom Managed Agent exists with a pre-staged sandbox environment."""
    agent_id = validate_agent_id(settings.managed_agent_id)

    if not force_recreate:
        try:
            existing = client.agents.get(id=agent_id)
        except Exception:  # noqa: BLE001
            existing = None

        if existing is not None:
            raw_base_env = _get_field(existing, "base_environment")
            base_env_id: str | None = None
            if isinstance(raw_base_env, str) and raw_base_env:
                base_env_id = raw_base_env
            elif raw_base_env is not None:
                maybe_env_id = _get_field(raw_base_env, "environment_id")
                if isinstance(maybe_env_id, str) and maybe_env_id:
                    base_env_id = maybe_env_id
            if base_env_id is None:
                saved = _read_saved_state(settings.forge_state_path)
                if saved.get("agent_id") == agent_id and isinstance(
                    saved.get("base_environment_id"), str
                ):
                    base_env_id = saved["base_environment_id"]

            raw_cfg = _get_field(existing, "agent_config")
            resolved_model = _get_field(raw_cfg, "model") or settings.forge_model
            state = ForgeAgentState(
                agent_id=agent_id,
                base_environment_id=base_env_id,
                model=str(resolved_model),
                created_now=False,
            )
            _write_saved_state(settings.forge_state_path, state, settings.base_agent)
            return state
    else:
        try:
            client.agents.delete(id=agent_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Ignoring delete failure while recreating agent '%s': %s", agent_id, exc)

    sources = load_bundle_inline_sources(bundle_dir)
    base_environment: str | dict[str, Any] = {"type": "remote", "sources": sources}
    base_environment_id: str | None = None

    try:
        bootstrap = client.interactions.create(
            agent=settings.base_agent,
            input=(
                "Run `python3 /workspace/kinematics_verify.py --self-test` to verify "
                "the pre-staged TonyPi kinematics verifier and reply with the JSON output."
            ),
            environment={"type": "remote", "sources": sources},
        )
        bootstrap = _wait_for_interaction(client, bootstrap, poll_interval_sec=poll_interval_sec)
        bootstrap_status = _get_field(bootstrap, "status")
        env_id = _get_field(bootstrap, "environment_id")
        if (
            isinstance(env_id, str)
            and env_id.strip()
            and bootstrap_status not in {"failed", "cancelled"}
        ):
            base_environment = env_id.strip()
            base_environment_id = env_id.strip()
        else:
            logger.warning(
                "Bootstrap interaction returned no usable environment_id (status=%s); "
                "falling back to inline base_environment.",
                bootstrap_status,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Bootstrap environment fork failed (%s); falling back to inline base_environment.",
            exc,
        )

    client.agents.create(
        id=agent_id,
        base_agent=settings.base_agent,
        agent_config={"type": "antigravity", "model": settings.forge_model},
        system_instruction=FORGE_SYSTEM_INSTRUCTION,
        base_environment=base_environment,
    )

    state = ForgeAgentState(
        agent_id=agent_id,
        base_environment_id=base_environment_id,
        model=settings.forge_model,
        created_now=True,
    )
    _write_saved_state(settings.forge_state_path, state, settings.base_agent)
    return state


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Resiliently extract a single JSON object from plain or markdown-fenced output."""
    text = raw_text.strip()
    if not text:
        raise ValueError("Forge returned empty output; expected ForgeResponse JSON.")

    # 1. Direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. Markdown code fences (prefer last valid JSON object block)
    fence_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    for block in reversed(fence_matches):
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    # 3. Scan for balanced JSON objects via JSONDecoder.raw_decode
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    idx = 0
    while idx < len(text):
        brace_pos = text.find("{", idx)
        if brace_pos == -1:
            break
        try:
            obj, end_pos = decoder.raw_decode(text, brace_pos)
            if isinstance(obj, dict):
                candidates.append(obj)
            idx = end_pos
        except json.JSONDecodeError:
            idx = brace_pos + 1

    if candidates:
        for cand in reversed(candidates):
            if "keyframes" in cand or "refused" in cand:
                return cand
        return candidates[-1]

    snippet = text[:200].replace("\n", " ")
    raise ValueError(f"Could not extract a JSON object from forge output: {snippet!r}")


def _build_forge_prompt(
    pose_seq: PoseSequence,
    skill_name: str,
    *,
    source_video: str = "",
    adaptation_hint: str | None = None,
    repair_feedback: str | None = None,
) -> str:
    sections: list[str] = [
        f"Target skill_name: {skill_name}",
    ]
    if source_video:
        sections.append(f"Source video: {source_video}")
    if adaptation_hint:
        sections.append(f"Adaptation hint: {adaptation_hint}")
    if repair_feedback:
        sections.append(
            f"Repair feedback from prior verification/execution gate: {repair_feedback}"
        )
    sections.append(
        "Follow .agents/AGENTS.md and .agents/skills/kinematics-verifier/SKILL.md.\n"
        "1. Retarget the PoseSequence below to 16-DOF TonyPi servo pulses.\n"
        "2. Write the candidate to /workspace/candidate_skill.json.\n"
        "3. Execute `python3 /workspace/kinematics_verify.py /workspace/candidate_skill.json` "
        "and repair until exit code 0 (`verified: true`, `joint_clipping_count == 0`, "
        "`peak_torso_tilt_deg < 30.0`, `max_slew_rate_pps <= 900.0`) or set `refused: true`.\n"
        "4. Return ONLY a single JSON object matching the ForgeResponse schema.\n\n"
        "Input PoseSequence JSON:\n"
        f"{pose_seq.model_dump_json(indent=2)}"
    )
    return "\n\n".join(sections)


def _repair_keyframes_locally(
    keyframes: list[ServoKeyframe],
) -> tuple[list[ServoKeyframe], bool, list[str]]:
    """Deterministically clamp bounds, project high-tilt legs, and fix slew durations."""
    repaired: list[ServoKeyframe] = []
    lower_body_projected = False
    notes: list[str] = []
    prev_pose = dict(NEUTRAL_POSE)

    for idx, frame in enumerate(keyframes):
        new_servos: dict[str, int] = dict(NEUTRAL_POSE)
        for name, pulse in frame.servos.items():
            spec = JOINTS_BY_NAME[name]
            clamped = max(spec.min_pulse, min(spec.max_pulse, int(pulse)))
            if clamped != int(pulse):
                notes.append(f"local repair frame[{idx}]: clamped {name} from {pulse} to {clamped}")
            new_servos[name] = clamped

        tilt_deg, _ = estimate_frame_tilt_and_com(new_servos)
        note_text = frame.note
        if tilt_deg >= AUTO_FIX_TILT_TRIGGER_DEG:
            lower_body_projected = True
            bounce = 24 if idx % 2 == 1 else 0
            for joint_name in LOWER_BODY_JOINTS:
                new_servos[joint_name] = 500
            new_servos["r_knee"] = 500 + bounce
            new_servos["l_knee"] = 500 + bounce
            new_servos["r_hip_pitch"] = 500 - bounce // 2
            new_servos["l_hip_pitch"] = 500 - bounce // 2
            new_servos["r_ankle_pitch"] = 500 + bounce // 2
            new_servos["l_ankle_pitch"] = 500 + bounce // 2
            notes.append(
                f"local repair frame[{idx}]: projected lower body (tilt {tilt_deg:.1f} deg >= {AUTO_FIX_TILT_TRIGGER_DEG:.1f} deg)"
            )
            if "[lower-body projected]" not in note_text:
                note_text = f"{note_text} [lower-body projected]".strip()

        max_delta = max(abs(new_servos[k] - prev_pose[k]) for k in new_servos)
        min_safe_ms = math.ceil((max_delta / (MAX_SLEW_PULSE_PER_SEC * 0.90)) * 1000.0)
        safe_duration = min(5000, max(frame.duration_ms, min_safe_ms, MIN_KEYFRAME_DURATION_MS))
        if safe_duration != frame.duration_ms:
            notes.append(
                f"local repair frame[{idx}]: adjusted duration_ms {frame.duration_ms} -> {safe_duration}"
            )

        repaired.append(ServoKeyframe(duration_ms=safe_duration, servos=new_servos, note=note_text))
        prev_pose = new_servos

    return repaired, lower_body_projected, notes


def forge_skill(
    client: Any,
    settings: RoboHarnessSettings,
    pose_seq: PoseSequence,
    skill_name: str,
    *,
    source_video: str = "",
    adaptation_hint: str | None = None,
    repair_feedback: str | None = None,
    use_remote_mcp: bool = False,
    auto_repair_locally: bool = False,
    ensure_agent: bool = True,
    poll_interval_sec: float = 1.0,
) -> ForgeResult:
    """Invoke the custom Managed Agent Skill Forge and verify/sign the returned skill."""
    agent_id = validate_agent_id(settings.managed_agent_id)
    if ensure_agent and hasattr(client, "agents"):
        ensure_forge_agent(client, settings, poll_interval_sec=poll_interval_sec)

    tools = build_forge_tools(settings, use_remote_mcp=use_remote_mcp)
    prompt = _build_forge_prompt(
        pose_seq,
        skill_name,
        source_video=source_video,
        adaptation_hint=adaptation_hint,
        repair_feedback=repair_feedback,
    )

    interaction = client.interactions.create(
        agent=agent_id,
        input=prompt,
        environment="remote",
        tools=tools,
    )
    interaction = _wait_for_interaction(client, interaction, poll_interval_sec=poll_interval_sec)

    interaction_id = _get_field(interaction, "id")
    env_id = _get_field(interaction, "environment_id")
    status = _get_field(interaction, "status")
    raw_output = _extract_interaction_text(interaction)

    if status in {"failed", "cancelled"} and not raw_output:
        errors = _get_field(interaction, "errors")
        return ForgeResult(
            artifact=None,
            refused=True,
            refusal_reason=f"Managed forge interaction ended with status '{status}' ({errors}).",
            environment_id=str(env_id) if env_id else None,
            interaction_id=str(interaction_id) if interaction_id else None,
            raw_output=raw_output,
        )

    payload = extract_json_object(raw_output)

    if payload.get("refused") is True:
        reason = str(payload.get("refusal_reason") or "Motion refused by skill forge.")
        return ForgeResult(
            artifact=None,
            refused=True,
            refusal_reason=reason,
            environment_id=str(env_id) if env_id else None,
            interaction_id=str(interaction_id) if interaction_id else None,
            raw_output=raw_output,
        )

    raw_keyframes = payload.get("keyframes")
    if not isinstance(raw_keyframes, list) or not raw_keyframes:
        raise ValueError("ForgeResponse JSON contains no keyframes and did not set refused=true.")

    extra_violations: list[str] = []
    extra_clipping = 0
    parsed_frames: list[ServoKeyframe] = []

    for idx, raw_frame in enumerate(raw_keyframes):
        if not isinstance(raw_frame, dict):
            raise TypeError(f"Keyframe[{idx}] must be a JSON object.")
        raw_dur = round(float(raw_frame.get("duration_ms", 400)))
        if raw_dur < MIN_KEYFRAME_DURATION_MS:
            extra_violations.append(
                f"frame[{idx}] duration_ms={raw_dur} < {MIN_KEYFRAME_DURATION_MS}ms"
            )
            safe_dur = MIN_KEYFRAME_DURATION_MS
        elif raw_dur > 5000:
            extra_violations.append(f"frame[{idx}] duration_ms={raw_dur} > 5000ms")
            safe_dur = 5000
        else:
            safe_dur = raw_dur

        raw_servos = raw_frame.get("servos")
        if not isinstance(raw_servos, dict):
            raise TypeError(f"Keyframe[{idx}] missing 'servos' dict.")

        clean_servos: dict[str, int] = {}
        for joint_name, raw_val in raw_servos.items():
            if joint_name not in JOINTS_BY_NAME:
                extra_clipping += 1
                extra_violations.append(f"frame[{idx}] unknown joint '{joint_name}'")
                continue
            clean_servos[joint_name] = round(float(raw_val))

        parsed_frames.append(
            ServoKeyframe(
                duration_ms=safe_dur,
                servos=clean_servos,
                note=str(raw_frame.get("note", "")),
            )
        )

    raw_kin = payload.get("kinematics") if isinstance(payload.get("kinematics"), dict) else {}
    lower_body_adapted = bool(
        payload.get("lower_body_adapted")
        or raw_kin.get("lower_body_adapted")
        or payload.get("adaptation_mode") == "upper_body_adapted"
    )
    raw_notes = payload.get("adaptation_notes") or raw_kin.get("adaptation_notes") or []
    adaptation_notes = [str(n) for n in raw_notes] if isinstance(raw_notes, list) else []

    # Deterministic local defense-in-depth verification (never blindly trust sandbox claims)
    report = check_skill_keyframes(
        parsed_frames,
        lower_body_adapted=lower_body_adapted,
        adaptation_notes=adaptation_notes,
        sandbox_environment_id=str(env_id) if env_id else None,
    )

    if extra_violations or extra_clipping:
        report = report.model_copy(
            update={
                "verified": False,
                "joint_clipping_count": report.joint_clipping_count + extra_clipping,
                "violations": [*report.violations, *extra_violations],
            }
        )

    final_frames = parsed_frames
    if not report.verified and auto_repair_locally:
        repaired_frames, projected, repair_notes = _repair_keyframes_locally(parsed_frames)
        lower_body_adapted = lower_body_adapted or projected
        adaptation_notes = [*adaptation_notes, *repair_notes]
        final_frames = repaired_frames
        report = check_skill_keyframes(
            final_frames,
            lower_body_adapted=lower_body_adapted,
            adaptation_notes=adaptation_notes,
            sandbox_environment_id=str(env_id) if env_id else None,
        )

    adaptation_mode: Literal["full_body", "upper_body_adapted"] = (
        "upper_body_adapted"
        if (lower_body_adapted or payload.get("adaptation_mode") == "upper_body_adapted")
        else "full_body"
    )

    artifact = SkillArtifact(
        name=skill_name,
        source_video=source_video,
        description=str(payload.get("description") or pose_seq.source_description),
        adaptation_mode=adaptation_mode,
        keyframes=final_frames,
        kinematics=report,
    )
    signed_artifact = sign_skill_artifact(artifact)

    return ForgeResult(
        artifact=signed_artifact,
        refused=False,
        refusal_reason="",
        environment_id=str(env_id) if env_id else None,
        interaction_id=str(interaction_id) if interaction_id else None,
        raw_output=raw_output,
    )


def _clamp_joint(joint_name: str, value: float) -> int:
    spec = JOINTS_BY_NAME[joint_name]
    return max(spec.min_pulse, min(spec.max_pulse, round(value)))


def deterministic_retarget_fallback(
    pose_seq: PoseSequence,
    skill_name: str,
    *,
    source_video: str = "",
    force_upper_body_adaptation: bool = False,
) -> SkillArtifact:
    """Pure-math offline 16-DOF retargeter for fast CI/unit tests.

    Uses the exact same angle-to-pulse rules (3.2 pulses/deg), joint bounds,
    slew limits, and grounded lower-body adaptation rules as the sandbox skill.
    Never used in place of the Managed Agent in ``--mode live``.
    """
    needs_adaptation = force_upper_body_adaptation or bool(pose_seq.impossible_elements)
    adaptation_notes: list[str] = []

    if pose_seq.impossible_elements:
        elements = ", ".join(el.element for el in pose_seq.impossible_elements)
        adaptation_notes.append(f"Adapted impossible motion elements ({elements}) to grounded knee-bounce.")

    for kf in pose_seq.keyframes:
        if kf.foot_contact == "airborne":
            needs_adaptation = True
            adaptation_notes.append(
                f"beat {kf.beat_index}: replaced airborne foot_contact with grounded knee-bounce."
            )
            break
        if (
            abs(kf.torso_pitch_deg) > 22.0
            or abs(kf.torso_roll_deg) > 18.0
            or abs(kf.l_hip_pitch_deg - kf.r_hip_pitch_deg) > 45.0
        ):
            needs_adaptation = True
            adaptation_notes.append(
                f"beat {kf.beat_index}: projected extreme lower-body posture to grounded stance."
            )
            break

    beat_ms = max(250, min(800, round(60000.0 / max(pose_seq.bpm, 30.0))))
    frames: list[ServoKeyframe] = []
    prev_pose = dict(NEUTRAL_POSE)
    lower_body_adapted = needs_adaptation

    for idx, kf in enumerate(pose_seq.keyframes):
        servos: dict[str, int] = dict(NEUTRAL_POSE)

        # Expressive mirrored upper body (3.2 pulses/deg clamped to joint spec)
        servos["r_shoulder_pitch"] = _clamp_joint(
            "r_shoulder_pitch", 500.0 + kf.r_shoulder_pitch_deg * PULSES_PER_DEG
        )
        servos["r_shoulder_roll"] = _clamp_joint(
            "r_shoulder_roll", 500.0 + kf.r_shoulder_roll_deg * PULSES_PER_DEG
        )
        servos["r_elbow"] = _clamp_joint(
            "r_elbow", 500.0 + kf.r_elbow_deg * PULSES_PER_DEG
        )
        servos["l_shoulder_pitch"] = _clamp_joint(
            "l_shoulder_pitch", 500.0 - kf.l_shoulder_pitch_deg * PULSES_PER_DEG
        )
        servos["l_shoulder_roll"] = _clamp_joint(
            "l_shoulder_roll", 500.0 - kf.l_shoulder_roll_deg * PULSES_PER_DEG
        )
        servos["l_elbow"] = _clamp_joint(
            "l_elbow", 500.0 - kf.l_elbow_deg * PULSES_PER_DEG
        )

        if needs_adaptation:
            bounce = 24 if idx % 2 == 1 else 0
            servos["r_knee"] = 500 + bounce
            servos["l_knee"] = 500 + bounce
            servos["r_hip_pitch"] = 500 - bounce // 2
            servos["l_hip_pitch"] = 500 - bounce // 2
            servos["r_ankle_pitch"] = 500 + bounce // 2
            servos["l_ankle_pitch"] = 500 + bounce // 2
        else:
            avg_knee_deg = min(20.0, 0.5 * (kf.l_knee_deg + kf.r_knee_deg) * 0.35)
            knee_delta = round(avg_knee_deg * PULSES_PER_DEG)
            lean_p = max(-8.0, min(8.0, kf.torso_pitch_deg * 0.35))
            lean_r = max(-8.0, min(8.0, kf.torso_roll_deg * 0.35))
            servos["r_knee"] = _clamp_joint("r_knee", 500 + knee_delta)
            servos["l_knee"] = _clamp_joint("l_knee", 500 + knee_delta)
            servos["r_hip_pitch"] = _clamp_joint(
                "r_hip_pitch", 500 - knee_delta // 2 + round(lean_p * 1.6)
            )
            servos["l_hip_pitch"] = _clamp_joint(
                "l_hip_pitch", 500 - knee_delta // 2 + round(lean_p * 1.6)
            )
            servos["r_ankle_pitch"] = _clamp_joint(
                "r_ankle_pitch", 500 + knee_delta // 2 + round(lean_p * 1.2)
            )
            servos["l_ankle_pitch"] = _clamp_joint(
                "l_ankle_pitch", 500 + knee_delta // 2 + round(lean_p * 1.2)
            )
            servos["r_hip_roll"] = _clamp_joint("r_hip_roll", 500 + round(lean_r * 1.6))
            servos["l_hip_roll"] = _clamp_joint("l_hip_roll", 500 + round(lean_r * 1.6))
            servos["r_ankle_roll"] = _clamp_joint("r_ankle_roll", 500 + round(lean_r * 1.2))
            servos["l_ankle_roll"] = _clamp_joint("l_ankle_roll", 500 + round(lean_r * 1.2))

            tilt_deg, _ = estimate_frame_tilt_and_com(servos)
            if tilt_deg >= AUTO_FIX_TILT_TRIGGER_DEG:
                lower_body_adapted = True
                bounce = 24 if idx % 2 == 1 else 0
                for joint_name in LOWER_BODY_JOINTS:
                    servos[joint_name] = 500
                servos["r_knee"] = 500 + bounce
                servos["l_knee"] = 500 + bounce
                servos["r_hip_pitch"] = 500 - bounce // 2
                servos["l_hip_pitch"] = 500 - bounce // 2
                servos["r_ankle_pitch"] = 500 + bounce // 2
                servos["l_ankle_pitch"] = 500 + bounce // 2
                adaptation_notes.append(
                    f"beat {kf.beat_index}: clamped high tilt ({tilt_deg:.1f} deg) to grounded bounce."
                )

        if idx == 0:
            base_ms = beat_ms
        else:
            delta_t = kf.t_ms - pose_seq.keyframes[idx - 1].t_ms
            base_ms = delta_t if delta_t >= MIN_KEYFRAME_DURATION_MS else beat_ms

        max_delta = max(abs(servos[k] - prev_pose[k]) for k in servos)
        min_safe_ms = math.ceil((max_delta / (MAX_SLEW_PULSE_PER_SEC * 0.90)) * 1000.0)
        duration_ms = min(5000, max(base_ms, min_safe_ms, MIN_KEYFRAME_DURATION_MS))

        note = kf.label or f"beat_{kf.beat_index}"
        if lower_body_adapted:
            note = f"{note} [grounded bounce]"

        frames.append(ServoKeyframe(duration_ms=duration_ms, servos=servos, note=note))
        prev_pose = servos

    report = check_skill_keyframes(
        frames,
        lower_body_adapted=lower_body_adapted,
        adaptation_notes=adaptation_notes,
    )
    artifact = SkillArtifact(
        name=skill_name,
        source_video=source_video,
        description=pose_seq.source_description,
        adaptation_mode="upper_body_adapted" if lower_body_adapted else "full_body",
        keyframes=frames,
        kinematics=report,
    )
    return sign_skill_artifact(artifact)
