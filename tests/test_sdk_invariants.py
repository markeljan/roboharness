"""Static, AST, and runtime invariant tests enforcing GEMINI.md & Gemini 4 Builder Guide."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.eval_5sigma import run_benchmark
from roboharness.cli import app as cli_app
from roboharness.config import (
    API_VERSION,
    BETA_API_VERSION,
    DECISION_CONFIDENCE_THRESHOLD,
    DECISION_MODEL,
    DECISION_SAFETY_PROB_THRESHOLD,
    FORBIDDEN_GEMINI4_PARAMS,
    FORGE_AGENT_ID,
    FORGE_BASE_AGENT,
    FORGE_DEEP_AGENT_ID,
    FORGE_DEEP_MODEL,
    FORGE_MODEL,
    REMOTE_MCP_NAME,
    RESERVED_AGENT_ID_PREFIXES,
    STABLE_API_VERSION,
    VALID_THINKING_LEVELS,
    RoboHarnessSettings,
)
from roboharness.decision_gate import (
    POST_RUN_QUESTIONS,
    PRE_EXECUTION_QUESTIONS,
    parse_decision_response,
)
from roboharness.managed_forge import (
    ensure_forge_agent,
    forge_skill,
    forge_skill_with_escalation,
    resolve_forge_environment,
)
from roboharness.mcp_client import TonypiMCPClient
from roboharness.mock_mcp import MockTonypiMCPServer
from roboharness.models import (
    DualGenAIClients,
    build_gemini4_generation_config,
    compact_json_schema_for_gemini,
    make_beta_client,
    make_dual_clients,
    make_v1_client,
    upload_video_agentic,
)
from roboharness.pipeline import MimicPipeline
from roboharness.schemas import NEUTRAL_POSE, HumanPoseKeyframe, PoseSequence
from roboharness.telemetry_sse import _render_strip_html

SRC_DIR = REPO_ROOT / "src" / "roboharness"
BUNDLE_DIR = REPO_ROOT / "agent_bundle"
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
GEMINI_MD_PATH = REPO_ROOT / "GEMINI.md"

FORBIDDEN_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "cachedContents",
        "cached_content",
        "max_output_tokens",
        "maxOutputTokens",
        "max_total_tokens",
        "maxTotalTokens",
    }
)

# Only config.py (constant definition) and models.py (runtime validation guard)
# may list forbidden parameter names in guard sets; actual SDK calls are ALWAYS scanned.
FORBIDDEN_PARAM_GUARD_EXEMPT_FILES: frozenset[str] = frozenset(
    {
        "src/roboharness/config.py",
        "src/roboharness/models.py",
    }
)

FORBIDDEN_IMPORT_ROOTS: frozenset[str] = frozenset(
    {
        "tonypi",
        "hiwondersdk",
        "streamlit",
    }
)

MCP_NAME_RE = re.compile(r"^[a-z0-9_-]+$")
INTERACTION_ALLOWED_THINKING_LEVELS: frozenset[str] = frozenset({"medium", "high"})


def _iter_python_files(*roots: Path) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def _iter_bundle_text_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    ]


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _dotted_call_name(func: ast.expr) -> str:
    parts: list[str] = []
    curr: ast.expr | None = func
    while isinstance(curr, ast.Attribute):
        parts.append(curr.attr)
        curr = curr.value
    if isinstance(curr, ast.Name):
        parts.append(curr.id)
    return ".".join(reversed(parts))


def _is_gemini_call_or_config_builder(call: ast.Call) -> bool:
    name = _dotted_call_name(call.func)
    if not name:
        return False
    return (
        name.endswith(
            (
                "interactions.create",
                "agents.create",
                "decisions.create",
                "models.generate_content",
                ".generate_content",
            )
        )
        or "generation_config" in name
    )


def _str_literal(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _collect_dict_assignments(tree: ast.AST) -> dict[str, ast.Dict]:
    mapping: dict[str, ast.Dict] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    mapping[target.id] = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.value, ast.Dict)
        ):
            mapping[node.target.id] = node.value
    return mapping


def _find_forbidden_keys_in_expr(
    expr: ast.expr,
    dict_vars: dict[str, ast.Dict],
    seen_vars: set[str] | None = None,
) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    seen = set() if seen_vars is None else seen_vars
    for sub in ast.walk(expr):
        if isinstance(sub, ast.Dict):
            for key_node in sub.keys:
                key_str = _str_literal(key_node)
                if key_str in FORBIDDEN_PARAM_NAMES:
                    found.append((getattr(key_node, "lineno", sub.lineno), key_str))
        elif isinstance(sub, ast.Name) and sub.id in dict_vars and sub.id not in seen:
            seen.add(sub.id)
            found.extend(_find_forbidden_keys_in_expr(dict_vars[sub.id], dict_vars, seen))
    return found


def _find_forbidden_gemini4_param_violations(tree: ast.AST, rel_path: str) -> list[str]:
    violations: list[str] = []
    dict_vars = _collect_dict_assignments(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_gemini_call_or_config_builder(node):
            callee = _dotted_call_name(node.func)
            for kw in node.keywords:
                if kw.arg in FORBIDDEN_PARAM_NAMES:
                    violations.append(
                        f"{rel_path}:{kw.lineno}: forbidden Gemini 4 kwarg '{kw.arg}' in {callee}()"
                    )
                for lineno, bad_key in _find_forbidden_keys_in_expr(kw.value, dict_vars):
                    violations.append(
                        f"{rel_path}:{lineno}: forbidden Gemini 4 dict key '{bad_key}' "
                        f"passed to {callee}()"
                    )
            for arg in node.args:
                for lineno, bad_key in _find_forbidden_keys_in_expr(arg, dict_vars):
                    violations.append(
                        f"{rel_path}:{lineno}: forbidden Gemini 4 dict key '{bad_key}' "
                        f"passed to {callee}()"
                    )

        if rel_path not in FORBIDDEN_PARAM_GUARD_EXEMPT_FILES and isinstance(node, ast.Dict):
            for key_node in node.keys:
                key_str = _str_literal(key_node)
                if key_str in FORBIDDEN_PARAM_NAMES:
                    violations.append(
                        f"{rel_path}:{getattr(key_node, 'lineno', node.lineno)}: "
                        f"forbidden Gemini 4 key '{key_str}' in dict literal"
                    )

    return violations


def _find_invalid_thinking_level_violations(tree: ast.AST, rel_path: str) -> list[str]:
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callee = _dotted_call_name(node.func)
            for kw in node.keywords:
                if kw.arg == "thinking_level":
                    val = _str_literal(kw.value)
                    if val is not None and val.lower() not in VALID_THINKING_LEVELS:
                        violations.append(
                            f"{rel_path}:{kw.lineno}: invalid thinking_level={val!r} in {callee}()"
                        )
            if callee.endswith("build_gemini4_generation_config") and node.args:
                val = _str_literal(node.args[0])
                if val is not None and val.lower() not in VALID_THINKING_LEVELS:
                    violations.append(
                        f"{rel_path}:{node.lineno}: invalid thinking_level={val!r} in {callee}()"
                    )

        elif isinstance(node, ast.Dict):
            for key_node, val_node in zip(node.keys, node.values, strict=False):
                if _str_literal(key_node) == "thinking_level":
                    val = _str_literal(val_node)
                    if val is not None and val.lower() not in VALID_THINKING_LEVELS:
                        violations.append(
                            f"{rel_path}:{getattr(val_node, 'lineno', node.lineno)}: "
                            f"invalid dict thinking_level={val!r}"
                        )

        elif isinstance(node, ast.Assign):
            val = _str_literal(node.value)
            if val is not None and val.lower() == "minimal":
                for target in node.targets:
                    if isinstance(target, ast.Name) and "thinking" in target.id.lower():
                        violations.append(
                            f"{rel_path}:{node.lineno}: 'minimal' assigned to {target.id}"
                        )

        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and "thinking" in node.target.id.lower()
        ):
            val = _str_literal(node.value)
            if val is not None and val.lower() == "minimal":
                violations.append(
                    f"{rel_path}:{node.lineno}: 'minimal' assigned to {node.target.id}"
                )

    return violations


def _find_forbidden_imports(tree: ast.AST, rel_path: str) -> list[str]:
    violations: list[str] = []

    def _check_module_name(mod_name: str, lineno: int) -> None:
        segments = [seg.lower() for seg in mod_name.split(".") if seg]
        if segments and segments[0] in FORBIDDEN_IMPORT_ROOTS:
            violations.append(f"{rel_path}:{lineno}: forbidden clean-room import '{mod_name}'")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_module_name(alias.name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                _check_module_name(node.module, node.lineno)
            for alias in node.names:
                if alias.name.lower() in FORBIDDEN_IMPORT_ROOTS:
                    violations.append(
                        f"{rel_path}:{node.lineno}: forbidden clean-room import '{alias.name}'"
                    )
        elif isinstance(node, ast.Call):
            callee = _dotted_call_name(node.func)
            if callee in {"__import__", "importlib.import_module", "import_module"} and node.args:
                mod_str = _str_literal(node.args[0])
                if mod_str:
                    _check_module_name(mod_str, node.lineno)

    return violations


def _get_func_def(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Expected top-level function '{name}'")


def _sample_pose_sequence() -> PoseSequence:
    return PoseSequence(
        source_description="Invariant verification wave",
        bpm=110.0,
        duration_ms=1000,
        keyframes=[
            HumanPoseKeyframe(
                t_ms=0,
                beat_index=0,
                torso_pitch_deg=0.0,
                torso_roll_deg=0.0,
                l_shoulder_pitch_deg=0.0,
                l_shoulder_roll_deg=0.0,
                l_elbow_deg=10.0,
                r_shoulder_pitch_deg=45.0,
                r_shoulder_roll_deg=20.0,
                r_elbow_deg=30.0,
                l_hip_pitch_deg=0.0,
                l_knee_deg=5.0,
                r_hip_pitch_deg=0.0,
                r_knee_deg=5.0,
                foot_contact="both",
                label="raise",
            ),
            HumanPoseKeyframe(
                t_ms=500,
                beat_index=1,
                torso_pitch_deg=0.0,
                torso_roll_deg=0.0,
                l_shoulder_pitch_deg=0.0,
                l_shoulder_roll_deg=0.0,
                l_elbow_deg=10.0,
                r_shoulder_pitch_deg=0.0,
                r_shoulder_roll_deg=0.0,
                r_elbow_deg=10.0,
                l_hip_pitch_deg=0.0,
                l_knee_deg=5.0,
                r_hip_pitch_deg=0.0,
                r_knee_deg=5.0,
                foot_contact="both",
                label="return",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Adjustment #1: Dual Clients (stable v1 + v1beta) & Gemini 4 parameter guards
# ---------------------------------------------------------------------------


def test_dual_clients_v1_and_v1beta_invariants() -> None:
    assert STABLE_API_VERSION == "v1"
    assert BETA_API_VERSION == "v1beta"
    assert API_VERSION == "v1beta"

    # Runtime verification against real google-genai Client construction
    v1_client = make_v1_client("invariant-test-key")
    beta_client = make_beta_client("invariant-test-key")
    dual = make_dual_clients("invariant-test-key")

    assert isinstance(dual, DualGenAIClients)
    assert v1_client._api_client._http_options.api_version == "v1"
    assert v1_client._api_client._http_options.api_version != "v1beta"
    assert beta_client._api_client._http_options.api_version == "v1beta"
    assert dual.client._api_client._http_options.api_version == "v1"
    assert dual.beta_client._api_client._http_options.api_version == "v1beta"

    # Static AST check: genai.Client instantiated ONLY in src/roboharness/models.py
    models_path = SRC_DIR / "models.py"
    models_tree = ast.parse(models_path.read_text(encoding="utf-8"), filename=str(models_path))

    v1_fn = _get_func_def(models_tree, "make_v1_client")
    v1_calls = [
        n
        for n in ast.walk(v1_fn)
        if isinstance(n, ast.Call) and _dotted_call_name(n.func) == "genai.Client"
    ]
    assert len(v1_calls) == 1
    assert all(kw.arg != "http_options" for kw in v1_calls[0].keywords), (
        "make_v1_client() must NOT pass http_options / api_version='v1beta'"
    )

    beta_fn = _get_func_def(models_tree, "make_beta_client")
    beta_calls = [
        n
        for n in ast.walk(beta_fn)
        if isinstance(n, ast.Call) and _dotted_call_name(n.func) == "genai.Client"
    ]
    assert len(beta_calls) == 1
    assert any(kw.arg == "http_options" for kw in beta_calls[0].keywords), (
        "make_beta_client() must explicitly pass http_options={'api_version': 'v1beta'}"
    )

    for path in _iter_python_files(SRC_DIR, BENCHMARKS_DIR):
        rel = _rel(path)
        if rel == "src/roboharness/models.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _dotted_call_name(node.func).endswith("genai.Client"):
                raise AssertionError(
                    f"{rel}:{node.lineno}: direct genai.Client() call outside models.py; "
                    "use make_v1_client / make_beta_client / make_dual_clients"
                )


def test_zero_forbidden_gemini4_params_and_zero_max_total_tokens() -> None:
    py_files = _iter_python_files(SRC_DIR, BUNDLE_DIR, BENCHMARKS_DIR)
    assert py_files, "Expected Python files under src/roboharness, agent_bundle, benchmarks"

    violations: list[str] = []
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        violations.extend(_find_forbidden_gemini4_param_violations(tree, _rel(path)))

    assert not violations, "Forbidden Gemini 4 params detected:\n" + "\n".join(violations)

    # Strict raw-text sweep: ZERO max_total_tokens anywhere outside guard definitions
    token_cap_offenders: list[str] = []
    for path in [*_iter_python_files(SRC_DIR, BENCHMARKS_DIR), *_iter_bundle_text_files(BUNDLE_DIR)]:
        rel = _rel(path)
        if rel in FORBIDDEN_PARAM_GUARD_EXEMPT_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        if "max_total_tokens" in text or "maxTotalTokens" in text:
            token_cap_offenders.append(rel)

    assert not token_cap_offenders, (
        "Forbidden 'max_total_tokens' found in: " + ", ".join(token_cap_offenders)
    )

    # Runtime guard verification on build_gemini4_generation_config
    assert FORBIDDEN_GEMINI4_PARAMS.issuperset(
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
    for bad_param in sorted(FORBIDDEN_PARAM_NAMES):
        with pytest.raises(ValueError, match="Forbidden Gemini 4"):
            build_gemini4_generation_config("medium", **{bad_param: 0.5})


def test_zero_minimal_thinking_level_and_interactions_use_medium_or_high() -> None:
    assert "minimal" not in VALID_THINKING_LEVELS
    assert VALID_THINKING_LEVELS == frozenset({"low", "medium", "high", "xhigh"})

    with pytest.raises(ValueError, match="Invalid thinking_level"):
        build_gemini4_generation_config("minimal")

    py_files = _iter_python_files(SRC_DIR, BUNDLE_DIR, BENCHMARKS_DIR)
    violations: list[str] = []
    for path in py_files:
        rel = _rel(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        violations.extend(_find_invalid_thinking_level_violations(tree, rel))

        # Every literal thinking_level passed to build_gemini4_generation_config in src/benchmarks
        # must be "medium" or "high" (never "low" or "minimal").
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _dotted_call_name(node.func).endswith(
                "build_gemini4_generation_config"
            ):
                level_arg: ast.expr | None = node.args[0] if node.args else None
                for kw in node.keywords:
                    if kw.arg == "thinking_level":
                        level_arg = kw.value
                lit = _str_literal(level_arg)
                if lit is not None and lit not in INTERACTION_ALLOWED_THINKING_LEVELS:
                    violations.append(
                        f"{rel}:{node.lineno}: Interactions call uses thinking_level={lit!r} "
                        "(expected 'medium' or 'high')"
                    )

    assert not violations, "Invalid thinking_level values detected:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# Adjustment #2: Agentic Video Processing (files.upload + processing="agentic")
# ---------------------------------------------------------------------------


def test_agentic_video_processing_ast_and_runtime(tmp_path: Path) -> None:
    # 1. AST verification in models.py
    models_path = SRC_DIR / "models.py"
    models_tree = ast.parse(models_path.read_text(encoding="utf-8"), filename=str(models_path))
    upload_fn = _get_func_def(models_tree, "upload_video_agentic")

    called_methods = {
        _dotted_call_name(n.func) for n in ast.walk(upload_fn) if isinstance(n, ast.Call)
    }
    assert any(m.endswith("files.upload") for m in called_methods), (
        "upload_video_agentic must upload via client.files.upload(...)"
    )
    assert any(m.endswith("files.get") for m in called_methods), (
        "upload_video_agentic must poll via client.files.get(...) until ACTIVE"
    )

    has_agentic_video_dict = False
    for node in ast.walk(upload_fn):
        if isinstance(node, ast.Dict):
            pairs = {
                _str_literal(k): _str_literal(v)
                for k, v in zip(node.keys, node.values, strict=False)
                if _str_literal(k) is not None
            }
            if pairs.get("type") == "video" and pairs.get("processing") == "agentic":
                has_agentic_video_dict = True
                break
    assert has_agentic_video_dict, (
        'upload_video_agentic must emit {"type": "video", "uri": ..., "processing": "agentic"}'
    )

    # 2. AST verification in motion_extractor.py
    motion_path = SRC_DIR / "motion_extractor.py"
    motion_tree = ast.parse(motion_path.read_text(encoding="utf-8"), filename=str(motion_path))
    for fn_name in (
        "extract_motion_from_video",
        "debug_trial_video",
        "generate_baseline_raw_keyframes",
    ):
        fn_node = _get_func_def(motion_tree, fn_name)
        fn_calls = {
            _dotted_call_name(n.func) for n in ast.walk(fn_node) if isinstance(n, ast.Call)
        }
        assert "upload_video_agentic" in fn_calls, (
            f"motion_extractor.{fn_name} must call upload_video_agentic(...)"
        )

    # 3. Runtime unit assertion proving files.upload + processing="agentic"
    clip_path = tmp_path / "sample.mp4"
    clip_path.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    upload_calls: list[str] = []

    class _FakeFilesAPI:
        def upload(self, *, file: str) -> Any:
            upload_calls.append(file)
            return SimpleNamespace(
                name="files/demo-clip",
                uri="https://generativelanguage.googleapis.com/v1/files/demo-clip",
                mime_type="video/mp4",
                state="PROCESSING",
            )

        def get(self, *, name: str) -> Any:
            assert name == "files/demo-clip"
            return SimpleNamespace(
                name=name,
                uri="https://generativelanguage.googleapis.com/v1/files/demo-clip",
                mime_type="video/mp4",
                state="ACTIVE",
            )

    fake_client = SimpleNamespace(files=_FakeFilesAPI())
    video_part = upload_video_agentic(fake_client, clip_path, poll_interval_s=0.0)
    assert upload_calls == [str(clip_path)]
    assert video_part == {
        "type": "video",
        "uri": "https://generativelanguage.googleapis.com/v1/files/demo-clip",
        "mime_type": "video/mp4",
        "processing": "agentic",
    }

    # 4. Compact Gemini 4 JSON Schema guard (prevents live HTTP 400 FSM state explosion)
    compact_pose_schema = compact_json_schema_for_gemini(PoseSequence)
    serialized_schema = json.dumps(compact_pose_schema)
    for heavy_key in ('"minimum"', '"maximum"', '"minItems"', '"maxItems"'):
        assert heavy_key not in serialized_schema, (
            f"compact_json_schema_for_gemini(PoseSequence) leaked FSM keyword {heavy_key}"
        )
    for rel_mod in ("motion_extractor.py", "spatial.py"):
        mod_path = SRC_DIR / rel_mod
        mod_tree = ast.parse(mod_path.read_text(encoding="utf-8"), filename=str(mod_path))
        for node in ast.walk(mod_tree):
            if isinstance(node, ast.Call) and _dotted_call_name(node.func).endswith(
                "model_json_schema"
            ):
                raise AssertionError(
                    f"src/roboharness/{rel_mod}:{node.lineno}: pass "
                    "compact_json_schema_for_gemini(...) instead of raw model_json_schema() "
                    "to avoid live Gemini 4 constrained-decoding 400 errors"
                )


# ---------------------------------------------------------------------------
# Adjustment #3: Managed Agent (remote / GEMINI_ENVIRONMENT_ID / Carbon escalation)
# ---------------------------------------------------------------------------


def test_forge_agent_id_never_uses_reserved_prefix_and_mcp_name_regex() -> None:
    reserved = ("gemini-", "google-", "antigravity-", "deep-research-", "gemma-")
    assert tuple(RESERVED_AGENT_ID_PREFIXES) == reserved
    assert FORGE_BASE_AGENT == "antigravity-preview-09-2026"
    assert FORGE_MODEL == "gemini-4-neon"
    assert FORGE_DEEP_MODEL == "gemini-4-carbon"
    assert FORGE_DEEP_AGENT_ID == "roboharness-skill-forge-carbon"

    settings = RoboHarnessSettings()
    for agent_id in (
        FORGE_AGENT_ID,
        FORGE_DEEP_AGENT_ID,
        settings.managed_agent_id,
        settings.forge_deep_agent_id,
    ):
        assert not agent_id.lower().startswith(reserved), f"Reserved prefix on agent_id={agent_id!r}"

    # Verify any literal id= passed to agents.create(...) avoids reserved prefixes
    for path in _iter_python_files(SRC_DIR, BENCHMARKS_DIR):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _dotted_call_name(node.func).endswith(
                "agents.create"
            ):
                for kw in node.keywords:
                    if kw.arg == "id":
                        val = _str_literal(kw.value)
                        if val is not None:
                            assert not val.lower().startswith(reserved), (
                                f"{_rel(path)}:{kw.lineno} reserved agent id prefix: {val!r}"
                            )

    assert MCP_NAME_RE.fullmatch(REMOTE_MCP_NAME), (
        f"REMOTE_MCP_NAME={REMOTE_MCP_NAME!r} must match ^[a-z0-9_-]+$"
    )
    for path in _iter_python_files(SRC_DIR, BENCHMARKS_DIR):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            pairs = {
                _str_literal(k): _str_literal(v)
                for k, v in zip(node.keys, node.values, strict=False)
                if _str_literal(k) is not None
            }
            if pairs.get("type") == "mcp_server" and pairs.get("name") is not None:
                assert MCP_NAME_RE.fullmatch(pairs["name"]), (
                    f"{_rel(path)}:{node.lineno} invalid mcp_server name {pairs['name']!r}"
                )


def test_managed_agent_self_hosted_env_omits_sources_and_carbon_escalation(
    tmp_path: Path,
) -> None:
    # Default environment is "remote"; GEMINI_ENVIRONMENT_ID overrides it directly
    default_settings = RoboHarnessSettings(
        gemini_environment_id=None,
        forge_state_path=tmp_path / "forge_default.json",
    )
    assert resolve_forge_environment(default_settings) == "remote"

    custom_env_id = "env-self-hosted-mujoco-42"
    self_hosted_settings = RoboHarnessSettings(
        gemini_environment_id=custom_env_id,
        forge_state_path=tmp_path / "forge_self_hosted.json",
    )
    assert resolve_forge_environment(self_hosted_settings) == custom_env_id

    # Verify GEMINI_ENVIRONMENT_ID omits hosted-only sources/allowlist on create & interact
    agent_create_calls: list[dict[str, Any]] = []
    interaction_calls: list[dict[str, Any]] = []
    safe_response = json.dumps(
        {
            "refused": False,
            "name": "wave",
            "description": "Safe wave",
            "adaptation_mode": "full_body",
            "keyframes": [{"duration_ms": 400, "servos": dict(NEUTRAL_POSE), "note": "neutral"}],
        }
    )
    bad_response = json.dumps(
        {
            "refused": False,
            "name": "wave",
            "description": "Clipped wave",
            "adaptation_mode": "full_body",
            "keyframes": [
                {
                    "duration_ms": 400,
                    "servos": {**NEUTRAL_POSE, "r_shoulder_pitch": 999},
                    "note": "clipped",
                }
            ],
        }
    )

    class _FakeAgentsAPI:
        def __init__(self) -> None:
            self._store: dict[str, Any] = {}

        def get(self, *, id: str) -> Any:
            return self._store.get(id)

        def create(self, **kwargs: Any) -> Any:
            agent_create_calls.append(kwargs)
            obj = SimpleNamespace(**kwargs)
            self._store[kwargs["id"]] = obj
            return obj

    class _FakeInteractionsAPI:
        def __init__(self, responses: list[str]) -> None:
            self._responses = list(responses)

        def create(self, **kwargs: Any) -> Any:
            interaction_calls.append(kwargs)
            text = self._responses.pop(0)
            return SimpleNamespace(
                id=f"int-{len(interaction_calls)}",
                status="completed",
                environment_id=kwargs.get("environment"),
                output_text=text,
            )

    beta_self_hosted = SimpleNamespace(
        agents=_FakeAgentsAPI(),
        interactions=_FakeInteractionsAPI([safe_response]),
    )
    ensure_forge_agent(beta_self_hosted, self_hosted_settings, poll_interval_sec=0.0)
    assert len(agent_create_calls) == 1
    create_kw = agent_create_calls[0]
    assert create_kw["base_environment"] == custom_env_id
    assert "sources" not in json.dumps(create_kw)
    assert "allowlist" not in json.dumps(create_kw)
    assert "max_total_tokens" not in json.dumps(create_kw)
    assert len(interaction_calls) == 0, "Self-hosted GEMINI_ENVIRONMENT_ID must skip hosted bootstrap"

    forge_skill(
        beta_self_hosted,
        self_hosted_settings,
        _sample_pose_sequence(),
        "wave",
        poll_interval_sec=0.0,
    )
    assert len(interaction_calls) == 1
    inter_kw = interaction_calls[0]
    assert inter_kw["environment"] == custom_env_id
    assert "sources" not in inter_kw
    assert "allowlist" not in json.dumps(inter_kw)
    assert "network" not in inter_kw
    assert "max_total_tokens" not in json.dumps(inter_kw)
    assert "model" not in inter_kw

    # Verify automatic Neon -> Carbon escalation after 2 failed local kinematics checks
    escalation_calls: list[dict[str, Any]] = []

    class _EscInteractionsAPI:
        def __init__(self, responses: list[str]) -> None:
            self._responses = list(responses)

        def create(self, **kwargs: Any) -> Any:
            escalation_calls.append(kwargs)
            return SimpleNamespace(
                id=f"esc-{len(escalation_calls)}",
                status="completed",
                environment_id="env-fork",
                output_text=self._responses.pop(0),
            )

    beta_esc = SimpleNamespace(
        agents=_FakeAgentsAPI(),
        interactions=_EscInteractionsAPI([bad_response, bad_response, safe_response]),
    )
    esc_res = forge_skill_with_escalation(
        beta_esc,
        self_hosted_settings,
        _sample_pose_sequence(),
        "wave",
        deep=False,
        poll_interval_sec=0.0,
    )
    assert [c["agent"] for c in escalation_calls] == [
        FORGE_AGENT_ID,
        FORGE_AGENT_ID,
        FORGE_DEEP_AGENT_ID,
    ]
    assert esc_res.escalated_to_carbon is True
    assert esc_res.model_used == FORGE_DEEP_MODEL
    assert esc_res.artifact is not None and esc_res.artifact.kinematics.verified

    # Verify --deep routes directly to Carbon on call 1
    deep_calls: list[dict[str, Any]] = []

    class _DeepInteractionsAPI:
        def create(self, **kwargs: Any) -> Any:
            deep_calls.append(kwargs)
            return SimpleNamespace(
                id="deep-1",
                status="completed",
                environment_id="env-fork",
                output_text=safe_response,
            )

    beta_deep = SimpleNamespace(agents=_FakeAgentsAPI(), interactions=_DeepInteractionsAPI())
    deep_res = forge_skill_with_escalation(
        beta_deep,
        self_hosted_settings,
        _sample_pose_sequence(),
        "wave",
        deep=True,
        poll_interval_sec=0.0,
    )
    assert [c["agent"] for c in deep_calls] == [FORGE_DEEP_AGENT_ID]
    assert deep_res.escalated_to_carbon is True
    assert deep_res.model_used == FORGE_DEEP_MODEL


# ---------------------------------------------------------------------------
# Adjustment #4: Decisions API (>= 0.75 conf & safety, "other") & Zero Mocked Live Paths
# ---------------------------------------------------------------------------


def test_decisions_api_gate_invariants() -> None:
    assert DECISION_MODEL == "gemini-decision-1"
    assert DECISION_CONFIDENCE_THRESHOLD == 0.75
    assert DECISION_SAFETY_PROB_THRESHOLD == 0.75

    # Zero "none_of_the_above" anywhere under src/, agent_bundle/, benchmarks/
    offenders: list[str] = []
    for path in [*_iter_python_files(SRC_DIR, BENCHMARKS_DIR), *_iter_bundle_text_files(BUNDLE_DIR)]:
        content = path.read_text(encoding="utf-8")
        if "none_of_the_above" in content.lower():
            offenders.append(_rel(path))
    assert not offenders, (
        "Found forbidden 'none_of_the_above' (must use 'other'): " + ", ".join(offenders)
    )

    for qset in (PRE_EXECUTION_QUESTIONS, POST_RUN_QUESTIONS):
        criteria = qset["route"]["criteria"]
        assert "other" in criteria
        assert "none_of_the_above" not in criteria

    # AST check: decision_gate calls beta_client.decisions.create(model=DECISION_MODEL, ...)
    dg_path = SRC_DIR / "decision_gate.py"
    dg_tree = ast.parse(dg_path.read_text(encoding="utf-8"), filename=str(dg_path))
    for fn_name in ("evaluate_pre_execution", "evaluate_post_run", "evaluate_baseline_skill"):
        fn_node = _get_func_def(dg_tree, fn_name)
        calls = [
            _dotted_call_name(n.func) for n in ast.walk(fn_node) if isinstance(n, ast.Call)
        ]
        assert "beta_client.decisions.create" in calls, (
            f"decision_gate.{fn_name} must call beta_client.decisions.create(...)"
        )

    # Boundary tests: auto-branch requires BOTH confidence >= 0.75 AND is_physically_safe >= 0.75
    def _resp(safe_noul: float, safe_conf: float, route: str, route_conf: float) -> dict[str, Any]:
        return {
            "answers": {
                "is_physically_safe": {"noul": safe_noul, "confidence": safe_conf},
                "needs_lower_body_adaptation": {"noul": 0.10, "confidence": 0.90},
                "move_fidelity": {
                    "score": 2.7,
                    "confidence": 0.90,
                    "legend": {"0": "bad", "3": "great"},
                },
                "route": {
                    "choice": route,
                    "confidence": route_conf,
                    "probabilities": {route: route_conf},
                },
            }
        }

    ok_res = parse_decision_response(_resp(0.75, 0.75, "execute_and_save", 0.75))
    assert ok_res.auto_branch_allowed is True
    assert ok_res.effective_action == "execute_and_save"

    low_conf = parse_decision_response(_resp(0.95, 0.74, "execute_and_save", 0.95))
    assert low_conf.auto_branch_allowed is False
    assert low_conf.effective_action == "escalate_human"

    low_route_conf = parse_decision_response(_resp(0.95, 0.95, "execute_and_save", 0.74))
    assert low_route_conf.auto_branch_allowed is False
    assert low_route_conf.effective_action == "escalate_human"

    low_safety = parse_decision_response(_resp(0.74, 0.95, "execute_and_save", 0.95))
    assert low_safety.auto_branch_allowed is False
    assert low_safety.effective_action in {"repair_in_forge", "refuse_impossible", "escalate_human"}

    other_route = parse_decision_response(_resp(0.99, 0.99, "other", 0.99))
    assert other_route.auto_branch_allowed is False
    assert other_route.effective_action == "escalate_human"

    # SSE telemetry strip HTML must reflect 0.75 threshold, never stale 0.70
    strip_html = _render_strip_html()
    assert "0.70" not in strip_html, (
        "src/roboharness/telemetry_sse.py contains stale '0.70' threshold text (must be 0.75)"
    )


def test_zero_mocked_fallbacks_in_live_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. MimicPipeline(offline_mode=False) with no clients raises RuntimeError immediately
    mock_server = MockTonypiMCPServer(mode="batch")
    with (
        TonypiMCPClient(transport=mock_server.as_httpx_transport()) as mcp,
        pytest.raises(RuntimeError, match="DualGenAIClients|GEMINI_API_KEY|--offline"),
    ):
        MimicPipeline(
            clients=None,
            mcp=mcp,
            settings=RoboHarnessSettings(gemini_api_key=None, skills_dir=tmp_path / "skills"),
            offline_mode=False,
        )

    # 2. benchmarks/eval_5sigma.py (--mode live) with missing API key raises RuntimeError
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        run_benchmark(
            mode="live",
            limit=1,
            settings=RoboHarnessSettings(gemini_api_key=None),
            quiet=True,
        )

    # 3. CLI `roboharness mimic --mock` WITHOUT `--offline` and WITHOUT key exits non-zero
    monkeypatch.setattr("roboharness.config.load_dotenv_if_present", lambda *_a, **_kw: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    dummy_vid = tmp_path / "dummy.mp4"
    dummy_vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    res = CliRunner().invoke(
        cli_app,
        ["mimic", "--video", str(dummy_vid), "--name", "must_fail", "--mock"],
    )
    assert res.exit_code == 1
    assert "GEMINI_API_KEY" in res.output


# ---------------------------------------------------------------------------
# Clean-room boundary & GEMINI.md constitution documentation
# ---------------------------------------------------------------------------


def test_clean_room_boundary_zero_forbidden_imports() -> None:
    py_files = _iter_python_files(SRC_DIR, BUNDLE_DIR, BENCHMARKS_DIR)
    assert py_files
    violations: list[str] = []
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        violations.extend(_find_forbidden_imports(tree, _rel(path)))

    assert not violations, "Clean-room import violations detected:\n" + "\n".join(violations)


def test_gemini_md_exists_and_documents_constitution() -> None:
    assert GEMINI_MD_PATH.is_file(), f"Missing {GEMINI_MD_PATH}"
    text = GEMINI_MD_PATH.read_text(encoding="utf-8")

    required_terms = (
        "Clean-Room",
        "make_v1_client",
        "make_beta_client",
        "v1beta",
        "agentic",
        "Managed Agents",
        "antigravity-preview-09-2026",
        "roboharness-skill-forge",
        "roboharness-skill-forge-carbon",
        "gemini-4-carbon",
        "GEMINI_ENVIRONMENT_ID",
        "max_total_tokens",
        "Decisions API",
        "gemini-decision-1",
        "0.75",
        "Safety Latches",
        "10.9",
        "70.0",
        "30.0",
        "joint_clipping_count",
    )
    missing = [term for term in required_terms if term not in text]
    assert not missing, f"GEMINI.md missing required documentation terms: {missing}"


# ---------------------------------------------------------------------------
# Self-tests proving our AST detectors catch genuine violations
# ---------------------------------------------------------------------------


def test_ast_detectors_catch_synthetic_violations() -> None:
    bad_params_src = """
def f(client, beta_client):
    cfg = {"thinking_level": "medium", "temperature": 0.2}
    client.interactions.create(model="gemini-4-hydrogen", generation_config=cfg, top_p=0.9)
    build_gemini4_generation_config("medium", max_output_tokens=1024)
    beta_client.agents.create(id="x", agent_config={"type": "antigravity", "max_total_tokens": 5000})
"""
    param_violations = _find_forbidden_gemini4_param_violations(
        ast.parse(bad_params_src), "src/roboharness/fake.py"
    )
    assert any("temperature" in v for v in param_violations)
    assert any("top_p" in v for v in param_violations)
    assert any("max_output_tokens" in v for v in param_violations)
    assert any("max_total_tokens" in v for v in param_violations)

    bad_thinking_src = """
def g(client):
    build_gemini4_generation_config("minimal")
    client.interactions.create(generation_config={"thinking_level": "minimal"})
"""
    thinking_violations = _find_invalid_thinking_level_violations(
        ast.parse(bad_thinking_src), "src/roboharness/fake.py"
    )
    assert len(thinking_violations) == 2

    bad_imports_src = """
import streamlit
import tonypi.daemon
from HiwonderSDK import Board
import importlib
importlib.import_module("tonypi")
"""
    import_violations = _find_forbidden_imports(
        ast.parse(bad_imports_src), "src/roboharness/fake.py"
    )
    assert len(import_violations) == 4
