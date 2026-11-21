"""Static and AST invariant tests enforcing GEMINI.md constitution across RoboHarness."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from roboharness.config import (
    API_VERSION,
    FORBIDDEN_GEMINI4_PARAMS,
    FORGE_AGENT_ID,
    FORGE_BASE_AGENT,
    FORGE_MODEL,
    REMOTE_MCP_NAME,
    RESERVED_AGENT_ID_PREFIXES,
    VALID_THINKING_LEVELS,
    RoboHarnessSettings,
)
from roboharness.decision_gate import POST_RUN_QUESTIONS, PRE_EXECUTION_QUESTIONS
from roboharness.genai_client import build_gemini4_generation_config

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src" / "roboharness"
BUNDLE_DIR = REPO_ROOT / "agent_bundle"
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
    }
)

# Only config.py (constant list) and genai_client.py (runtime validation guard)
# are exempt from raw dict/constant mention checks; actual SDK calls are still scanned.
FORBIDDEN_PARAM_GUARD_EXEMPT_FILES: frozenset[str] = frozenset(
    {
        "src/roboharness/config.py",
        "src/roboharness/genai_client.py",
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
        name.endswith(("interactions.create", "models.generate_content", ".generate_content"))
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


# ---------------------------------------------------------------------------
# Codebase-wide invariant tests
# ---------------------------------------------------------------------------


def test_zero_forbidden_gemini4_params_in_codebase() -> None:
    py_files = _iter_python_files(SRC_DIR, BUNDLE_DIR)
    assert py_files, "Expected Python files under src/roboharness and agent_bundle"

    violations: list[str] = []
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        violations.extend(_find_forbidden_gemini4_param_violations(tree, _rel(path)))

    assert not violations, "Forbidden Gemini 4 params detected:\n" + "\n".join(violations)

    # Runtime guard verification on build_gemini4_generation_config
    for bad_param in sorted(FORBIDDEN_PARAM_NAMES):
        with pytest.raises(ValueError, match="Forbidden Gemini 4"):
            build_gemini4_generation_config("medium", **{bad_param: 0.5})


def test_zero_minimal_thinking_level_in_codebase() -> None:
    assert "minimal" not in VALID_THINKING_LEVELS
    assert VALID_THINKING_LEVELS == frozenset({"low", "medium", "high", "xhigh"})

    with pytest.raises(ValueError, match="Invalid thinking_level"):
        build_gemini4_generation_config("minimal")

    py_files = _iter_python_files(SRC_DIR, BUNDLE_DIR)
    violations: list[str] = []
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        violations.extend(_find_invalid_thinking_level_violations(tree, _rel(path)))

    assert not violations, "Invalid thinking_level values detected:\n" + "\n".join(violations)


def test_zero_none_of_the_above_in_src() -> None:
    py_files = _iter_python_files(SRC_DIR, BUNDLE_DIR)
    offenders: list[str] = []
    for path in py_files:
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


def test_forge_agent_id_never_uses_reserved_prefix() -> None:
    reserved = ("gemini-", "google-", "antigravity-", "deep-research-", "gemma-")
    assert tuple(RESERVED_AGENT_ID_PREFIXES) == reserved
    assert API_VERSION == "v1beta"
    assert FORGE_BASE_AGENT == "antigravity-preview-09-2026"
    assert FORGE_MODEL == "gemini-4-neon"
    assert not FORGE_AGENT_ID.startswith(reserved)
    assert not RoboHarnessSettings().managed_agent_id.startswith(reserved)

    # Verify any literal id= passed to agents.create(...) avoids reserved prefixes
    for path in _iter_python_files(SRC_DIR):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _dotted_call_name(node.func).endswith(
                "agents.create"
            ):
                for kw in node.keywords:
                    if kw.arg == "id":
                        val = _str_literal(kw.value)
                        if val is not None:
                            assert not val.startswith(reserved), (
                                f"{_rel(path)}:{kw.lineno} reserved agent id prefix: {val!r}"
                            )


def test_remote_mcp_name_matches_regex() -> None:
    assert MCP_NAME_RE.fullmatch(REMOTE_MCP_NAME), (
        f"REMOTE_MCP_NAME={REMOTE_MCP_NAME!r} must match ^[a-z0-9_-]+$"
    )

    for path in _iter_python_files(SRC_DIR):
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


def test_clean_room_boundary_zero_forbidden_imports() -> None:
    py_files = _iter_python_files(SRC_DIR, BUNDLE_DIR)
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
        "v1beta",
        "Managed Agents",
        "antigravity-preview-09-2026",
        "roboharness-skill-forge",
        "Decisions API",
        "gemini-decision-1",
        "Safety Latches",
        "10.9",
        "70.0",
        "30.0",
        "joint_clipping_count",
    )
    missing = [term for term in required_terms if term not in text]
    assert not missing, f"GEMINI.md missing required documentation terms: {missing}"
    assert FORBIDDEN_GEMINI4_PARAMS.issuperset({"temperature", "top_p", "top_k", "cachedContents"})


# ---------------------------------------------------------------------------
# Self-tests proving our AST detectors catch genuine violations
# ---------------------------------------------------------------------------


def test_ast_detectors_catch_synthetic_violations() -> None:
    bad_params_src = """
def f(client):
    cfg = {"thinking_level": "medium", "temperature": 0.2}
    client.interactions.create(model="gemini-4-hydrogen", generation_config=cfg, top_p=0.9)
    build_gemini4_generation_config("medium", max_output_tokens=1024)
"""
    param_violations = _find_forbidden_gemini4_param_violations(
        ast.parse(bad_params_src), "src/roboharness/fake.py"
    )
    assert any("temperature" in v for v in param_violations)
    assert any("top_p" in v for v in param_violations)
    assert any("max_output_tokens" in v for v in param_violations)

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
