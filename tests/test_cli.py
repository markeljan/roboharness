"""Offline CLI tests for `roboharness` Typer + Rich commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from roboharness.capture import make_synthetic_test_video
from roboharness.cli import app
from roboharness.config import (
    FORGE_AGENT_ID,
    FORGE_DEEP_AGENT_ID,
    FORGE_DEEP_MODEL,
    FORGE_MODEL,
    RoboHarnessSettings,
)
from roboharness.managed_forge import ForgeAgentState
from roboharness.models import DualGenAIClients
from roboharness.pipeline import CheckItem, MimicPipeline, PreflightReport

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_from_repo_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep offline CLI tests hermetic from repo `.env` and ambient API keys."""
    monkeypatch.setattr("roboharness.config.load_dotenv_if_present", lambda *_a, **_kw: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_ENVIRONMENT_ID", raising=False)


@pytest.fixture(scope="module")
def synth_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a single compact synthetic MP4 reused across offline CLI tests."""
    return make_synthetic_test_video(
        tmp_path_factory.mktemp("cli_vid") / "wave.mp4",
        duration_sec=0.4,
    )


def _settings_with_ephemeral_sse(
    skills_dir: Path,
    *,
    gemini_api_key: str | None = None,
    gemini_environment_id: str | None = None,
) -> RoboHarnessSettings:
    return RoboHarnessSettings(
        gemini_api_key=gemini_api_key,
        skills_dir=skills_dir,
        gemini_environment_id=gemini_environment_id,
        sse_host="127.0.0.1",
        sse_port=0,
    )


def test_cli_help_lists_all_commands_and_options() -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for cmd in ("mimic", "run", "list", "check", "forge-setup", "debug-video", "sim-server"):
        assert cmd in res.output

    mimic_help = runner.invoke(app, ["mimic", "--help"])
    assert mimic_help.exit_code == 0
    assert "--deep" in mimic_help.output

    forge_help = runner.invoke(app, ["forge-setup", "--help"])
    assert forge_help.exit_code == 0
    assert "--deep" in forge_help.output

    dbg_help = runner.invoke(app, ["debug-video", "--help"])
    assert dbg_help.exit_code == 0
    for flag in ("--video", "--name", "--expected"):
        assert flag in dbg_help.output

    sim_help = runner.invoke(app, ["sim-server", "--help"])
    assert sim_help.exit_code == 0
    for flag in ("--host", "--port", "--viewer", "--friction"):
        assert flag in sim_help.output


def test_cli_list_empty_then_populated_via_mimic_and_run(
    tmp_path: Path,
    synth_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skills_dir = tmp_path / "skills"
    monkeypatch.setenv("ROBOHARNESS_SKILLS_DIR", str(skills_dir))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    # 1. Empty skill library
    empty_res = runner.invoke(app, ["list"])
    assert empty_res.exit_code == 0
    assert "No saved skills found" in empty_res.output

    # 2. Learn & execute skill from synthetic video via explicit --mock --offline
    mimic_res = runner.invoke(
        app,
        [
            "mimic",
            "--video",
            str(synth_video),
            "--name",
            "test_wave",
            "--mock",
            "--offline",
            "--no-spatial",
        ],
    )
    assert mimic_res.exit_code == 0, mimic_res.output
    assert "test_wave" in mimic_res.output
    assert "executed_and_saved" in mimic_res.output
    assert "Kinematics & Safety Verification" in mimic_res.output
    assert "Hardware Execution Trace" in mimic_res.output
    assert (skills_dir / "test_wave.json").is_file()

    # 3. `roboharness list` now displays the saved signed skill
    list_res = runner.invoke(app, ["list"])
    assert list_res.exit_code == 0, list_res.output
    assert "test_wave" in list_res.output
    assert "Peak Tilt" in list_res.output
    assert "SHA-256" in list_res.output

    # 4. Replay saved skill via explicit `roboharness run test_wave --mock --offline`
    run_res = runner.invoke(
        app,
        ["run", "test_wave", "--mock", "--offline", "--no-spatial"],
    )
    assert run_res.exit_code == 0, run_res.output
    assert "test_wave" in run_res.output
    assert "executed_and_saved" in run_res.output

    # 5. Missing saved skill exits cleanly with non-zero code
    missing_res = runner.invoke(app, ["run", "does_not_exist", "--mock", "--offline"])
    assert missing_res.exit_code != 0
    assert "Run skill failed" in missing_res.output


def test_cli_live_mode_fails_loudly_without_api_key_even_with_mock(
    tmp_path: Path,
    synth_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero silent mocked Gemini fallback: --mock without --offline requires GEMINI_API_KEY."""
    skills_dir = tmp_path / "skills"
    monkeypatch.setenv("ROBOHARNESS_SKILLS_DIR", str(skills_dir))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    mimic_res = runner.invoke(
        app,
        ["mimic", "--video", str(synth_video), "--name", "live_must_fail", "--mock"],
    )
    assert mimic_res.exit_code == 1
    assert "GEMINI_API_KEY" in mimic_res.output
    assert "--offline" in mimic_res.output

    run_res = runner.invoke(app, ["run", "live_must_fail", "--mock"])
    assert run_res.exit_code == 1
    assert "GEMINI_API_KEY" in run_res.output


def test_cli_mimic_validation_deep_and_webcam_with_ui(
    tmp_path: Path,
    synth_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skills_dir = tmp_path / "skills"
    monkeypatch.setenv("ROBOHARNESS_SKILLS_DIR", str(skills_dir))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    # Neither --video nor --webcam -> exit 2
    bad_none = runner.invoke(app, ["mimic", "--name", "bad", "--mock", "--offline"])
    assert bad_none.exit_code == 2
    assert "Specify exactly one of --video or --webcam" in bad_none.output

    # Both --video and --webcam -> exit 2
    bad_both = runner.invoke(
        app,
        [
            "mimic",
            "--video",
            str(synth_video),
            "--webcam",
            "3",
            "--name",
            "bad",
            "--mock",
            "--offline",
        ],
    )
    assert bad_both.exit_code == 2

    # Webcam path + --ui + --deep passes deep=True and stops SSE cleanly
    orig_mimic_video = MimicPipeline.mimic_video
    captured_deep: list[bool] = []

    def _spy_mimic_video(self, *args, **kwargs):
        captured_deep.append(bool(kwargs.get("deep", False)))
        return orig_mimic_video(self, *args, **kwargs)

    with (
        patch("roboharness.cli.record_webcam_clip", return_value=synth_video) as mock_rec,
        patch(
            "roboharness.cli.RoboHarnessSettings.from_env",
            return_value=_settings_with_ephemeral_sse(skills_dir),
        ),
        patch.object(MimicPipeline, "mimic_video", new=_spy_mimic_video),
    ):
        res = runner.invoke(
            app,
            [
                "mimic",
                "--webcam",
                "2",
                "--device",
                "0",
                "--name",
                "cam_wave",
                "--dry-run",
                "--ui",
                "--deep",
                "--mock",
                "--offline",
            ],
        )
    assert res.exit_code == 0, res.output
    mock_rec.assert_called_once_with(duration_sec=2, device="0")
    assert captured_deep == [True]
    assert "Live SSE telemetry strip:" in res.output
    assert "dry_run_saved" in res.output


def test_cli_check_shows_dual_clients_and_env_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_ENVIRONMENT_ID", "env-docker-mujoco-99")

    ok_report = PreflightReport(
        all_ok=True,
        items=[
            CheckItem(component="gemini-4-hydrogen", ok=True, detail="pong (v1)", latency_ms=120.0),
            CheckItem(
                component="roboharness-skill-forge",
                ok=True,
                detail="ready (v1beta)",
                latency_ms=90.0,
            ),
            CheckItem(
                component="gemini-decision-1",
                ok=True,
                detail="noul=0.95 (v1beta)",
                latency_ms=80.0,
            ),
            CheckItem(
                component="gemini-robotics-er-2-preview",
                ok=True,
                detail="centered (v1)",
                latency_ms=140.0,
            ),
            CheckItem(component="tonypi_mcp", ok=True, detail="11.6V OK", latency_ms=5.0),
        ],
    )
    with patch("roboharness.cli.run_preflight_check", return_value=ok_report) as mock_check:
        res_ok = runner.invoke(app, ["check", "--mock", "--skip-forge-create"])
    assert res_ok.exit_code == 0, res_ok.output
    assert mock_check.call_args.kwargs["clients"] is None
    assert "v1" in res_ok.output and "v1beta" in res_ok.output
    assert "env-docker-mujoco-99" in res_ok.output
    assert "PASS" in res_ok.output

    fail_report = PreflightReport(
        all_ok=False,
        items=[
            CheckItem(component="gemini-4-hydrogen", ok=False, detail="Missing key", latency_ms=0.0),
            CheckItem(component="tonypi_mcp", ok=True, detail="11.6V OK", latency_ms=4.0),
        ],
    )
    with patch("roboharness.cli.run_preflight_check", return_value=fail_report):
        res_fail = runner.invoke(app, ["check", "--mock"])
    assert res_fail.exit_code == 1
    assert "FAIL" in res_fail.output


def test_cli_forge_setup_neon_and_deep_carbon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_ENVIRONMENT_ID", raising=False)

    no_key = runner.invoke(app, ["forge-setup"])
    assert no_key.exit_code == 1
    assert "GEMINI_API_KEY" in no_key.output

    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")
    v1_stub = object()
    beta_stub = object()
    dual_stub = DualGenAIClients(client=v1_stub, beta_client=beta_stub)

    def _fake_ensure(client, settings, *, force_recreate=False, deep=False):
        assert client is beta_stub
        if deep:
            return ForgeAgentState(
                agent_id=FORGE_DEEP_AGENT_ID,
                base_environment_id="env-fork-carbon",
                model=FORGE_DEEP_MODEL,
                created_now=True,
            )
        return ForgeAgentState(
            agent_id=FORGE_AGENT_ID,
            base_environment_id="env-fork-neon",
            model=FORGE_MODEL,
            created_now=True,
        )

    with (
        patch("roboharness.cli.make_dual_clients", return_value=dual_stub),
        patch("roboharness.cli.ensure_forge_agent", side_effect=_fake_ensure) as mock_ensure,
    ):
        res = runner.invoke(app, ["forge-setup", "--force", "--deep"])

    assert res.exit_code == 0, res.output
    assert mock_ensure.call_count == 2
    assert mock_ensure.call_args_list[0].kwargs == {"force_recreate": True, "deep": False}
    assert mock_ensure.call_args_list[1].kwargs == {"force_recreate": True, "deep": True}
    assert FORGE_AGENT_ID in res.output
    assert FORGE_MODEL in res.output
    assert FORGE_DEEP_AGENT_ID in res.output
    assert FORGE_DEEP_MODEL in res.output
    assert 'environment="remote"' in res.output


def test_cli_debug_video_uses_stable_v1_client_agentic(
    synth_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    no_key = runner.invoke(
        app,
        ["debug-video", "--video", str(synth_video), "--name", "floss"],
    )
    assert no_key.exit_code == 1
    assert "GEMINI_API_KEY" in no_key.output

    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")
    v1_stub = object()
    beta_stub = object()
    dual_stub = DualGenAIClients(client=v1_stub, beta_client=beta_stub)
    fake_diag = {
        "skill_name": "floss",
        "root_cause": "Right ankle slip at 00:01.4",
        "recommended_fix": "Reduce hip_roll amplitude by 15%",
    }

    with (
        patch("roboharness.cli.make_dual_clients", return_value=dual_stub),
        patch("roboharness.cli.debug_trial_video", return_value=fake_diag) as mock_dbg,
    ):
        res = runner.invoke(
            app,
            [
                "debug-video",
                "--video",
                str(synth_video),
                "--name",
                "floss",
                "--expected",
                "Upright torso with smooth arm swing",
            ],
        )

    assert res.exit_code == 0, res.output
    assert mock_dbg.call_args.args[0] is v1_stub
    assert mock_dbg.call_args.args[1] == synth_video
    assert mock_dbg.call_args.kwargs == {
        "skill_name": "floss",
        "expected_behavior": "Upright torso with smooth arm swing",
        "thinking_level": "high",
    }
    assert 'processing="agentic"' in res.output
    assert "Right ankle slip at 00:01.4" in res.output
