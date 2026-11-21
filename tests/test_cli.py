"""Offline CLI tests for `roboharness` Typer + Rich commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from roboharness.capture import make_synthetic_test_video
from roboharness.cli import app
from roboharness.config import FORGE_AGENT_ID, FORGE_MODEL
from roboharness.managed_forge import ForgeAgentState
from roboharness.pipeline import CheckItem, PreflightReport

runner = CliRunner()


def test_cli_help_lists_all_commands() -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for cmd in ("mimic", "run", "list", "check", "forge-setup"):
        assert cmd in res.output


def test_cli_list_empty_then_populated_via_mimic_and_run(
    tmp_path: Path,
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

    # 2. Learn & execute skill from synthetic video via --mock --offline
    video_path = make_synthetic_test_video(tmp_path / "wave.mp4", duration_sec=0.8)
    mimic_res = runner.invoke(
        app,
        [
            "mimic",
            "--video",
            str(video_path),
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

    # 4. Replay saved skill via `roboharness run test_wave --mock --offline`
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


def test_cli_mimic_validation_and_webcam_with_ui(
    tmp_path: Path,
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
    vid = tmp_path / "dummy.mp4"
    vid.write_bytes(b"00")
    bad_both = runner.invoke(
        app,
        ["mimic", "--video", str(vid), "--webcam", "3", "--name", "bad", "--mock", "--offline"],
    )
    assert bad_both.exit_code == 2

    # Webcam path + --ui starts and stops TelemetrySSEServer cleanly on ephemeral port
    synth = make_synthetic_test_video(tmp_path / "cam.mp4", duration_sec=0.6)
    with (
        patch("roboharness.cli.record_webcam_clip", return_value=synth) as mock_rec,
        patch(
            "roboharness.cli.RoboHarnessSettings.from_env",
            return_value=_settings_with_ephemeral_sse(skills_dir),
        ),
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
                "--mock",
                "--offline",
            ],
        )
    assert res.exit_code == 0, res.output
    mock_rec.assert_called_once_with(duration_sec=2, device="0")
    assert "Live SSE telemetry strip:" in res.output
    assert "dry_run_saved" in res.output


def _settings_with_ephemeral_sse(skills_dir: Path):
    from roboharness.config import RoboHarnessSettings

    return RoboHarnessSettings(
        gemini_api_key=None,
        skills_dir=skills_dir,
        sse_host="127.0.0.1",
        sse_port=0,
    )


def test_cli_check_pass_and_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    ok_report = PreflightReport(
        all_ok=True,
        items=[
            CheckItem(component="gemini-4-hydrogen", ok=True, detail="pong", latency_ms=120.0),
            CheckItem(component="roboharness-skill-forge", ok=True, detail="ready", latency_ms=90.0),
            CheckItem(component="gemini-decision-1", ok=True, detail="noul=0.95", latency_ms=80.0),
            CheckItem(
                component="gemini-robotics-er-2-preview", ok=True, detail="centered", latency_ms=140.0
            ),
            CheckItem(component="tonypi_mcp", ok=True, detail="11.6V OK", latency_ms=5.0),
        ],
    )
    with patch("roboharness.cli.run_preflight_check", return_value=ok_report):
        res_ok = runner.invoke(app, ["check", "--mock", "--skip-forge-create"])
    assert res_ok.exit_code == 0, res_ok.output
    assert "PASS" in res_ok.output
    assert "gemini-4-hydrogen" in res_ok.output
    assert "tonypi_mcp" in res_ok.output

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


def test_cli_forge_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    no_key = runner.invoke(app, ["forge-setup"])
    assert no_key.exit_code == 1
    assert "GEMINI_API_KEY" in no_key.output

    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")
    fake_state = ForgeAgentState(
        agent_id=FORGE_AGENT_ID,
        base_environment_id="env-fork-12345",
        model=FORGE_MODEL,
        created_now=True,
    )
    with (
        patch("roboharness.cli.make_client", return_value=object()),
        patch("roboharness.cli.ensure_forge_agent", return_value=fake_state) as mock_ensure,
    ):
        res = runner.invoke(app, ["forge-setup", "--force"])

    assert res.exit_code == 0, res.output
    assert mock_ensure.call_args.kwargs["force_recreate"] is True
    assert FORGE_AGENT_ID in res.output
    assert FORGE_MODEL in res.output
    assert "env-fork-12345" in res.output
