"""Typer + Rich CLI for RoboHarness (`roboharness`).

Provides judge-ready commands for learning skills from live webcam or video clips,
replaying signed skills, inspecting the local skill library, preflight checking all
4 Gemini 4 models + `TONYPI_MCP_URL`, and pre-staging the Managed Agent Skill Forge.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from roboharness.capture import CaptureError, record_webcam_clip
from roboharness.config import RoboHarnessSettings
from roboharness.genai_client import make_client
from roboharness.managed_forge import ensure_forge_agent
from roboharness.mcp_client import MCPClientError, TonypiMCPClient
from roboharness.mock_mcp import MockTonypiMCPServer
from roboharness.pipeline import MimicPipeline, MimicResult, PreflightReport, run_preflight_check
from roboharness.skill_store import list_skills
from roboharness.telemetry_sse import TelemetryHub, TelemetrySSEServer

app = typer.Typer(
    name="roboharness",
    help="Gemini 4 native hardware agent harness for the Hiwonder TonyPi humanoid robot.",
    no_args_is_help=True,
    add_completion=False,
)

_STATUS_STYLES: dict[str, str] = {
    "executed_and_saved": "bold green",
    "dry_run_saved": "bold cyan",
    "refused": "bold yellow",
    "escalated_human": "bold magenta",
    "safety_aborted": "bold red",
}


def _console() -> Console:
    """Create a fresh Rich Console bound to current stdout for clean CLI & test capture."""
    if sys.stdout.isatty():
        return Console(highlight=False)
    width = max(120, shutil.get_terminal_size((120, 40)).columns)
    return Console(highlight=False, width=width)


def _build_mcp_client(
    settings: RoboHarnessSettings,
    *,
    mock: bool,
) -> tuple[TonypiMCPClient, MockTonypiMCPServer | None]:
    if mock:
        mock_server = MockTonypiMCPServer(mode="batch")
        return (
            TonypiMCPClient("http://mock.local/mcp", transport=mock_server.as_httpx_transport()),
            mock_server,
        )
    return TonypiMCPClient(settings.tonypi_mcp_url), None


def _render_mimic_result(console: Console, result: MimicResult) -> None:
    """Render a compact, judge-friendly Rich summary of a MimicPipeline run."""
    style = _STATUS_STYLES.get(result.status, "bold white")
    header_lines = [
        f"[bold]Skill:[/bold] {result.skill_name}",
        f"[bold]Status:[/bold] [{style}]{result.status}[/{style}]",
    ]
    if result.reason:
        header_lines.append(f"[bold]Reason:[/bold] {result.reason}")
    if result.saved_path:
        header_lines.append(f"[bold]Saved Path:[/bold] {result.saved_path}")
    if result.forge_attempts:
        header_lines.append(f"[bold]Forge Attempts:[/bold] {result.forge_attempts}")
    if result.announcement:
        header_lines.append(f'[bold]Robot Announcement:[/bold] "{result.announcement}"')

    console.print(Panel("\n".join(header_lines), title="RoboHarness Result", border_style="cyan"))

    if result.spatial is not None:
        sp = result.spatial
        sp_table = Table(title="Spatial Workspace Framing (gemini-robotics-er-2-preview)", box=None)
        sp_table.add_column("Field", style="bold dim")
        sp_table.add_column("Value")
        sp_table.add_row("Centered on Table", str(sp.centered_on_table))
        sp_table.add_row("Workspace Clear", str(sp.workspace_clear))
        sp_table.add_row("Clearance Margin", sp.clearance_margin)
        sp_table.add_row("Summary", sp.summary)
        console.print(sp_table)

    if result.artifact is not None:
        art = result.artifact
        kin = art.kinematics
        kin_table = Table(title="Kinematics & Safety Verification", show_header=True)
        kin_table.add_column("Verified")
        kin_table.add_column("Adaptation Mode")
        kin_table.add_column("Keyframes", justify="right")
        kin_table.add_column("Peak Tilt (deg)", justify="right")
        kin_table.add_column("CoM Offset", justify="right")
        kin_table.add_column("Clipping", justify="right")
        kin_table.add_column("Max Slew (pps)", justify="right")
        kin_table.add_column("SHA-256")
        kin_table.add_row(
            "[green]YES[/green]" if kin.verified else "[red]NO[/red]",
            art.adaptation_mode,
            str(len(art.keyframes)),
            f"{kin.peak_torso_tilt_deg:.1f}",
            f"{kin.max_com_offset_norm:.2f}",
            str(kin.joint_clipping_count),
            f"{kin.max_slew_rate_pps:.1f}",
            art.artifact_sha256[:12] or "--",
        )
        console.print(kin_table)

    if result.pre_decision is not None or result.post_decision is not None:
        dec_table = Table(title="Decisions API Gate (gemini-decision-1, 0 output tokens)")
        dec_table.add_column("Phase")
        dec_table.add_column("Safe Prob", justify="right")
        dec_table.add_column("Adapt Prob", justify="right")
        dec_table.add_column("Fidelity (0..3)", justify="right")
        dec_table.add_column("Route")
        dec_table.add_column("Effective Action")
        dec_table.add_column("Min Conf", justify="right")
        for label, dec in (
            ("Pre-Run", result.pre_decision),
            ("Post-Run", result.post_decision),
        ):
            if dec is None:
                continue
            dec_table.add_row(
                label,
                f"{dec.is_physically_safe_prob:.2f}",
                f"{dec.needs_lower_body_adaptation_prob:.2f}",
                f"{dec.move_fidelity_score:.2f}",
                dec.route,
                dec.effective_action,
                f"{dec.min_confidence:.2f}",
            )
        console.print(dec_table)

    if result.execution is not None:
        ex = result.execution
        st = ex.final_status
        ex_table = Table(title="Hardware Execution Trace")
        ex_table.add_column("Transport")
        ex_table.add_column("Frames", justify="right")
        ex_table.add_column("Aborted")
        ex_table.add_column("Peak Tilt (deg)", justify="right")
        ex_table.add_column("Battery (V)", justify="right")
        ex_table.add_column("Max Temp (C)", justify="right")
        ex_table.add_column("Abort Reason")
        ex_table.add_row(
            ex.transport_mode,
            str(ex.frames_executed),
            "[red]YES[/red]" if ex.aborted else "[green]NO[/green]",
            f"{ex.peak_observed_tilt_deg:.1f}",
            f"{st.battery_v:.2f}",
            f"{st.max_servo_temp_c:.1f}",
            ex.abort_reason or "--",
        )
        console.print(ex_table)


def _render_preflight_report(console: Console, report: PreflightReport) -> None:
    """Render a Rich preflight diagnostics table."""
    table = Table(title="RoboHarness Preflight Check")
    table.add_column("Component", style="bold")
    table.add_column("Status")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Detail")
    for item in report.items:
        badge = "[bold green]PASS[/bold green]" if item.ok else "[bold red]FAIL[/bold red]"
        table.add_row(item.component, badge, f"{item.latency_ms:.0f}", item.detail)
    console.print(table)


@app.command()
def mimic(
    name: Annotated[str, typer.Option("--name", help="Skill name to save in skills/<slug>.json.")],
    video: Annotated[
        Path | None,
        typer.Option("--video", help="Path to an MP4/MOV video clip (exclusive with --webcam)."),
    ] = None,
    webcam: Annotated[
        int | None,
        typer.Option(
            "--webcam",
            min=1,
            help="Record duration in seconds from webcam (exclusive with --video).",
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="macOS avfoundation camera device index (default: '0')."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run/--no-dry-run", help="Verify and save skill without moving servos."),
    ] = False,
    spatial: Annotated[
        bool,
        typer.Option(
            "--spatial/--no-spatial",
            help="Run gemini-robotics-er-2-preview spatial table framing check.",
        ),
    ] = True,
    ui: Annotated[
        bool,
        typer.Option(
            "--ui/--no-ui",
            help="Serve minimal live SSE telemetry strip on 127.0.0.1:8770.",
        ),
    ] = False,
    mock: Annotated[
        bool,
        typer.Option(
            "--mock/--no-mock",
            help="Run against in-process MockTonypiMCPServer instead of physical TonyPi.",
        ),
    ] = False,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline/--no-offline",
            help="Skip live Gemini API calls and use deterministic offline mode.",
        ),
    ] = False,
    remote_mcp: Annotated[
        bool,
        typer.Option(
            "--remote-mcp/--no-remote-mcp",
            help="Expose TONYPI_REMOTE_MCP_URL to the Managed Agent sandbox.",
        ),
    ] = False,
    speed_scale: Annotated[
        float,
        typer.Option("--speed-scale", min=0.1, help="Playback speed multiplier for execution."),
    ] = 1.0,
) -> None:
    """Learn a verified 16-DOF TonyPi skill from a webcam clip or video and execute it."""
    console = _console()
    if (video is None and webcam is None) or (video is not None and webcam is not None):
        console.print("[bold red]Error:[/bold red] Specify exactly one of --video or --webcam.")
        raise typer.Exit(code=2)

    settings = RoboHarnessSettings.from_env()
    resolved_device = device if device is not None else settings.webcam_device
    use_offline = offline or (mock and not settings.gemini_api_key)

    client = None
    if not use_offline:
        if not settings.gemini_api_key:
            console.print(
                "[bold red]Error:[/bold red] GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. "
                "Export an API key or pass --offline / --mock."
            )
            raise typer.Exit(code=1)
        client = make_client(settings.gemini_api_key)

    video_path: Path
    if webcam is not None:
        console.print(
            f"[cyan]Recording {webcam}s webcam clip from device '{resolved_device}'...[/cyan]"
        )
        try:
            video_path = record_webcam_clip(duration_sec=webcam, device=resolved_device)
        except CaptureError as exc:
            console.print(f"[bold red]Webcam capture failed:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc
    else:
        assert video is not None
        if not video.is_file():
            console.print(f"[bold red]Error:[/bold red] Video file not found: {video}")
            raise typer.Exit(code=2)
        video_path = video

    hub: TelemetryHub | None = None
    sse_server: TelemetrySSEServer | None = None
    if ui:
        hub = TelemetryHub()
        sse_server = TelemetrySSEServer(hub, host=settings.sse_host, port=settings.sse_port)
        ui_url = sse_server.start()
        console.print(f"[bold cyan]Live SSE telemetry strip:[/bold cyan] {ui_url}")

    mcp_client, _mock_server = _build_mcp_client(settings, mock=mock)
    try:
        with mcp_client:
            pipeline = MimicPipeline(
                client=client,
                mcp=mcp_client,
                settings=settings,
                hub=hub,
                offline_mode=use_offline,
            )
            result = pipeline.mimic_video(
                video_path,
                skill_name=name,
                dry_run=dry_run,
                run_spatial=spatial,
                use_remote_mcp=remote_mcp,
                speed_scale=speed_scale,
            )
    except (MCPClientError, CaptureError, ValueError, FileNotFoundError) as exc:
        console.print(f"[bold red]Mimic pipeline error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        if sse_server is not None:
            sse_server.stop()

    _render_mimic_result(console, result)
    if result.status == "safety_aborted":
        raise typer.Exit(code=1)


@app.command()
def run(
    skill_name: Annotated[str, typer.Argument(help="Saved skill name or slug to execute.")],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run/--no-dry-run", help="Verify telemetry & signature without moving."),
    ] = False,
    spatial: Annotated[
        bool,
        typer.Option(
            "--spatial/--no-spatial",
            help="Run gemini-robotics-er-2-preview spatial table framing check.",
        ),
    ] = True,
    ui: Annotated[
        bool,
        typer.Option(
            "--ui/--no-ui",
            help="Serve minimal live SSE telemetry strip on 127.0.0.1:8770.",
        ),
    ] = False,
    mock: Annotated[
        bool,
        typer.Option(
            "--mock/--no-mock",
            help="Run against in-process MockTonypiMCPServer instead of physical TonyPi.",
        ),
    ] = False,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline/--no-offline",
            help="Skip live Gemini API calls (spatial/announcement/post-run gate).",
        ),
    ] = False,
    speed_scale: Annotated[
        float,
        typer.Option("--speed-scale", min=0.1, help="Playback speed multiplier."),
    ] = 1.0,
) -> None:
    """Load a SHA-256 signed skill from skills/ and execute it on TonyPi."""
    console = _console()
    settings = RoboHarnessSettings.from_env()
    use_offline = offline or (mock and not settings.gemini_api_key)

    client = None
    if not use_offline and settings.gemini_api_key:
        client = make_client(settings.gemini_api_key)

    hub: TelemetryHub | None = None
    sse_server: TelemetrySSEServer | None = None
    if ui:
        hub = TelemetryHub()
        sse_server = TelemetrySSEServer(hub, host=settings.sse_host, port=settings.sse_port)
        ui_url = sse_server.start()
        console.print(f"[bold cyan]Live SSE telemetry strip:[/bold cyan] {ui_url}")

    mcp_client, _mock_server = _build_mcp_client(settings, mock=mock)
    try:
        with mcp_client:
            pipeline = MimicPipeline(
                client=client,
                mcp=mcp_client,
                settings=settings,
                hub=hub,
                offline_mode=use_offline,
            )
            result = pipeline.run_saved_skill(
                skill_name,
                dry_run=dry_run,
                run_spatial=spatial,
                speed_scale=speed_scale,
            )
    except (FileNotFoundError, ValueError, MCPClientError) as exc:
        console.print(f"[bold red]Run skill failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        if sse_server is not None:
            sse_server.stop()

    _render_mimic_result(console, result)
    if result.status == "safety_aborted":
        raise typer.Exit(code=1)


@app.command(name="list")
def list_cmd() -> None:
    """List verified, SHA-256 signed skills stored in the local skill library."""
    console = _console()
    settings = RoboHarnessSettings.from_env()
    skills = list_skills(settings.skills_dir)

    if not skills:
        console.print(
            f"[dim]No saved skills found in '{settings.skills_dir}'. "
            "Run `roboharness mimic --video <path> --name <skill>` to create one.[/dim]"
        )
        return

    table = Table(title=f"Saved TonyPi Skills ({settings.skills_dir})")
    table.add_column("Name", style="bold cyan", no_wrap=True)
    table.add_column("Mode", no_wrap=True)
    table.add_column("Keyframes", justify="right")
    table.add_column("Peak Tilt (deg)", justify="right")
    table.add_column("Clipping", justify="right")
    table.add_column("Fidelity (0..3)", justify="right")
    table.add_column("SHA-256", no_wrap=True)
    table.add_column("Created", no_wrap=True)

    for art in skills:
        fidelity_val = art.post_run_fidelity_score
        if fidelity_val is None and art.decision_gate is not None:
            fidelity_val = art.decision_gate.move_fidelity_score
        fidelity_str = f"{fidelity_val:.2f}" if fidelity_val is not None else "--"
        table.add_row(
            art.name,
            art.adaptation_mode,
            str(len(art.keyframes)),
            f"{art.kinematics.peak_torso_tilt_deg:.1f}",
            str(art.kinematics.joint_clipping_count),
            fidelity_str,
            art.artifact_sha256[:12] or "--",
            art.created_at[:19],
        )

    console.print(table)


@app.command()
def check(
    skip_forge_create: Annotated[
        bool,
        typer.Option(
            "--skip-forge-create/--no-skip-forge-create",
            help="Check for existing Managed Agent without creating/bootstrapping if absent.",
        ),
    ] = False,
    mock: Annotated[
        bool,
        typer.Option(
            "--mock/--no-mock",
            help="Check MCP connectivity against in-process MockTonypiMCPServer.",
        ),
    ] = False,
) -> None:
    """Run live preflight verification across all 4 Gemini 4 models and TONYPI_MCP_URL."""
    console = _console()
    settings = RoboHarnessSettings.from_env()
    client = make_client(settings.gemini_api_key) if settings.gemini_api_key else None
    mcp_client, _mock_server = _build_mcp_client(settings, mock=mock)

    with mcp_client:
        report = run_preflight_check(
            client,
            mcp_client,
            settings,
            skip_forge_create=skip_forge_create,
        )

    _render_preflight_report(console, report)
    if not report.all_ok:
        raise typer.Exit(code=1)


@app.command(name="forge-setup")
def forge_setup(
    force: Annotated[
        bool,
        typer.Option(
            "--force/--no-force",
            help="Delete and recreate the Managed Agent and forked sandbox environment.",
        ),
    ] = False,
) -> None:
    """Create or verify the Managed Agent Skill Forge and pre-staged sandbox environment."""
    console = _console()
    settings = RoboHarnessSettings.from_env()
    if not settings.gemini_api_key:
        console.print(
            "[bold red]Error:[/bold red] GEMINI_API_KEY (or GOOGLE_API_KEY) is required "
            "for `roboharness forge-setup`."
        )
        raise typer.Exit(code=1)

    client = make_client(settings.gemini_api_key)
    try:
        state = ensure_forge_agent(client, settings, force_recreate=force)
    except (ValueError, RuntimeError, OSError) as exc:
        console.print(f"[bold red]Forge setup failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    action = "Created & pre-staged" if state.created_now else "Reused existing"
    summary = (
        f"[bold]Status:[/bold] [green]{action}[/green]\n"
        f"[bold]Agent ID:[/bold] {state.agent_id}\n"
        f"[bold]Model:[/bold] {state.model}\n"
        f"[bold]Base Environment ID:[/bold] {state.base_environment_id or '--'}"
    )
    console.print(Panel(summary, title="Managed Agent Skill Forge", border_style="green"))


if __name__ == "__main__":
    app()
