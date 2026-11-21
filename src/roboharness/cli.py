"""Typer + Rich CLI for RoboHarness (`roboharness`).

Provides judge-ready commands for learning skills from live webcam or video clips,
replaying signed skills, inspecting the local skill library, preflight checking the
dual-client (`v1` + `v1beta`) Gemini 4 stack + `TONYPI_MCP_URL`, pre-staging the
Managed Agent Skill Forge (`gemini-4-neon` + `gemini-4-carbon`), and debugging robot
trial videos with Agentic Video Processing (`processing="agentic"`).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from roboharness.capture import CaptureError, record_webcam_clip
from roboharness.config import (
    BETA_API_VERSION,
    DECISION_CONFIDENCE_THRESHOLD,
    DECISION_SAFETY_PROB_THRESHOLD,
    STABLE_API_VERSION,
    RoboHarnessSettings,
)
from roboharness.managed_forge import ForgeAgentState, ensure_forge_agent
from roboharness.mcp_client import MCPClientError, TonypiMCPClient
from roboharness.mock_mcp import MockTonypiMCPServer
from roboharness.models import DualGenAIClients, make_dual_clients
from roboharness.motion_extractor import debug_trial_video
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


def _sandbox_mode_label(settings: RoboHarnessSettings) -> str:
    """Describe whether self-hosted GEMINI_ENVIRONMENT_ID or hosted 'remote' sandbox is active."""
    if settings.gemini_environment_id:
        return (
            f"Self-hosted Docker/MuJoCo "
            f"(GEMINI_ENVIRONMENT_ID={settings.gemini_environment_id})"
        )
    return 'Hosted remote sandbox (environment="remote")'


def _client_split_label() -> str:
    """Describe the Builder Guide dual-client split between v1 and v1beta."""
    return (
        f"Dual Clients: {STABLE_API_VERSION} (GA Interactions: hydrogen, debug-video) | "
        f"{BETA_API_VERSION} (Managed Agents, Environments, Decisions, spatial preview)"
    )


def _require_live_clients(
    console: Console,
    settings: RoboHarnessSettings,
    *,
    command_hint: str,
    allow_offline_hint: bool = True,
) -> DualGenAIClients:
    """Instantiate live DualGenAIClients or exit loudly with zero mocked fallback."""
    if not settings.gemini_api_key:
        offline_note = (
            " Pass `--offline` explicitly for deterministic offline CI "
            "(`--mock` only mocks TonyPi MCP hardware)."
            if allow_offline_hint
            else ""
        )
        console.print(
            f"[bold red]Error:[/bold red] GEMINI_API_KEY (or GOOGLE_API_KEY) is required "
            f"for live `roboharness {command_hint}`.{offline_note}"
        )
        raise typer.Exit(code=1)
    return make_dual_clients(settings.gemini_api_key)


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
        model_tag = f" ({result.forge_model_used})" if result.forge_model_used else ""
        esc_tag = " [yellow](escalated to gemini-4-carbon)[/yellow]" if result.escalated_to_carbon else ""
        header_lines.append(
            f"[bold]Forge Attempts:[/bold] {result.forge_attempts}{model_tag}{esc_tag}"
        )
    if result.announcement:
        header_lines.append(f'[bold]Robot Announcement:[/bold] "{result.announcement}"')

    console.print(Panel("\n".join(header_lines), title="RoboHarness Result", border_style="cyan"))

    if result.spatial is not None:
        sp = result.spatial
        sp_table = Table(
            title=f"Spatial Workspace Framing (gemini-robotics-er-2-preview, {BETA_API_VERSION})",
            box=None,
        )
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
        dec_title = (
            f"Decisions API Gate (gemini-decision-1, {BETA_API_VERSION}, 0 output tokens, "
            f"conf>={DECISION_CONFIDENCE_THRESHOLD:.2f} & safe>={DECISION_SAFETY_PROB_THRESHOLD:.2f})"
        )
        dec_table = Table(title=dec_title)
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


def _render_preflight_report(
    console: Console,
    report: PreflightReport,
    settings: RoboHarnessSettings,
) -> None:
    """Render a Rich preflight diagnostics panel and status table."""
    console.print(
        Panel(
            f"[bold]Client Split:[/bold] {_client_split_label()}\n"
            f"[bold]Sandbox Mode:[/bold] {_sandbox_mode_label(settings)}",
            title="RoboHarness Runtime Topology",
            border_style="cyan",
        )
    )
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
            help="Mock TonyPi MCP hardware only (Gemini calls stay live unless --offline is set).",
        ),
    ] = False,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline/--no-offline",
            help="Explicitly skip live Gemini API calls for deterministic offline CI.",
        ),
    ] = False,
    remote_mcp: Annotated[
        bool,
        typer.Option(
            "--remote-mcp/--no-remote-mcp",
            help="Expose TONYPI_REMOTE_MCP_URL to the Managed Agent sandbox.",
        ),
    ] = False,
    deep: Annotated[
        bool,
        typer.Option(
            "--deep/--no-deep",
            help="Route directly to gemini-4-carbon Managed Agent (roboharness-skill-forge-carbon).",
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

    dual_clients: DualGenAIClients | None = None
    if not offline:
        dual_clients = _require_live_clients(console, settings, command_hint="mimic")

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
                clients=dual_clients,
                mcp=mcp_client,
                settings=settings,
                hub=hub,
                offline_mode=offline,
            )
            result = pipeline.mimic_video(
                video_path,
                skill_name=name,
                dry_run=dry_run,
                run_spatial=spatial,
                use_remote_mcp=remote_mcp,
                speed_scale=speed_scale,
                deep=deep,
            )
    except (MCPClientError, CaptureError, ValueError, FileNotFoundError, RuntimeError) as exc:
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
            help="Mock TonyPi MCP hardware only (Gemini calls stay live unless --offline is set).",
        ),
    ] = False,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline/--no-offline",
            help="Explicitly skip live Gemini API calls (spatial/announcement/post-run gate).",
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

    dual_clients: DualGenAIClients | None = None
    if not offline:
        dual_clients = _require_live_clients(console, settings, command_hint="run")

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
                clients=dual_clients,
                mcp=mcp_client,
                settings=settings,
                hub=hub,
                offline_mode=offline,
            )
            result = pipeline.run_saved_skill(
                skill_name,
                dry_run=dry_run,
                run_spatial=spatial,
                speed_scale=speed_scale,
            )
    except (FileNotFoundError, ValueError, MCPClientError, RuntimeError) as exc:
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
    """Run live preflight verification across v1 + v1beta Gemini 4 models and TONYPI_MCP_URL."""
    console = _console()
    settings = RoboHarnessSettings.from_env()
    dual_clients = make_dual_clients(settings.gemini_api_key) if settings.gemini_api_key else None
    mcp_client, _mock_server = _build_mcp_client(settings, mock=mock)

    with mcp_client:
        report = run_preflight_check(
            clients=dual_clients,
            mcp=mcp_client,
            settings=settings,
            skip_forge_create=skip_forge_create,
        )

    _render_preflight_report(console, report, settings)
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
    deep: Annotated[
        bool,
        typer.Option(
            "--deep/--no-deep",
            help="Also pre-stage the gemini-4-carbon Managed Agent (roboharness-skill-forge-carbon).",
        ),
    ] = False,
) -> None:
    """Create or verify the v1beta Managed Agent Skill Forge and sandbox environment."""
    console = _console()
    settings = RoboHarnessSettings.from_env()
    dual_clients = _require_live_clients(
        console,
        settings,
        command_hint="forge-setup",
        allow_offline_hint=False,
    )

    states: list[ForgeAgentState] = []
    try:
        states.append(
            ensure_forge_agent(
                dual_clients.beta_client,
                settings,
                force_recreate=force,
                deep=False,
            )
        )
        if deep:
            states.append(
                ensure_forge_agent(
                    dual_clients.beta_client,
                    settings,
                    force_recreate=force,
                    deep=True,
                )
            )
    except (ValueError, RuntimeError, OSError) as exc:
        console.print(f"[bold red]Forge setup failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    lines = [
        f"[bold]Client Split:[/bold] {_client_split_label()}",
        f"[bold]Sandbox Mode:[/bold] {_sandbox_mode_label(settings)}",
    ]
    for state in states:
        action = "Created & pre-staged" if state.created_now else "Reused existing"
        lines.append(
            f"[bold]{state.agent_id}[/bold] ({state.model}): [green]{action}[/green] "
            f"| env={state.base_environment_id or '--'}"
        )
    console.print(Panel("\n".join(lines), title="Managed Agent Skill Forge", border_style="green"))


@app.command(name="debug-video")
def debug_video(
    video: Annotated[
        Path,
        typer.Option("--video", help="Path to a recorded robot trial MP4/MOV video to diagnose."),
    ],
    name: Annotated[
        str,
        typer.Option("--name", help="Skill name being analyzed in the trial video."),
    ],
    expected: Annotated[
        str,
        typer.Option(
            "--expected",
            help="Expected robot posture, motion timing, or stability criteria.",
        ),
    ] = "Balanced execution without tilt >= 30 deg, joint clipping, or table slip.",
) -> None:
    """Diagnose a trial recording via Agentic Video Processing (`processing='agentic'`, high)."""
    console = _console()
    if not video.is_file():
        console.print(f"[bold red]Error:[/bold red] Video file not found: {video}")
        raise typer.Exit(code=2)

    settings = RoboHarnessSettings.from_env()
    dual_clients = _require_live_clients(
        console,
        settings,
        command_hint="debug-video",
        allow_offline_hint=False,
    )

    try:
        diagnosis = debug_trial_video(
            dual_clients.client,
            video,
            skill_name=name,
            expected_behavior=expected,
            thinking_level="high",
        )
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        console.print(f"[bold red]Debug video failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    header = (
        f"[bold]Skill:[/bold] {name}\n"
        f"[bold]Video:[/bold] {video}\n"
        f"[bold]Client / Mode:[/bold] {STABLE_API_VERSION} | processing=\"agentic\" | "
        "thinking_level=\"high\""
    )
    console.print(Panel(header, title="Agentic Video Trial Debugger", border_style="cyan"))
    console.print(json.dumps(diagnosis, indent=2, sort_keys=True))


@app.command(name="sim-server")
def sim_server(
    host: Annotated[
        str,
        typer.Option("--host", help="Bind host for the MuJoCo MCP HTTP server."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", min=1, max=65535, help="Bind port for the MuJoCo MCP HTTP server."),
    ] = 8765,
    viewer: Annotated[
        bool,
        typer.Option(
            "--viewer/--no-viewer",
            help="Launch interactive 3D MuJoCo passive viewer (use `mjpython` on macOS).",
        ),
    ] = False,
    friction: Annotated[
        float,
        typer.Option(
            "--friction",
            min=0.05,
            help="Floor sliding friction scale (1.0=normal, 0.25=low-friction slip).",
        ),
    ] = 1.0,
    xml: Annotated[
        Path | None,
        typer.Option("--xml", help="Optional path to TonyPi MuJoCo scene.xml asset."),
    ] = None,
) -> None:
    """Launch the clean-room MuJoCo Streamable HTTP MCP server + live MJPEG stream."""
    console = _console()
    try:
        from roboharness.mujoco_mcp_server import run_server
    except ImportError as exc:
        console.print(
            "[bold red]Error:[/bold red] MuJoCo simulator dependency is not installed. "
            "Run `uv sync --extra sim` first."
        )
        raise typer.Exit(code=1) from exc

    info_lines = [
        f"[bold]MCP Endpoint:[/bold] http://{host}:{port}/mcp",
        f"[bold]Live MJPEG Stream:[/bold] http://{host}:{port}/stream.mjpeg",
        f"[bold]Latest Snapshot:[/bold] http://{host}:{port}/frame.jpg",
        f"[bold]Floor Friction Scale:[/bold] {friction:.2f}x",
        (
            "[dim]macOS 3D interactive viewer tip: "
            f"uv run mjpython -m roboharness.mujoco_mcp_server --port {port} --viewer[/dim]"
        ),
    ]
    console.print(
        Panel("\n".join(info_lines), title="RoboHarness MuJoCo MCP Server", border_style="cyan")
    )
    try:
        run_server(host=host, port=port, viewer=viewer, friction=friction, xml_path=xml)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        console.print(f"[bold red]MuJoCo server error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
