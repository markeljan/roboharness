"""Dear ImGui telemetry overlay plugin for MuJoCo 4.0.1 Studio (`mujoco.studio`).

Renders a compact "RoboHarness Telemetry" panel inside MuJoCo Studio's WebAssembly
(Filament/WASM) and Native 3D viewers, displaying the active skill, live torso tilt
against the 30.0 deg hard latch, foot contacts, floor friction scale, and safety state.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import mujoco
from mujoco.studio import messages, plugin
from mujoco.studio.dear_imgui import dear_imgui as imgui

from roboharness.config import MAX_TORSO_TILT_DEG

_COLOR_CYAN = (0.22, 0.74, 0.97, 1.0)
_COLOR_GREEN = (0.20, 0.83, 0.60, 1.0)
_COLOR_AMBER = (0.98, 0.75, 0.14, 1.0)
_COLOR_RED = (0.97, 0.44, 0.44, 1.0)
_COLOR_MUTED = (0.58, 0.64, 0.72, 1.0)


@dataclass(frozen=True)
class StudioTelemetrySnapshot:
    """Thread-safe snapshot of live RoboHarness telemetry rendered by Studio."""

    skill_name: str = "idle"
    tilt_deg: float = 0.0
    contacts: int = 0
    friction: float = 1.0
    status_label: str = "STANDING"
    safety_ok: bool = True
    mujoco_version: str = mujoco.__version__


class RoboHarnessStudioPlugin(plugin.ViewerPlugin):
    """MuJoCo 4.0.1 Studio ViewerPlugin drawing the RoboHarness Telemetry ImGui panel."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = StudioTelemetrySnapshot()

    def update_telemetry(
        self,
        skill_name: str,
        tilt_deg: float,
        contacts: int,
        friction: float,
        status_label: str,
        safety_ok: bool,
    ) -> None:
        """Update live telemetry atomically from physics / MCP server threads."""
        with self._lock:
            self._state = StudioTelemetrySnapshot(
                skill_name=str(skill_name or "idle"),
                tilt_deg=round(float(tilt_deg), 2),
                contacts=int(contacts),
                friction=round(float(friction), 3),
                status_label=str(status_label or "IDLE"),
                safety_ok=bool(safety_ok),
                mujoco_version=mujoco.__version__,
            )

    def snapshot(self) -> StudioTelemetrySnapshot:
        """Return the latest immutable telemetry snapshot."""
        with self._lock:
            return self._state

    @messages.handler
    def on_build_gui(self, _event: messages.BuildGuiEvent) -> None:
        """Render the RoboHarness Telemetry Dear ImGui window each Studio frame."""
        snap = self.snapshot()

        imgui.SetNextWindowPos(imgui.Vec2(16.0, 56.0), int(imgui.Cond.FirstUseEver))
        imgui.SetNextWindowSize(imgui.Vec2(320.0, 210.0), int(imgui.Cond.FirstUseEver))

        expanded, _ = imgui.Begin("RoboHarness Telemetry")
        try:
            if not expanded:
                return

            imgui.TextColored(
                imgui.Vec4(*_COLOR_CYAN),
                f"MuJoCo {snap.mujoco_version} Studio | RoboHarness",
            )
            imgui.Separator()

            imgui.Text(f"Active Skill: {snap.skill_name}")
            imgui.Text(f"Phase:        {snap.status_label}")

            if snap.tilt_deg >= MAX_TORSO_TILT_DEG:
                tilt_col = _COLOR_RED
            elif snap.tilt_deg >= MAX_TORSO_TILT_DEG * 0.75:
                tilt_col = _COLOR_AMBER
            else:
                tilt_col = _COLOR_GREEN

            imgui.TextColored(
                imgui.Vec4(*tilt_col),
                f"Torso Tilt:   {snap.tilt_deg:.1f} deg (latch < {MAX_TORSO_TILT_DEG:.0f} deg)",
            )
            fraction = max(0.0, min(1.0, snap.tilt_deg / MAX_TORSO_TILT_DEG))
            imgui.ProgressBar(
                fraction,
                imgui.Vec2(-1.0, 14.0),
                f"{snap.tilt_deg:.1f} / {MAX_TORSO_TILT_DEG:.0f} deg",
            )

            imgui.Text(f"Foot Contacts:  {snap.contacts}")
            imgui.Text(f"Floor Friction: mu = {snap.friction:.2f}x")
            imgui.Separator()

            if snap.safety_ok:
                imgui.TextColored(imgui.Vec4(*_COLOR_GREEN), "Safety Latch: SAFE")
            else:
                imgui.TextColored(imgui.Vec4(*_COLOR_RED), "Safety Latch: ABORTED")

            imgui.TextColored(
                imgui.Vec4(*_COLOR_MUTED),
                "Timeline scrubbing enabled via SimHistory",
            )
        finally:
            imgui.End()
