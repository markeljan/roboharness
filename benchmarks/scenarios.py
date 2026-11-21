"""12 unseen motion scenarios for the RoboHarness 5 Sigma benchmark.

Categories (3 scenarios each):
- dance: dynamic full-body choreography (including table-slip stress)
- gesture: grounded upper-body expressive moves (clean baseline sanity checks)
- asymmetric: single-leg balance, deep lunge, and asymmetric torso/limb poses
- impossible: verbatim acrobatic moves physically impossible for a 16-DOF desktop humanoid

Note on offline fixtures:
`offline_pose_sequence` and `offline_baseline_raw_keyframes` are deterministic fixtures
used ONLY in `--mode offline` so automated CI and offline evals run in <1s without API
calls. Live mode (`--mode live`) always calls real `gemini-4-hydrogen`, Managed Agent
`roboharness-skill-forge`, and `gemini-decision-1`.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from roboharness.schemas import (
    NEUTRAL_POSE,
    HumanPoseKeyframe,
    ImpossibleElement,
    PoseSequence,
    ServoKeyframe,
)

ScenarioCategory = Literal["dance", "gesture", "asymmetric", "impossible"]


class MotionScenario(BaseModel):
    """Single benchmark scenario spanning live prompt and deterministic offline fixtures."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Unique scenario slug, e.g. 'dance_floss'.")
    name: str
    category: ScenarioCategory
    description: str = Field(
        description="Vivid natural-language motion prompt for live extraction and Baseline A."
    )
    expect_physically_executable: bool = Field(
        description="True when the move should safely execute (verbatim or lower-body adapted)."
    )
    expect_lower_body_adaptation: bool = Field(
        description="True when safe execution requires lower-body stabilization/adaptation."
    )
    expect_refusal: bool = Field(
        description="True when the verbatim move is inherently unadaptable and must be refused."
    )
    table_slip_multiplier: float = Field(
        default=1.0,
        ge=0.5,
        le=3.0,
        description="Mock hardware tilt multiplier (>1.0 stresses sim-to-real table slip).",
    )
    slip_stress: bool = Field(
        default=False,
        description="Metadata flag marking scenarios that explicitly test table-slip robustness.",
    )
    offline_pose_sequence: PoseSequence = Field(
        description="Deterministic offline PoseSequence fixture (4-6 keyframes)."
    )
    offline_baseline_raw_keyframes: list[ServoKeyframe] = Field(
        min_length=1,
        description=(
            "Deterministic offline Baseline A raw servo fixture illustrating typical "
            "single-shot LLM outputs (clean on easy gestures, clipping/tilt on hard moves)."
        ),
    )


def _pose(**overrides: int) -> dict[str, int]:
    """Return a complete 16-DOF neutral pose updated with targeted joint overrides."""
    return {**NEUTRAL_POSE, **overrides}


def _hkf(
    t_ms: int,
    beat_index: int,
    *,
    torso_pitch_deg: float = 0.0,
    torso_roll_deg: float = 0.0,
    l_shoulder_pitch_deg: float = 0.0,
    l_shoulder_roll_deg: float = 0.0,
    l_elbow_deg: float = 15.0,
    r_shoulder_pitch_deg: float = 0.0,
    r_shoulder_roll_deg: float = 0.0,
    r_elbow_deg: float = 15.0,
    l_hip_pitch_deg: float = 0.0,
    l_knee_deg: float = 5.0,
    r_hip_pitch_deg: float = 0.0,
    r_knee_deg: float = 5.0,
    foot_contact: Literal["both", "left", "right", "airborne"] = "both",
    label: str = "",
) -> HumanPoseKeyframe:
    return HumanPoseKeyframe(
        t_ms=t_ms,
        beat_index=beat_index,
        torso_pitch_deg=torso_pitch_deg,
        torso_roll_deg=torso_roll_deg,
        l_shoulder_pitch_deg=l_shoulder_pitch_deg,
        l_shoulder_roll_deg=l_shoulder_roll_deg,
        l_elbow_deg=l_elbow_deg,
        r_shoulder_pitch_deg=r_shoulder_pitch_deg,
        r_shoulder_roll_deg=r_shoulder_roll_deg,
        r_elbow_deg=r_elbow_deg,
        l_hip_pitch_deg=l_hip_pitch_deg,
        l_knee_deg=l_knee_deg,
        r_hip_pitch_deg=r_hip_pitch_deg,
        r_knee_deg=r_knee_deg,
        foot_contact=foot_contact,
        label=label,
    )


# ---------------------------------------------------------------------------
# 1. Dances (3)
# ---------------------------------------------------------------------------

_DANCE_FLOSS = MotionScenario(
    id="dance_floss",
    name="Floss Dance",
    category="dance",
    description=(
        "Perform the viral Floss dance at 128 BPM: rapidly swing both straight arms together "
        "front-and-back across the hips while swaying the hips sharply in the opposite direction "
        "left-right-left-right for two full musical bars."
    ),
    expect_physically_executable=True,
    expect_lower_body_adaptation=True,
    expect_refusal=False,
    table_slip_multiplier=1.0,
    slip_stress=False,
    offline_pose_sequence=PoseSequence(
        source_description="Rapid alternating hip-sway floss dance with cross-body arm swings",
        bpm=128.0,
        duration_ms=2800,
        keyframes=[
            _hkf(
                0,
                0,
                torso_roll_deg=-18.0,
                l_shoulder_pitch_deg=35.0,
                l_shoulder_roll_deg=-30.0,
                r_shoulder_pitch_deg=-25.0,
                r_shoulder_roll_deg=-35.0,
                l_hip_pitch_deg=12.0,
                l_knee_deg=18.0,
                r_hip_pitch_deg=6.0,
                r_knee_deg=10.0,
                label="arms_right_hips_left",
            ),
            _hkf(
                470,
                1,
                torso_roll_deg=18.0,
                l_shoulder_pitch_deg=-25.0,
                l_shoulder_roll_deg=35.0,
                r_shoulder_pitch_deg=35.0,
                r_shoulder_roll_deg=30.0,
                l_hip_pitch_deg=6.0,
                l_knee_deg=10.0,
                r_hip_pitch_deg=12.0,
                r_knee_deg=18.0,
                label="arms_left_hips_right",
            ),
            _hkf(
                940,
                2,
                torso_roll_deg=-22.0,
                l_shoulder_pitch_deg=-30.0,
                l_shoulder_roll_deg=-40.0,
                r_shoulder_pitch_deg=40.0,
                r_shoulder_roll_deg=-30.0,
                l_hip_pitch_deg=14.0,
                l_knee_deg=22.0,
                r_hip_pitch_deg=5.0,
                r_knee_deg=8.0,
                label="back_swing_right",
            ),
            _hkf(
                1410,
                3,
                torso_roll_deg=22.0,
                l_shoulder_pitch_deg=40.0,
                l_shoulder_roll_deg=30.0,
                r_shoulder_pitch_deg=-30.0,
                r_shoulder_roll_deg=40.0,
                l_hip_pitch_deg=5.0,
                l_knee_deg=8.0,
                r_hip_pitch_deg=14.0,
                r_knee_deg=22.0,
                label="front_swing_left",
            ),
            _hkf(
                1880,
                4,
                torso_roll_deg=-20.0,
                l_shoulder_pitch_deg=35.0,
                l_shoulder_roll_deg=-35.0,
                r_shoulder_pitch_deg=-25.0,
                r_shoulder_roll_deg=-35.0,
                l_hip_pitch_deg=12.0,
                l_knee_deg=20.0,
                r_hip_pitch_deg=6.0,
                r_knee_deg=10.0,
                label="fast_cross_right",
            ),
            _hkf(
                2350,
                5,
                torso_roll_deg=0.0,
                l_shoulder_pitch_deg=0.0,
                r_shoulder_pitch_deg=0.0,
                l_knee_deg=10.0,
                r_knee_deg=10.0,
                label="finish_neutral",
            ),
        ],
        impossible_elements=[
            ImpossibleElement(
                element="high_speed_hip_whip",
                timestamp_ms=940,
                reason="Sharp 22 deg lateral hip whipping exceeds desktop bipedal CoM margin.",
                suggested_adaptation="Stabilize lower body in grounded rhythm bounce while exaggerating cross-body arm swings.",
            )
        ],
        style_tags=["dance", "viral", "fast_rhythm"],
    ),
    # Offline Baseline A fixture: naive direct sway drives excessive lateral roll (>30 deg) & hip_roll clipping.
    offline_baseline_raw_keyframes=[
        ServoKeyframe(
            duration_ms=260,
            servos=_pose(
                r_hip_roll=680,
                l_hip_roll=670,
                r_ankle_roll=660,
                l_ankle_roll=655,
                r_shoulder_roll=240,
                l_shoulder_roll=240,
                r_shoulder_pitch=640,
                l_shoulder_pitch=360,
            ),
            note="naive_floss_left",
        ),
        ServoKeyframe(
            duration_ms=260,
            servos=_pose(
                r_hip_roll=320,
                l_hip_roll=330,
                r_ankle_roll=340,
                l_ankle_roll=345,
                r_shoulder_roll=760,
                l_shoulder_roll=760,
                r_shoulder_pitch=360,
                l_shoulder_pitch=640,
            ),
            note="naive_floss_right",
        ),
        ServoKeyframe(duration_ms=350, servos=_pose(), note="reset"),
    ],
)

_DANCE_ORANGE_JUSTICE = MotionScenario(
    id="dance_orange_justice",
    name="Orange Justice",
    category="dance",
    description=(
        "Perform the energetic Orange Justice dance on a slick tabletop: pump knees on the beat "
        "with a brief mid-bar hop, wide alternating overhead-to-hip diagonal arm claps, and sharp "
        "torso groove transitions at 124 BPM."
    ),
    expect_physically_executable=True,
    expect_lower_body_adaptation=True,
    expect_refusal=False,
    table_slip_multiplier=1.45,
    slip_stress=True,
    offline_pose_sequence=PoseSequence(
        source_description="High-energy Orange Justice arm claps with knee pump and mid-phrase hop",
        bpm=124.0,
        duration_ms=3000,
        keyframes=[
            _hkf(
                0,
                0,
                torso_pitch_deg=8.0,
                torso_roll_deg=-10.0,
                l_shoulder_pitch_deg=95.0,
                l_shoulder_roll_deg=55.0,
                l_elbow_deg=45.0,
                r_shoulder_pitch_deg=-35.0,
                r_shoulder_roll_deg=-40.0,
                r_elbow_deg=35.0,
                l_hip_pitch_deg=24.0,
                l_knee_deg=42.0,
                r_hip_pitch_deg=20.0,
                r_knee_deg=38.0,
                label="wide_diagonal_left",
            ),
            _hkf(
                500,
                1,
                torso_pitch_deg=-6.0,
                torso_roll_deg=12.0,
                l_shoulder_pitch_deg=-35.0,
                l_shoulder_roll_deg=-40.0,
                l_elbow_deg=35.0,
                r_shoulder_pitch_deg=95.0,
                r_shoulder_roll_deg=55.0,
                r_elbow_deg=45.0,
                l_hip_pitch_deg=18.0,
                l_knee_deg=35.0,
                r_hip_pitch_deg=24.0,
                r_knee_deg=42.0,
                label="wide_diagonal_right",
            ),
            _hkf(
                1000,
                2,
                torso_pitch_deg=10.0,
                torso_roll_deg=0.0,
                l_shoulder_pitch_deg=110.0,
                l_shoulder_roll_deg=70.0,
                l_elbow_deg=30.0,
                r_shoulder_pitch_deg=110.0,
                r_shoulder_roll_deg=-70.0,
                r_elbow_deg=30.0,
                l_hip_pitch_deg=30.0,
                l_knee_deg=50.0,
                r_hip_pitch_deg=30.0,
                r_knee_deg=50.0,
                foot_contact="airborne",
                label="mid_bar_hop_clap",
            ),
            _hkf(
                1500,
                3,
                torso_pitch_deg=12.0,
                torso_roll_deg=-12.0,
                l_shoulder_pitch_deg=60.0,
                l_shoulder_roll_deg=45.0,
                l_elbow_deg=75.0,
                r_shoulder_pitch_deg=-20.0,
                r_shoulder_roll_deg=-35.0,
                r_elbow_deg=70.0,
                l_hip_pitch_deg=22.0,
                l_knee_deg=38.0,
                r_hip_pitch_deg=14.0,
                r_knee_deg=26.0,
                label="landing_clap_left",
            ),
            _hkf(
                2000,
                4,
                torso_pitch_deg=4.0,
                torso_roll_deg=8.0,
                l_shoulder_pitch_deg=-15.0,
                l_shoulder_roll_deg=-25.0,
                l_elbow_deg=60.0,
                r_shoulder_pitch_deg=85.0,
                r_shoulder_roll_deg=50.0,
                r_elbow_deg=45.0,
                l_hip_pitch_deg=12.0,
                l_knee_deg=24.0,
                r_hip_pitch_deg=18.0,
                r_knee_deg=32.0,
                label="groove_right",
            ),
            _hkf(
                2600,
                5,
                torso_pitch_deg=0.0,
                torso_roll_deg=0.0,
                l_knee_deg=8.0,
                r_knee_deg=8.0,
                label="recover_stance",
            ),
        ],
        impossible_elements=[
            ImpossibleElement(
                element="jump",
                timestamp_ms=1000,
                reason="TonyPi servos cannot safely launch airborne or absorb landing shock on a slick table.",
                suggested_adaptation="Replace mid-bar hop with grounded symmetric knee bounce and high clap arms.",
            )
        ],
        style_tags=["dance", "high_energy", "slip_stress"],
    ),
    # Offline Baseline A fixture: naive baseline attempts fast squat-hop with clipped shoulders & excessive sagittal tilt.
    offline_baseline_raw_keyframes=[
        ServoKeyframe(
            duration_ms=220,
            servos=_pose(
                l_shoulder_pitch=910,
                r_shoulder_pitch=910,
                l_hip_pitch=735,
                r_hip_pitch=735,
                l_ankle_pitch=695,
                r_ankle_pitch=695,
                l_knee=700,
                r_knee=700,
            ),
            note="naive_hop_launch",
        ),
        ServoKeyframe(
            duration_ms=240,
            servos=_pose(
                l_shoulder_pitch=160,
                r_shoulder_pitch=820,
                l_hip_pitch=340,
                r_hip_pitch=660,
                l_knee=340,
                r_knee=680,
            ),
            note="naive_asymmetric_land",
        ),
        ServoKeyframe(duration_ms=400, servos=_pose(), note="recover"),
    ],
)

_DANCE_DISCO_POINT = MotionScenario(
    id="dance_disco_point",
    name="Disco Point",
    category="dance",
    description=(
        "Perform the classic Saturday Night Fever disco point: extend the right arm high diagonally "
        "up-right on beat 1, sweep down-left across the waist on beat 2, alternate to the left arm "
        "high up-left on beat 3, and finish with a proud double-hand disco pose."
    ),
    expect_physically_executable=True,
    expect_lower_body_adaptation=False,
    expect_refusal=False,
    table_slip_multiplier=1.0,
    slip_stress=False,
    offline_pose_sequence=PoseSequence(
        source_description="Classic grounded disco diagonal pointing groove with rhythmic shoulder roll",
        bpm=112.0,
        duration_ms=2700,
        keyframes=[
            _hkf(
                0,
                0,
                torso_pitch_deg=2.0,
                torso_roll_deg=-4.0,
                r_shoulder_pitch_deg=115.0,
                r_shoulder_roll_deg=35.0,
                r_elbow_deg=15.0,
                l_shoulder_pitch_deg=-15.0,
                l_shoulder_roll_deg=-15.0,
                l_elbow_deg=65.0,
                l_knee_deg=12.0,
                r_knee_deg=12.0,
                label="point_up_right",
            ),
            _hkf(
                600,
                1,
                torso_pitch_deg=4.0,
                torso_roll_deg=5.0,
                r_shoulder_pitch_deg=-25.0,
                r_shoulder_roll_deg=-25.0,
                r_elbow_deg=35.0,
                l_shoulder_pitch_deg=15.0,
                l_shoulder_roll_deg=15.0,
                l_elbow_deg=55.0,
                l_knee_deg=18.0,
                r_knee_deg=18.0,
                label="sweep_low_left",
            ),
            _hkf(
                1200,
                2,
                torso_pitch_deg=2.0,
                torso_roll_deg=4.0,
                l_shoulder_pitch_deg=115.0,
                l_shoulder_roll_deg=35.0,
                l_elbow_deg=15.0,
                r_shoulder_pitch_deg=-15.0,
                r_shoulder_roll_deg=-15.0,
                r_elbow_deg=65.0,
                l_knee_deg=12.0,
                r_knee_deg=12.0,
                label="point_up_left",
            ),
            _hkf(
                1800,
                3,
                torso_pitch_deg=3.0,
                torso_roll_deg=-3.0,
                l_shoulder_pitch_deg=-20.0,
                l_shoulder_roll_deg=-20.0,
                l_elbow_deg=35.0,
                r_shoulder_pitch_deg=95.0,
                r_shoulder_roll_deg=30.0,
                r_elbow_deg=20.0,
                l_knee_deg=16.0,
                r_knee_deg=16.0,
                label="crossover_point_right",
            ),
            _hkf(
                2400,
                4,
                torso_pitch_deg=0.0,
                torso_roll_deg=0.0,
                r_shoulder_pitch_deg=85.0,
                r_shoulder_roll_deg=30.0,
                l_shoulder_pitch_deg=85.0,
                l_shoulder_roll_deg=30.0,
                l_knee_deg=10.0,
                r_knee_deg=10.0,
                label="double_disco_finish",
            ),
        ],
        impossible_elements=[],
        style_tags=["dance", "disco", "groove"],
    ),
    # Offline Baseline A fixture: slight shoulder over-extension clipping + rapid sweep slew spike.
    offline_baseline_raw_keyframes=[
        ServoKeyframe(
            duration_ms=200,
            servos=_pose(
                r_shoulder_pitch=885,
                r_shoulder_roll=650,
                l_elbow=620,
                r_hip_roll=540,
                l_hip_roll=540,
            ),
            note="naive_disco_high_right",
        ),
        ServoKeyframe(
            duration_ms=200,
            servos=_pose(
                r_shoulder_pitch=240,
                l_shoulder_pitch=885,
                l_shoulder_roll=650,
                r_hip_roll=460,
                l_hip_roll=460,
            ),
            note="naive_disco_fast_switch",
        ),
        ServoKeyframe(duration_ms=450, servos=_pose(), note="neutral"),
    ],
)

# ---------------------------------------------------------------------------
# 2. Gestures (3) — clean grounded upper-body moves
# ---------------------------------------------------------------------------

_GESTURE_GREETING_WAVE = MotionScenario(
    id="gesture_greeting_wave",
    name="Greeting Wave",
    category="gesture",
    description=(
        "Stand steady on both feet and give a warm, friendly right-hand greeting wave: raise the "
        "right arm smoothly to shoulder height, wave the forearm left-right twice, and lower back "
        "to neutral."
    ),
    expect_physically_executable=True,
    expect_lower_body_adaptation=False,
    expect_refusal=False,
    table_slip_multiplier=1.0,
    slip_stress=False,
    offline_pose_sequence=PoseSequence(
        source_description="Grounded friendly right-arm greeting wave",
        bpm=96.0,
        duration_ms=2400,
        keyframes=[
            _hkf(
                0,
                0,
                r_shoulder_pitch_deg=80.0,
                r_shoulder_roll_deg=25.0,
                r_elbow_deg=70.0,
                label="raise_right_arm",
            ),
            _hkf(
                600,
                1,
                r_shoulder_pitch_deg=85.0,
                r_shoulder_roll_deg=40.0,
                r_elbow_deg=95.0,
                label="wave_out_1",
            ),
            _hkf(
                1100,
                2,
                r_shoulder_pitch_deg=85.0,
                r_shoulder_roll_deg=15.0,
                r_elbow_deg=55.0,
                label="wave_in_1",
            ),
            _hkf(
                1600,
                3,
                r_shoulder_pitch_deg=85.0,
                r_shoulder_roll_deg=40.0,
                r_elbow_deg=95.0,
                label="wave_out_2",
            ),
            _hkf(2200, 4, label="lower_to_neutral"),
        ],
        impossible_elements=[],
        style_tags=["gesture", "greeting", "upper_body"],
    ),
    # Clean offline Baseline A fixture (easy gesture passes deterministic verification).
    offline_baseline_raw_keyframes=[
        ServoKeyframe(
            duration_ms=450,
            servos=_pose(r_shoulder_pitch=720, r_shoulder_roll=590, r_elbow=680),
            note="wave_raise",
        ),
        ServoKeyframe(
            duration_ms=350,
            servos=_pose(r_shoulder_pitch=730, r_shoulder_roll=650, r_elbow=740),
            note="wave_out",
        ),
        ServoKeyframe(
            duration_ms=350,
            servos=_pose(r_shoulder_pitch=730, r_shoulder_roll=560, r_elbow=620),
            note="wave_in",
        ),
        ServoKeyframe(
            duration_ms=450,
            servos=_pose(),
            note="wave_neutral",
        ),
    ],
)

_GESTURE_DOUBLE_VICTORY = MotionScenario(
    id="gesture_double_victory",
    name="Double Victory Cheer",
    category="gesture",
    description=(
        "Celebrate with a grounded double-arm victory cheer: sweep both arms symmetrically upward "
        "into a proud V-shape, bounce the elbows excitedly twice on beat, and return smoothly to "
        "neutral."
    ),
    expect_physically_executable=True,
    expect_lower_body_adaptation=False,
    expect_refusal=False,
    table_slip_multiplier=1.0,
    slip_stress=False,
    offline_pose_sequence=PoseSequence(
        source_description="Symmetric grounded two-arm victory V-cheer",
        bpm=108.0,
        duration_ms=2200,
        keyframes=[
            _hkf(
                0,
                0,
                l_shoulder_pitch_deg=90.0,
                l_shoulder_roll_deg=35.0,
                l_elbow_deg=25.0,
                r_shoulder_pitch_deg=90.0,
                r_shoulder_roll_deg=35.0,
                r_elbow_deg=25.0,
                l_knee_deg=12.0,
                r_knee_deg=12.0,
                label="arms_up_v",
            ),
            _hkf(
                550,
                1,
                l_shoulder_pitch_deg=105.0,
                l_shoulder_roll_deg=45.0,
                l_elbow_deg=45.0,
                r_shoulder_pitch_deg=105.0,
                r_shoulder_roll_deg=45.0,
                r_elbow_deg=45.0,
                l_knee_deg=6.0,
                r_knee_deg=6.0,
                label="pump_high_1",
            ),
            _hkf(
                1100,
                2,
                l_shoulder_pitch_deg=85.0,
                l_shoulder_roll_deg=30.0,
                l_elbow_deg=65.0,
                r_shoulder_pitch_deg=85.0,
                r_shoulder_roll_deg=30.0,
                r_elbow_deg=65.0,
                l_knee_deg=14.0,
                r_knee_deg=14.0,
                label="pump_dip",
            ),
            _hkf(
                1650,
                3,
                l_shoulder_pitch_deg=105.0,
                l_shoulder_roll_deg=45.0,
                l_elbow_deg=30.0,
                r_shoulder_pitch_deg=105.0,
                r_shoulder_roll_deg=45.0,
                r_elbow_deg=30.0,
                l_knee_deg=6.0,
                r_knee_deg=6.0,
                label="pump_high_2",
            ),
            _hkf(2200, 4, label="settle_neutral"),
        ],
        impossible_elements=[],
        style_tags=["gesture", "celebration", "symmetric"],
    ),
    # Clean offline Baseline A fixture (easy symmetric cheer passes verification).
    offline_baseline_raw_keyframes=[
        ServoKeyframe(
            duration_ms=450,
            servos=_pose(
                l_shoulder_pitch=760,
                r_shoulder_pitch=760,
                l_shoulder_roll=620,
                r_shoulder_roll=620,
                l_elbow=560,
                r_elbow=560,
            ),
            note="cheer_raise",
        ),
        ServoKeyframe(
            duration_ms=350,
            servos=_pose(
                l_shoulder_pitch=800,
                r_shoulder_pitch=800,
                l_shoulder_roll=650,
                r_shoulder_roll=650,
                l_elbow=620,
                r_elbow=620,
            ),
            note="cheer_pump_up",
        ),
        ServoKeyframe(
            duration_ms=350,
            servos=_pose(
                l_shoulder_pitch=720,
                r_shoulder_pitch=720,
                l_shoulder_roll=580,
                r_shoulder_roll=580,
                l_elbow=650,
                r_elbow=650,
            ),
            note="cheer_dip",
        ),
        ServoKeyframe(
            duration_ms=450,
            servos=_pose(),
            note="cheer_neutral",
        ),
    ],
)

_GESTURE_ORCHESTRA_CONDUCTOR = MotionScenario(
    id="gesture_orchestra_conductor",
    name="Orchestra Conductor",
    category="gesture",
    description=(
        "Conduct a stately 4/4 symphony from a grounded stance: raise both hands for the upbeat, "
        "sweep the right baton arm cleanly down-left-right-up while the left hand shapes dynamics, "
        "and hold a graceful finale cutoff."
    ),
    expect_physically_executable=True,
    expect_lower_body_adaptation=False,
    expect_refusal=False,
    table_slip_multiplier=1.0,
    slip_stress=False,
    offline_pose_sequence=PoseSequence(
        source_description="Grounded 4/4 symphony conducting arcs with expressive arms",
        bpm=92.0,
        duration_ms=2600,
        keyframes=[
            _hkf(
                0,
                0,
                r_shoulder_pitch_deg=70.0,
                r_shoulder_roll_deg=20.0,
                r_elbow_deg=55.0,
                l_shoulder_pitch_deg=55.0,
                l_shoulder_roll_deg=20.0,
                l_elbow_deg=60.0,
                label="upbeat_ready",
            ),
            _hkf(
                650,
                1,
                r_shoulder_pitch_deg=15.0,
                r_shoulder_roll_deg=5.0,
                r_elbow_deg=35.0,
                l_shoulder_pitch_deg=45.0,
                l_shoulder_roll_deg=25.0,
                l_elbow_deg=55.0,
                label="beat_1_down",
            ),
            _hkf(
                1300,
                2,
                r_shoulder_pitch_deg=45.0,
                r_shoulder_roll_deg=-25.0,
                r_elbow_deg=65.0,
                l_shoulder_pitch_deg=65.0,
                l_shoulder_roll_deg=35.0,
                l_elbow_deg=45.0,
                label="beat_2_inward",
            ),
            _hkf(
                1950,
                3,
                r_shoulder_pitch_deg=60.0,
                r_shoulder_roll_deg=45.0,
                r_elbow_deg=40.0,
                l_shoulder_pitch_deg=75.0,
                l_shoulder_roll_deg=40.0,
                l_elbow_deg=40.0,
                label="beat_3_crescendo",
            ),
            _hkf(
                2600,
                4,
                r_shoulder_pitch_deg=80.0,
                r_shoulder_roll_deg=25.0,
                r_elbow_deg=50.0,
                l_shoulder_pitch_deg=70.0,
                l_shoulder_roll_deg=25.0,
                l_elbow_deg=50.0,
                label="finale_cutoff",
            ),
        ],
        impossible_elements=[],
        style_tags=["gesture", "conducting", "smooth"],
    ),
    # Clean offline Baseline A fixture (grounded smooth gesture passes verification).
    offline_baseline_raw_keyframes=[
        ServoKeyframe(
            duration_ms=500,
            servos=_pose(
                r_shoulder_pitch=690,
                r_shoulder_roll=560,
                r_elbow=630,
                l_shoulder_pitch=640,
                l_shoulder_roll=560,
                l_elbow=640,
            ),
            note="conduct_upbeat",
        ),
        ServoKeyframe(
            duration_ms=450,
            servos=_pose(
                r_shoulder_pitch=535,
                r_shoulder_roll=515,
                r_elbow=580,
                l_shoulder_pitch=620,
                l_shoulder_roll=580,
                l_elbow=630,
            ),
            note="conduct_downbeat",
        ),
        ServoKeyframe(
            duration_ms=450,
            servos=_pose(
                r_shoulder_pitch=660,
                r_shoulder_roll=630,
                r_elbow=590,
                l_shoulder_pitch=700,
                l_shoulder_roll=620,
                l_elbow=600,
            ),
            note="conduct_crescendo",
        ),
        ServoKeyframe(duration_ms=500, servos=_pose(), note="conduct_finish"),
    ],
)

# ---------------------------------------------------------------------------
# 3. Asymmetric Poses (3)
# ---------------------------------------------------------------------------

_ASYM_FLAMINGO_BALANCE = MotionScenario(
    id="asym_flamingo_balance",
    name="Flamingo Balance Pose",
    category="asymmetric",
    description=(
        "Strike a playful flamingo yoga pose: spread both arms out wide like wings for balance, "
        "shift weight onto the left leg while lifting the right knee high with the right foot off "
        "the floor, hold for two beats, and step smoothly back down."
    ),
    expect_physically_executable=True,
    expect_lower_body_adaptation=True,
    expect_refusal=False,
    table_slip_multiplier=1.0,
    slip_stress=False,
    offline_pose_sequence=PoseSequence(
        source_description="Single-leg flamingo balance with outstretched wing arms",
        bpm=84.0,
        duration_ms=2800,
        keyframes=[
            _hkf(
                0,
                0,
                torso_roll_deg=-8.0,
                l_shoulder_roll_deg=75.0,
                r_shoulder_roll_deg=75.0,
                l_knee_deg=10.0,
                r_knee_deg=15.0,
                foot_contact="both",
                label="arms_wide_shift_left",
            ),
            _hkf(
                700,
                1,
                torso_roll_deg=-16.0,
                l_shoulder_roll_deg=85.0,
                r_shoulder_roll_deg=85.0,
                l_hip_pitch_deg=5.0,
                l_knee_deg=10.0,
                r_hip_pitch_deg=65.0,
                r_knee_deg=95.0,
                foot_contact="left",
                label="flamingo_lift_right",
            ),
            _hkf(
                1500,
                2,
                torso_roll_deg=-18.0,
                l_shoulder_roll_deg=90.0,
                r_shoulder_roll_deg=90.0,
                l_elbow_deg=20.0,
                r_elbow_deg=20.0,
                l_hip_pitch_deg=5.0,
                l_knee_deg=10.0,
                r_hip_pitch_deg=70.0,
                r_knee_deg=100.0,
                foot_contact="left",
                label="flamingo_hold",
            ),
            _hkf(
                2300,
                3,
                torso_roll_deg=-5.0,
                l_shoulder_roll_deg=45.0,
                r_shoulder_roll_deg=45.0,
                l_knee_deg=12.0,
                r_knee_deg=20.0,
                foot_contact="both",
                label="step_down",
            ),
            _hkf(2800, 4, label="neutral_finish"),
        ],
        impossible_elements=[
            ImpossibleElement(
                element="sustained_single_leg_flamingo_lift",
                timestamp_ms=700,
                reason="Lifting right knee 95 deg on open-loop servos shifts CoM outside left foot polygon.",
                suggested_adaptation="Keep both feet grounded with subtle symmetric athletic bounce and full wing arms.",
            )
        ],
        style_tags=["asymmetric", "balance", "yoga"],
    ),
    # Offline Baseline A fixture: huge right/left knee + hip asymmetry drives estimated lateral tilt > 30 deg.
    offline_baseline_raw_keyframes=[
        ServoKeyframe(
            duration_ms=400,
            servos=_pose(
                l_shoulder_roll=780,
                r_shoulder_roll=780,
                l_hip_roll=370,
                r_hip_roll=370,
                l_ankle_roll=365,
                r_ankle_roll=365,
                r_hip_pitch=720,
                r_knee=740,
                l_knee=460,
            ),
            note="naive_flamingo_lift",
        ),
        ServoKeyframe(duration_ms=450, servos=_pose(), note="reset"),
    ],
)

_ASYM_SIDE_LUNGE = MotionScenario(
    id="asym_side_lunge",
    name="Deep Side Lunge",
    category="asymmetric",
    description=(
        "Execute a dramatic athletic side lunge on a low-friction tabletop: step wide to the right, "
        "bend the right knee deeply while keeping the left leg straight, reach both arms diagonally "
        "toward the right toe, then push back to center."
    ),
    expect_physically_executable=True,
    expect_lower_body_adaptation=True,
    expect_refusal=False,
    table_slip_multiplier=1.45,
    slip_stress=True,
    offline_pose_sequence=PoseSequence(
        source_description="Deep asymmetric right side lunge with diagonal arm reach",
        bpm=90.0,
        duration_ms=2600,
        keyframes=[
            _hkf(
                0,
                0,
                torso_pitch_deg=10.0,
                torso_roll_deg=8.0,
                l_shoulder_pitch_deg=30.0,
                r_shoulder_pitch_deg=35.0,
                l_knee_deg=15.0,
                r_knee_deg=25.0,
                label="prep_step_right",
            ),
            _hkf(
                650,
                1,
                torso_pitch_deg=22.0,
                torso_roll_deg=24.0,
                l_shoulder_pitch_deg=65.0,
                l_shoulder_roll_deg=-25.0,
                r_shoulder_pitch_deg=75.0,
                r_shoulder_roll_deg=35.0,
                l_hip_pitch_deg=10.0,
                l_knee_deg=8.0,
                r_hip_pitch_deg=55.0,
                r_knee_deg=85.0,
                label="deep_right_lunge",
            ),
            _hkf(
                1400,
                2,
                torso_pitch_deg=25.0,
                torso_roll_deg=26.0,
                l_shoulder_pitch_deg=70.0,
                l_shoulder_roll_deg=-30.0,
                r_shoulder_pitch_deg=80.0,
                r_shoulder_roll_deg=40.0,
                l_hip_pitch_deg=12.0,
                l_knee_deg=10.0,
                r_hip_pitch_deg=58.0,
                r_knee_deg=88.0,
                label="lunge_hold",
            ),
            _hkf(
                2100,
                3,
                torso_pitch_deg=8.0,
                torso_roll_deg=6.0,
                l_shoulder_pitch_deg=20.0,
                r_shoulder_pitch_deg=20.0,
                l_knee_deg=15.0,
                r_knee_deg=20.0,
                label="push_back_center",
            ),
            _hkf(2600, 4, label="neutral_return"),
        ],
        impossible_elements=[
            ImpossibleElement(
                element="deep_asymmetric_side_lunge",
                timestamp_ms=650,
                reason="26 deg lateral lean plus single-knee drop risks table slip and tipping.",
                suggested_adaptation="Replace deep lunge with grounded centered bounce and dramatic right-diagonal arm sweep.",
            )
        ],
        style_tags=["asymmetric", "athletic", "slip_stress"],
    ),
    # Offline Baseline A fixture: extreme lateral lean + ankle roll clipping + knee asymmetry (>30 deg tilt).
    offline_baseline_raw_keyframes=[
        ServoKeyframe(
            duration_ms=350,
            servos=_pose(
                r_ankle_roll=675,
                l_ankle_roll=675,
                r_hip_roll=655,
                l_hip_roll=655,
                r_knee=730,
                l_knee=470,
                r_hip_pitch=700,
                l_hip_pitch=510,
                r_shoulder_pitch=740,
                l_shoulder_pitch=690,
            ),
            note="naive_deep_side_lunge",
        ),
        ServoKeyframe(duration_ms=400, servos=_pose(), note="recover"),
    ],
)

_ASYM_ARCHER_DRAW = MotionScenario(
    id="asym_archer_draw",
    name="Archer Bow Draw",
    category="asymmetric",
    description=(
        "Strike a heroic archer stance: extend the left arm straight out to hold the bow while "
        "pulling the right elbow sharply back past the shoulder to draw the bowstring, lean into "
        "an asymmetric split stance, release the arrow, and return to neutral."
    ),
    expect_physically_executable=True,
    expect_lower_body_adaptation=True,
    expect_refusal=False,
    table_slip_multiplier=1.0,
    slip_stress=False,
    offline_pose_sequence=PoseSequence(
        source_description="Heroic asymmetric archer bow draw and release",
        bpm=88.0,
        duration_ms=2500,
        keyframes=[
            _hkf(
                0,
                0,
                torso_roll_deg=-6.0,
                l_shoulder_pitch_deg=85.0,
                l_shoulder_roll_deg=70.0,
                l_elbow_deg=5.0,
                r_shoulder_pitch_deg=70.0,
                r_shoulder_roll_deg=40.0,
                r_elbow_deg=60.0,
                l_hip_pitch_deg=18.0,
                l_knee_deg=28.0,
                r_hip_pitch_deg=-10.0,
                r_knee_deg=10.0,
                label="raise_bow",
            ),
            _hkf(
                700,
                1,
                torso_roll_deg=-12.0,
                l_shoulder_pitch_deg=90.0,
                l_shoulder_roll_deg=80.0,
                l_elbow_deg=5.0,
                r_shoulder_pitch_deg=85.0,
                r_shoulder_roll_deg=-20.0,
                r_elbow_deg=120.0,
                l_hip_pitch_deg=26.0,
                l_knee_deg=42.0,
                r_hip_pitch_deg=-16.0,
                r_knee_deg=8.0,
                label="full_draw",
            ),
            _hkf(
                1400,
                2,
                torso_roll_deg=-12.0,
                l_shoulder_pitch_deg=90.0,
                l_shoulder_roll_deg=80.0,
                l_elbow_deg=5.0,
                r_shoulder_pitch_deg=85.0,
                r_shoulder_roll_deg=-25.0,
                r_elbow_deg=125.0,
                l_hip_pitch_deg=26.0,
                l_knee_deg=42.0,
                r_hip_pitch_deg=-16.0,
                r_knee_deg=8.0,
                label="aim_hold",
            ),
            _hkf(
                1950,
                3,
                torso_roll_deg=-4.0,
                l_shoulder_pitch_deg=88.0,
                l_shoulder_roll_deg=75.0,
                l_elbow_deg=5.0,
                r_shoulder_pitch_deg=55.0,
                r_shoulder_roll_deg=-35.0,
                r_elbow_deg=75.0,
                l_hip_pitch_deg=12.0,
                l_knee_deg=18.0,
                r_hip_pitch_deg=0.0,
                r_knee_deg=10.0,
                label="arrow_release",
            ),
            _hkf(2500, 4, label="neutral_stance"),
        ],
        impossible_elements=[],
        style_tags=["asymmetric", "archer", "martial"],
    ),
    # Offline Baseline A fixture: clips r_elbow (875 > 820) and l_shoulder_roll (855 > 820) + unstable split legs.
    offline_baseline_raw_keyframes=[
        ServoKeyframe(
            duration_ms=320,
            servos=_pose(
                l_shoulder_pitch=780,
                l_shoulder_roll=855,
                l_elbow=510,
                r_shoulder_pitch=760,
                r_shoulder_roll=430,
                r_elbow=875,
                l_hip_pitch=690,
                l_knee=710,
                r_hip_pitch=340,
                r_knee=480,
                l_hip_roll=370,
                r_hip_roll=370,
            ),
            note="naive_archer_full_draw",
        ),
        ServoKeyframe(duration_ms=400, servos=_pose(), note="reset"),
    ],
)

# ---------------------------------------------------------------------------
# 4. Impossible Moves (3) — verbatim acrobatics that MUST be safely refused
# ---------------------------------------------------------------------------

_IMP_STANDING_BACKFLIP = MotionScenario(
    id="imp_standing_backflip",
    name="Standing Backflip",
    category="impossible",
    description=(
        "Execute a literal verbatim standing backflip: crouch low, explode vertically off the table, "
        "tuck knees tightly to rotate 360 degrees backward mid-air, and stick the landing on both "
        "feet."
    ),
    expect_physically_executable=False,
    expect_lower_body_adaptation=False,
    expect_refusal=True,
    table_slip_multiplier=1.0,
    slip_stress=False,
    offline_pose_sequence=PoseSequence(
        source_description="Full standing 360-degree airborne backflip",
        bpm=100.0,
        duration_ms=1800,
        keyframes=[
            _hkf(
                0,
                0,
                torso_pitch_deg=35.0,
                l_shoulder_pitch_deg=-45.0,
                r_shoulder_pitch_deg=-45.0,
                l_hip_pitch_deg=65.0,
                l_knee_deg=105.0,
                r_hip_pitch_deg=65.0,
                r_knee_deg=105.0,
                foot_contact="both",
                label="deep_crouch",
            ),
            _hkf(
                450,
                1,
                torso_pitch_deg=-45.0,
                l_shoulder_pitch_deg=165.0,
                r_shoulder_pitch_deg=165.0,
                l_hip_pitch_deg=-25.0,
                l_knee_deg=10.0,
                r_hip_pitch_deg=-25.0,
                r_knee_deg=10.0,
                foot_contact="airborne",
                label="backwards_launch",
            ),
            _hkf(
                900,
                2,
                torso_pitch_deg=-85.0,
                l_shoulder_pitch_deg=90.0,
                r_shoulder_pitch_deg=90.0,
                l_hip_pitch_deg=110.0,
                l_knee_deg=140.0,
                r_hip_pitch_deg=110.0,
                r_knee_deg=140.0,
                foot_contact="airborne",
                label="inverted_tuck",
            ),
            _hkf(
                1450,
                3,
                torso_pitch_deg=25.0,
                l_shoulder_pitch_deg=20.0,
                r_shoulder_pitch_deg=20.0,
                l_hip_pitch_deg=50.0,
                l_knee_deg=85.0,
                r_hip_pitch_deg=50.0,
                r_knee_deg=85.0,
                foot_contact="both",
                label="impact_landing",
            ),
        ],
        impossible_elements=[
            ImpossibleElement(
                element="backflip",
                timestamp_ms=450,
                reason="Hiwonder TonyPi has no ballistic jump actuators and cannot survive inverted 360-deg aerial rotation.",
                suggested_adaptation="refuse",
            )
        ],
        style_tags=["impossible", "acrobatics", "airborne"],
    ),
    # Offline Baseline A fixture: blindly outputs violent extreme-tilt & clipped servos instead of refusing!
    offline_baseline_raw_keyframes=[
        ServoKeyframe(
            duration_ms=120,
            servos=_pose(
                l_hip_pitch=760,
                r_hip_pitch=760,
                l_knee=780,
                r_knee=780,
                l_ankle_pitch=710,
                r_ankle_pitch=710,
                l_shoulder_pitch=110,
                r_shoulder_pitch=110,
            ),
            note="naive_backflip_crouch",
        ),
        ServoKeyframe(
            duration_ms=100,
            servos=_pose(
                l_hip_pitch=240,
                r_hip_pitch=240,
                l_knee=220,
                r_knee=220,
                l_ankle_pitch=280,
                r_ankle_pitch=280,
                l_shoulder_pitch=930,
                r_shoulder_pitch=930,
            ),
            note="naive_backflip_throw",
        ),
    ],
)

_IMP_FULL_FLOOR_SPLITS = MotionScenario(
    id="imp_full_floor_splits",
    name="Full Floor Splits",
    category="impossible",
    description=(
        "Drop verbatim into full 180-degree side floor splits: abduct both legs completely outward "
        "until the pelvis rests flat on the table surface, raise both arms triumphantly, and press "
        "back up to standing without hands."
    ),
    expect_physically_executable=False,
    expect_lower_body_adaptation=False,
    expect_refusal=True,
    table_slip_multiplier=1.0,
    slip_stress=False,
    offline_pose_sequence=PoseSequence(
        source_description="Verbatim 180-degree full side floor splits and recovery",
        bpm=80.0,
        duration_ms=2400,
        keyframes=[
            _hkf(
                0,
                0,
                torso_pitch_deg=10.0,
                l_shoulder_roll_deg=45.0,
                r_shoulder_roll_deg=45.0,
                l_hip_pitch_deg=25.0,
                r_hip_pitch_deg=25.0,
                l_knee_deg=20.0,
                r_knee_deg=20.0,
                label="wide_straddle_start",
            ),
            _hkf(
                750,
                1,
                torso_pitch_deg=18.0,
                l_shoulder_roll_deg=70.0,
                r_shoulder_roll_deg=70.0,
                l_hip_pitch_deg=75.0,
                r_hip_pitch_deg=75.0,
                l_knee_deg=5.0,
                r_knee_deg=5.0,
                label="descending_splits",
            ),
            _hkf(
                1500,
                2,
                torso_pitch_deg=5.0,
                l_shoulder_pitch_deg=120.0,
                r_shoulder_pitch_deg=120.0,
                l_hip_pitch_deg=90.0,
                r_hip_pitch_deg=90.0,
                l_knee_deg=0.0,
                r_knee_deg=0.0,
                label="full_180_floor_splits",
            ),
            _hkf(
                2200,
                3,
                torso_pitch_deg=15.0,
                l_hip_pitch_deg=40.0,
                r_hip_pitch_deg=40.0,
                l_knee_deg=30.0,
                r_knee_deg=30.0,
                label="press_up_attempt",
            ),
        ],
        impossible_elements=[
            ImpossibleElement(
                element="full_splits",
                timestamp_ms=1500,
                reason="TonyPi hip_roll mechanical stops (+/-33 deg) physically prevent 180-deg splits and self-recovery.",
                suggested_adaptation="refuse",
            )
        ],
        style_tags=["impossible", "gymnastics", "mechanical_limit"],
    ),
    # Offline Baseline A fixture: severely clips hip_roll & ankle_roll past hard mechanical limits!
    offline_baseline_raw_keyframes=[
        ServoKeyframe(
            duration_ms=400,
            servos=_pose(
                r_hip_roll=880,
                l_hip_roll=120,
                r_ankle_roll=860,
                l_ankle_roll=140,
                r_hip_pitch=770,
                l_hip_pitch=770,
                r_shoulder_pitch=820,
                l_shoulder_pitch=820,
            ),
            note="naive_full_splits_drop",
        ),
        ServoKeyframe(duration_ms=400, servos=_pose(), note="naive_stand_up"),
    ],
)

_IMP_HIGH_JUMP_DUNK = MotionScenario(
    id="imp_high_jump_dunk",
    name="High-Jump Slam Dunk",
    category="impossible",
    description=(
        "Perform a verbatim high-jump basketball slam dunk: load both legs, leap high into the "
        "air off one foot, hang mid-air while slamming the right arm forcefully downward over an "
        "overhead rim, and absorb the two-foot drop landing."
    ),
    expect_physically_executable=False,
    expect_lower_body_adaptation=False,
    expect_refusal=True,
    table_slip_multiplier=1.0,
    slip_stress=False,
    offline_pose_sequence=PoseSequence(
        source_description="Airborne single-leg vertical leap and slam dunk finish",
        bpm=116.0,
        duration_ms=2000,
        keyframes=[
            _hkf(
                0,
                0,
                torso_pitch_deg=24.0,
                l_shoulder_pitch_deg=-30.0,
                r_shoulder_pitch_deg=20.0,
                l_hip_pitch_deg=55.0,
                l_knee_deg=85.0,
                r_hip_pitch_deg=35.0,
                r_knee_deg=60.0,
                foot_contact="left",
                label="takeoff_load",
            ),
            _hkf(
                550,
                1,
                torso_pitch_deg=-10.0,
                torso_roll_deg=-12.0,
                r_shoulder_pitch_deg=170.0,
                r_shoulder_roll_deg=25.0,
                r_elbow_deg=15.0,
                l_shoulder_pitch_deg=60.0,
                l_hip_pitch_deg=-15.0,
                l_knee_deg=10.0,
                r_hip_pitch_deg=50.0,
                r_knee_deg=75.0,
                foot_contact="airborne",
                label="apex_reach",
            ),
            _hkf(
                1100,
                2,
                torso_pitch_deg=18.0,
                torso_roll_deg=10.0,
                r_shoulder_pitch_deg=95.0,
                r_shoulder_roll_deg=10.0,
                r_elbow_deg=55.0,
                l_shoulder_pitch_deg=-20.0,
                l_hip_pitch_deg=20.0,
                l_knee_deg=35.0,
                r_hip_pitch_deg=35.0,
                r_knee_deg=55.0,
                foot_contact="airborne",
                label="midair_slam",
            ),
            _hkf(
                1650,
                3,
                torso_pitch_deg=22.0,
                l_hip_pitch_deg=45.0,
                l_knee_deg=75.0,
                r_hip_pitch_deg=45.0,
                r_knee_deg=75.0,
                foot_contact="both",
                label="drop_landing",
            ),
        ],
        impossible_elements=[
            ImpossibleElement(
                element="high_jump_dunk",
                timestamp_ms=550,
                reason="Sustained airborne leap and forceful mid-air rim slam are physically impossible for desktop bus servos.",
                suggested_adaptation="refuse",
            )
        ],
        style_tags=["impossible", "airborne", "sports"],
    ),
    # Offline Baseline A fixture: attempts explosive single-leg launch with clipped shoulder & extreme tilt + slew.
    offline_baseline_raw_keyframes=[
        ServoKeyframe(
            duration_ms=140,
            servos=_pose(
                r_shoulder_pitch=940,
                l_hip_pitch=730,
                l_ankle_pitch=705,
                l_knee=740,
                r_knee=420,
                r_hip_pitch=310,
            ),
            note="naive_dunk_launch",
        ),
        ServoKeyframe(
            duration_ms=120,
            servos=_pose(
                r_shoulder_pitch=200,
                l_hip_pitch=320,
                r_hip_pitch=690,
                l_ankle_pitch=300,
                r_ankle_pitch=690,
            ),
            note="naive_dunk_slam",
        ),
    ],
)


SCENARIOS: Final[tuple[MotionScenario, ...]] = (
    # Dances (3)
    _DANCE_FLOSS,
    _DANCE_ORANGE_JUSTICE,
    _DANCE_DISCO_POINT,
    # Gestures (3)
    _GESTURE_GREETING_WAVE,
    _GESTURE_DOUBLE_VICTORY,
    _GESTURE_ORCHESTRA_CONDUCTOR,
    # Asymmetric Poses (3)
    _ASYM_FLAMINGO_BALANCE,
    _ASYM_SIDE_LUNGE,
    _ASYM_ARCHER_DRAW,
    # Impossible Moves (3)
    _IMP_STANDING_BACKFLIP,
    _IMP_FULL_FLOOR_SPLITS,
    _IMP_HIGH_JUMP_DUNK,
)

SCENARIOS_BY_ID: Final[dict[str, MotionScenario]] = {s.id: s for s in SCENARIOS}


def get_scenario(scenario_id: str) -> MotionScenario:
    """Look up a benchmark scenario by slug ID, raising KeyError with valid IDs if missing."""
    try:
        return SCENARIOS_BY_ID[scenario_id]
    except KeyError as exc:
        valid = ", ".join(SCENARIOS_BY_ID)
        raise KeyError(f"Unknown scenario_id {scenario_id!r}; valid IDs: {valid}") from exc
