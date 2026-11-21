#!/usr/bin/env python3
"""Standalone 16-DOF Hiwonder TonyPi kinematics & CoM verifier for RoboHarness.

Zero external dependencies (Python 3 stdlib only) so it runs identically inside
Google's Ubuntu Managed Agent sandbox and local CI/unit tests.

Thresholds and kinematic model match ``src/roboharness/safety.py`` and
``src/roboharness/schemas.py`` verbatim.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# Hard physical safety limits (matching roboharness.config & roboharness.safety)
MIN_KEYFRAME_DURATION_MS: int = 80
MAX_KEYFRAME_DURATION_MS: int = 5000
MAX_TORSO_TILT_DEG: float = 30.0
AUTO_FIX_TILT_TRIGGER_DEG: float = 28.0
MAX_SLEW_PULSE_PER_SEC: float = 900.0
DEG_PER_PULSE: float = 0.24

# Canonical Hiwonder TonyPi 16-DOF servo specs (servo_id, name, min, max, neutral, role)
TONYPI_JOINT_SPECS: tuple[dict[str, Any], ...] = (
    # Right leg (IDs 1-5)
    {"servo_id": 1, "name": "r_ankle_roll", "min_pulse": 350, "max_pulse": 650, "neutral_pulse": 500, "role": "lower_body"},
    {"servo_id": 2, "name": "r_ankle_pitch", "min_pulse": 320, "max_pulse": 680, "neutral_pulse": 500, "role": "lower_body"},
    {"servo_id": 3, "name": "r_knee", "min_pulse": 260, "max_pulse": 740, "neutral_pulse": 500, "role": "lower_body"},
    {"servo_id": 4, "name": "r_hip_pitch", "min_pulse": 280, "max_pulse": 720, "neutral_pulse": 500, "role": "lower_body"},
    {"servo_id": 5, "name": "r_hip_roll", "min_pulse": 360, "max_pulse": 640, "neutral_pulse": 500, "role": "lower_body"},
    # Right arm (IDs 6-8)
    {"servo_id": 6, "name": "r_shoulder_pitch", "min_pulse": 150, "max_pulse": 850, "neutral_pulse": 500, "role": "upper_body"},
    {"servo_id": 7, "name": "r_shoulder_roll", "min_pulse": 180, "max_pulse": 820, "neutral_pulse": 500, "role": "upper_body"},
    {"servo_id": 8, "name": "r_elbow", "min_pulse": 180, "max_pulse": 820, "neutral_pulse": 500, "role": "upper_body"},
    # Left leg (IDs 9-13)
    {"servo_id": 9, "name": "l_ankle_roll", "min_pulse": 350, "max_pulse": 650, "neutral_pulse": 500, "role": "lower_body"},
    {"servo_id": 10, "name": "l_ankle_pitch", "min_pulse": 320, "max_pulse": 680, "neutral_pulse": 500, "role": "lower_body"},
    {"servo_id": 11, "name": "l_knee", "min_pulse": 260, "max_pulse": 740, "neutral_pulse": 500, "role": "lower_body"},
    {"servo_id": 12, "name": "l_hip_pitch", "min_pulse": 280, "max_pulse": 720, "neutral_pulse": 500, "role": "lower_body"},
    {"servo_id": 13, "name": "l_hip_roll", "min_pulse": 360, "max_pulse": 640, "neutral_pulse": 500, "role": "lower_body"},
    # Left arm (IDs 14-16)
    {"servo_id": 14, "name": "l_shoulder_pitch", "min_pulse": 150, "max_pulse": 850, "neutral_pulse": 500, "role": "upper_body"},
    {"servo_id": 15, "name": "l_shoulder_roll", "min_pulse": 180, "max_pulse": 820, "neutral_pulse": 500, "role": "upper_body"},
    {"servo_id": 16, "name": "l_elbow", "min_pulse": 180, "max_pulse": 820, "neutral_pulse": 500, "role": "upper_body"},
)

JOINTS_BY_NAME: dict[str, dict[str, Any]] = {spec["name"]: spec for spec in TONYPI_JOINT_SPECS}
LOWER_BODY_JOINTS: frozenset[str] = frozenset(
    spec["name"] for spec in TONYPI_JOINT_SPECS if spec["role"] == "lower_body"
)
NEUTRAL_POSE: dict[str, int] = {spec["name"]: int(spec["neutral_pulse"]) for spec in TONYPI_JOINT_SPECS}


def estimate_frame_tilt_and_com(servos: dict[str, int]) -> tuple[float, float]:
    """Estimate torso tilt (deg) and normalized CoM offset from 16-DOF servo pulses."""
    pose: dict[str, float] = {k: float(v) for k, v in NEUTRAL_POSE.items()}
    for name, val in servos.items():
        if name in JOINTS_BY_NAME:
            pose[name] = float(val)

    r_hip_p = (pose["r_hip_pitch"] - 500.0) * DEG_PER_PULSE
    l_hip_p = (pose["l_hip_pitch"] - 500.0) * DEG_PER_PULSE
    r_ankle_p = (pose["r_ankle_pitch"] - 500.0) * DEG_PER_PULSE
    l_ankle_p = (pose["l_ankle_pitch"] - 500.0) * DEG_PER_PULSE
    r_knee = (pose["r_knee"] - 500.0) * DEG_PER_PULSE
    l_knee = (pose["l_knee"] - 500.0) * DEG_PER_PULSE

    avg_ankle_p = 0.5 * (r_ankle_p + l_ankle_p)
    avg_hip_p = 0.5 * (r_hip_p + l_hip_p)
    avg_knee = 0.5 * (r_knee + l_knee)
    sagittal_tilt = abs(avg_ankle_p + 0.55 * avg_hip_p - 0.35 * avg_knee)

    r_ankle_r = (pose["r_ankle_roll"] - 500.0) * DEG_PER_PULSE
    l_ankle_r = (pose["l_ankle_roll"] - 500.0) * DEG_PER_PULSE
    r_hip_r = (pose["r_hip_roll"] - 500.0) * DEG_PER_PULSE
    l_hip_r = (pose["l_hip_roll"] - 500.0) * DEG_PER_PULSE
    knee_asym = abs(r_knee - l_knee)
    hip_asym = abs(r_hip_p - l_hip_p)

    lateral_tilt = (
        abs(0.5 * (r_ankle_r + l_ankle_r) + 0.45 * (r_hip_r + l_hip_r))
        + 0.22 * knee_asym
        + 0.12 * hip_asym
    )

    r_sh_p = (pose["r_shoulder_pitch"] - 500.0) * DEG_PER_PULSE
    l_sh_p = (pose["l_shoulder_pitch"] - 500.0) * DEG_PER_PULSE
    r_sh_r = (pose["r_shoulder_roll"] - 500.0) * DEG_PER_PULSE
    l_sh_r = (pose["l_shoulder_roll"] - 500.0) * DEG_PER_PULSE
    arm_pitch_bias = 0.06 * abs(0.5 * (r_sh_p + l_sh_p))
    arm_roll_bias = 0.05 * abs(r_sh_r - l_sh_r)

    total_tilt_deg = math.hypot(sagittal_tilt + arm_pitch_bias, lateral_tilt + arm_roll_bias)
    com_offset_norm = total_tilt_deg / MAX_TORSO_TILT_DEG
    return round(total_tilt_deg, 2), round(com_offset_norm, 3)


def _extract_keyframes(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        kfs = payload.get("keyframes")
        if isinstance(kfs, list):
            return kfs
    raise ValueError("Input JSON must be a list of keyframes or an object with a 'keyframes' list.")


def verify_keyframes(keyframes: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate 16-DOF joint bounds, durations, slew rates, CoM, and torso tilt."""
    if not keyframes:
        return {
            "verified": False,
            "peak_torso_tilt_deg": 0.0,
            "max_com_offset_norm": 0.0,
            "joint_clipping_count": 0,
            "max_slew_rate_pps": 0.0,
            "violations": ["Skill contains zero keyframes"],
            "suggested_fixes": ["Provide at least 1 valid ServoKeyframe or return refused=true."],
        }

    clipping_count = 0
    peak_tilt = 0.0
    max_com = 0.0
    max_slew = 0.0
    violations: list[str] = []
    suggested_fixes: list[str] = []

    prev_pose = dict(NEUTRAL_POSE)

    for idx, frame in enumerate(keyframes):
        if not isinstance(frame, dict):
            violations.append(f"frame[{idx}] is not a JSON object")
            suggested_fixes.append(f"frame[{idx}]: represent each keyframe as {{duration_ms, servos, note}}")
            continue

        raw_duration = frame.get("duration_ms")
        if not isinstance(raw_duration, (int, float)) or isinstance(raw_duration, bool):
            violations.append(f"frame[{idx}] missing numeric duration_ms")
            suggested_fixes.append(f"frame[{idx}]: set duration_ms between {MIN_KEYFRAME_DURATION_MS} and {MAX_KEYFRAME_DURATION_MS}")
            duration_ms = MIN_KEYFRAME_DURATION_MS
        else:
            duration_ms = int(raw_duration)
            if duration_ms < MIN_KEYFRAME_DURATION_MS:
                violations.append(
                    f"frame[{idx}] duration_ms={duration_ms} < {MIN_KEYFRAME_DURATION_MS}ms"
                )
                suggested_fixes.append(
                    f"frame[{idx}]: increase duration_ms to at least {MIN_KEYFRAME_DURATION_MS}"
                )
            elif duration_ms > MAX_KEYFRAME_DURATION_MS:
                violations.append(
                    f"frame[{idx}] duration_ms={duration_ms} > {MAX_KEYFRAME_DURATION_MS}ms"
                )
                suggested_fixes.append(
                    f"frame[{idx}]: split transition or clamp duration_ms <= {MAX_KEYFRAME_DURATION_MS}"
                )

        servos = frame.get("servos")
        if not isinstance(servos, dict):
            violations.append(f"frame[{idx}] missing 'servos' dictionary")
            suggested_fixes.append(f"frame[{idx}]: provide a 'servos' dict mapping joint names to pulses")
            continue

        dt_sec = max(duration_ms / 1000.0, 0.001)
        merged_pose = dict(NEUTRAL_POSE)

        for joint_name, raw_pulse in servos.items():
            spec = JOINTS_BY_NAME.get(joint_name)
            if spec is None:
                clipping_count += 1
                violations.append(f"frame[{idx}] unknown joint '{joint_name}'")
                suggested_fixes.append(f"frame[{idx}]: remove unknown joint '{joint_name}'")
                continue

            if not isinstance(raw_pulse, (int, float)) or isinstance(raw_pulse, bool):
                clipping_count += 1
                violations.append(f"frame[{idx}] {joint_name} has non-numeric pulse {raw_pulse!r}")
                suggested_fixes.append(
                    f"frame[{idx}]: set {joint_name} to an integer in [{spec['min_pulse']}, {spec['max_pulse']}]"
                )
                continue

            pulse = round(raw_pulse)
            merged_pose[joint_name] = pulse

            if pulse < spec["min_pulse"] or pulse > spec["max_pulse"]:
                clipping_count += 1
                clamped = max(spec["min_pulse"], min(spec["max_pulse"], pulse))
                violations.append(
                    f"frame[{idx}] {joint_name}={pulse} outside [{spec['min_pulse']}, {spec['max_pulse']}]"
                )
                suggested_fixes.append(f"frame[{idx}]: clamp {joint_name} from {pulse} to {clamped}")

            slew = abs(pulse - prev_pose[joint_name]) / dt_sec
            max_slew = max(max_slew, slew)
            if slew > MAX_SLEW_PULSE_PER_SEC:
                min_safe_ms = math.ceil((abs(pulse - prev_pose[joint_name]) / (MAX_SLEW_PULSE_PER_SEC * 0.90)) * 1000.0)
                violations.append(
                    f"frame[{idx}] {joint_name} slew {slew:.1f} pps > {MAX_SLEW_PULSE_PER_SEC:.0f} pps"
                )
                suggested_fixes.append(
                    f"frame[{idx}]: increase duration_ms from {duration_ms} to >= {min_safe_ms} for {joint_name}"
                )

        tilt_deg, com_norm = estimate_frame_tilt_and_com(merged_pose)
        peak_tilt = max(peak_tilt, tilt_deg)
        max_com = max(max_com, com_norm)
        if tilt_deg >= MAX_TORSO_TILT_DEG:
            violations.append(
                f"frame[{idx}] estimated torso tilt {tilt_deg:.1f} deg >= {MAX_TORSO_TILT_DEG:.1f} deg"
            )
            suggested_fixes.append(
                f"frame[{idx}]: project lower body to grounded rhythmic knee-bounce (r_knee/l_knee 500..540, counter-pitched hips/ankles) to bring tilt < {MAX_TORSO_TILT_DEG:.1f} deg"
            )

        prev_pose = merged_pose

    verified = (
        clipping_count == 0
        and peak_tilt < MAX_TORSO_TILT_DEG
        and max_slew <= MAX_SLEW_PULSE_PER_SEC
        and len(violations) == 0
    )

    return {
        "verified": verified,
        "peak_torso_tilt_deg": round(peak_tilt, 2),
        "max_com_offset_norm": round(max_com, 3),
        "joint_clipping_count": clipping_count,
        "max_slew_rate_pps": round(max_slew, 1),
        "violations": violations,
        "suggested_fixes": suggested_fixes,
    }


def auto_fix_keyframes(keyframes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool, list[str]]:
    """Deterministically clamp limits, project lower body when tilt >= 28 deg, and adjust durations."""
    fixed_frames: list[dict[str, Any]] = []
    lower_body_adapted = False
    notes: list[str] = []
    prev_pose = dict(NEUTRAL_POSE)

    for idx, frame in enumerate(keyframes):
        if not isinstance(frame, dict):
            continue
        raw_servos = frame.get("servos") if isinstance(frame.get("servos"), dict) else {}
        new_servos: dict[str, int] = dict(NEUTRAL_POSE)

        # Clamp all known joints into hardware bounds
        for joint_name, raw_pulse in raw_servos.items():
            spec = JOINTS_BY_NAME.get(joint_name)
            if spec is None:
                notes.append(f"frame[{idx}]: dropped unknown joint '{joint_name}'")
                continue
            if isinstance(raw_pulse, (int, float)) and not isinstance(raw_pulse, bool):
                rounded = round(raw_pulse)
                clamped = max(spec["min_pulse"], min(spec["max_pulse"], rounded))
                if clamped != rounded:
                    notes.append(f"frame[{idx}]: clamped {joint_name} to {clamped}")
                new_servos[joint_name] = clamped

        # Check tilt; project lower body to grounded athletic bounce if tilt >= 28 deg
        tilt_deg, _ = estimate_frame_tilt_and_com(new_servos)
        note_str = str(frame.get("note", ""))
        if tilt_deg >= AUTO_FIX_TILT_TRIGGER_DEG:
            lower_body_adapted = True
            bounce = 24 if idx % 2 == 1 else 0
            for j in LOWER_BODY_JOINTS:
                new_servos[j] = 500
            new_servos["r_knee"] = 500 + bounce
            new_servos["l_knee"] = 500 + bounce
            new_servos["r_hip_pitch"] = 500 - bounce // 2
            new_servos["l_hip_pitch"] = 500 - bounce // 2
            new_servos["r_ankle_pitch"] = 500 + bounce // 2
            new_servos["l_ankle_pitch"] = 500 + bounce // 2
            notes.append(
                f"frame[{idx}]: projected lower body to grounded knee-bounce (pre-fix tilt {tilt_deg:.1f} deg >= {AUTO_FIX_TILT_TRIGGER_DEG:.1f} deg)"
            )
            if "[lower-body projected]" not in note_str:
                note_str = f"{note_str} [lower-body projected]".strip()

        raw_duration = frame.get("duration_ms", 400)
        base_duration = (
            round(raw_duration)
            if isinstance(raw_duration, (int, float)) and not isinstance(raw_duration, bool)
            else 400
        )

        max_delta = max(abs(new_servos[k] - prev_pose[k]) for k in new_servos)
        min_safe_ms = math.ceil((max_delta / (MAX_SLEW_PULSE_PER_SEC * 0.90)) * 1000.0)
        safe_duration = min(MAX_KEYFRAME_DURATION_MS, max(base_duration, min_safe_ms, MIN_KEYFRAME_DURATION_MS))
        if safe_duration != base_duration:
            notes.append(f"frame[{idx}]: adjusted duration_ms from {base_duration} to {safe_duration}")

        fixed_frames.append(
            {
                "duration_ms": safe_duration,
                "servos": new_servos,
                "note": note_str,
            }
        )
        prev_pose = new_servos

    return fixed_frames, lower_body_adapted, notes


def run_self_test() -> dict[str, Any]:
    """Execute built-in deterministic verification sanity checks."""
    checks_passed = 0

    # 1. Neutral pose passes
    neutral_report = verify_keyframes([{"duration_ms": 400, "servos": dict(NEUTRAL_POSE), "note": "neutral"}])
    if not neutral_report["verified"] or neutral_report["peak_torso_tilt_deg"] != 0.0:
        raise AssertionError(f"Neutral self-test failed: {neutral_report}")
    checks_passed += 1

    # 2. Clipped servo fails
    clipped_servos = dict(NEUTRAL_POSE)
    clipped_servos["r_shoulder_pitch"] = 960
    clipped_report = verify_keyframes([{"duration_ms": 400, "servos": clipped_servos, "note": "clipped"}])
    if clipped_report["verified"] or clipped_report["joint_clipping_count"] != 1:
        raise AssertionError(f"Clipped self-test failed: {clipped_report}")
    checks_passed += 1

    # 3. High tilt fails
    tilted_servos = dict(NEUTRAL_POSE)
    tilted_servos["r_ankle_pitch"] = 680
    tilted_servos["l_ankle_pitch"] = 680
    tilted_servos["r_hip_pitch"] = 720
    tilted_servos["l_hip_pitch"] = 720
    tilted_report = verify_keyframes([{"duration_ms": 600, "servos": tilted_servos, "note": "tilted"}])
    if tilted_report["verified"] or tilted_report["peak_torso_tilt_deg"] < MAX_TORSO_TILT_DEG:
        raise AssertionError(f"High-tilt self-test failed: {tilted_report}")
    checks_passed += 1

    # 4. Excessive slew fails
    fast_servos = dict(NEUTRAL_POSE)
    fast_servos["r_shoulder_pitch"] = 800
    fast_report = verify_keyframes([{"duration_ms": 100, "servos": fast_servos, "note": "fast"}])
    if fast_report["verified"] or fast_report["max_slew_rate_pps"] <= MAX_SLEW_PULSE_PER_SEC:
        raise AssertionError(f"Slew self-test failed: {fast_report}")
    checks_passed += 1

    # 5. Auto-fix repairs clipped + high-tilt + fast keyframes into verified output
    bad_frames = [
        {"duration_ms": 100, "servos": {**tilted_servos, "r_shoulder_pitch": 960}, "note": "bad0"},
        {"duration_ms": 100, "servos": {**tilted_servos, "l_shoulder_pitch": 100}, "note": "bad1"},
    ]
    fixed, adapted, _ = auto_fix_keyframes(bad_frames)
    fixed_report = verify_keyframes(fixed)
    if not fixed_report["verified"] or not adapted or fixed_report["joint_clipping_count"] != 0:
        raise AssertionError(f"Auto-fix self-test failed: {fixed_report}")
    checks_passed += 1

    return {"self_test": "ok", "checks_passed": checks_passed, "joints": len(TONYPI_JOINT_SPECS)}


def _read_json_input(input_path: str) -> Any:
    if input_path == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(input_path).read_text(encoding="utf-8")
    return json.loads(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify or auto-fix 16-DOF Hiwonder TonyPi skill keyframes."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Path to candidate skill JSON file, or '-' to read JSON from stdin.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in sanity checks and exit 0 on success.",
    )
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Deterministically clamp limits, fix slew durations, and project lower body if tilt >= 28 deg.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for --auto-fix repaired JSON.",
    )

    args = parser.parse_args(argv)

    if args.self_test:
        try:
            result = run_self_test()
        except AssertionError as exc:
            print(json.dumps({"self_test": "failed", "error": str(exc)}, indent=2))
            return 1
        print(json.dumps(result, indent=2))
        return 0

    if not args.input_path:
        parser.error("Provide an input JSON path (or '-') or pass --self-test.")

    try:
        payload = _read_json_input(args.input_path)
        keyframes = _extract_keyframes(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "verified": False,
                    "peak_torso_tilt_deg": 0.0,
                    "max_com_offset_norm": 0.0,
                    "joint_clipping_count": 0,
                    "max_slew_rate_pps": 0.0,
                    "violations": [f"Failed to parse input JSON: {exc}"],
                    "suggested_fixes": ["Supply valid JSON containing a 'keyframes' array."],
                },
                indent=2,
            )
        )
        return 1

    if args.auto_fix:
        fixed_keyframes, lower_body_adapted, notes = auto_fix_keyframes(keyframes)
        report = verify_keyframes(fixed_keyframes)
        report["lower_body_adapted"] = lower_body_adapted
        report["adaptation_notes"] = notes

        if isinstance(payload, dict):
            out_payload = dict(payload)
            out_payload["keyframes"] = fixed_keyframes
            if lower_body_adapted:
                out_payload["adaptation_mode"] = "upper_body_adapted"
                out_payload["lower_body_adapted"] = True
            existing_notes = list(out_payload.get("adaptation_notes", [])) if isinstance(out_payload.get("adaptation_notes"), list) else []
            out_payload["adaptation_notes"] = existing_notes + notes
            out_payload["kinematics"] = report
        else:
            out_payload = {"keyframes": fixed_keyframes, "kinematics": report}

        if args.out and args.out != "-":
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(out_payload, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2))
        else:
            print(json.dumps(out_payload, indent=2))
        return 0 if report["verified"] else 1

    report = verify_keyframes(keyframes)
    print(json.dumps(report, indent=2))
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
