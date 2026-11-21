---
name: kinematics-verifier
description: Retarget human motion keyframes to Hiwonder TonyPi 16-DOF servo bounds and verify CoM, tilt < 30 deg, and zero clipping
---

# Hiwonder TonyPi 16-DOF Kinematics Retargeting & Verification Skill

Use this skill to retarget extracted human motion (`PoseSequence`) into safe, beat-synced 16-DOF Hiwonder TonyPi servo keyframes and deterministically verify physical feasibility inside the sandbox.

## 1. Hiwonder TonyPi 16-DOF Servo Bounds (`0..1000` pulses, `~0.24 deg/pulse`)

Every keyframe `servos` dictionary MUST use ONLY these canonical joint names and stay strictly inside `[min_pulse, max_pulse]`:

| Servo ID | Joint Name | Min Pulse | Neutral Pulse | Max Pulse | Role |
|---:|---|---:|---:|---:|---|
| 1 | `r_ankle_roll` | 350 | 500 | 650 | `lower_body` |
| 2 | `r_ankle_pitch` | 320 | 500 | 680 | `lower_body` |
| 3 | `r_knee` | 260 | 500 | 740 | `lower_body` |
| 4 | `r_hip_pitch` | 280 | 500 | 720 | `lower_body` |
| 5 | `r_hip_roll` | 360 | 500 | 640 | `lower_body` |
| 6 | `r_shoulder_pitch` | 150 | 500 | 850 | `upper_body` |
| 7 | `r_shoulder_roll` | 180 | 500 | 820 | `upper_body` |
| 8 | `r_elbow` | 180 | 500 | 820 | `upper_body` |
| 9 | `l_ankle_roll` | 350 | 500 | 650 | `lower_body` |
| 10 | `l_ankle_pitch` | 320 | 500 | 680 | `lower_body` |
| 11 | `l_knee` | 260 | 500 | 740 | `lower_body` |
| 12 | `l_hip_pitch` | 280 | 500 | 720 | `lower_body` |
| 13 | `l_hip_roll` | 360 | 500 | 640 | `lower_body` |
| 14 | `l_shoulder_pitch` | 150 | 500 | 850 | `upper_body` |
| 15 | `l_shoulder_roll` | 180 | 500 | 820 | `upper_body` |
| 16 | `l_elbow` | 180 | 500 | 820 | `upper_body` |

## 2. Hard Physical Safety Limits

- **Minimum keyframe duration**: `duration_ms >= 80` (and `<= 5000`).
- **Peak torso tilt**: `peak_torso_tilt_deg < 30.0` deg (target `< 20.0` deg for comfortable margin).
- **Normalized CoM offset**: `max_com_offset_norm < 1.0` (`total_tilt_deg / 30.0`).
- **Zero joint clipping**: `joint_clipping_count == 0` across all 16 servos.
- **Peak servo slew rate**: `max_slew_rate_pps <= 900.0` pulses/sec (including transition from neutral `500` into frame 0).

## 3. Retargeting Heuristics

### Angle-to-Pulse Mapping
Convert human joint degrees to TonyPi servo pulse deltas around neutral `500`:

```text
pulse = clamp(round(500 + sign * deg * 3.2), min_pulse, max_pulse)
```

Recommended sign conventions (mirrored left/right upper limbs):
- Right arm: `r_shoulder_pitch` (`+1`), `r_shoulder_roll` (`+1`), `r_elbow` (`+1`).
- Left arm: `l_shoulder_pitch` (`-1`), `l_shoulder_roll` (`-1`), `l_elbow` (`-1`).
- Sagittal legs (coordinated squat/bounce): positive knee bend `d = 0..40` pulses paired with counter-pitched hips (`500 - d // 2`) and counter-pitched ankles (`500 + d // 2`) to keep the torso upright.
- Timing: compute `dt_ms` from beat timestamps (`t_ms[i] - t_ms[i-1]`), clamped to `[80, 5000]`. Increase `duration_ms` whenever `max(|pulse[i] - pulse[i-1]|) / (duration_ms / 1000.0) > 900.0`.

### Impossible-Motion & Lower-Body Adaptation Rules
1. **Airborne / Jump / Hop**:
   - TonyPi cannot leave the table surface safely.
   - Replace airborne phases with a grounded rhythmic knee bounce (`r_knee` and `l_knee` oscillating in `500..540`, counter-pitched `r_hip_pitch`/`l_hip_pitch = 500 - bounce // 2`, `r_ankle_pitch`/`l_ankle_pitch = 500 + bounce // 2`, roll joints neutral `500`) paired with expressive upper-body arm choreography.
   - Set `"adaptation_mode": "upper_body_adapted"` and `"lower_body_adapted": true`.
2. **Single-Leg Flamingo / Deep Lunge**:
   - Shift `hip_roll` and `ankle_roll` gently so total torso tilt stays strictly `< 20.0` deg, OR lock the lower body to a grounded athletic stance if leg asymmetry pushes tilt near the limit.
3. **Backflip / Full Splits / Airborne Somersault**:
   - If the prompt explicitly requests an upper-body tribute (via `adaptation_hint`), produce a grounded upper-body tribute (`"adaptation_mode": "upper_body_adapted"`).
   - Otherwise safely refuse (`"refused": true`, concise `"refusal_reason"`, `"keyframes": []`).

## 4. Deterministic Verification CLI

Run the verifier inside the sandbox before returning your final JSON:

```bash
# Built-in self-test (sanity check)
python3 /workspace/kinematics_verify.py --self-test

# Verify candidate skill JSON (exits 0 if verified, 1 if violations exist)
python3 /workspace/kinematics_verify.py /workspace/candidate_skill.json

# Optional deterministic auto-repair (clamps bounds, fixes slew durations, projects lower body if tilt >= 28 deg)
python3 /workspace/kinematics_verify.py --auto-fix /workspace/candidate_skill.json --out /workspace/candidate_skill.json
```

Always re-run `python3 /workspace/kinematics_verify.py /workspace/candidate_skill.json` after any edit and confirm exit code `0` (`"verified": true`) before emitting your final JSON response.
