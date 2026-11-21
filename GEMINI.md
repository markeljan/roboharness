# RoboHarness — Clean-Room Constitution & Gemini 4 API Invariants

## 1. Clean-Room Boundary
- **Net-New Code Only**: All code inside `Projects/roboharness` is written from scratch for the Gemini 4 NYC Hackathon.
- **External Hardware Driver Boundary**: Treat `markeljan/tonypi` (daemon + MuJoCo bridge) strictly as an external, unchanged MIT hardware driver accessed **ONLY** over Streamable HTTP MCP (`TONYPI_MCP_URL`, default `http://127.0.0.1:8765/mcp`).
- **Zero Copy-Paste**: Never copy-paste files, kinematics implementations, or modules from `markeljan/tonypi`.
- **Offline Determinism**: Always provide and test against `MockTonypiMCPServer` so automated tests & offline evals run in `<2s` without physical hardware.
- **UI Discipline**: No Streamlit. Any visual monitor must be a minimal live SSE telemetry strip strictly subordinate to the physical robot.

## 2. Gemini 4 `v1beta` SDK Invariants (`google-genai >= 3.2.0`)
- Always instantiate clients via `genai.Client(http_options={"api_version": "v1beta"})`.
- **Forbidden Parameters on Gemini 4 Models** (`gemini-4-neon`, `gemini-4-hydrogen`, `gemini-4-carbon`, `gemini-4-argon`):
  - NEVER pass `temperature`, `top_p`, `top_k`, or `cachedContents`.
  - Leave `max_output_tokens` unset (65536 default).
  - Valid `thinking_level` values ONLY: `"low"`, `"medium"`, `"high"`, `"xhigh"` (NEVER `"minimal"`).
- **Managed Agents (`antigravity-preview-09-2026`)**:
  - Custom agent `id` MUST NOT start with reserved prefixes (`gemini-`, `google-`, `antigravity-`, etc.). Use `id="roboharness-skill-forge"`.
  - Lock model via `agent_config={"type": "antigravity", "model": "gemini-4-neon"}` at `client.agents.create(...)` time (never override at interaction time).
  - Pre-stage `.agents/AGENTS.md`, `.agents/skills/kinematics-verifier/SKILL.md`, and `/workspace/kinematics_verify.py` via environment forking (`base_environment=interaction.environment_id`) for fast live demo execution.
  - Any remote MCP tool `name` MUST match `^[a-z0-9_-]+$` (e.g., `"tonypi_mcp"`).
- **Decisions API (`gemini-decision-1`)**:
  - Call `client.decisions.create(model="gemini-decision-1", input=[...], questions={...})` (0 output tokens).
  - Supported question types:
    - `"noul"`: returns `answer.noul` (`0.01..0.99`) and `answer.confidence`.
    - `"choice"`: `criteria` dict; returns `answer.choice`, `answer.probabilities`, `answer.confidence`. **NEVER** use `"none_of_the_above"` — always use `"other"`.
    - `"score"`: ordered list `criteria`; returns `answer.score`, `answer.legend`, `answer.confidence`.
  - Pass camera/video frames as `{"type": "image", "data": b64_jpg, "mime_type": "image/jpeg", "resolution": "low"}` and robot telemetry as `{"type": "text", "text": json.dumps(state)}`.
  - Auto-branch **ONLY** when `confidence >= 0.70`; otherwise escalate safely without moving servos.

## 3. Hardware Safety Latches (Deterministic, Non-LLM)
Every skill must pass local deterministic safety verification before any servo command reaches `TONYPI_MCP_URL`:
- **Battery Voltage**: `battery_v >= 10.9` V
- **Servo Temperature**: `max_servo_temp_c < 70.0` °C
- **Peak Torso Tilt**: `peak_torso_tilt_deg < 30.0` deg
- **Joint-Limit Clipping**: `joint_clipping_count == 0` across all 16 TonyPi servos

## 4. 4-Model Architecture (Zero Overlap)
1. `gemini-4-hydrogen` (`thinking_level="medium"`): Video motion extraction (`src/roboharness/motion_extractor.py`) + fast live reflex coordinator (`status` + `perceive` + `say`).
2. Managed Agent `roboharness-skill-forge` (`antigravity-preview-09-2026` + `gemini-4-neon`): Sandboxed 16-DOF retargeting, lower-body adaptation, and Python physics verification (`src/roboharness/managed_forge.py` + `agent_bundle/`).
3. `gemini-decision-1`: 0-output-token multimodal safety, feasibility, and fidelity gate (`src/roboharness/decision_gate.py`).
4. `gemini-robotics-er-2-preview` (`thinking_level="medium"`): Live spatial workspace framing check on `perceive` frames (`src/roboharness/spatial.py`).
