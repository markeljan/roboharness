# RoboHarness — Clean-Room Constitution & Gemini 4 Builder Guide Invariants

## 1. Clean-Room Boundary
- **Net-New Code Only**: All code inside `Projects/roboharness` is written from scratch for the Gemini 4 NYC Hackathon.
- **External Hardware Driver Boundary**: Treat `markeljan/tonypi` (daemon + MuJoCo bridge) strictly as an external, unchanged MIT hardware driver accessed **ONLY** over Streamable HTTP MCP (`TONYPI_MCP_URL`, default `http://127.0.0.1:8765/mcp`).
- **Zero Copy-Paste**: Never copy-paste files, kinematics implementations, or modules from `markeljan/tonypi`.
- **Offline Determinism & Zero Mocked Live Paths**: Always provide and test against `MockTonypiMCPServer` under explicit `--offline` mode so automated tests & offline evals run in `<2s` without physical hardware. Live paths (`offline_mode=False`, `--mode live`, CLI without `--offline`) MUST NEVER fall back to mocked or synthetic demo responses.
- **UI Discipline**: No Streamlit. Any visual monitor must be a minimal live SSE telemetry strip strictly subordinate to the physical robot.

## 2. Gemini 4 Dual-Client SDK Invariants (`google-genai >= 3.2.0`)
- **Dual Clients (`src/roboharness/models.py`)**:
  - **Stable `v1` client (`make_v1_client`)**: `genai.Client(api_key=...)` WITHOUT `api_version="v1beta"` — used for all GA Gemini 4 Interactions API model calls (`gemini-4-hydrogen`, `gemini-4-neon`, `gemini-4-carbon`).
  - **Beta `v1beta` client (`make_beta_client`)**: `genai.Client(api_key=..., http_options={"api_version": "v1beta"})` — used for Managed Agents (`antigravity-preview-09-2026`), Environments, Decisions API (`gemini-decision-1`), and preview robotics model (`gemini-robotics-er-2-preview`).
- **Forbidden Parameters on Gemini 4 Models** (`gemini-4-neon`, `gemini-4-hydrogen`, `gemini-4-carbon`, `gemini-4-argon`):
  - NEVER pass `temperature`, `top_p`, `top_k`, `cachedContents`, `cached_content`, `max_output_tokens`, or `max_total_tokens` (leave token budgets uncapped on Gemini Ultra).
  - Valid `thinking_level` values ONLY: `"low"`, `"medium"`, `"high"`, `"xhigh"` (NEVER `"minimal"`). Standard Interactions calls use `"medium"` or `"high"`.
- **Agentic Video Processing**:
  - Upload video clips via `client.files.upload(...)`, poll until `ACTIVE`, and pass `{"type": "video", "uri": clip.uri, "mime_type": mime, "processing": "agentic"}` so Gemini 4 actively scrubs timestamps and zooms into limbs during motion extraction and trial debugging.
- **Managed Agents (`antigravity-preview-09-2026`, `v1beta`)**:
  - Custom agent `id` MUST NOT start with reserved prefixes (`gemini-`, `google-`, `antigravity-`, `deep-research-`, `gemma-`). Use `id="roboharness-skill-forge"` (`gemini-4-neon`) and `id="roboharness-skill-forge-carbon"` (`gemini-4-carbon`).
  - Lock model via `agent_config={"type": "antigravity", "model": ...}` at `beta_client.agents.create(...)` time (never override at interaction time, and NEVER set `max_total_tokens`).
  - **Neon -> Carbon Escalation**: Run `gemini-4-neon` by default; escalate automatically to `gemini-4-carbon` (`roboharness-skill-forge-carbon`) if `gemini-4-neon` fails deterministic kinematics verification twice OR when `--deep` is requested.
  - **Hosted vs Self-Hosted Environment**: Default to `environment="remote"` with pre-staged `.agents/AGENTS.md`, `.agents/skills/kinematics-verifier/SKILL.md`, and `/workspace/kinematics_verify.py` via environment forking (`base_environment=bootstrap.environment_id`). When `GEMINI_ENVIRONMENT_ID` is set (self-hosted Docker/MuJoCo environment), pass `environment=settings.gemini_environment_id` and `base_environment=settings.gemini_environment_id` while strictly **omitting** hosted-only `sources` and `network.allowlist`.
  - Any remote MCP tool `name` MUST match `^[a-z0-9_-]+$` (e.g., `"tonypi_mcp"`).
- **Decisions API (`gemini-decision-1`, `v1beta`)**:
  - Call `beta_client.decisions.create(model="gemini-decision-1", input=[...], questions={...})` (0 output tokens).
  - Supported question types:
    - `"noul"`: returns `answer.noul` (`0.01..0.99`) and `answer.confidence`.
    - `"choice"`: `criteria` dict; returns `answer.choice`, `answer.probabilities`, `answer.confidence`. **NEVER** use `"none_of_the_above"` — always use `"other"`.
    - `"score"`: ordered list `criteria`; returns `answer.score`, `answer.legend`, `answer.confidence`.
  - Pass camera/video frames as `{"type": "image", "data": b64_jpg, "mime_type": "image/jpeg", "resolution": "low"}` and robot telemetry as `{"type": "text", "text": json.dumps(state)}`.
  - Auto-branch execution **ONLY** when `confidence >= 0.75` **AND** `is_physically_safe.noul >= 0.75`; otherwise escalate safely to Managed Forge or human review without moving servos.

## 3. Hardware Safety Latches (Deterministic, Non-LLM)
Every skill must pass local deterministic safety verification before any servo command reaches `TONYPI_MCP_URL`:
- **Battery Voltage**: `battery_v >= 10.9` V
- **Servo Temperature**: `max_servo_temp_c < 70.0` °C
- **Peak Torso Tilt**: `peak_torso_tilt_deg < 30.0` deg
- **Joint-Limit Clipping**: `joint_clipping_count == 0` across all 16 TonyPi servos

## 4. 4-Model (+ Carbon Escalation) Architecture (Zero Overlap)
1. `gemini-4-hydrogen` (`v1`, `thinking_level="medium"` / `"high"` + `processing="agentic"`): Video motion extraction, trial debugging (`src/roboharness/motion_extractor.py`), and live reflex speech.
2. Managed Agent `roboharness-skill-forge` (`v1beta`, `antigravity-preview-09-2026` + `gemini-4-neon`, escalating to `roboharness-skill-forge-carbon` on `gemini-4-carbon` after 2 verification failures or `--deep`): Sandboxed 16-DOF retargeting, lower-body adaptation, and Python physics verification (`src/roboharness/managed_forge.py` + `agent_bundle/`).
3. `gemini-decision-1` (`v1beta`, 0 output tokens): Multimodal safety (`>= 0.75`), confidence (`>= 0.75`), feasibility, and fidelity gate (`src/roboharness/decision_gate.py`).
4. `gemini-robotics-er-2-preview` (`v1beta`, `thinking_level="medium"`): Live spatial workspace framing check on `perceive` frames (`src/roboharness/spatial.py`).
