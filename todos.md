# openenv-fintech — Build TODO
**Stack:** FastAPI + Python 3.11 | **Target:** OpenEnv submission on HuggingFace Spaces  
**Order:** Do Phase 0 entirely before touching any code. Then phases 1–6 in sequence.

---

## PHASE 0 — Pre-Build Setup (Do This First, No Code Yet)

These are decisions and installations that must be locked before you write a single line.

### 0.1 — Environment & Tooling
- [ ] Install Python 3.11 (not 3.12 — some deps lag) and confirm `python --version`
- [ ] Install Docker Desktop and confirm `docker build` works on a test image
- [ ] Install `openenv-core` globally: `pip install openenv-core` and confirm `openenv --version`
- [ ] Install `uv` or confirm `pip` + `venv` setup — pick one and stick to it
- [ ] Create the project repo: `openenv-fintech/` with git initialized
- [ ] Create `.env.example` file now with these keys (never commit real values):
  ```
  API_BASE_URL=https://router.huggingface.co/v1
  MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
  HF_TOKEN=
  IMAGE_NAME=
  ```

### 0.2 — Architecture Decisions (Lock These Before Coding)

**Decision 1: Env instantiation model**  
The sample inference script uses `MyEnvV4Env.from_docker_image(IMAGE_NAME)` with async/await.  
Your env will run as a **FastAPI server inside Docker**, and inference.py will talk to it over HTTP.  
→ Lock this: inference.py hits `http://localhost:7860/reset`, `/step`, `/state` via HTTP — NOT direct Python import.

**Decision 2: Async vs sync**  
FastAPI supports both. Your env logic (data gen, graders) is CPU-bound and sync.  
→ Decision: Use `async def` endpoints in FastAPI but call sync env logic with `asyncio.run_in_executor` for the portfolio simulator (it's the only heavy one).

**Decision 3: Session/state management**  
Multiple episodes can't share a single global env object.  
→ Decision: Each `reset()` call generates a `session_id` (UUID). State is stored in a dict keyed by session_id. inference.py passes session_id on every `/step` and `/state` call.

**Decision 4: LLM judge**  
The 20-minute runtime cap on 2 vCPU / 8GB RAM kills per-episode LLM judge calls.  
→ Decision: LLM judge is OFF by default. Rule-based grader is the submission baseline. Add `--llm-judge` flag to inference.py only.

**Decision 5: Score clamping**  
Your reward functions can go negative (bad loans, missed fraud). The spec requires [0.0, 1.0].  
→ Decision: Every grader returns raw reward internally, then a `normalize_score(raw, min_possible, max_possible)` utility clamps and scales to [0.0, 1.0] before returning to the env.

### 0.3 — Confirm Final Project Structure
Lock this structure now. Do not deviate mid-build:
```
openenv-fintech/
├── .env.example
├── .gitignore                   # include .env, __pycache__, *.pyc
├── openenv.yaml
├── README.md
├── Dockerfile
├── requirements.txt
├── inference.py                 # MUST be at root, MUST be named exactly this
├── app.py                       # FastAPI app entry point (also Gradio for HF Spaces)
├── baseline.py
├── openenv_fintech/
│   ├── __init__.py
│   ├── env.py                   # FintechEnv class — the session manager
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── base.py              # BaseTask abstract class
│   │   ├── loan_underwriting.py
│   │   ├── fraud_detection.py
│   │   └── portfolio_rebalancing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── observations.py      # Pydantic v2 observation models
│   │   ├── actions.py           # Pydantic v2 action models
│   │   └── results.py           # StepResult, EpisodeResult models
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generators.py        # All synthetic data generation
│   │   └── seeds.py             # Seed management utility
│   └── graders/
│       ├── __init__.py
│       ├── base.py              # BaseGrader abstract class
│       ├── rule_based.py        # One grader per task
│       └── llm_judge.py         # Optional, disabled by default
└── tests/
    ├── test_env.py
    ├── test_graders.py
    ├── test_reproducibility.py
    └── test_api.py              # FastAPI endpoint tests
```

### 0.4 — Write `openenv.yaml` First (Validator Checks This)
Create this file before any Python. The `openenv validate` command reads it first:
```yaml
name: openenv-fintech
version: 1.0.0
tasks:
  - id: loan_underwriting
    difficulty: easy
    max_steps: 3
    score_range: [0.0, 1.0]
  - id: fraud_detection
    difficulty: medium
    max_steps: 20
    score_range: [0.0, 1.0]
  - id: portfolio_rebalancing
    difficulty: hard
    max_steps: 20
    score_range: [0.0, 1.0]
observation_space: typed_dict
action_space: typed_dict
reward: partial_credit
reproducible: seed_deterministic
```

---

## PHASE 1 — Pydantic Models (No Logic, Just Schemas)

Build all typed models first. Zero business logic here — just data shapes.

### 1.1 — Observation Models (`models/observations.py`)
- [ ] `LoanObservation` — all fields from the design doc (applicant_id, credit_score, annual_income, debt_to_income_ratio, employment_years, loan_amount, loan_purpose enum, previous_defaults, documents_submitted)
- [ ] `TransactionObservation` — transaction fields + `transaction_history_summary` nested model + `velocity_signals` nested model
- [ ] `PortfolioAsset` — nested model (current_weight, target_weight, current_price, shares_held, unrealized_pnl)
- [ ] `MarketContext` — volatility_regime enum, recent_returns dict, correlation_matrix
- [ ] `PortfolioConstraints` — max_single_trade_pct, min_cash_reserve, tax_lot_consideration, rebalancing_budget_usd
- [ ] `AccountState` — cash, total_value, days_remaining
- [ ] `PortfolioObservation` — step int, portfolio dict, market_context, constraints, account

### 1.2 — Action Models (`models/actions.py`)
- [ ] `LoanAction` — decision enum (approve/reject/request_info), reasoning str, risk_tier enum (low/medium/high), interest_rate_suggestion optional float
- [ ] `TradeOrder` — nested model: asset_id, direction enum (buy/sell/hold), amount_usd, rationale
- [ ] `FraudAction` — flag bool, confidence float, hold bool, reason_code enum, notes str
- [ ] `PortfolioAction` — trades list[TradeOrder], defer_rebalancing bool, risk_comment str
- [ ] Add Pydantic validators: interest_rate_suggestion must be 0.03–0.35, confidence must be 0.0–1.0, amount_usd must be positive

### 1.3 — Result Models (`models/results.py`)
- [ ] `StepResult` — observation (Union of all obs types), reward float, done bool, info dict, session_id str
- [ ] `ResetResult` — observation, session_id str, task str, seed int
- [ ] `StateSnapshot` — full serializable state (task, step, session_id, seed, episode_history list)
- [ ] `EpisodeResult` — final score float, total_reward float, steps int, success bool, breakdown dict

---

## PHASE 2 — Data Generators (Deterministic, Seeded)

All synthetic. No external API calls. Must be fully reproducible given same seed.

### 2.1 — Seed Management (`data/seeds.py`)
- [ ] `SeedManager` class — wraps `numpy.random.Generator` with `numpy.random.default_rng(seed)`
- [ ] Method: `get_rng(task: str, episode: int, seed: int) -> np.random.Generator` — derives a sub-seed per task+episode so tasks don't interfere
- [ ] Write a test: same seed → identical output across 3 calls

### 2.2 — Loan Applicant Generator (`data/generators.py`)
- [ ] `generate_loan_applicant(rng, episode_difficulty="easy") -> dict` 
- [ ] Credit score from beta distribution scaled to 300–850
- [ ] Default probability as a deterministic function of features (transparent ground truth): `p_default = sigmoid(-0.01*(credit_score-600) + 0.5*dti + 0.3*(previous_defaults>0))`
- [ ] `loan_purpose` sampled from weighted enum
- [ ] `documents_submitted` sampled from predefined list based on loan_purpose
- [ ] Store `_ground_truth` dict with `p_default` — used by grader, not exposed to agent

### 2.3 — Transaction Stream Generator (`data/generators.py`)
- [ ] `generate_transaction_stream(rng, n_transactions=15, fraud_rate=0.2) -> list[dict]`
- [ ] Markov chain over merchant categories (grocery → gas → restaurant → online → etc.)
- [ ] Amounts sampled from log-normal distribution per category
- [ ] Inject fraud patterns at generation time: velocity spike (5+ txns in 1 hour), location jump (>500km from home), amount outlier (10x avg)
- [ ] Each transaction has `_is_fraud` label stored in ground truth — not in observation
- [ ] `transaction_history_summary` computed from preceding 30 days of synthetic history

### 2.4 — Asset Price Generator (`data/generators.py`)
- [ ] `generate_price_path(rng, n_assets=10, n_steps=20) -> dict`
- [ ] Geometric Brownian Motion: `S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)`
- [ ] Regime-switching volatility: sample volatility regime at start, switch probability 0.1 per step
- [ ] Correlation matrix: generate once from seeded Cholesky decomposition
- [ ] Full price path pre-generated at `reset()` — revealed one step at a time during episode
- [ ] Store entire path in session state — do NOT regenerate per step

---

## PHASE 3 — Task Logic & Graders

One task at a time. Build → grade → test before moving to next.

### 3.1 — BaseTask (`tasks/base.py`)
- [ ] Abstract class with: `reset(seed, episode) -> Observation`, `step(action) -> StepResult`, `state() -> dict`, `is_done() -> bool`
- [ ] Each task stores: `current_step`, `max_steps`, `episode_rewards: list[float]`, `ground_truth: dict`

### 3.2 — Loan Underwriting Task (`tasks/loan_underwriting.py`)
- [ ] `reset()` — generate 1 applicant, set step=0, store ground truth p_default
- [ ] `step(action: LoanAction)` — compute reward using rule-based grader, return StepResult
- [ ] Episode ends after 1 step (approve/reject) or 2 steps (request_info → follow-up decision)
- [ ] **Reward logic:**
  - `p_default < 0.20` = good applicant, `p_default > 0.70` = bad applicant, between = ambiguous
  - Correct approve (good): +1.0 | Correct reject (bad): +1.0
  - Approve bad: -0.8 | Reject good: -0.4
  - Request info when ambiguous (0.30–0.70): +0.3
  - Risk tier matches actual band: +0.2
  - Interest rate within ±2% of actuarially fair: +0.1
  - **Clamp raw reward to [-1.0, 1.5], normalize to [0.0, 1.0]**

### 3.3 — Fraud Detection Task (`tasks/fraud_detection.py`)
- [ ] `reset()` — generate full transaction stream (15 txns), store all ground truth labels, set step=0
- [ ] `step(action: FraudAction)` — record flag for current transaction, advance step, return intermediate reward=0.0 until done
- [ ] On final step (`done=True`): run episode-level grader, return final score as reward
- [ ] **Episode-level grader:**
  - Compute precision, recall, F1 from accumulated flags vs ground truth labels
  - Confidence calibration: ECE (Expected Calibration Error) between confidence scores and accuracy
  - Early detection bonus: +0.15 if caught fraud before highest-value transaction in stream
  - Missed high-value fraud penalty: -0.2 per missed transaction with amount > $5000
  - Weighted sum: TPR*0.4 + (1-FPR)*0.3 + F1*0.2 + calibration*0.1 + bonuses/penalties
  - **Clamp to [0.0, 1.0]**

### 3.4 — Portfolio Rebalancing Task (`tasks/portfolio_rebalancing.py`)
- [ ] `reset()` — generate 10-asset price path for 20 steps, set random initial weights, set target weights, step=0
- [ ] `step(action: PortfolioAction)` — execute trades (validate constraints), advance price by one step, compute per-step partial credit, return StepResult
- [ ] Constraint violations: check max_single_trade_pct, min_cash_reserve, budget — each violation recorded
- [ ] Per-step partial credit: +0.02 per asset currently within 2% of target weight
- [ ] On final step: run episode-level grader
- [ ] **Episode-level grader:**
  - Tracking error: RMSE of final weights vs target weights, scaled to [0, 1]
  - Transaction cost efficiency: (budget_used / budget_allocated), penalize overrun
  - Sharpe ratio: compute from daily returns over episode, normalize
  - Constraint violations: -0.15 per violation
  - Early completion bonus: +0.1 if target reached with ≥3 days remaining
  - Drift penalty: -0.1 if any asset drifted >15% from target at any point
  - **Clamp to [0.0, 1.0]**

### 3.5 — LLM Judge (`graders/llm_judge.py`)
- [ ] `score_reasoning(reasoning_text: str, context: dict) -> float` — returns 0.0–0.2
- [ ] Uses Anthropic client (`claude-sonnet-4-20250514`)
- [ ] Scores on: relevance, correctness, completeness (each 0–0.067)
- [ ] Gated behind `LLM_JUDGE_ENABLED=false` env var — off by default
- [ ] Timeout: 10 seconds max per call, returns 0.0 on timeout

---

## PHASE 4 — FastAPI Server (`app.py` + `env.py`)

### 4.1 — Session Manager (`env.py`)
- [ ] `FintechEnv` class with:
  - `sessions: dict[str, BaseTask]` — active sessions
  - `create_session(task: str, seed: int) -> str` — returns session_id
  - `get_session(session_id: str) -> BaseTask` — raises 404 if not found
  - `cleanup_session(session_id: str)` — delete after episode ends
- [ ] Session TTL: auto-expire sessions older than 30 minutes (use a background task)

### 4.2 — FastAPI Endpoints (`app.py`)
Build these endpoints in this exact order:

- [ ] `POST /reset` — body: `{task: str, seed: int = 42}` → returns `ResetResult` with session_id
  - This is the endpoint the validator pings. It MUST return HTTP 200.
  - Must work even with empty body `{}` (validator sends `{}`)
  - Default task to `loan_underwriting` if not specified
- [ ] `POST /step` — body: `{session_id: str, action: dict}` → returns `StepResult`
  - Route action dict to correct Pydantic model based on session's task type
  - Return 422 on invalid action schema (Pydantic validation error)
- [ ] `GET /state/{session_id}` — returns `StateSnapshot`
- [ ] `POST /close/{session_id}` — cleanup session, return `{success: true}`
- [ ] `GET /health` — returns `{status: "ok", tasks: [...]}` — useful for debugging
- [ ] `GET /` — Gradio UI (see Phase 5) OR a simple HTML page if Gradio is separate

### 4.3 — Error Handling
- [ ] Global exception handler: all errors return `{error: str, detail: str}` with appropriate HTTP code
- [ ] 404 for unknown session_id
- [ ] 400 for wrong action type for the current task
- [ ] 422 for Pydantic validation failures (FastAPI does this automatically — confirm it's working)
- [ ] Never let an unhandled exception crash the server — wrap task `step()` calls in try/except

### 4.4 — Middleware
- [ ] CORS middleware enabled (HF Spaces iframe needs it)
- [ ] Request logging middleware: log `[REQUEST] method=POST path=/step session_id=xxx` to stdout

---

## PHASE 5 — inference.py (The Submission Script)

This file is evaluated by the judges. Get it exactly right.

### 5.1 — Required Structure
- [ ] File named exactly `inference.py`, at project root
- [ ] Read env vars at top: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` — no hardcoded values
- [ ] Use `OpenAI` client (not `anthropic` client) for all LLM calls
- [ ] `asyncio.run(main())` at bottom

### 5.2 — Stdout Format (Zero Tolerance for Deviation)
- [ ] `log_start(task, env, model)` → `[START] task=loan_underwriting env=openenv-fintech model=Qwen2.5-72B`
- [ ] `log_step(step, action, reward, done, error)` → `[STEP] step=1 action=... reward=0.00 done=false error=null`
  - `action=` must be a single-line string — JSON-encode the action dict and strip newlines
  - `reward` always 2 decimal places
  - `done` is lowercase `true` or `false`
  - `error` is `null` if no error, else the raw error string
- [ ] `log_end(success, steps, score, rewards)` → `[END] success=true steps=3 score=0.82 rewards=0.00,0.00,0.82`
  - `[END]` is ALWAYS emitted — put it in a `finally` block
  - `score` to 3 decimal places, individual rewards to 2 decimal places

### 5.3 — Agent Logic
- [ ] `async def run_episode(client, task_name, seed) -> EpisodeResult`
  - Call `POST /reset` to get session_id
  - Loop: call LLM → parse action → call `POST /step` → log step → check done
  - Call `POST /close/{session_id}` after episode
- [ ] System prompt per task — explain the task, observation format, and what a good action looks like
- [ ] User prompt per step — include current observation as JSON + step number + last reward
- [ ] LLM response must be parsed as JSON matching the action schema — add retry (max 2) on parse failure
- [ ] Hard max steps: loan=3, fraud=20, portfolio=20 — enforce in inference.py too, not just server

### 5.4 — Score Normalization in inference.py
- [ ] `score = final_reward` — already [0.0, 1.0] from server
- [ ] `success = score >= 0.1` (match the sample script's threshold)
- [ ] Collect `rewards` list across all steps for `[END]` line

---

## PHASE 6 — Docker + HF Spaces + Validation

### 6.1 — Dockerfile
- [ ] Base: `FROM python:3.11-slim`
- [ ] `WORKDIR /app`
- [ ] Copy `requirements.txt` first, then `pip install` — Docker cache optimization
- [ ] Copy rest of project
- [ ] `EXPOSE 7860`
- [ ] `CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]`
- [ ] Build and run locally: `docker build -t openenv-fintech . && docker run -p 7860:7860 openenv-fintech`
- [ ] Test: `curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'` must return 200

### 6.2 — requirements.txt (Pin All Versions)
```
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic>=2.0,<3.0
numpy>=1.26,<2.0
pandas>=2.0,<3.0
openai>=1.0
anthropic>=0.25       # only for llm_judge, optional
gradio>=4.0           # for HF Spaces UI
httpx>=0.27           # for async HTTP in inference.py
```

### 6.3 — HF Spaces Setup
- [ ] Create HF Space: type = Docker, hardware = CPU Basic (free tier)
- [ ] Add `README.md` with YAML frontmatter for HF Spaces:
  ```yaml
  ---
  title: OpenEnv Fintech
  emoji: 💹
  colorFrom: blue
  colorTo: green
  sdk: docker
  pinned: false
  ---
  ```
- [ ] Set HF Space secrets: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- [ ] Confirm Space URL responds: `curl https://<your-space>.hf.space/reset`

### 6.4 — Run the Validator
Run this only after 6.1–6.3 are done:
```bash
./validate-submission.sh https://<your-space>.hf.space .
```
- [ ] Step 1 passes: HF Space live, `/reset` returns 200
- [ ] Step 2 passes: Docker build succeeds
- [ ] Step 3 passes: `openenv validate` passes

---

## PHASE 7 — Tests (Write As You Build, Not After)

### 7.1 — Reproducibility Tests (`tests/test_reproducibility.py`)
- [ ] Same seed → identical loan applicant across 3 calls
- [ ] Same seed → identical transaction stream
- [ ] Same seed → identical price path
- [ ] Different seeds → different outputs

### 7.2 — Score Range Tests (`tests/test_graders.py`)
- [ ] Loan grader: score always in [0.0, 1.0] across 100 random actions
- [ ] Fraud grader: score always in [0.0, 1.0] across 100 random action sequences
- [ ] Portfolio grader: score always in [0.0, 1.0] across 100 random action sequences
- [ ] Random agent scores ~0.2 (floor test) — run 50 episodes, assert mean > 0.1
- [ ] Perfect agent scores > 0.9 (ceiling test) — hardcode optimal actions, assert score > 0.85

### 7.3 — API Tests (`tests/test_api.py`)
- [ ] `POST /reset` with `{}` returns 200
- [ ] `POST /reset` with unknown task returns 422
- [ ] `POST /step` with wrong session_id returns 404
- [ ] `POST /step` with invalid action returns 422
- [ ] Full episode run for each task completes without error
- [ ] `GET /state/{session_id}` returns correct step count

### 7.4 — Inference Script Test
- [ ] Run `inference.py` with `--task loan_underwriting` — confirm stdout matches exact format
- [ ] `[END]` line appears even if an exception is raised mid-episode
- [ ] Score is in [0.0, 1.0]

---

## Final Pre-Submission Checklist

Run through this in order before submitting:

- [ ] `inference.py` is at root, uses `OpenAI` client, reads env vars
- [ ] All `[START]` / `[STEP]` / `[END]` stdout lines match format exactly
- [ ] `POST /reset` with empty body `{}` returns HTTP 200
- [ ] `openenv validate` passes locally
- [ ] `docker build` succeeds with no errors
- [ ] HF Space is live and responds to ping
- [ ] All 3 tasks have graders returning scores in [0.0, 1.0]
- [ ] Inference script completes in < 20 minutes on a slow machine (test with throttled CPU)
- [ ] No hardcoded API keys anywhere in the repo
- [ ] `validate-submission.sh` all 3 steps pass

---

## Known Landmines (Read Before Building)

1. **The validator sends `POST /reset` with body `{}`** — your FastAPI route must handle missing `task` and `seed` with defaults. If you use `task: str` without a default it will 422.

2. **`[END]` must always emit** — wrap the entire episode loop in `try/finally`. If the LLM throws, the server is unreachable, or parsing fails, `[END]` still fires.

3. **action= in [STEP] must be single-line** — JSON dumps the action, then `replace('\n', ' ')`. Multi-line action strings will break the log parser.

4. **Portfolio reward can go very negative** — multiple constraint violations + drift penalty + tracking error can stack to below -1.0 raw. Your `normalize_score()` must know the theoretical minimum. Pre-compute it from the reward formula in the design doc.

5. **Pydantic v2 breaking change** — `@validator` is deprecated, use `@field_validator`. `orm_mode=True` is now `model_config = ConfigDict(from_attributes=True)`. Don't mix v1 and v2 syntax.

6. **Docker on HF Spaces CPU Basic has 2 vCPU / 16GB** — but the evaluation machine is 2 vCPU / 8GB per the spec. Test inference.py on a constrained machine before submitting.

7. **Session cleanup** — if inference.py crashes without calling `/close`, sessions accumulate. Add a 30-minute TTL background cleanup or the server will OOM on long leaderboard runs.
