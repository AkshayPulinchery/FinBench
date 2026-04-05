# OpenEnv Fintech — Design Document
**Status:** DRAFT  
**Project:** openenv-fintech  
**Audience:** AI researchers, ML engineers, HF open-source ecosystem  
**Version:** 1.0.0

---

## 1. One-Liner

> The first OpenEnv benchmark where LLM agents make sequential financial decisions — loan underwriting, fraud detection, and portfolio rebalancing — with partial-credit reward signals and fully reproducible, seed-deterministic scoring.

---

## 2. Problem Statement

Existing agent benchmarks are either:
- **Too gamified** (GridWorld, CartPole) — reward functions don't map to real business cost
- **Too narrow** (single API call tasks) — no sequential decision-making under uncertainty
- **Too opaque** (no partial credit) — binary pass/fail tells agents nothing about *why* they failed

Fintech is the ideal domain because:
- Every decision has measurable financial consequence (approving a bad loan costs X, missing fraud costs Y)
- Partial progress is natural and meaningful (risk score calibration, portfolio drift, precision/recall tradeoff)
- Tasks form a natural easy → medium → hard ladder without arbitrary difficulty tuning
- Data can be fully synthetic yet structurally identical to real underwriting/fraud/portfolio problems

---

## 3. Environment Overview

**Name:** `openenv-fintech`  
**API:** OpenEnv spec — `step()` / `reset()` / `state()`  
**Tasks:** 3 (LoanUnderwriting → FraudDetection → PortfolioRebalancing)  
**Scoring:** 0.0–1.0 per task, partial credit at every step  
**Data:** Fully synthetic, seeded, no external API dependencies  
**Deploy:** Docker + Hugging Face Spaces  

---

## 4. Task Specifications

### Task 1: Loan Underwriting (Easy)
**Difficulty:** Easy  
**Agent goal:** Review a loan application and decide: Approve / Reject / Request More Info  
**Episode length:** 1–3 steps (single applicant, possibly with follow-up)

**Observation space:**
```
{
  applicant_id: str,
  credit_score: int (300–850),
  annual_income: float,
  debt_to_income_ratio: float (0.0–1.0),
  employment_years: float,
  loan_amount: float,
  loan_purpose: str (enum: mortgage, auto, personal, business),
  previous_defaults: int,
  documents_submitted: list[str]
}
```

**Action space:**
```
{
  decision: enum(approve, reject, request_info),
  reasoning: str,  # agent must provide rationale
  risk_tier: enum(low, medium, high),  # must match decision
  interest_rate_suggestion: float  # 0.03–0.35, only if approved
}
```

**Reward function:**
- Correct approve on good applicant: +1.0
- Correct reject on bad applicant: +1.0
- Approved bad applicant (default risk >70%): -0.8
- Rejected good applicant (default risk <20%): -0.4 (false negative is less costly)
- Request info when genuinely ambiguous (30–70% risk zone): +0.3
- Risk tier matches actual risk band: +0.2 bonus
- Interest rate within ±2% of actuarially fair rate: +0.1 bonus
- Reasoning quality (graded by LLM judge): 0.0–0.2

**Grader:** Rule-based ground truth from synthetic applicant generator with known default probabilities. LLM judge scores reasoning quality.

---

### Task 2: Fraud Detection (Medium)
**Difficulty:** Medium  
**Agent goal:** Analyze a transaction stream (10–20 transactions) and flag fraudulent ones  
**Episode length:** 10–20 steps (one transaction per step)

**Observation space (per step):**
```
{
  transaction_id: str,
  amount: float,
  merchant_category: str,
  timestamp: datetime,
  location: str,
  device_fingerprint: str,
  distance_from_home: float (km),
  time_since_last_transaction: float (minutes),
  transaction_history_summary: {
    avg_amount_30d: float,
    num_transactions_30d: int,
    usual_merchants: list[str],
    usual_locations: list[str]
  },
  velocity_signals: {
    transactions_last_hour: int,
    unique_merchants_last_24h: int,
    amount_last_24h: float
  }
}
```

**Action space (per step):**
```
{
  flag: bool,  # fraud or not
  confidence: float (0.0–1.0),
  hold: bool,  # freeze card pending review
  reason_code: enum(velocity, location_anomaly, amount_anomaly, merchant_risk, pattern_break, none),
  notes: str
}
```

**Reward function (per episode):**
- True positive rate (caught fraud): weight 0.4
- False positive rate (wrongly flagged): penalty weight 0.3 (false positives hurt customer experience)
- Precision × Recall F1: weight 0.2
- Confidence calibration (does confidence match actual accuracy?): weight 0.1
- Bonus: caught fraud *before* high-value transaction (early detection): +0.15
- Penalty: missed fraud on transactions >$5,000: -0.2

**Grader:** Episode-level scorer computes F1, calibration, and early-detection bonus from labeled ground truth.

---

### Task 3: Portfolio Rebalancing (Hard)
**Difficulty:** Hard  
**Agent goal:** Rebalance a 10-asset portfolio over 20 time steps to hit target allocation while minimizing transaction cost and tracking error  
**Episode length:** 20 steps (each step = 1 trading day)

**Observation space (per step):**
```
{
  step: int,
  portfolio: {
    [asset_id]: {
      current_weight: float,
      target_weight: float,
      current_price: float,
      shares_held: float,
      unrealized_pnl: float
    }
  },
  market_context: {
    volatility_regime: enum(low, medium, high),
    recent_returns: dict[asset_id, float],  # 5-day returns
    correlation_matrix: list[list[float]]
  },
  constraints: {
    max_single_trade_pct: float,  # % of portfolio per trade
    min_cash_reserve: float,
    tax_lot_consideration: bool,
    rebalancing_budget_usd: float  # total transaction cost budget
  },
  account: {
    cash: float,
    total_value: float,
    days_remaining: int
  }
}
```

**Action space (per step):**
```
{
  trades: list[{
    asset_id: str,
    direction: enum(buy, sell, hold),
    amount_usd: float,
    rationale: str
  }],
  defer_rebalancing: bool,  # agent can choose to wait
  risk_comment: str
}
```

**Reward function (per episode):**
- Tracking error vs target allocation (final state): weight 0.35
- Transaction cost efficiency (stayed within budget): weight 0.25
- Sharpe ratio of portfolio during episode: weight 0.20
- Constraint violations (each violation): -0.15 each
- Partial credit: progress toward target weight at each step: +0.02 per step per asset within 2% of target
- Bonus: completed rebalancing with ≥3 days remaining: +0.1
- Penalty: drift >15% from target at any point: -0.1

**Grader:** Multi-period financial simulator computes all metrics against known price path (seeded).

---

## 5. OpenEnv Spec Implementation

### Core API

```python
class FintechEnv:
    def reset(self, task: str, seed: int = 42) -> Observation
    def step(self, action: Action) -> StepResult  # (obs, reward, done, info)
    def state(self) -> EnvState  # full serializable state snapshot
```

### openenv.yaml structure
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

### Typed Models (Pydantic)
All observations, actions, and step results are Pydantic v2 models. No raw dicts in the public API. This enables:
- Schema validation on agent actions
- JSON serialization for logging
- Auto-generated OpenAPI-style documentation

---

## 6. Data Generation

All data is synthetic and seed-deterministic:

- **Loan applicants:** Generated from parameterized distributions matching real credit score distributions. Default probability is a known function of features (not hidden) — ground truth is transparent.
- **Transaction streams:** Markov chain over merchant categories and amounts, with injected anomaly patterns (velocity spike, location jump, amount outlier). Fraud labels are set at generation time.
- **Asset prices:** Geometric Brownian Motion with regime-switching volatility. Correlation matrix is seeded. Prices are deterministic given seed.

No external APIs. No internet required at runtime.

---

## 7. Graders

Each task has two grader types:

**Rule-based grader:** Computes numerical scores from ground truth (fast, deterministic).

**LLM judge (optional):** Scores reasoning quality for loan underwriting and fraud notes. Uses `claude-sonnet-4-20250514` via Anthropic API. Can be disabled with `--no-llm-judge` flag. Baseline scores use rule-based only for full reproducibility.

---

## 8. Baseline Inference Script

```bash
python baseline.py \
  --task loan_underwriting \
  --model gpt-4o \
  --episodes 100 \
  --seed 42 \
  --output results/loan_gpt4o_seed42.json
```

Baseline script:
- Runs N episodes per task
- Records per-step observations, actions, rewards
- Computes mean ± std score across episodes
- Saves full trace for reproducibility
- Prints leaderboard-style summary table

**Expected baseline scores (rule-based grader, seed=42, N=100):**

| Model | Loan Underwriting | Fraud Detection | Portfolio Rebalancing |
|-------|-------------------|-----------------|----------------------|
| Random agent | 0.21 ± 0.08 | 0.31 ± 0.06 | 0.18 ± 0.05 |
| Rule-based heuristic | 0.64 ± 0.04 | 0.58 ± 0.05 | 0.42 ± 0.07 |
| GPT-4o (target) | ~0.78 | ~0.71 | ~0.55 |

---

## 9. Deployment

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]
```

### Hugging Face Spaces
- Gradio interface: interactive demo where users can run one episode per task in-browser
- Leaderboard tab: shows community-submitted scores
- API tab: exposes REST endpoint wrapping the OpenEnv API

### requirements.txt (key deps)
```
openenv>=0.2.0
pydantic>=2.0
numpy>=1.26
pandas>=2.0
gradio>=4.0
anthropic>=0.25  # optional, for LLM judge
```

---

## 10. README Structure

1. Environment description (what, why, who)
2. Quick start (pip install + 5-line example)
3. Task descriptions with observation/action space tables
4. Reward function explanation with examples
5. Baseline scores table
6. Running the baseline script
7. Submitting to leaderboard
8. Docker / HF Spaces deployment
9. Citation / BibTeX

---

## 11. Project Structure

```
openenv-fintech/
├── openenv.yaml
├── README.md
├── Dockerfile
├── requirements.txt
├── app.py                  # Gradio HF Spaces app
├── baseline.py             # Reproducible inference script
├── openenv_fintech/
│   ├── __init__.py
│   ├── env.py              # FintechEnv main class
│   ├── tasks/
│   │   ├── loan_underwriting.py
│   │   ├── fraud_detection.py
│   │   └── portfolio_rebalancing.py
│   ├── models/             # Pydantic typed models
│   │   ├── observations.py
│   │   ├── actions.py
│   │   └── results.py
│   ├── data/
│   │   ├── generators.py   # Synthetic data generators
│   │   └── seeds.py        # Seed management
│   └── graders/
│       ├── rule_based.py
│       └── llm_judge.py
└── tests/
    ├── test_env.py
    ├── test_graders.py
    └── test_reproducibility.py
```

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| OpenEnv spec changes break API | Pin openenv version, abstract behind wrapper |
| Synthetic data feels too easy | Inject edge cases (borderline applicants, adversarial transactions, volatile markets) in hard episodes |
| LLM judge introduces variance | Baseline scores explicitly exclude LLM judge; it's an opt-in quality signal |
| Portfolio task too hard to get signal | Partial credit per-step (per-asset weight progress) ensures gradient even for bad agents |
| HF Spaces cold start latency | Pre-generate 10 demo episodes at build time; cache them for instant playback |

---

## 13. Success Metrics

- [ ] All 3 tasks run end-to-end with `step()` / `reset()` / `state()`
- [ ] Scores are deterministic given same seed across 3 runs
- [ ] Random agent scores ~0.2 (floor is meaningful)
- [ ] Best possible agent scores <0.95 (ceiling is not saturated)
- [ ] HF Spaces demo loads in <5 seconds
- [ ] Baseline script produces results table in <2 minutes for 100 episodes
- [ ] README has copy-paste quickstart that works on first try

---

*Generated by office-hours builder mode. Next step: `/plan-eng-review` to lock architecture and edge cases.*
