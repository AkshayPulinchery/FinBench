---
title: OpenEnv Fintech
emoji: 💹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - finance
  - benchmark
  - fastapi
---

# OpenEnv Fintech

`openenv-fintech` is a seeded financial decision benchmark with three tasks:
loan underwriting, fraud detection, and portfolio rebalancing.

## Motivation

This environment benchmarks realistic financial operations work rather than toy control tasks. The three tasks represent decisions human analysts and operators actually make:

- loan underwriting
- transaction fraud review
- portfolio rebalancing

The benchmark is designed to provide partial credit throughout the trajectory, deterministic seeded replay, and clear programmatic grading so model comparisons stay reproducible.

## Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 7860
```

For the full local runbook, API examples, test commands, baseline usage, inference setup, and Docker notes, see [RUNNING.md](RUNNING.md).

For the full architecture, file map, requirement audit, and OpenEnv flow diagrams, see [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md).

For frontend integration, see [API_REFERENCE.md](API_REFERENCE.md).

The validator-facing API is:

- `POST /reset`
- `POST /step`
- `GET /state/{session_id}`
- `POST /close/{session_id}`
- `GET /health`

The Hugging Face demo is mounted at `/demo`.

## Runtime configuration

Copy `.env.example` to `.env` and provide values as needed:

- `API_BASE_URL`: model provider base URL used by `inference.py`
- `ENV_URL`/`SPACE_URL`: environment server URL used by `inference.py`
- `MODEL_NAME`: chat model name
- `HF_TOKEN`: provider API token used by the OpenAI client
- `OPENAI_API_KEY`: optional compatibility fallback for provider token
- `LLM_JUDGE_ENABLED`: optional reasoning scorer toggle

## Tasks

| Task | Difficulty | What the agent does | Reward style |
|---|---|---|---|
| `loan_underwriting` | Easy | approve, reject, or request more information for a loan applicant | per-decision partial credit with risk-tier and pricing bonuses |
| `fraud_detection` | Medium | review a transaction stream and flag fraud while controlling false positives | episode-level score with precision/recall/calibration and penalties |
| `portfolio_rebalancing` | Hard | rebalance a 10-asset portfolio over time under trading and cash constraints | per-step progress credit plus final portfolio quality score |

## Observation And Action Spaces

### Loan Underwriting

Observation fields:
- applicant identity and credit features
- leverage and income fields
- purpose and submitted documents

Action fields:
- `decision`
- `reasoning`
- `risk_tier`
- `interest_rate_suggestion`

### Fraud Detection

Observation fields:
- transaction metadata
- location and timing signals
- 30-day history summary
- velocity indicators

Action fields:
- `flag`
- `confidence`
- `hold`
- `reason_code`
- `notes`

### Portfolio Rebalancing

Observation fields:
- per-asset weights, prices, shares, and unrealized PnL
- market volatility regime and recent returns
- account cash and value
- trading constraints

Action fields:
- `trades`
- `defer_rebalancing`
- `risk_comment`

## Baseline Scores

The local deterministic heuristic baseline can be run with:

```bash
python3 baseline.py --task all --episodes 25 --seed 42
```

Current local heuristic baseline sample (`--task all --episodes 10 --seed 42`):

| Task | Mean | Std | Min | Max |
|---|---|---|---|---|
| `loan_underwriting` | 0.672 | 0.115 | 0.600 | 0.920 |
| `fraud_detection` | 0.990 | 0.024 | 0.918 | 0.990 |
| `portfolio_rebalancing` | 0.528 | 0.421 | 0.010 | 0.990 |

Exact scores are seed-dependent but reproducible for the same seed and episode count for the heuristic baseline.

## Setup And Usage

Start the environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 7860
```

Run the heuristic baseline across all tasks:

```bash
python3 baseline.py --task all --episodes 10 --seed 42
```

Run the OpenAI-client inference script:

```bash
export HF_TOKEN=your_key_here
export ENV_URL=http://localhost:7860
python3 inference.py --task all --episodes 1 --seed 42
```

## Local validation

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{}'
pytest
openenv validate
```
