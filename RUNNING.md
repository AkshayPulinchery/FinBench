# Run And Test Guide

This guide covers local setup, running the API, exercising the tasks, running tests, and preparing for Docker and Hugging Face Spaces.

## 1. Prerequisites

- Python `3.11`
- `pip` or a virtualenv-capable Python install
- Docker Desktop if you want to build the image
- Optional: `openenv` CLI if you want to run submission validation

Check the basics:

```bash
python3.11 --version
docker --version
```

If you plan to run `inference.py` against a hosted model endpoint, you need a valid `HF_TOKEN`. `OPENAI_API_KEY` is supported as a compatibility fallback.

## 2. Create A Virtual Environment

From repo root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Configure Environment Variables

Copy the example file:

```bash
cp .env.example .env
```

The key variables are:

- `API_BASE_URL`: model provider base URL used by `inference.py` (LLM endpoint)
- `ENV_URL` (or `SPACE_URL`): environment API base URL, default `http://localhost:7860`
- `MODEL_NAME`: model identifier for the OpenAI-compatible endpoint
- `HF_TOKEN`: token for the OpenAI-compatible provider
- `OPENAI_API_KEY`: optional compatibility fallback token
- `LLM_JUDGE_ENABLED`: optional reasoning scorer toggle, default `false`

For local development, the default `.env.example` values are already correct for the API server.

## 4. Run The API Server

Start the FastAPI app from repo root:

```bash
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

What this gives you:

- API root: `http://localhost:7860/`
- Health check: `http://localhost:7860/health`
- Demo UI: `http://localhost:7860/demo` if `gradio` is installed

## 5. Smoke Test The API

Reset a default session:

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{}'
```

Example reset for a specific task:

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task":"loan_underwriting","seed":42}'
```

Use the returned `session_id` in the next call:

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id":"<SESSION_ID>",
    "action":{
      "decision":"approve",
      "reasoning":"Strong repayment profile.",
      "risk_tier":"low",
      "interest_rate_suggestion":0.07
    }
  }'
```

Important: the action payload must match the task selected in `/reset`.

Fraud detection step example:

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id":"<SESSION_ID>",
    "action":{
      "flag":false,
      "confidence":0.12,
      "hold":false,
      "reason_code":"none",
      "notes":"No anomaly signals."
    }
  }'
```

Portfolio rebalancing step example:

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id":"<SESSION_ID>",
    "action":{
      "trades":[
        {
          "asset_id":"AST01",
          "direction":"hold",
          "amount_usd":0,
          "rationale":"No trade this step."
        }
      ],
      "defer_rebalancing":false,
      "risk_comment":"Monitoring drift before trading."
    }
  }'
```

Inspect state:

```bash
curl http://localhost:7860/state/<SESSION_ID>
```

Close the session:

```bash
curl -X POST http://localhost:7860/close/<SESSION_ID>
```

## 6. Run The Test Suite

Run everything:

```bash
source .venv/bin/activate
pytest -q
```

Run a narrower slice when iterating:

```bash
pytest -q tests/test_api.py
pytest -q tests/test_graders.py
pytest -q tests/test_inference.py
```

What the suite covers:

- schema validation and action-model rules
- deterministic seeded generation
- task flow and grading behavior
- FastAPI endpoint behavior
- inference logging format

## 7. Run The Baseline Locally

The baseline runner uses deterministic heuristics and does not require an external model.

Example:

```bash
source .venv/bin/activate
python3 baseline.py --task loan_underwriting --episodes 10
python3 baseline.py --task fraud_detection --episodes 10
python3 baseline.py --task portfolio_rebalancing --episodes 10
python3 baseline.py --task all --episodes 10
```

Save results to disk:

```bash
python3 baseline.py \
  --task loan_underwriting \
  --episodes 25 \
  --seed 42 \
  --output results/loan_baseline.json
```

## 8. Run The Judged Inference Script

`inference.py` expects:

- the env server to be running at `ENV_URL` (or `SPACE_URL`)
- an OpenAI-compatible model endpoint at `API_BASE_URL`
- a valid `HF_TOKEN` (or fallback `OPENAI_API_KEY`)

Example:

```bash
source .venv/bin/activate
export ENV_URL=http://localhost:7860
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=your_token_here

python3 inference.py --task all --episodes 1 --seed 42
```

Expected stdout shape:

- `[START] ...`
- `[STEP] ...`
- `[END] ...`

`[END]` is always emitted, even on failure.

## 9. Build And Run With Docker

Build the image:

```bash
docker build -t openenv-fintech .
```

Run the container:

```bash
docker run --rm -p 7860:7860 openenv-fintech
```

Then verify:

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{}'
```

If you see `Cannot connect to the Docker daemon`, start Docker Desktop first and retry.

## 10. Run OpenEnv Validation

If the `openenv` CLI is installed:

```bash
openenv --help
openenv validate .
```

If you also have the official validator script:

```bash
./validate-submission.sh https://<your-space>.hf.space .
```

## 11. Hugging Face Spaces Notes

- Space type should be `Docker`
- The app entrypoint is the container command in the `Dockerfile`
- Required secrets are typically:
  - `ENV_URL` (or `SPACE_URL`)
  - `API_BASE_URL`
  - `MODEL_NAME`
  - `HF_TOKEN`
  - `OPENAI_API_KEY` (optional fallback)

If the Space itself is hosting the environment API, `ENV_URL` should point at that deployed app URL.

## 12. Common Issues

`ModuleNotFoundError` during tests:
- Run tests from repo root
- Activate the virtual environment
- Install dependencies from `requirements.txt`

`/demo` does not load:
- Confirm `gradio` is installed in the active environment

`inference.py` fails immediately:
- Check `OPENAI_API_KEY`
- Check `API_BASE_URL`
- Confirm the API server is reachable at `ENV_URL` (or `SPACE_URL`)

Docker build or run fails:
- Make sure Docker Desktop is running

`openenv validate` is missing:
- Install the CLI first; it is not bundled with this repo
