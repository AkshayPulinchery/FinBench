# API Reference

This document is the frontend integration contract for `openenv-fintech`.

Base URL for local development:

```text
http://localhost:7860
```

## API Overview

| Method | Path | What it does |
|---|---|---|
| `POST` | `/reset` | Starts a new task episode and returns the first observation. Creates a unique `session_id` that the frontend must keep for the rest of the episode. |
| `POST` | `/step` | Submits one agent action for the current session. Returns the next observation, reward, and whether the episode is complete. |
| `GET` | `/state/{session_id}` | Returns the current serializable state of an active episode. Useful for debugging, refresh flows, or session inspection. |
| `POST` | `/close/{session_id}` | Explicitly closes an active session. Call this when the UI is done with an episode to free server memory. |
| `GET` | `/health` | Returns service health and supported tasks. Good for startup checks and diagnostics in the frontend. |
| `GET` | `/` | Returns a small HTML landing page. This is informational only, not part of the JSON integration surface. |
| `GET` | `/demo` | Gradio demo UI mounted on the same app. This is for humans, not for frontend API integration. |

---

## 1. `POST /reset`

Starts a new episode for a selected task and seed.
Returns the initial observation and the `session_id` the frontend must store.

### Request Body

```json
{
  "task": "loan_underwriting",
  "seed": 42
}
```

### Fields

- `task`: optional, one of `loan_underwriting`, `fraud_detection`, `portfolio_rebalancing`
- `seed`: optional integer, defaults to `42`

### Notes

- `{}` is valid and defaults to `loan_underwriting` with seed `42`
- each call creates a new session

### Example Curl

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task":"loan_underwriting","seed":42}'
```

### Example Response

```json
{
  "observation": {
    "applicant_id": "app-624391",
    "credit_score": 702,
    "annual_income": 85620.17,
    "debt_to_income_ratio": 0.271,
    "employment_years": 6.4,
    "loan_amount": 27836.56,
    "loan_purpose": "auto",
    "previous_defaults": 0,
    "documents_submitted": ["id", "pay_stub"]
  },
  "session_id": "ed6bd23c-4f53-4bc3-9d6f-33c8c2de7d0e",
  "task": "loan_underwriting",
  "seed": 42
}
```

---

## 2. `POST /step`

Submits one action for the current session.
Returns reward, next observation, and the `done` flag for episode completion.

### Request Body

```json
{
  "session_id": "ed6bd23c-4f53-4bc3-9d6f-33c8c2de7d0e",
  "action": {}
}
```

### Notes

- `action` must match the schema for the current task
- wrong shape returns `422`
- unknown session returns `404`

---

## 2A. Loan Underwriting Action

### Action Shape

```json
{
  "decision": "approve",
  "reasoning": "Strong repayment profile with acceptable leverage.",
  "risk_tier": "low",
  "interest_rate_suggestion": 0.07
}
```

### Rules

- `decision`: `approve` | `reject` | `request_info`
- `risk_tier`: `low` | `medium` | `high`
- `interest_rate_suggestion` is required only for `approve`
- `interest_rate_suggestion` must be between `0.03` and `0.35`

### Example Curl

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id":"ed6bd23c-4f53-4bc3-9d6f-33c8c2de7d0e",
    "action":{
      "decision":"approve",
      "reasoning":"Strong repayment profile with acceptable leverage.",
      "risk_tier":"low",
      "interest_rate_suggestion":0.07
    }
  }'
```

### Example Response

```json
{
  "observation": {
    "applicant_id": "app-624391",
    "credit_score": 702,
    "annual_income": 85620.17,
    "debt_to_income_ratio": 0.271,
    "employment_years": 6.4,
    "loan_amount": 27836.56,
    "loan_purpose": "auto",
    "previous_defaults": 0,
    "documents_submitted": ["id", "pay_stub"]
  },
  "reward": 0.76,
  "done": true,
  "info": {
    "stage": "decision_complete",
    "breakdown": {
      "raw_reward": 0.9,
      "normalized_reward": 0.76,
      "p_default": 0.22,
      "actuarial_rate": 0.0784,
      "risk_band": "medium",
      "ambiguous": false
    }
  },
  "session_id": "ed6bd23c-4f53-4bc3-9d6f-33c8c2de7d0e"
}
```

---

## 2B. Fraud Detection Action

### Action Shape

```json
{
  "flag": true,
  "confidence": 0.91,
  "hold": true,
  "reason_code": "velocity",
  "notes": "Rapid high-risk pattern inconsistent with normal behavior."
}
```

### Rules

- `confidence` must be between `0.0` and `1.0`
- `reason_code`: `velocity`, `location_anomaly`, `amount_anomaly`, `merchant_risk`, `pattern_break`, `none`
- if `flag` is `false`, `reason_code` must be `none`

### Example Curl

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id":"b4f07466-df6a-4f45-99b6-f9b52af573f0",
    "action":{
      "flag":true,
      "confidence":0.91,
      "hold":true,
      "reason_code":"velocity",
      "notes":"Rapid high-risk pattern inconsistent with normal behavior."
    }
  }'
```

### Example Response

```json
{
  "observation": {
    "transaction_id": "txn-002",
    "amount": 67.24,
    "merchant_category": "grocery",
    "timestamp": "2025-01-18T10:11:00",
    "location": "New York",
    "device_fingerprint": "dev-3187-1",
    "distance_from_home": 3.8,
    "time_since_last_transaction": 18.0,
    "transaction_history_summary": {
      "avg_amount_30d": 92.14,
      "num_transactions_30d": 84,
      "usual_merchants": ["gas", "grocery", "online", "restaurant"],
      "usual_locations": ["New York"]
    },
    "velocity_signals": {
      "transactions_last_hour": 2,
      "unique_merchants_last_24h": 2,
      "amount_last_24h": 184.79
    }
  },
  "reward": 0.0,
  "done": false,
  "info": {
    "progress": "1/15"
  },
  "session_id": "b4f07466-df6a-4f45-99b6-f9b52af573f0"
}
```

---

## 2C. Portfolio Rebalancing Action

### Action Shape

```json
{
  "trades": [
    {
      "asset_id": "AST01",
      "direction": "sell",
      "amount_usd": 1800,
      "rationale": "Trim overweight asset toward target allocation."
    },
    {
      "asset_id": "AST07",
      "direction": "buy",
      "amount_usd": 1800,
      "rationale": "Increase underweight asset toward target allocation."
    }
  ],
  "defer_rebalancing": false,
  "risk_comment": "Gradual rebalance to preserve cash and budget."
}
```

### Rules

- `direction`: `buy` | `sell` | `hold`
- `amount_usd` must be `0` for `hold`
- `amount_usd` must be positive for `buy` and `sell`

### Example Curl

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id":"f6f98c0d-c478-4df8-8ec9-cd86730a36de",
    "action":{
      "trades":[
        {
          "asset_id":"AST01",
          "direction":"sell",
          "amount_usd":1800,
          "rationale":"Trim overweight asset toward target allocation."
        },
        {
          "asset_id":"AST07",
          "direction":"buy",
          "amount_usd":1800,
          "rationale":"Increase underweight asset toward target allocation."
        }
      ],
      "defer_rebalancing":false,
      "risk_comment":"Gradual rebalance to preserve cash and budget."
    }
  }'
```

### Example Response

```json
{
  "observation": {
    "step": 1,
    "portfolio": {
      "AST01": {
        "current_weight": 0.071423,
        "target_weight": 0.052004,
        "current_price": 91.4471,
        "shares_held": 74.230118,
        "unrealized_pnl": 123.18
      },
      "AST07": {
        "current_weight": 0.097813,
        "target_weight": 0.114002,
        "current_price": 64.1473,
        "shares_held": 162.319921,
        "unrealized_pnl": 84.77
      }
    },
    "market_context": {
      "volatility_regime": "medium",
      "recent_returns": {
        "AST01": 0.0114,
        "AST07": 0.0081
      },
      "correlation_matrix": [[1.0, 0.14], [0.14, 1.0]]
    },
    "constraints": {
      "max_single_trade_pct": 0.15,
      "min_cash_reserve": 6000.0,
      "tax_lot_consideration": true,
      "rebalancing_budget_usd": 3200.0
    },
    "account": {
      "cash": 12340.27,
      "total_value": 108901.65,
      "days_remaining": 19
    }
  },
  "reward": 0.06,
  "done": false,
  "info": {
    "violations": []
  },
  "session_id": "f6f98c0d-c478-4df8-8ec9-cd86730a36de"
}
```

---

## 3. `GET /state/{session_id}`

Returns the current server-side snapshot of a live session.
Useful if the frontend needs to recover state after a refresh or inspect episode history.

### Example Curl

```bash
curl http://localhost:7860/state/ed6bd23c-4f53-4bc3-9d6f-33c8c2de7d0e
```

### Example Response

```json
{
  "task": "loan_underwriting",
  "step": 1,
  "session_id": "ed6bd23c-4f53-4bc3-9d6f-33c8c2de7d0e",
  "seed": 42,
  "episode_history": [
    {
      "step": 1,
      "action": {
        "decision": "approve",
        "reasoning": "Strong repayment profile with acceptable leverage.",
        "risk_tier": "low",
        "interest_rate_suggestion": 0.07
      },
      "reward": 0.76,
      "done": true
    }
  ],
  "done": true,
  "max_steps": 3
}
```

---

## 4. `POST /close/{session_id}`

Closes and deletes a server-side session.
This should be called when the frontend leaves an episode or finishes displaying results.

### Example Curl

```bash
curl -X POST http://localhost:7860/close/ed6bd23c-4f53-4bc3-9d6f-33c8c2de7d0e
```

### Example Response

```json
{
  "success": true
}
```

---

## 5. `GET /health`

Returns service health and the currently supported tasks.
Useful for frontend boot checks and environment diagnostics.

### Example Curl

```bash
curl http://localhost:7860/health
```

### Example Response

```json
{
  "status": "ok",
  "tasks": [
    "loan_underwriting",
    "fraud_detection",
    "portfolio_rebalancing"
  ],
  "active_sessions": 2
}
```

---

## 6. `GET /`

Returns a lightweight HTML landing page.
This endpoint is useful only as a browser sanity check.

### Example Curl

```bash
curl http://localhost:7860/
```

### Example Response

```html
<html>
  <head><title>OpenEnv Fintech</title></head>
  <body>
    <h1>OpenEnv Fintech</h1>
    <p>Validator-facing API is live.</p>
  </body>
</html>
```

---

## Error Responses

### `404` Unknown Session

```json
{
  "error": "session_not_found",
  "detail": "unknown session_id: missing-session-id"
}
```

### `422` Validation Error

```json
{
  "error": "validation_error",
  "detail": [
    {
      "type": "value_error",
      "loc": ["interest_rate_suggestion"],
      "msg": "interest_rate_suggestion is required when approving"
    }
  ]
}
```

### `400` Invalid Runtime State

```json
{
  "error": "http_error",
  "detail": "episode is already complete"
}
```

---

## Frontend Integration Notes

- Keep `session_id` in frontend state after `/reset`
- Always send task-appropriate `action` payloads to `/step`
- Use `done` to switch the UI from episode interaction to results view
- Call `/close/{session_id}` when the user abandons or completes a session
- Use `/health` for initial availability checks
- Treat `/demo` as a developer demo, not as an API dependency
