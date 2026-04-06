# Frontend UI for OpenEnv Fintech

This frontend is a static retro-style control panel inspired by classic terminal dashboards and retroui aesthetics.

## Features

- Health check against `GET /health`
- Session lifecycle:
  - `POST /reset`
  - `POST /step`
  - `GET /state/{session_id}`
  - `POST /close/{session_id}`
- Task-specific action composers for:
  - `loan_underwriting`
  - `fraud_detection`
  - `portfolio_rebalancing`
- Observation/result JSON panes and event log

## Run

1. Start backend API from `JAS_BACKEND/FinBench` on port `7860`.
2. Start the frontend static server from `frontend`:

```powershell
powershell -ExecutionPolicy Bypass -File .\start-frontend.ps1
```

3. Open `http://localhost:5500` in your browser.
4. Keep API base URL as `http://localhost:7860` unless your server is elsewhere.

Optional custom port:

```powershell
powershell -ExecutionPolicy Bypass -File .\start-frontend.ps1 -Port 8080
```

## Notes

- The form validations and payload shapes follow backend models in `openenv_fintech/models/actions.py`.
- The UI defaults are aligned with examples from `API_REFERENCE.md`.
