# Frontend UI for OpenEnv Fintech

This frontend is a React + TypeScript + Tailwind CSS control panel inspired by retro and neo-brutalist interfaces.

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
2. Install frontend dependencies from `frontend`:

```powershell
npm install
```

3. Start the Vite dev server:

```powershell
npm run dev
```

4. Open the local URL shown by Vite (usually `http://localhost:5173`).
4. Keep API base URL as `http://localhost:7860` unless your server is elsewhere.

## Notes

- The form validations and payload shapes follow backend models in `openenv_fintech/models/actions.py`.
- The UI defaults are aligned with examples from `API_REFERENCE.md`.
