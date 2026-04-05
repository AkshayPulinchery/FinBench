"""FastAPI server and Gradio demo for OpenEnv Fintech."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, ValidationError

from openenv_fintech.env import FintechEnv, SessionNotFoundError
from openenv_fintech.models.actions import FraudAction, LoanAction, PortfolioAction

try:
    import gradio as gr
except ImportError:  # pragma: no cover - exercised only in minimal local envs
    gr = None


env_manager = FintechEnv()


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Literal["loan_underwriting", "fraud_detection", "portfolio_rebalancing"] = (
        "loan_underwriting"
    )
    seed: int = 42


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    action: dict[str, Any]


ACTION_MODEL_REGISTRY = {
    "loan_underwriting": LoanAction,
    "fraud_detection": FraudAction,
    "portfolio_rebalancing": PortfolioAction,
}

@asynccontextmanager
async def lifespan(_: FastAPI):
    await env_manager.start()
    try:
        yield
    finally:
        await env_manager.stop()


app = FastAPI(title="OpenEnv Fintech", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_action(task_name: str, action_payload: dict[str, Any]) -> Any:
    model_cls = ACTION_MODEL_REGISTRY[task_name]
    return model_cls.model_validate(action_payload)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    session_id = None
    if request.method == "POST":
        try:
            body = await request.body()
            if body:
                payload = json.loads(body.decode("utf-8"))
                session_id = payload.get("session_id")
        except Exception:
            session_id = None
    print(f"[REQUEST] method={request.method} path={request.url.path} session_id={session_id}")
    response = await call_next(request)
    return response


@app.exception_handler(SessionNotFoundError)
async def session_not_found_handler(_: Request, exc: SessionNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"error": "session_not_found", "detail": f"unknown session_id: {exc.args[0]}"},
    )


@app.exception_handler(RequestValidationError)
async def request_validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "detail": exc.errors()},
    )


@app.exception_handler(ValidationError)
async def validation_handler(_: Request, exc: ValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "detail": exc.errors()},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "detail": detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


@app.post("/reset")
async def reset(payload: ResetRequest | None = Body(default=None)):
    data = payload or ResetRequest()
    result = env_manager.create_session(task=data.task, seed=data.seed)
    return result.model_dump(mode="json")


@app.post("/step")
async def step(payload: StepRequest):
    record = env_manager.get_session(payload.session_id)
    try:
        action = parse_action(record.task_name, payload.action)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    try:
        result = env_manager.step(payload.session_id, action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump(mode="json")


@app.get("/state/{session_id}")
async def state(session_id: str):
    return env_manager.state(session_id).model_dump(mode="json")


@app.post("/close/{session_id}")
async def close(session_id: str):
    env_manager.cleanup_session(session_id)
    return {"success": True}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "tasks": env_manager.supported_tasks(),
        "active_sessions": len(env_manager.sessions),
    }


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return """
    <html>
      <head><title>OpenEnv Fintech</title></head>
      <body>
        <h1>OpenEnv Fintech</h1>
        <p>Validator-facing API is live.</p>
        <ul>
          <li>POST /reset</li>
          <li>POST /step</li>
          <li>GET /state/{session_id}</li>
          <li>POST /close/{session_id}</li>
          <li>GET /health</li>
        </ul>
        <p>Interactive demo: <a href="/demo">/demo</a></p>
      </body>
    </html>
    """


def build_demo():
    if gr is None:
        raise RuntimeError("gradio is not installed")

    def demo_reset(task_name: str, seed: int) -> tuple[str, str]:
        result = env_manager.create_session(task=task_name, seed=seed)
        return result.session_id, json.dumps(result.observation.model_dump(mode="json"), indent=2)

    def demo_step(session_id: str, action_json: str) -> str:
        record = env_manager.get_session(session_id)
        payload = json.loads(action_json)
        action = parse_action(record.task_name, payload)
        result = env_manager.step(session_id, action)
        return json.dumps(result.model_dump(mode="json"), indent=2)

    with gr.Blocks(title="OpenEnv Fintech Demo") as demo:
        gr.Markdown("# OpenEnv Fintech Demo")
        task = gr.Dropdown(
            choices=env_manager.supported_tasks(),
            value="loan_underwriting",
            label="Task",
        )
        seed = gr.Number(value=42, precision=0, label="Seed")
        start_btn = gr.Button("Start Episode")
        session_id = gr.Textbox(label="Session ID")
        observation = gr.Code(label="Observation", language="json")
        action_box = gr.Code(label="Action JSON", language="json", value="{}")
        step_btn = gr.Button("Submit Step")
        result_box = gr.Code(label="Step Result", language="json")
        start_btn.click(demo_reset, inputs=[task, seed], outputs=[session_id, observation])
        step_btn.click(demo_step, inputs=[session_id, action_box], outputs=result_box)
    return demo


if gr is not None:
    app = gr.mount_gradio_app(app, build_demo(), path="/demo")
