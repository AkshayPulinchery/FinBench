"""Judged inference runner for OpenEnv Fintech."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

import httpx
from openai import AsyncOpenAI

from openenv_fintech.models.results import EpisodeResult

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") or os.getenv("HF_TOKEN", "")
ENV_NAME = "openenv-fintech"
MAX_STEPS = {
    "loan_underwriting": 3,
    "fraud_detection": 20,
    "portfolio_rebalancing": 20,
}


def single_line_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True).replace("\n", " ")


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}")


def log_step(step: int, action: dict[str, Any], reward: float, done: bool, error: str | None) -> None:
    error_text = "null" if error is None else error.replace("\n", " ")
    print(
        f"[STEP] step={step} action={single_line_json(action)} reward={reward:.2f} "
        f"done={'true' if done else 'false'} error={error_text}"
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    reward_text = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={'true' if success else 'false'} steps={steps} "
        f"score={score:.3f} rewards={reward_text}"
    )


SYSTEM_PROMPTS = {
    "loan_underwriting": (
        "You are evaluating a loan application. Return JSON only matching the action schema with "
        "decision, reasoning, risk_tier, and optional interest_rate_suggestion."
    ),
    "fraud_detection": (
        "You are reviewing one transaction at a time for fraud. Return JSON only matching the action "
        "schema with flag, confidence, hold, reason_code, and notes."
    ),
    "portfolio_rebalancing": (
        "You are rebalancing a portfolio under budget and cash constraints. Return JSON only matching "
        "the action schema with trades, defer_rebalancing, and risk_comment."
    ),
}


def build_user_prompt(task_name: str, observation: dict[str, Any], step: int, last_reward: float) -> str:
    return (
        f"Task: {task_name}\n"
        f"Step: {step}\n"
        f"Last reward: {last_reward:.2f}\n"
        "Observation JSON:\n"
        f"{json.dumps(observation, indent=2, sort_keys=True)}\n\n"
        "Respond with JSON only."
    )


def extract_json_block(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("model response did not contain a JSON object")
    return json.loads(cleaned[start : end + 1])


async def call_llm_for_action(
    client: AsyncOpenAI,
    task_name: str,
    observation: dict[str, Any],
    step: int,
    last_reward: float,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[task_name]},
        {"role": "user", "content": build_user_prompt(task_name, observation, step, last_reward)},
    ]
    for attempt in range(3):
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
        )
        content = completion.choices[0].message.content or ""
        try:
            return extract_json_block(content)
        except (ValueError, json.JSONDecodeError):
            if attempt == 2:
                raise
            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": "Return valid JSON only. Do not include markdown or explanations.",
                }
            )
    raise RuntimeError("unreachable")


async def run_episode(client: AsyncOpenAI, task_name: str, seed: int) -> EpisodeResult:
    session_id: str | None = None
    rewards: list[float] = []
    steps = 0
    final_score = 0.0
    success = False
    breakdown: dict[str, Any] = {}
    log_start(task_name, ENV_NAME, MODEL_NAME.split("/")[-1])

    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as env_client:
        try:
            reset_response = await env_client.post("/reset", json={"task": task_name, "seed": seed})
            reset_response.raise_for_status()
            reset_payload = reset_response.json()
            session_id = reset_payload["session_id"]
            observation = reset_payload["observation"]
            last_reward = 0.0
            done = False

            while not done and steps < MAX_STEPS[task_name]:
                action = await call_llm_for_action(
                    client=client,
                    task_name=task_name,
                    observation=observation,
                    step=steps + 1,
                    last_reward=last_reward,
                )
                step_response = await env_client.post(
                    "/step",
                    json={"session_id": session_id, "action": action},
                )
                step_response.raise_for_status()
                step_payload = step_response.json()
                reward = float(step_payload["reward"])
                done = bool(step_payload["done"])
                observation = step_payload["observation"]
                last_reward = reward
                rewards.append(reward)
                steps += 1
                breakdown = step_payload.get("info", {})
                log_step(steps, action, reward, done, None)

            if steps >= MAX_STEPS[task_name] and not done:
                breakdown["terminated"] = "max_steps"

            final_score = rewards[-1] if rewards else 0.0
            success = final_score >= 0.1
            return EpisodeResult(
                final_score=final_score,
                total_reward=sum(rewards),
                steps=steps,
                success=success,
                breakdown=breakdown,
            )
        except Exception as exc:
            log_step(steps + 1, {}, 0.0, True, str(exc))
            return EpisodeResult(
                final_score=0.0,
                total_reward=sum(rewards),
                steps=steps,
                success=False,
                breakdown={"error": str(exc)},
            )
        finally:
            if session_id is not None:
                try:
                    await env_client.post(f"/close/{session_id}")
                except Exception:
                    pass
            log_end(success, steps, final_score, rewards)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="all",
        choices=["all", *list(MAX_STEPS)],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required")

    client = AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    task_names = list(MAX_STEPS) if args.task == "all" else [args.task]
    aggregate: dict[str, list[float]] = {task_name: [] for task_name in task_names}
    for task_name in task_names:
        for episode in range(args.episodes):
            result = await run_episode(client=client, task_name=task_name, seed=args.seed + episode)
            aggregate[task_name].append(result.final_score)

    if len(task_names) > 1 or args.episodes > 1:
        summary = {
            task_name: {
                "episodes": len(scores),
                "mean_score": round(sum(scores) / len(scores), 4),
                "scores": [round(score, 4) for score in scores],
            }
            for task_name, scores in aggregate.items()
        }
        print(json.dumps({"environment": ENV_NAME, "summary": summary}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
