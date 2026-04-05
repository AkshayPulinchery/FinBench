"""Deterministic baseline runner for local benchmarking."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from openenv_fintech.env import TASK_REGISTRY
from openenv_fintech.models.actions import (
    FraudAction,
    LoanAction,
    PortfolioAction,
    TradeOrder,
)
from openenv_fintech.models.observations import (
    LoanObservation,
    PortfolioObservation,
    TransactionObservation,
)


def estimate_loan_risk(observation: LoanObservation) -> float:
    logit = (
        -0.01 * (observation.credit_score - 600)
        + 0.5 * observation.debt_to_income_ratio
        + 0.3 * int(observation.previous_defaults > 0)
    )
    return 1.0 / (1.0 + math.exp(-logit))


def loan_policy(observation: LoanObservation) -> LoanAction:
    risk = estimate_loan_risk(observation)
    if 0.30 <= risk <= 0.70 and len(observation.documents_submitted) < 3:
        return LoanAction(
            decision="request_info",
            reasoning="Borderline application with incomplete documentation.",
            risk_tier="medium",
        )
    decision = "approve" if risk < 0.52 else "reject"
    risk_tier = "low" if risk < 0.20 else "high" if risk > 0.70 else "medium"
    interest = None if decision != "approve" else round(min(0.35, max(0.03, 0.03 + risk * 0.22)), 4)
    return LoanAction(
        decision=decision,
        reasoning="Decision based on credit score, leverage, and default history.",
        risk_tier=risk_tier,
        interest_rate_suggestion=interest,
    )


def fraud_policy(observation: TransactionObservation) -> FraudAction:
    suspicious = []
    if observation.velocity_signals.transactions_last_hour >= 4:
        suspicious.append("velocity")
    if observation.distance_from_home > 500:
        suspicious.append("location_anomaly")
    if observation.amount > max(3 * observation.transaction_history_summary.avg_amount_30d, 5000):
        suspicious.append("amount_anomaly")
    flag = bool(suspicious)
    reason = suspicious[0] if suspicious else "none"
    confidence = 0.85 if flag else 0.18
    return FraudAction(
        flag=flag,
        confidence=confidence,
        hold=flag and confidence > 0.75,
        reason_code=reason,
        notes="Flagged due to anomaly signals." if flag else "No major anomaly detected.",
    )


def portfolio_policy(observation: PortfolioObservation) -> PortfolioAction:
    account_value = observation.account.total_value
    max_trade = observation.constraints.max_single_trade_pct * account_value
    buy_orders = []
    sell_orders = []
    for asset_id, asset in observation.portfolio.items():
        diff = asset.target_weight - asset.current_weight
        trade_value = min(abs(diff) * account_value * 0.7, max_trade)
        if trade_value < 250:
            continue
        if diff > 0:
            buy_orders.append(
                TradeOrder(
                    asset_id=asset_id,
                    direction="buy",
                    amount_usd=round(trade_value, 2),
                    rationale="Move underweight asset toward target allocation.",
                )
            )
        elif diff < 0:
            sell_orders.append(
                TradeOrder(
                    asset_id=asset_id,
                    direction="sell",
                    amount_usd=round(trade_value, 2),
                    rationale="Trim overweight asset toward target allocation.",
                )
            )
    trades = sell_orders[:2] + buy_orders[:2]
    defer = not trades
    return PortfolioAction(
        trades=trades,
        defer_rebalancing=defer,
        risk_comment="Prefer gradual moves while preserving cash and transaction budget.",
    )


def policy_for(task_name: str, observation: Any):
    if task_name == "loan_underwriting":
        return loan_policy(observation)
    if task_name == "fraud_detection":
        return fraud_policy(observation)
    return portfolio_policy(observation)


def run_episode(task_name: str, seed: int, episode: int) -> dict[str, Any]:
    task = TASK_REGISTRY[task_name](seed, episode)
    observation = task.reset()
    rewards: list[float] = []
    trace: list[dict[str, Any]] = []
    done = False
    while not done:
        action = policy_for(task_name, observation)
        result = task.step(action, session_id=f"baseline-{episode}")
        rewards.append(result.reward)
        trace.append(
            {
                "observation": observation.model_dump(mode="json"),
                "action": action.model_dump(mode="json"),
                "reward": result.reward,
                "done": result.done,
            }
        )
        observation = result.observation
        done = result.done
    final_score = rewards[-1] if rewards else 0.0
    return {
        "task": task_name,
        "seed": seed,
        "episode": episode,
        "steps": len(rewards),
        "final_score": final_score,
        "total_reward": sum(rewards),
        "trace": trace,
    }


def run_task_summary(task_name: str, episodes: int, seed: int) -> dict[str, Any]:
    results = [run_episode(task_name, seed + idx, idx + 1) for idx in range(episodes)]
    scores = [episode["final_score"] for episode in results]
    return {
        "task": task_name,
        "episodes": episodes,
        "mean_score": statistics.mean(scores),
        "std_score": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "min_score": min(scores),
        "max_score": max(scores),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", choices=["all", *list(TASK_REGISTRY)])
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    if args.task == "all":
        summary = {
            "environment": "openenv-fintech",
            "episodes_per_task": args.episodes,
            "tasks": {
                task_name: run_task_summary(task_name, args.episodes, args.seed)
                for task_name in TASK_REGISTRY
            },
        }
        for task_name, task_summary in summary["tasks"].items():
            print(
                f"{task_name}: mean={task_summary['mean_score']:.3f} "
                f"std={task_summary['std_score']:.3f} "
                f"min={task_summary['min_score']:.3f} max={task_summary['max_score']:.3f}"
            )
    else:
        summary = run_task_summary(args.task, args.episodes, args.seed)
        print(
            f"{args.task}: mean={summary['mean_score']:.3f} std={summary['std_score']:.3f} "
            f"min={summary['min_score']:.3f} max={summary['max_score']:.3f}"
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
