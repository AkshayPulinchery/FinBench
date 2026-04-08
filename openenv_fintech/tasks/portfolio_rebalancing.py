"""Portfolio rebalancing task."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from openenv_fintech.data.generators import generate_price_path
from openenv_fintech.data.seeds import SeedManager
from openenv_fintech.graders.rule_based import PortfolioRebalancingGrader
from openenv_fintech.models.actions import PortfolioAction
from openenv_fintech.models.observations import (
    AccountState,
    MarketContext,
    PortfolioAsset,
    PortfolioConstraints,
    PortfolioObservation,
)
from openenv_fintech.models.results import StepResult
from openenv_fintech.scoring import MIN_SCORE

from .base import BaseTask


class PortfolioRebalancingTask(BaseTask[PortfolioObservation, PortfolioAction]):
    task_name = "portfolio_rebalancing"

    def __init__(self, seed: int, episode: int) -> None:
        super().__init__(seed=seed, episode=episode, max_steps=20)
        self.rng = SeedManager.get_rng(self.task_name, episode, seed)
        self.grader = PortfolioRebalancingGrader()
        self.asset_ids: list[str] = []
        self.price_history: dict[str, list[float]] = {}
        self.volatility_regimes: list[str] = []
        self.correlation_matrix: list[list[float]] = []
        self.holdings: dict[str, float] = {}
        self.target_weights: dict[str, float] = {}
        self.initial_prices: dict[str, float] = {}
        self.constraints: PortfolioConstraints | None = None
        self.cash = 0.0
        self.transaction_cost_budget_used = 0.0
        self.constraint_violations: list[str] = []
        self.daily_returns: list[float] = []
        self.partial_credit_total = 0.0
        self.reached_target_day: int | None = None
        self.drift_breached = False

    def reset(self) -> PortfolioObservation:
        generated = generate_price_path(self.rng, n_assets=10, n_steps=self.max_steps)
        self.asset_ids = generated["asset_ids"]
        self.price_history = generated["price_history"]
        self.volatility_regimes = generated["volatility_regimes"]
        self.correlation_matrix = generated["correlation_matrix"]
        self.initial_prices = generated["initial_prices"]

        initial_weights = self.rng.dirichlet(np.ones(len(self.asset_ids)))
        target_weights = self.rng.dirichlet(np.ones(len(self.asset_ids)))
        investable_value = 90000.0
        self.cash = float(round(self.rng.uniform(10000.0, 18000.0), 2))
        self.holdings = {
            asset: float(round((investable_value * initial_weights[idx]) / self.price_history[asset][0], 6))
            for idx, asset in enumerate(self.asset_ids)
        }
        self.target_weights = {
            asset: float(round(target_weights[idx], 6)) for idx, asset in enumerate(self.asset_ids)
        }
        self.constraints = PortfolioConstraints(
            max_single_trade_pct=float(round(self.rng.uniform(0.10, 0.20), 3)),
            min_cash_reserve=float(round(self.rng.uniform(4000.0, 9000.0), 2)),
            tax_lot_consideration=bool(self.rng.integers(0, 2)),
            rebalancing_budget_usd=float(round(self.rng.uniform(1800.0, 4500.0), 2)),
        )
        self.transaction_cost_budget_used = 0.0
        self.constraint_violations = []
        self.daily_returns = []
        self.partial_credit_total = 0.0
        self.reached_target_day = None
        self.drift_breached = False
        self.current_step = 0
        self.done = False
        self.episode_rewards.clear()
        self.episode_history.clear()
        self.current_observation = self._build_observation()
        return self.current_observation

    def _prices_for_step(self, step: int) -> dict[str, float]:
        return {asset: self.price_history[asset][step] for asset in self.asset_ids}

    def _portfolio_values(self, prices: dict[str, float]) -> tuple[dict[str, float], float]:
        asset_values = {asset: self.holdings[asset] * prices[asset] for asset in self.asset_ids}
        total_value = self.cash + sum(asset_values.values())
        return asset_values, total_value

    def _current_weights(self, prices: dict[str, float]) -> tuple[dict[str, float], float]:
        asset_values, total_value = self._portfolio_values(prices)
        if total_value <= 0:
            raise RuntimeError("portfolio total value must remain positive")
        return {asset: asset_values[asset] / total_value for asset in self.asset_ids}, total_value

    def _recent_returns(self, step: int) -> dict[str, float]:
        returns = {}
        for asset in self.asset_ids:
            start_idx = max(0, step - 5)
            start_price = self.price_history[asset][start_idx]
            end_price = self.price_history[asset][step]
            returns[asset] = 0.0 if start_price == 0 else round((end_price - start_price) / start_price, 4)
        return returns

    def _build_observation(self) -> PortfolioObservation:
        prices = self._prices_for_step(self.current_step)
        weights, total_value = self._current_weights(prices)
        portfolio = {}
        for asset in self.asset_ids:
            initial_price = self.initial_prices[asset]
            current_price = prices[asset]
            pnl = (current_price - initial_price) * self.holdings[asset]
            portfolio[asset] = PortfolioAsset(
                current_weight=round(weights[asset], 6),
                target_weight=round(self.target_weights[asset], 6),
                current_price=round(current_price, 4),
                shares_held=round(self.holdings[asset], 6),
                unrealized_pnl=round(pnl, 2),
            )

        assert self.constraints is not None
        return PortfolioObservation(
            step=self.current_step,
            portfolio=portfolio,
            market_context=MarketContext(
                volatility_regime=self.volatility_regimes[self.current_step],
                recent_returns=self._recent_returns(self.current_step),
                correlation_matrix=deepcopy(self.correlation_matrix),
            ),
            constraints=self.constraints,
            account=AccountState(
                cash=round(self.cash, 2),
                total_value=round(total_value, 2),
                days_remaining=max(self.max_steps - self.current_step, 0),
            ),
        )

    def _apply_trade(self, asset_id: str, direction: str, amount_usd: float, total_value: float) -> None:
        assert self.constraints is not None
        prices = self._prices_for_step(self.current_step)
        if asset_id not in self.holdings:
            self.constraint_violations.append(f"unknown_asset:{asset_id}")
            return
        if amount_usd > self.constraints.max_single_trade_pct * total_value:
            self.constraint_violations.append(f"max_single_trade_pct:{asset_id}")
            return

        fee = amount_usd * 0.001
        if self.transaction_cost_budget_used + fee > self.constraints.rebalancing_budget_usd:
            self.constraint_violations.append(f"budget:{asset_id}")
            return

        price = prices[asset_id]
        if direction == "buy":
            if self.cash - amount_usd - fee < self.constraints.min_cash_reserve:
                self.constraint_violations.append(f"min_cash_reserve:{asset_id}")
                return
            self.holdings[asset_id] += amount_usd / price
            self.cash -= amount_usd + fee
            self.transaction_cost_budget_used += fee
        elif direction == "sell":
            available_value = self.holdings[asset_id] * price
            if amount_usd > available_value:
                self.constraint_violations.append(f"oversell:{asset_id}")
                return
            self.holdings[asset_id] -= amount_usd / price
            self.cash += amount_usd - fee
            self.transaction_cost_budget_used += fee

    def step(self, action: PortfolioAction, session_id: str) -> StepResult:
        if self.current_observation is None:
            raise RuntimeError("task has not been reset")
        if self.done:
            raise RuntimeError("episode is already complete")

        pre_trade_prices = self._prices_for_step(self.current_step)
        _, total_value = self._portfolio_values(pre_trade_prices)
        if not action.defer_rebalancing:
            for trade in action.trades:
                if trade.direction.value != "hold":
                    self._apply_trade(trade.asset_id, trade.direction.value, trade.amount_usd, total_value)

        next_step = min(self.current_step + 1, self.max_steps)
        post_trade_value = self._portfolio_values(pre_trade_prices)[1]
        next_prices = self._prices_for_step(next_step)
        next_value = self._portfolio_values(next_prices)[1]
        self.daily_returns.append((next_value - post_trade_value) / max(post_trade_value, 1.0))
        self.current_step = next_step

        weights, _ = self._current_weights(next_prices)
        within_band = sum(
            1 for asset in self.asset_ids if abs(weights[asset] - self.target_weights[asset]) <= 0.02
        )
        step_reward = max(MIN_SCORE, 0.02 * within_band)
        self.partial_credit_total += step_reward
        if all(abs(weights[asset] - self.target_weights[asset]) <= 0.02 for asset in self.asset_ids):
            if self.reached_target_day is None:
                self.reached_target_day = self.current_step
        if any(abs(weights[asset] - self.target_weights[asset]) > 0.15 for asset in self.asset_ids):
            self.drift_breached = True

        done = self.current_step >= self.max_steps
        info: dict[str, object] = {"violations": list(self.constraint_violations)}
        if done:
            self.done = True
            breakdown = self.grader.score(
                final_weights=weights,
                target_weights=self.target_weights,
                budget_used=self.transaction_cost_budget_used,
                budget_allocated=self.constraints.rebalancing_budget_usd if self.constraints else 1.0,
                daily_returns=self.daily_returns,
                violation_count=len(self.constraint_violations),
                partial_credit_total=self.partial_credit_total,
                reached_target_day=self.reached_target_day,
                max_steps=self.max_steps,
                drift_breached=self.drift_breached,
            )
            reward = breakdown["final_score"]
            info["breakdown"] = breakdown
        else:
            reward = step_reward

        self.current_observation = self._build_observation()
        self.episode_rewards.append(reward)
        self.episode_history.append(
            {
                "step": self.current_step,
                "action": action.model_dump(mode="json"),
                "reward": reward,
                "done": done,
            }
        )
        return StepResult(
            observation=self.current_observation,
            reward=reward,
            done=done,
            info=info,
            session_id=session_id,
        )
