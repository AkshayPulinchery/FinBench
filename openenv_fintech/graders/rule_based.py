"""Deterministic task graders."""

from __future__ import annotations

import math
from statistics import mean, pstdev

from openenv_fintech.models.actions import FraudAction, LoanAction
from openenv_fintech.scoring import MIN_SCORE, MAX_SCORE, clamp01, normalize_score, safe_score

from .base import BaseGrader


class LoanUnderwritingGrader(BaseGrader):
    min_raw = -1.0
    max_raw = 1.5

    @staticmethod
    def _risk_band(p_default: float) -> str:
        if p_default < 0.20:
            return "low"
        if p_default > 0.70:
            return "high"
        return "medium"

    def score(self, action: LoanAction, p_default: float, actuarial_rate: float) -> dict[str, float]:
        raw = 0.0
        ambiguous = 0.30 <= p_default <= 0.70
        if action.decision.value == "approve":
            if p_default < 0.20:
                raw += 1.0
            elif p_default > 0.70:
                raw -= 0.8
            else:
                raw += 0.25
        elif action.decision.value == "reject":
            if p_default > 0.70:
                raw += 1.0
            elif p_default < 0.20:
                raw -= 0.4
            else:
                raw += 0.35
        else:
            raw += 0.3 if ambiguous else -0.2
        if action.risk_tier.value == self._risk_band(p_default):
            raw += 0.2
        if action.decision.value == "approve" and action.interest_rate_suggestion is not None:
            if abs(action.interest_rate_suggestion - actuarial_rate) <= 0.02:
                raw += 0.1
        normalized = normalize_score(raw, self.min_raw, self.max_raw)
        return {
            "raw_reward": raw,
            "normalized_reward": normalized,
            "p_default": p_default,
            "actuarial_rate": actuarial_rate,
            "risk_band": self._risk_band(p_default),
            "ambiguous": ambiguous,
        }


class FraudDetectionGrader(BaseGrader):
    @staticmethod
    def _expected_calibration_error(actions: list[FraudAction], labels: list[bool], bins: int = 5) -> float:
        if not actions:
            return MAX_SCORE
        bucket_size = 1.0 / bins
        ece = 0.0
        for idx in range(bins):
            low = idx * bucket_size
            high = 1.0 if idx == bins - 1 else (idx + 1) * bucket_size
            bucket = [
                (action, label)
                for action, label in zip(actions, labels)
                if low <= action.confidence <= high
            ]
            if not bucket:
                continue
            avg_conf = mean(action.confidence for action, _ in bucket)
            accuracy = mean(MAX_SCORE if action.flag == label else MIN_SCORE for action, label in bucket)
            ece += abs(avg_conf - accuracy) * (len(bucket) / len(actions))
        return ece

    def score(self, actions: list[FraudAction], transactions: list[dict]) -> dict[str, float]:
        labels = [bool(tx["_is_fraud"]) for tx in transactions]
        flags = [action.flag for action in actions]
        amounts = [float(tx["amount"]) for tx in transactions]
        tp = sum(flag and label for flag, label in zip(flags, labels))
        tn = sum((not flag) and (not label) for flag, label in zip(flags, labels))
        fp = sum(flag and (not label) for flag, label in zip(flags, labels))
        fn = sum((not flag) and label for flag, label in zip(flags, labels))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = MIN_SCORE if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        fpr = fp / max(fp + tn, 1)
        calibration = safe_score(MAX_SCORE - self._expected_calibration_error(actions, labels))

        fraud_indices = [idx for idx, label in enumerate(labels) if label]
        high_value_index = None
        if fraud_indices:
            high_value_index = max(fraud_indices, key=lambda idx: amounts[idx])
        caught_early = (
            high_value_index is not None
            and any(
                idx <= high_value_index and labels[idx] and flags[idx]
                for idx in range(len(labels))
            )
        )
        early_bonus = 0.15 if caught_early else MIN_SCORE
        missed_high_value = sum(
            1
            for label, flag, amount in zip(labels, flags, amounts)
            if label and (not flag) and amount > 5000
        )
        penalty = 0.2 * missed_high_value
        raw = recall * 0.4 + (1.0 - fpr) * 0.3 + f1 * 0.2 + calibration * 0.1 + early_bonus - penalty
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fpr,
            "calibration": calibration,
            "early_detection_bonus": early_bonus,
            "missed_high_value_penalty": penalty,
            "final_score": safe_score(raw),
        }


class PortfolioRebalancingGrader(BaseGrader):
    @staticmethod
    def _sharpe_ratio(returns: list[float]) -> float:
        if len(returns) < 2:
            return MIN_SCORE
        volatility = pstdev(returns)
        if volatility == 0:
            return MIN_SCORE
        return mean(returns) / volatility * math.sqrt(252)

    def score(
        self,
        final_weights: dict[str, float],
        target_weights: dict[str, float],
        budget_used: float,
        budget_allocated: float,
        daily_returns: list[float],
        violation_count: int,
        partial_credit_total: float,
        reached_target_day: int | None,
        max_steps: int,
        drift_breached: bool,
    ) -> dict[str, float]:
        diffs = [final_weights[asset] - target_weights[asset] for asset in final_weights]
        rmse = math.sqrt(sum(diff * diff for diff in diffs) / max(len(diffs), 1))
        tracking_score = clamp01(1.0 - rmse / 0.25)

        utilization = budget_used / max(budget_allocated, 1.0)
        if utilization <= 1.0:
            cost_efficiency = clamp01(1.0 - utilization * 0.5)
        else:
            cost_efficiency = clamp01(0.5 - min(utilization - 1.0, 1.0) * 0.5)

        sharpe = self._sharpe_ratio(daily_returns)
        sharpe_score = clamp01((sharpe + MAX_SCORE) / 3.0)
        early_bonus = 0.1 if reached_target_day is not None and (max_steps - reached_target_day) >= 3 else MIN_SCORE
        drift_penalty = 0.1 if drift_breached else MIN_SCORE
        violation_penalty = 0.15 * violation_count

        raw = (
            tracking_score * 0.35
            + cost_efficiency * 0.25
            + sharpe_score * 0.20
            + partial_credit_total
            + early_bonus
            - violation_penalty
            - drift_penalty
        )
        return {
            "tracking_error_rmse": rmse,
            "tracking_score": safe_score(tracking_score),
            "cost_efficiency": safe_score(cost_efficiency),
            "budget_used": budget_used,
            "budget_allocated": budget_allocated,
            "sharpe_ratio": sharpe,
            "sharpe_score": safe_score(sharpe_score),
            "violation_penalty": violation_penalty,
            "drift_penalty": drift_penalty,
            "early_completion_bonus": early_bonus,
            "partial_credit_total": partial_credit_total,
            "final_score": safe_score(raw),
        }

