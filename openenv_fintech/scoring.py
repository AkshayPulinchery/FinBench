"""Scoring utilities shared by tasks and graders."""

from __future__ import annotations


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def clamp01(value: float) -> float:
    # Hackathon requires scores strictly between 0 and 1
    return clamp(value, 1e-6, 1.0 - 1e-6)


def normalize_score(raw: float, min_possible: float, max_possible: float) -> float:
    if max_possible <= min_possible:
        raise ValueError("max_possible must be greater than min_possible")
    scaled = (raw - min_possible) / (max_possible - min_possible)
    return clamp01(scaled)
