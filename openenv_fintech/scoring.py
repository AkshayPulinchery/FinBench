"""Scoring utilities shared by tasks and graders."""

from __future__ import annotations


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


MIN_SCORE = 0.01
MAX_SCORE = 0.99


def clamp01(value: float) -> float:
    # Hackathon requires scores strictly between 0 and 1
    # Using (0.01, 0.99) to avoid issues with rounding to 0.00
    return clamp(value, MIN_SCORE, MAX_SCORE)


def normalize_score(raw: float, min_possible: float, max_possible: float) -> float:
    if max_possible <= min_possible:
        raise ValueError("max_possible must be greater than min_possible")
    scaled = (raw - min_possible) / (max_possible - min_possible)
    return clamp01(scaled)
