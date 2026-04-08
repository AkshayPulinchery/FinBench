"""Pydantic result models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .observations import LoanObservation, PortfolioObservation, TransactionObservation

ObservationUnion = LoanObservation | TransactionObservation | PortfolioObservation


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: ObservationUnion
    reward: float
    done: bool
    info: dict[str, Any]
    session_id: str


class ResetResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: ObservationUnion
    session_id: str
    task: str
    seed: int


class StateSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    step: int = Field(ge=0)
    session_id: str
    seed: int
    episode_history: list[dict[str, Any]]
    done: bool
    max_steps: int


class EpisodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_score: float = Field(default=0.0)
    total_reward: float = Field(default=0.0)
    steps: int = Field(ge=0)
    success: bool
    breakdown: dict[str, Any]

