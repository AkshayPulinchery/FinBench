"""In-memory session manager for OpenEnv Fintech."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable
from uuid import uuid4

from openenv_fintech.models.results import ResetResult, StateSnapshot, StepResult
from openenv_fintech.tasks import (
    FraudDetectionTask,
    LoanUnderwritingTask,
    PortfolioRebalancingTask,
)
from openenv_fintech.tasks.base import BaseTask


class SessionNotFoundError(KeyError):
    pass


TASK_REGISTRY: dict[str, Callable[[int, int], BaseTask]] = {
    "loan_underwriting": LoanUnderwritingTask,
    "fraud_detection": FraudDetectionTask,
    "portfolio_rebalancing": PortfolioRebalancingTask,
}


@dataclass
class SessionRecord:
    task_name: str
    task: BaseTask
    seed: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FintechEnv:
    def __init__(self, session_ttl_minutes: int = 30) -> None:
        self.sessions: dict[str, SessionRecord] = {}
        self.session_ttl = timedelta(minutes=session_ttl_minutes)
        self.episode_counters: dict[str, int] = {task_name: 0 for task_name in TASK_REGISTRY}
        self._cleanup_task: asyncio.Task | None = None

    def supported_tasks(self) -> list[str]:
        return list(TASK_REGISTRY)

    def reset(self, task: str = "loan_underwriting", seed: int = 42) -> ResetResult:
        return self.create_session(task=task, seed=seed)

    def create_session(self, task: str, seed: int) -> ResetResult:
        if task not in TASK_REGISTRY:
            raise ValueError(f"unknown task '{task}'")
        self.episode_counters[task] += 1
        episode = self.episode_counters[task]
        task_instance = TASK_REGISTRY[task](seed, episode)
        observation = task_instance.reset()
        session_id = str(uuid4())
        self.sessions[session_id] = SessionRecord(task_name=task, task=task_instance, seed=seed)
        return ResetResult(observation=observation, session_id=session_id, task=task, seed=seed)

    def get_session(self, session_id: str) -> SessionRecord:
        record = self.sessions.get(session_id)
        if record is None:
            raise SessionNotFoundError(session_id)
        record.last_accessed_at = datetime.now(timezone.utc)
        return record

    def step(self, session_id: str, action) -> StepResult:
        record = self.get_session(session_id)
        return record.task.step(action, session_id)

    def state(self, session_id: str) -> StateSnapshot:
        record = self.get_session(session_id)
        snapshot = record.task.state()
        return StateSnapshot(
            task=record.task_name,
            step=snapshot["step"],
            session_id=session_id,
            seed=record.seed,
            episode_history=snapshot["episode_history"],
            done=snapshot["done"],
            max_steps=snapshot["max_steps"],
        )

    def cleanup_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            del self.sessions[session_id]

    def cleanup_expired_sessions(self) -> int:
        now = datetime.now(timezone.utc)
        expired = [
            session_id
            for session_id, record in self.sessions.items()
            if now - record.last_accessed_at > self.session_ttl
        ]
        for session_id in expired:
            del self.sessions[session_id]
        return len(expired)

    async def cleanup_expired_sessions_loop(self, interval_seconds: int = 60) -> None:
        while True:
            self.cleanup_expired_sessions()
            await asyncio.sleep(interval_seconds)

    async def start(self) -> None:
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self.cleanup_expired_sessions_loop())

    async def stop(self) -> None:
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
