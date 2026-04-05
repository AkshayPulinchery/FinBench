"""Abstract task base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

ObservationT = TypeVar("ObservationT", bound=BaseModel)
ActionT = TypeVar("ActionT", bound=BaseModel)


class BaseTask(ABC, Generic[ObservationT, ActionT]):
    task_name: str

    def __init__(self, seed: int, episode: int, max_steps: int) -> None:
        self.seed = seed
        self.episode = episode
        self.max_steps = max_steps
        self.current_step = 0
        self.done = False
        self.episode_rewards: list[float] = []
        self.episode_history: list[dict[str, Any]] = []
        self.ground_truth: dict[str, Any] = {}
        self.current_observation: ObservationT | None = None

    @abstractmethod
    def reset(self) -> ObservationT:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: ActionT, session_id: str) -> Any:
        raise NotImplementedError

    def state(self) -> dict[str, Any]:
        if self.current_observation is None:
            raise RuntimeError("task has not been reset")
        return {
            "task": self.task_name,
            "step": self.current_step,
            "seed": self.seed,
            "episode_history": self.episode_history,
            "done": self.done,
            "max_steps": self.max_steps,
            "observation": self.current_observation.model_dump(mode="json"),
        }

    def is_done(self) -> bool:
        return self.done

