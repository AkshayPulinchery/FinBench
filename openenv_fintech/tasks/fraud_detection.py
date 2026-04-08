"""Fraud detection task."""

from __future__ import annotations

from openenv_fintech.data.generators import _sanitize_payload, generate_transaction_stream
from openenv_fintech.data.seeds import SeedManager
from openenv_fintech.graders.rule_based import FraudDetectionGrader
from openenv_fintech.models.actions import FraudAction
from openenv_fintech.models.observations import TransactionObservation
from openenv_fintech.models.results import StepResult
from openenv_fintech.scoring import NUDGE

from .base import BaseTask


class FraudDetectionTask(BaseTask[TransactionObservation, FraudAction]):
    task_name = "fraud_detection"

    def __init__(self, seed: int, episode: int) -> None:
        super().__init__(seed=seed, episode=episode, max_steps=20)
        self.rng = SeedManager.get_rng(self.task_name, episode, seed)
        self.grader = FraudDetectionGrader()
        self.transactions: list[dict] = []
        self.actions: list[FraudAction] = []

    def reset(self) -> TransactionObservation:
        n_transactions = int(self.rng.integers(10, 21))
        self.transactions = generate_transaction_stream(self.rng, n_transactions=n_transactions)
        self.actions = []
        self.current_step = 0
        self.done = False
        self.episode_rewards.clear()
        self.episode_history.clear()
        self.current_observation = TransactionObservation.model_validate(
            _sanitize_payload(self.transactions[0])
        )
        return self.current_observation

    def step(self, action: FraudAction, session_id: str) -> StepResult:
        if self.current_observation is None:
            raise RuntimeError("task has not been reset")
        if self.done:
            raise RuntimeError("episode is already complete")

        self.actions.append(action)
        self.current_step += 1
        is_final = self.current_step >= len(self.transactions)
        reward = NUDGE
        info: dict[str, object] = {"progress": f"{self.current_step}/{len(self.transactions)}"}

        if is_final:
            breakdown = self.grader.score(self.actions, self.transactions)
            reward = breakdown["final_score"]
            self.done = True
            info["breakdown"] = breakdown
        else:
            self.current_observation = TransactionObservation.model_validate(
                _sanitize_payload(self.transactions[self.current_step])
            )

        self.episode_rewards.append(reward)
        self.episode_history.append(
            {
                "step": self.current_step,
                "action": action.model_dump(mode="json"),
                "reward": reward,
                "done": is_final,
            }
        )
        return StepResult(
            observation=self.current_observation,
            reward=reward,
            done=is_final,
            info=info,
            session_id=session_id,
        )

