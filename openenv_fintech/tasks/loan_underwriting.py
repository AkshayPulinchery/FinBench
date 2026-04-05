"""Loan underwriting task."""

from __future__ import annotations

from openenv_fintech.data.generators import _sanitize_payload, generate_loan_applicant
from openenv_fintech.data.seeds import SeedManager
from openenv_fintech.graders.rule_based import LoanUnderwritingGrader
from openenv_fintech.models.actions import LoanAction
from openenv_fintech.models.observations import LoanObservation
from openenv_fintech.models.results import StepResult

from .base import BaseTask


class LoanUnderwritingTask(BaseTask[LoanObservation, LoanAction]):
    task_name = "loan_underwriting"

    def __init__(self, seed: int, episode: int) -> None:
        super().__init__(seed=seed, episode=episode, max_steps=3)
        self.rng = SeedManager.get_rng(self.task_name, episode, seed)
        self.grader = LoanUnderwritingGrader()
        self.follow_up_available = False

    def reset(self) -> LoanObservation:
        payload = generate_loan_applicant(self.rng)
        self.ground_truth = payload.pop("_ground_truth")
        self.current_observation = LoanObservation.model_validate(_sanitize_payload(payload))
        self.follow_up_available = True
        self.current_step = 0
        self.done = False
        self.episode_rewards.clear()
        self.episode_history.clear()
        return self.current_observation

    def step(self, action: LoanAction, session_id: str) -> StepResult:
        if self.current_observation is None:
            raise RuntimeError("task has not been reset")
        if self.done:
            raise RuntimeError("episode is already complete")

        using_follow_up = self.current_step > 0 and not self.follow_up_available
        p_default = self.ground_truth["p_default_follow_up"] if using_follow_up else self.ground_truth["p_default"]
        actuarial_rate = (
            self.ground_truth["actuarial_rate_follow_up"]
            if using_follow_up
            else self.ground_truth["actuarial_rate"]
        )
        breakdown = self.grader.score(action, p_default=p_default, actuarial_rate=actuarial_rate)
        reward = breakdown["normalized_reward"]

        if (
            action.decision.value == "request_info"
            and self.follow_up_available
            and breakdown["ambiguous"]
        ):
            self.current_step += 1
            self.follow_up_available = False
            self.current_observation = LoanObservation.model_validate(
                self.ground_truth["follow_up_observation"]
            )
            done = False
            info = {"stage": "follow_up", "breakdown": breakdown}
        else:
            self.current_step += 1
            self.done = True
            done = True
            info = {"stage": "decision_complete", "breakdown": breakdown}

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

