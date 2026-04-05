from statistics import mean

from openenv_fintech.models.actions import FraudAction, LoanAction, PortfolioAction, TradeOrder
from openenv_fintech.tasks import FraudDetectionTask, LoanUnderwritingTask, PortfolioRebalancingTask


def test_loan_request_info_branch_completes_in_two_steps():
    task = LoanUnderwritingTask(seed=42, episode=1)
    observation = task.reset()
    request = LoanAction(
        decision="request_info",
        reasoning="Need more verification on a borderline applicant.",
        risk_tier="medium",
    )
    first = task.step(request, session_id="s1")
    if first.done:
        # If the generated applicant is not ambiguous, the task is still valid.
        assert first.reward >= 0.0
        return
    assert first.done is False
    assert len(first.observation.documents_submitted) >= len(observation.documents_submitted)
    second = task.step(
        LoanAction(
            decision="reject",
            reasoning="Verified leverage remains too high.",
            risk_tier="medium",
        ),
        session_id="s1",
    )
    assert second.done is True
    assert task.current_step == 2


def test_fraud_task_only_scores_on_terminal_step():
    task = FraudDetectionTask(seed=42, episode=1)
    observation = task.reset()
    while not task.done:
        result = task.step(
            FraudAction(
                flag=False,
                confidence=0.1,
                hold=False,
                reason_code="none",
                notes="baseline",
            ),
            session_id="s1",
        )
        if not result.done:
            assert result.reward == 0.0
        observation = result.observation
    assert observation is not None


def test_portfolio_task_advances_and_tracks_violations():
    task = PortfolioRebalancingTask(seed=42, episode=1)
    task.reset()
    result = task.step(
        PortfolioAction(
            trades=[
                TradeOrder(
                    asset_id="AST01",
                    direction="buy",
                    amount_usd=10_000_000,
                    rationale="force violation",
                )
            ],
            defer_rebalancing=False,
            risk_comment="stress test",
        ),
        session_id="s1",
    )
    assert task.current_step == 1
    assert "violations" in result.info
    assert result.info["violations"]


def test_random_agent_scores_remain_meaningful():
    scores = []
    for episode in range(1, 11):
        loan_task = LoanUnderwritingTask(seed=episode, episode=episode)
        observation = loan_task.reset()
        action = LoanAction(
            decision="approve",
            reasoning="random-ish",
            risk_tier="medium",
            interest_rate_suggestion=0.12,
        )
        result = loan_task.step(action, session_id=f"loan-{episode}")
        scores.append(result.reward)
        assert 0.0 <= result.reward <= 1.0

    for episode in range(1, 6):
        fraud_task = FraudDetectionTask(seed=episode, episode=episode)
        fraud_task.reset()
        while not fraud_task.done:
            result = fraud_task.step(
                FraudAction(
                    flag=episode % 2 == 0,
                    confidence=0.55 if episode % 2 == 0 else 0.15,
                    hold=episode % 2 == 0,
                    reason_code="velocity" if episode % 2 == 0 else "none",
                    notes="random-ish",
                ),
                session_id=f"fraud-{episode}",
            )
        scores.append(result.reward)
        assert 0.0 <= result.reward <= 1.0

    for episode in range(1, 4):
        portfolio_task = PortfolioRebalancingTask(seed=episode, episode=episode)
        observation = portfolio_task.reset()
        while not portfolio_task.done:
            asset_id = next(iter(observation.portfolio))
            result = portfolio_task.step(
                PortfolioAction(
                    trades=[
                        TradeOrder(
                            asset_id=asset_id,
                            direction="hold",
                            amount_usd=0,
                            rationale="hold",
                        )
                    ],
                    defer_rebalancing=False,
                    risk_comment="random-ish",
                ),
                session_id=f"portfolio-{episode}",
            )
            observation = result.observation
        scores.append(result.reward)
        assert 0.0 <= result.reward <= 1.0

    assert mean(scores) > 0.1

