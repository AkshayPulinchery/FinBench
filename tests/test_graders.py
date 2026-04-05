from baseline import fraud_policy, loan_policy, portfolio_policy
from openenv_fintech.models.actions import FraudAction, LoanAction, PortfolioAction, TradeOrder
from openenv_fintech.tasks import FraudDetectionTask, LoanUnderwritingTask, PortfolioRebalancingTask


def test_loan_grader_score_range_over_random_actions():
    for episode in range(1, 51):
        task = LoanUnderwritingTask(seed=episode, episode=episode)
        observation = task.reset()
        action = LoanAction(
            decision="approve" if episode % 2 == 0 else "reject",
            reasoning="test",
            risk_tier="medium",
            interest_rate_suggestion=0.08 if episode % 2 == 0 else None,
        )
        result = task.step(action, session_id=f"s-{episode}")
        assert 0.0 <= result.reward <= 1.0
        if not result.done:
            result = task.step(
                LoanAction(
                    decision="reject",
                    reasoning="follow-up",
                    risk_tier="medium",
                ),
                session_id=f"s-{episode}",
            )
            assert 0.0 <= result.reward <= 1.0


def test_fraud_grader_score_range_over_random_sequences():
    for episode in range(1, 21):
        task = FraudDetectionTask(seed=episode, episode=episode)
        observation = task.reset()
        while not task.done:
            flagged = (task.current_step + episode) % 3 == 0
            result = task.step(
                FraudAction(
                    flag=flagged,
                    confidence=0.75 if flagged else 0.15,
                    hold=flagged,
                    reason_code="velocity" if flagged else "none",
                    notes="test",
                ),
                session_id=f"s-{episode}",
            )
            observation = result.observation
        assert observation is not None
        assert 0.0 <= result.reward <= 1.0


def test_portfolio_grader_score_range_over_random_sequences():
    for episode in range(1, 11):
        task = PortfolioRebalancingTask(seed=episode, episode=episode)
        observation = task.reset()
        while not task.done:
            asset_id = next(iter(observation.portfolio))
            result = task.step(
                PortfolioAction(
                    trades=[
                        TradeOrder(
                            asset_id=asset_id,
                            direction="hold",
                            amount_usd=0,
                            rationale="no-op",
                        )
                    ],
                    defer_rebalancing=False,
                    risk_comment="test",
                ),
                session_id=f"s-{episode}",
            )
            observation = result.observation
        assert 0.0 <= result.reward <= 1.0


def test_heuristic_agents_score_above_minimum_floor():
    loan = LoanUnderwritingTask(seed=42, episode=1)
    loan_obs = loan.reset()
    loan_result = loan.step(loan_policy(loan_obs), session_id="loan")
    if not loan_result.done:
        loan_result = loan.step(loan_policy(loan_result.observation), session_id="loan")
    assert loan_result.reward > 0.4

    fraud = FraudDetectionTask(seed=42, episode=1)
    fraud_obs = fraud.reset()
    while not fraud.done:
        fraud_result = fraud.step(fraud_policy(fraud_obs), session_id="fraud")
        fraud_obs = fraud_result.observation
    assert fraud_result.reward > 0.1

    portfolio = PortfolioRebalancingTask(seed=42, episode=1)
    portfolio_obs = portfolio.reset()
    while not portfolio.done:
        portfolio_result = portfolio.step(portfolio_policy(portfolio_obs), session_id="portfolio")
        portfolio_obs = portfolio_result.observation
    assert portfolio_result.reward > 0.1

