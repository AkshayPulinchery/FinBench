from pydantic import ValidationError

from openenv_fintech.models.actions import FraudAction, LoanAction, PortfolioAction, TradeOrder
from openenv_fintech.models.observations import LoanObservation
from openenv_fintech.models.results import ResetResult


def test_loan_action_requires_interest_rate_for_approve():
    try:
        LoanAction(decision="approve", reasoning="test", risk_tier="low")
    except ValidationError:
        pass
    else:
        raise AssertionError("approve should require interest_rate_suggestion")


def test_loan_action_forbids_interest_rate_for_reject():
    try:
        LoanAction(
            decision="reject",
            reasoning="test",
            risk_tier="high",
            interest_rate_suggestion=0.1,
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("reject should not allow interest_rate_suggestion")


def test_fraud_action_requires_none_reason_code_when_not_flagged():
    try:
        FraudAction(
            flag=False,
            confidence=0.1,
            hold=False,
            reason_code="velocity",
            notes="n/a",
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("unflagged transaction should require reason_code='none'")


def test_trade_order_amount_rules():
    try:
        TradeOrder(asset_id="AST01", direction="hold", amount_usd=5, rationale="wait")
    except ValidationError:
        pass
    else:
        raise AssertionError("hold order should require zero amount")

    trade = TradeOrder(asset_id="AST01", direction="buy", amount_usd=25, rationale="buy")
    assert trade.amount_usd == 25


def test_result_models_round_trip():
    observation = LoanObservation(
        applicant_id="app-1",
        credit_score=720,
        annual_income=100000,
        debt_to_income_ratio=0.2,
        employment_years=8,
        loan_amount=25000,
        loan_purpose="auto",
        previous_defaults=0,
        documents_submitted=["id", "pay_stub"],
    )
    result = ResetResult(observation=observation, session_id="abc", task="loan_underwriting", seed=42)
    payload = result.model_dump(mode="json")
    assert payload["session_id"] == "abc"


def test_portfolio_action_model_accepts_valid_trade_list():
    action = PortfolioAction(
        trades=[TradeOrder(asset_id="AST01", direction="sell", amount_usd=50, rationale="trim")],
        defer_rebalancing=False,
        risk_comment="balanced",
    )
    assert len(action.trades) == 1

