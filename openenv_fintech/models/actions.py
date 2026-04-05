"""Pydantic action models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LoanDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_INFO = "request_info"


class RiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TradeDirection(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class FraudReasonCode(str, Enum):
    VELOCITY = "velocity"
    LOCATION_ANOMALY = "location_anomaly"
    AMOUNT_ANOMALY = "amount_anomaly"
    MERCHANT_RISK = "merchant_risk"
    PATTERN_BREAK = "pattern_break"
    NONE = "none"


class LoanAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: LoanDecision
    reasoning: str = Field(min_length=1)
    risk_tier: RiskTier
    interest_rate_suggestion: float | None = None

    @model_validator(mode="after")
    def validate_interest_rate(self) -> "LoanAction":
        if self.decision == LoanDecision.APPROVE:
            if self.interest_rate_suggestion is None:
                raise ValueError("interest_rate_suggestion is required when approving")
            if not 0.03 <= self.interest_rate_suggestion <= 0.35:
                raise ValueError("interest_rate_suggestion must be between 0.03 and 0.35")
        elif self.interest_rate_suggestion is not None:
            raise ValueError("interest_rate_suggestion is only allowed when approving")
        return self


class TradeOrder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    direction: TradeDirection
    amount_usd: float = Field(ge=0.0)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_amount(self) -> "TradeOrder":
        if self.direction == TradeDirection.HOLD and self.amount_usd != 0:
            raise ValueError("amount_usd must be 0 for hold orders")
        if self.direction != TradeDirection.HOLD and self.amount_usd <= 0:
            raise ValueError("amount_usd must be positive for buy and sell orders")
        return self


class FraudAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flag: bool
    confidence: float = Field(ge=0.0, le=1.0)
    hold: bool
    reason_code: FraudReasonCode
    notes: str

    @model_validator(mode="after")
    def validate_reason_code(self) -> "FraudAction":
        if not self.flag and self.reason_code != FraudReasonCode.NONE:
            raise ValueError("reason_code must be 'none' when flag is false")
        if self.flag and self.reason_code == FraudReasonCode.NONE:
            raise ValueError("flagged transactions require a reason_code")
        return self


class PortfolioAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trades: list[TradeOrder]
    defer_rebalancing: bool
    risk_comment: str

