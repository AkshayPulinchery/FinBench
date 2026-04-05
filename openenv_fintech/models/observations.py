"""Pydantic observation models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class LoanPurpose(str, Enum):
    MORTGAGE = "mortgage"
    AUTO = "auto"
    PERSONAL = "personal"
    BUSINESS = "business"


class VolatilityRegime(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LoanObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applicant_id: str
    credit_score: int = Field(ge=300, le=850)
    annual_income: float = Field(gt=0)
    debt_to_income_ratio: float = Field(ge=0.0, le=1.0)
    employment_years: float = Field(ge=0.0)
    loan_amount: float = Field(gt=0)
    loan_purpose: LoanPurpose
    previous_defaults: int = Field(ge=0)
    documents_submitted: list[str]


class TransactionHistorySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    avg_amount_30d: float = Field(ge=0.0)
    num_transactions_30d: int = Field(ge=0)
    usual_merchants: list[str]
    usual_locations: list[str]


class VelocitySignals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions_last_hour: int = Field(ge=0)
    unique_merchants_last_24h: int = Field(ge=0)
    amount_last_24h: float = Field(ge=0.0)


class TransactionObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: str
    amount: float = Field(gt=0)
    merchant_category: str
    timestamp: datetime
    location: str
    device_fingerprint: str
    distance_from_home: float = Field(ge=0.0)
    time_since_last_transaction: float = Field(ge=0.0)
    transaction_history_summary: TransactionHistorySummary
    velocity_signals: VelocitySignals


class PortfolioAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_weight: float = Field(ge=0.0, le=1.0)
    target_weight: float = Field(ge=0.0, le=1.0)
    current_price: float = Field(gt=0.0)
    shares_held: float = Field(ge=0.0)
    unrealized_pnl: float


class MarketContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    volatility_regime: VolatilityRegime
    recent_returns: dict[str, float]
    correlation_matrix: list[list[float]]


class PortfolioConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_single_trade_pct: float = Field(gt=0.0, le=1.0)
    min_cash_reserve: float = Field(ge=0.0)
    tax_lot_consideration: bool
    rebalancing_budget_usd: float = Field(gt=0.0)


class AccountState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cash: float = Field(ge=0.0)
    total_value: float = Field(gt=0.0)
    days_remaining: int = Field(ge=0)


class PortfolioObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0)
    portfolio: dict[str, PortfolioAsset]
    market_context: MarketContext
    constraints: PortfolioConstraints
    account: AccountState

