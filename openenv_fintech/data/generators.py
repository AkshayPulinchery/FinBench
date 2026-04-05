"""Deterministic synthetic generators for all tasks."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

from openenv_fintech.scoring import clamp


LOAN_PURPOSE_DOCS = {
    "mortgage": ["id", "pay_stub", "bank_statement", "property_contract"],
    "auto": ["id", "pay_stub", "vehicle_quote"],
    "personal": ["id", "bank_statement"],
    "business": ["id", "tax_return", "bank_statement", "business_plan"],
}

CITIES = [
    ("New York", (40.7128, -74.0060)),
    ("Chicago", (41.8781, -87.6298)),
    ("Austin", (30.2672, -97.7431)),
    ("Miami", (25.7617, -80.1918)),
    ("Seattle", (47.6062, -122.3321)),
    ("San Francisco", (37.7749, -122.4194)),
]

MERCHANTS = {
    "grocery": ["Whole Basket", "Fresh Mart", "Daily Foods"],
    "gas": ["FuelPoint", "QuickGas", "RoadStop"],
    "restaurant": ["Pasta Corner", "Urban Grill", "Spice Route"],
    "online": ["QuickCart", "DeviceHub", "MarketSquare"],
    "travel": ["SkyRoute", "MetroStay", "TripFoundry"],
    "electronics": ["Gizmo Center", "Volt Warehouse", "Chip Avenue"],
}

TRANSITIONS = {
    "grocery": {"gas": 0.25, "restaurant": 0.30, "online": 0.20, "grocery": 0.25},
    "gas": {"grocery": 0.30, "restaurant": 0.15, "online": 0.10, "gas": 0.45},
    "restaurant": {"grocery": 0.20, "online": 0.25, "travel": 0.10, "restaurant": 0.45},
    "online": {"electronics": 0.25, "grocery": 0.10, "restaurant": 0.20, "online": 0.45},
    "travel": {"restaurant": 0.25, "online": 0.20, "travel": 0.55},
    "electronics": {"online": 0.55, "grocery": 0.10, "electronics": 0.35},
}

CATEGORY_LOGNORMAL = {
    "grocery": (3.4, 0.35),
    "gas": (3.7, 0.30),
    "restaurant": (3.8, 0.40),
    "online": (4.0, 0.55),
    "travel": (5.1, 0.35),
    "electronics": (5.3, 0.45),
}


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _risk_band(p_default: float) -> str:
    if p_default < 0.20:
        return "low"
    if p_default > 0.70:
        return "high"
    return "medium"


def _distance_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) * 111.0


def _sanitize_payload(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def generate_loan_applicant(
    rng: np.random.Generator, episode_difficulty: str = "easy"
) -> dict:
    purpose = rng.choice(
        ["mortgage", "auto", "personal", "business"],
        p=[0.28, 0.22, 0.35, 0.15],
    ).item()
    shape_adjustment = 0.9 if episode_difficulty == "hard" else 1.0
    credit_score = int(round(300 + 550 * rng.beta(5.5 * shape_adjustment, 2.5)))
    annual_income = float(round(np.exp(rng.normal(math.log(78000), 0.45)), 2))
    debt_to_income_ratio = float(round(clamp(rng.beta(2.3, 4.2), 0.05, 0.95), 3))
    employment_years = float(round(rng.gamma(2.8, 1.9), 1))
    previous_defaults = int(rng.binomial(2, 0.08 if credit_score > 660 else 0.18))
    amount_multiplier = {
        "mortgage": rng.uniform(2.4, 4.6),
        "auto": rng.uniform(0.25, 0.6),
        "personal": rng.uniform(0.08, 0.25),
        "business": rng.uniform(0.4, 1.4),
    }[purpose]
    loan_amount = float(round(annual_income * amount_multiplier, 2))
    required_docs = LOAN_PURPOSE_DOCS[purpose]
    docs_submitted = sorted(
        set(rng.choice(required_docs, size=rng.integers(1, len(required_docs) + 1), replace=False).tolist())
    )

    p_default = _sigmoid(
        -0.01 * (credit_score - 600)
        + 0.5 * debt_to_income_ratio
        + 0.3 * int(previous_defaults > 0)
    )
    actuarial_rate = float(round(clamp(0.03 + 0.22 * p_default, 0.03, 0.35), 4))
    verified_income = float(round(annual_income * rng.uniform(0.92, 1.06), 2))
    verified_dti = float(
        round(
            clamp(
                debt_to_income_ratio * rng.uniform(0.88, 1.08)
                + (0.02 if previous_defaults else 0.0),
                0.03,
                0.98,
            ),
            3,
        )
    )
    follow_up_p_default = _sigmoid(
        -0.01 * (credit_score - 600)
        + 0.5 * verified_dti
        + 0.3 * int(previous_defaults > 0)
    )
    payload = {
        "applicant_id": f"app-{rng.integers(100000, 999999)}",
        "credit_score": credit_score,
        "annual_income": annual_income,
        "debt_to_income_ratio": debt_to_income_ratio,
        "employment_years": employment_years,
        "loan_amount": loan_amount,
        "loan_purpose": purpose,
        "previous_defaults": previous_defaults,
        "documents_submitted": docs_submitted,
        "_ground_truth": {
            "p_default": p_default,
            "risk_band": _risk_band(p_default),
            "actuarial_rate": actuarial_rate,
            "follow_up_observation": {
                "applicant_id": f"app-{rng.integers(100000, 999999)}",
                "credit_score": credit_score,
                "annual_income": verified_income,
                "debt_to_income_ratio": verified_dti,
                "employment_years": employment_years,
                "loan_amount": loan_amount,
                "loan_purpose": purpose,
                "previous_defaults": previous_defaults,
                "documents_submitted": sorted(set(required_docs)),
            },
            "p_default_follow_up": follow_up_p_default,
            "risk_band_follow_up": _risk_band(follow_up_p_default),
            "actuarial_rate_follow_up": float(
                round(clamp(0.03 + 0.22 * follow_up_p_default, 0.03, 0.35), 4)
            ),
        },
    }
    return payload


def generate_transaction_stream(
    rng: np.random.Generator, n_transactions: int = 15, fraud_rate: float = 0.2
) -> list[dict]:
    home_city, home_coord = CITIES[int(rng.integers(0, len(CITIES)))]
    history_count = int(rng.integers(45, 120))
    history_categories = rng.choice(list(MERCHANTS), size=history_count).tolist()
    history_amounts = []
    history_locations = []
    for category in history_categories:
        mean, sigma = CATEGORY_LOGNORMAL[category]
        history_amounts.append(float(round(np.exp(rng.normal(mean, sigma)), 2)))
        history_locations.append(home_city if rng.random() < 0.9 else CITIES[int(rng.integers(0, len(CITIES)))][0])
    history_summary = {
        "avg_amount_30d": float(round(float(np.mean(history_amounts)), 2)),
        "num_transactions_30d": history_count,
        "usual_merchants": sorted(set(history_categories))[:4],
        "usual_locations": sorted(set(history_locations))[:3],
    }

    fraud_count = max(1, int(round(n_transactions * fraud_rate)))
    fraud_start = int(rng.integers(2, max(3, n_transactions - fraud_count + 1)))
    fraud_indices = set(range(fraud_start, min(n_transactions, fraud_start + fraud_count)))
    timestamps: list[datetime] = []
    start = datetime(2025, 1, 1, 9, 0, 0) + timedelta(days=int(rng.integers(0, 60)))
    category = rng.choice(list(MERCHANTS)).item()
    transactions: list[dict] = []
    recent_categories: list[str] = []

    for index in range(n_transactions):
        if index > 0:
            next_map = TRANSITIONS.get(category, {"online": 1.0})
            keys = list(next_map)
            category = rng.choice(keys, p=[next_map[key] for key in keys]).item()
        is_fraud = index in fraud_indices
        mean, sigma = CATEGORY_LOGNORMAL[category]
        amount = float(round(np.exp(rng.normal(mean, sigma)), 2))
        minutes_since_last = float(round(rng.uniform(4, 360), 2))

        if timestamps and is_fraud:
            minutes_since_last = float(round(rng.uniform(1, 12), 2))
        current_time = start if not timestamps else timestamps[-1] + timedelta(minutes=minutes_since_last)
        timestamps.append(current_time)

        city, coord = home_city, home_coord
        reason_patterns: list[str] = []
        if is_fraud:
            fraud_city, fraud_coord = CITIES[int(rng.integers(0, len(CITIES)))]
            city, coord = fraud_city, fraud_coord
            if fraud_city != home_city:
                reason_patterns.append("location_anomaly")
            if index == max(fraud_indices):
                amount = float(round(amount * rng.uniform(4.5, 8.5), 2))
                reason_patterns.append("amount_anomaly")
            if len([ts for ts in timestamps[:-1] if (current_time - ts).total_seconds() <= 3600]) >= 3:
                reason_patterns.append("velocity")
        elif rng.random() < 0.08:
            city, coord = CITIES[int(rng.integers(0, len(CITIES)))]

        recent_window = [tx for tx in transactions if (current_time - datetime.fromisoformat(tx["timestamp"])).total_seconds() <= 3600]
        recent_day = [tx for tx in transactions if (current_time - datetime.fromisoformat(tx["timestamp"])).total_seconds() <= 86400]
        distance = float(round(_distance_km(home_coord, coord), 2))
        recent_categories.append(category)
        payload = {
            "transaction_id": f"txn-{index + 1:03d}",
            "amount": amount,
            "merchant_category": category,
            "timestamp": current_time.isoformat(),
            "location": city,
            "device_fingerprint": f"dev-{rng.integers(1000, 9999)}-{index}",
            "distance_from_home": distance,
            "time_since_last_transaction": 0.0 if index == 0 else minutes_since_last,
            "transaction_history_summary": history_summary,
            "velocity_signals": {
                "transactions_last_hour": len(recent_window) + 1,
                "unique_merchants_last_24h": len({tx["merchant_category"] for tx in recent_day} | {category}),
                "amount_last_24h": float(round(sum(tx["amount"] for tx in recent_day) + amount, 2)),
            },
            "_is_fraud": is_fraud,
            "_fraud_patterns": reason_patterns,
        }
        transactions.append(payload)

    return transactions


def generate_price_path(
    rng: np.random.Generator, n_assets: int = 10, n_steps: int = 20
) -> dict:
    asset_ids = [f"AST{i:02d}" for i in range(1, n_assets + 1)]
    mus = rng.uniform(0.04, 0.14, size=n_assets)
    base_vols = rng.uniform(0.12, 0.28, size=n_assets)
    initial_prices = rng.uniform(40.0, 180.0, size=n_assets)
    raw = rng.normal(size=(n_assets, n_assets))
    cov = raw @ raw.T
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr + np.eye(n_assets) * 1e-6)
    dt = 1.0 / 252.0
    regime_multipliers = {"low": 0.75, "medium": 1.0, "high": 1.45}
    current_regime = rng.choice(["low", "medium", "high"], p=[0.3, 0.45, 0.25]).item()
    regimes = [current_regime]
    prices = [initial_prices]

    for _ in range(n_steps):
        if rng.random() < 0.1:
            current_regime = rng.choice(["low", "medium", "high"], p=[0.3, 0.45, 0.25]).item()
        regimes.append(current_regime)
        shocks = chol @ rng.normal(size=n_assets)
        sigma = base_vols * regime_multipliers[current_regime]
        previous = prices[-1]
        next_prices = previous * np.exp((mus - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * shocks)
        prices.append(next_prices)

    price_matrix = np.vstack(prices)
    return {
        "asset_ids": asset_ids,
        "price_history": {
            asset: [float(round(value, 4)) for value in price_matrix[:, idx]]
            for idx, asset in enumerate(asset_ids)
        },
        "volatility_regimes": regimes,
        "correlation_matrix": [[float(round(value, 4)) for value in row] for row in corr.tolist()],
        "initial_prices": {asset: float(round(initial_prices[idx], 4)) for idx, asset in enumerate(asset_ids)},
    }


__all__ = [
    "generate_loan_applicant",
    "generate_price_path",
    "generate_transaction_stream",
    "_sanitize_payload",
]
