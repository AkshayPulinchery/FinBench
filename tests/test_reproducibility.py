from openenv_fintech.data.generators import (
    generate_loan_applicant,
    generate_price_path,
    generate_transaction_stream,
)
from openenv_fintech.data.seeds import SeedManager


def test_same_seed_produces_identical_loan_applicants():
    outputs = [
        generate_loan_applicant(SeedManager.get_rng("loan_underwriting", 1, 42))
        for _ in range(3)
    ]
    assert outputs[0] == outputs[1] == outputs[2]


def test_same_seed_produces_identical_transaction_streams():
    outputs = [
        generate_transaction_stream(SeedManager.get_rng("fraud_detection", 1, 42))
        for _ in range(3)
    ]
    assert outputs[0] == outputs[1] == outputs[2]


def test_same_seed_produces_identical_price_paths():
    outputs = [
        generate_price_path(SeedManager.get_rng("portfolio_rebalancing", 1, 42))
        for _ in range(3)
    ]
    assert outputs[0] == outputs[1] == outputs[2]


def test_different_seeds_produce_different_outputs():
    loan_a = generate_loan_applicant(SeedManager.get_rng("loan_underwriting", 1, 42))
    loan_b = generate_loan_applicant(SeedManager.get_rng("loan_underwriting", 1, 43))
    fraud_a = generate_transaction_stream(SeedManager.get_rng("fraud_detection", 1, 42))
    fraud_b = generate_transaction_stream(SeedManager.get_rng("fraud_detection", 1, 43))
    portfolio_a = generate_price_path(SeedManager.get_rng("portfolio_rebalancing", 1, 42))
    portfolio_b = generate_price_path(SeedManager.get_rng("portfolio_rebalancing", 1, 43))
    assert loan_a != loan_b
    assert fraud_a != fraud_b
    assert portfolio_a != portfolio_b

