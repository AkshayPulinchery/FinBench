from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_reset_with_empty_body_returns_200():
    response = client.post("/reset", json={})
    assert response.status_code == 200
    payload = response.json()
    assert payload["task"] == "loan_underwriting"
    client.post(f"/close/{payload['session_id']}")


def test_reset_with_unknown_task_returns_422():
    response = client.post("/reset", json={"task": "unknown"})
    assert response.status_code == 422


def test_step_with_wrong_session_id_returns_404():
    response = client.post("/step", json={"session_id": "missing", "action": {}})
    assert response.status_code == 404


def test_step_with_invalid_action_returns_422():
    reset = client.post("/reset", json={"task": "loan_underwriting", "seed": 42})
    session_id = reset.json()["session_id"]
    response = client.post("/step", json={"session_id": session_id, "action": {"decision": "approve"}})
    assert response.status_code == 422
    client.post(f"/close/{session_id}")


def test_full_episode_run_for_each_task_completes():
    tasks = {
        "loan_underwriting": {
            "decision": "approve",
            "reasoning": "Strong profile",
            "risk_tier": "low",
            "interest_rate_suggestion": 0.07,
        },
        "fraud_detection": {
            "flag": False,
            "confidence": 0.1,
            "hold": False,
            "reason_code": "none",
            "notes": "baseline",
        },
        "portfolio_rebalancing": {
            "trades": [
                {
                    "asset_id": "AST01",
                    "direction": "hold",
                    "amount_usd": 0,
                    "rationale": "wait",
                }
            ],
            "defer_rebalancing": False,
            "risk_comment": "baseline",
        },
    }
    for task_name, action in tasks.items():
        reset = client.post("/reset", json={"task": task_name, "seed": 42})
        assert reset.status_code == 200
        payload = reset.json()
        session_id = payload["session_id"]
        done = False
        while not done:
            step = client.post("/step", json={"session_id": session_id, "action": action})
            assert step.status_code == 200
            done = step.json()["done"]
        close = client.post(f"/close/{session_id}")
        assert close.status_code == 200


def test_state_endpoint_returns_correct_step_count():
    reset = client.post("/reset", json={"task": "loan_underwriting", "seed": 42})
    session_id = reset.json()["session_id"]
    client.post(
        "/step",
        json={
            "session_id": session_id,
            "action": {
                "decision": "approve",
                "reasoning": "Strong profile",
                "risk_tier": "low",
                "interest_rate_suggestion": 0.08,
            },
        },
    )
    state = client.get(f"/state/{session_id}")
    assert state.status_code == 200
    assert state.json()["step"] == 1
    client.post(f"/close/{session_id}")

