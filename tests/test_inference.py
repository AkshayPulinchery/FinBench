import asyncio

import inference


def test_log_step_is_single_line(capsys):
    inference.log_step(1, {"decision": "approve", "reasoning": "one\nline"}, 0.5, False, None)
    output = capsys.readouterr().out.strip()
    assert output.startswith("[STEP] step=1")
    assert "\n" not in output


def test_end_line_emits_even_on_exception(monkeypatch, capsys):
    class FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, path, json=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(inference.httpx, "AsyncClient", lambda **kwargs: FailingClient())
    result = asyncio.run(inference.run_episode(client=object(), task_name="loan_underwriting", seed=42))
    output = capsys.readouterr().out.strip().splitlines()
    assert result.success is False
    assert output[0].startswith("[START]")
    assert output[-1].startswith("[END]")
