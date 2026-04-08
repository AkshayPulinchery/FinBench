"""Optional Anthropic-based reasoning scorer."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from openenv_fintech.scoring import MIN_SCORE, safe_score


class LLMJudge:
    def __init__(self, model: str = "claude-sonnet-4-20250514", timeout_seconds: int = 10) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.enabled = os.getenv("LLM_JUDGE_ENABLED", "false").lower() == "true"

    async def score_reasoning(self, reasoning_text: str, context: dict[str, Any]) -> float:
        if not self.enabled or not reasoning_text.strip():
            return MIN_SCORE
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            return MIN_SCORE

        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        prompt = (
            "Score the reasoning quality from 0.0 to 0.2.\n"
            "Evaluate relevance, correctness, and completeness.\n"
            "Return only the numeric score.\n\n"
            f"Context: {context}\n"
            f"Reasoning: {reasoning_text}"
        )
        try:
            response = await asyncio.wait_for(
                client.messages.create(
                    model=self.model,
                    max_tokens=16,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=self.timeout_seconds,
            )
        except Exception:
            return MIN_SCORE

        try:
            text = response.content[0].text.strip()
            return safe_score(float(text))
        except Exception:
            return MIN_SCORE
