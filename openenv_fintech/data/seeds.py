"""Seed derivation helpers."""

from __future__ import annotations

import hashlib

import numpy as np


class SeedManager:
    """Derives stable per-task generators from a base seed."""

    @staticmethod
    def derive_seed(task: str, episode: int, seed: int) -> int:
        payload = f"{task}:{episode}:{seed}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    @classmethod
    def get_rng(cls, task: str, episode: int, seed: int) -> np.random.Generator:
        return np.random.default_rng(cls.derive_seed(task, episode, seed))

