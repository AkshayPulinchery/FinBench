"""Base grader abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseGrader(ABC):
    @abstractmethod
    def score(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

