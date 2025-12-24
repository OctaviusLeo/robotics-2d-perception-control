# estimation.py
# Simple filters for perception signals.
from __future__ import annotations

from typing import Optional


class ExponentialSmoother:
    """1D exponential moving average with reset support."""

    def __init__(self, alpha: float = 0.35):
        self.alpha = float(alpha)
        self._value: Optional[float] = None

    def reset(self) -> None:
        self._value = None

    def update(self, x: float) -> float:
        if self._value is None:
            self._value = x
        else:
            self._value = self.alpha * x + (1.0 - self.alpha) * self._value
        return self._value
