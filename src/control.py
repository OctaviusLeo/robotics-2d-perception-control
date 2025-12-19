from __future__ import annotations
import numpy as np

class HeadingController:
    """Steer using normalized image error (target center vs frame center)."""
    def __init__(self, kp: float = 2.0, kd: float = 0.2):
        self.kp = kp
        self.kd = kd
        self._prev = 0.0

    def compute(self, err: float, dt: float) -> float:
        derr = (err - self._prev) / max(dt, 1e-6)
        self._prev = err
        return float(self.kp * err + self.kd * derr)

def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))
