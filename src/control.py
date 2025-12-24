# control.py
# Heading control and a simple SEARCH/TRACK/APPROACH state machine with speed scheduling.
from __future__ import annotations
import enum
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

    def reset(self) -> None:
        self._prev = 0.0


class ControlMode(enum.Enum):
    SEARCH = "search"
    TRACK = "track"
    APPROACH = "approach"


class StateMachineController:
    """Simple FSM: SEARCH when no detection, TRACK when far, APPROACH when close."""

    def __init__(
        self,
        heading_ctrl: HeadingController,
        v_max: float,
        w_max: float,
        approach_radius: float = 140.0,
        lost_frames_for_search: int = 20,
    ):
        self.heading_ctrl = heading_ctrl
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.approach_radius = float(approach_radius)
        self.lost_frames_for_search = int(lost_frames_for_search)
        self.mode = ControlMode.SEARCH
        self._lost_counter = 0

    def step(self, detected: bool, err: float | None, dist: float, dt: float) -> tuple[float, float, ControlMode]:
        # Update mode based on detections and distance
        if not detected:
            self._lost_counter += 1
            if self._lost_counter >= self.lost_frames_for_search:
                self.mode = ControlMode.SEARCH
        else:
            self._lost_counter = 0
            if dist < self.approach_radius:
                self.mode = ControlMode.APPROACH
            else:
                self.mode = ControlMode.TRACK

        # Compute commands per mode
        if self.mode == ControlMode.SEARCH:
            v_cmd = self.v_max * 0.25
            w_cmd = self.w_max * 0.7
        else:
            if err is None:
                v_cmd = self.v_max * 0.2
                w_cmd = self.w_max * 0.5
            else:
                w_cmd = clamp(self.heading_ctrl.compute(err, dt), -self.w_max, self.w_max)
                heading_mag = abs(err)
                if self.mode == ControlMode.TRACK:
                    speed_scale = max(0.6, 1.0 - 0.6 * heading_mag)
                    v_cmd = self.v_max * 0.95 * speed_scale
                else:  # APPROACH
                    dist_scale = max(0.15, min(1.0, dist / self.approach_radius))
                    speed_scale = max(0.25, 1.0 - 0.5 * heading_mag)
                    v_cmd = self.v_max * 0.6 * dist_scale * speed_scale

        return float(v_cmd), float(w_cmd), self.mode


def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))
