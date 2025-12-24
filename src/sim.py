# sim.py
# This file contains the simulation environment for the robot and target.
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from config import SimConfig

@dataclass
class RobotState:
    x: float
    y: float
    theta: float  # radians

@dataclass
class TargetState:
    x: float
    y: float

def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

class World:
    def __init__(self, cfg: SimConfig, with_obstacles: bool = True):
        self.cfg = cfg
        self.robot = RobotState(x=cfg.width * 0.2, y=cfg.height * 0.5, theta=0.0)
        self.target = TargetState(x=cfg.width * 0.8, y=cfg.height * 0.5)
        # Optional circular obstacles (static). Not used in control yet.
        if with_obstacles:
            self.obstacles = [
                (cfg.width * 0.45, cfg.height * 0.35, 28),
                (cfg.width * 0.55, cfg.height * 0.7, 26),
            ]
        else:
            self.obstacles = []

    def step(self, v: float, w: float) -> None:
        # clamp
        v = float(np.clip(v, -self.cfg.v_max, self.cfg.v_max))
        w = float(np.clip(w, -self.cfg.w_max, self.cfg.w_max))

        self.robot.theta = wrap_pi(self.robot.theta + w * self.cfg.dt)
        self.robot.x += v * math.cos(self.robot.theta) * self.cfg.dt
        self.robot.y += v * math.sin(self.robot.theta) * self.cfg.dt

        # keep in bounds
        self.robot.x = float(np.clip(self.robot.x, 20, self.cfg.width - 20))
        self.robot.y = float(np.clip(self.robot.y, 20, self.cfg.height - 20))

    def distance_to_target(self) -> float:
        dx = self.target.x - self.robot.x
        dy = self.target.y - self.robot.y
        return float(math.sqrt(dx*dx + dy*dy))

    def bearing_to_target(self) -> float:
        dx = self.target.x - self.robot.x
        dy = self.target.y - self.robot.y
        ang = math.atan2(dy, dx)
        return wrap_pi(ang - self.robot.theta)
