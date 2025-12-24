# config.py
# Configuration for the simulation environment
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class SimConfig:
    width: int = 1920
    height: int = 1080
    dt: float = 1.0 / 60.0

    # Robot
    wheel_base: float = 40.0
    v_max: float = 140.0  # px/s
    w_max: float = 3.2    # rad/s

    # Camera
    cam_w: int = 160
    cam_h: int = 120
    cam_fov: float = 1.2  # radians (roughly)

    # Target
    target_radius: int = 16
