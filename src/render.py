from __future__ import annotations
import math
import numpy as np
import pygame
from config import SimConfig
from sim import World

TARGET_BGR = (255, 255, 0)  # BGR for OpenCV: cyan-ish
TARGET_RGB = (TARGET_BGR[2], TARGET_BGR[1], TARGET_BGR[0])

def init_pygame(headless: bool) -> None:
    if headless:
        import os
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()

def draw_world(screen: pygame.Surface, cfg: SimConfig, world: World) -> None:
    screen.fill((245, 245, 245))

    # Target
    pygame.draw.circle(screen, TARGET_RGB, (int(world.target.x), int(world.target.y)), cfg.target_radius)

    # Robot
    rx, ry, th = world.robot.x, world.robot.y, world.robot.theta
    pygame.draw.circle(screen, (30, 30, 30), (int(rx), int(ry)), 18, width=2)
    hx = rx + 26 * math.cos(th)
    hy = ry + 26 * math.sin(th)
    pygame.draw.line(screen, (30, 30, 30), (int(rx), int(ry)), (int(hx), int(hy)), width=3)

def get_camera_frame_bgr(cfg: SimConfig, screen: pygame.Surface) -> np.ndarray:
    # Downsample the full screen into a small "camera" frame
    surf_small = pygame.transform.smoothscale(screen, (cfg.cam_w, cfg.cam_h))
    rgb = pygame.surfarray.array3d(surf_small)  # (w,h,3) RGB
    rgb = np.transpose(rgb, (1, 0, 2))          # -> (h,w,3)
    bgr = rgb[:, :, ::-1].copy()
    return bgr
