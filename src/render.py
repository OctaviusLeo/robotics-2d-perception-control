# render.yp
# Rendering functions for the simulation environment
from __future__ import annotations
import math
import numpy as np
import pygame
import cv2
from config import SimConfig
from sim import World

# Cache distractor layouts per world size
_DISTRACTOR_CACHE = {}

TARGET_BGR = (255, 255, 0)  # BGR for OpenCV: cyan-ish
TARGET_RGB = (TARGET_BGR[2], TARGET_BGR[1], TARGET_BGR[0])

def init_pygame(headless: bool) -> None:
    if headless:
        import os
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()

def _get_distractors(cfg: SimConfig):
    key = (cfg.width, cfg.height)
    if key in _DISTRACTOR_CACHE:
        return _DISTRACTOR_CACHE[key]

    rng = np.random.default_rng(7)
    count = 4
    margin = 50
    distractors = []
    for _ in range(count):
        x = float(rng.uniform(margin, cfg.width - margin))
        y = float(rng.uniform(margin, cfg.height - margin))
        r = rng.uniform(8, 14)
        distractors.append((x, y, r))

    _DISTRACTOR_CACHE[key] = distractors
    return distractors


def draw_world(
    screen: pygame.Surface,
    cfg: SimConfig,
    world: World,
    *,
    draw_distractors: bool = True,
    draw_obstacles: bool = True,
) -> None:
    screen.fill((245, 245, 245))

    # Targets: draw remaining; highlight current
    current = world.current_target()
    if hasattr(world, "targets"):
        for idx, (tx, ty) in enumerate(world.targets):
            if world.targets_done or idx < world.target_index:
                continue
            color = TARGET_RGB if idx == world.target_index else (180, 220, 220)
            radius = cfg.target_radius if idx == world.target_index else int(cfg.target_radius * 0.8)
            pygame.draw.circle(screen, color, (int(tx), int(ty)), radius)

    # Obstacles (gray)
    if draw_obstacles and hasattr(world, "obstacles"):
        for ox, oy, r in world.obstacles:
            pygame.draw.circle(screen, (120, 120, 120), (int(ox), int(oy)), int(r))

    # Distractors (same hue, smaller blobs) to challenge perception robustness
    if draw_distractors:
        for dx, dy, dr in _get_distractors(cfg):
            pygame.draw.circle(screen, TARGET_RGB, (int(dx), int(dy)), int(dr))

    # Robot
    rx, ry, th = world.robot.x, world.robot.y, world.robot.theta
    pygame.draw.circle(screen, (30, 30, 30), (int(rx), int(ry)), 18, width=2)
    hx = rx + 26 * math.cos(th)
    hy = ry + 26 * math.sin(th)
    pygame.draw.line(screen, (30, 30, 30), (int(rx), int(ry)), (int(hx), int(hy)), width=3)


def draw_debug_overlay(
    screen: pygame.Surface,
    cfg: SimConfig,
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    center: tuple[int, int] | None,
    *,
    mode: str,
    conf: float,
    scale: int = 2,
    margin: int = 10,
) -> None:
    """Draw a small debug overlay with camera frame + mask + centroid + mode/conf."""

    # Convert BGR->RGB for display
    rgb = frame_bgr[:, :, ::-1].copy()
    H, W = rgb.shape[:2]

    # Mask -> RGB
    if mask.ndim == 2:
        mask_rgb = np.repeat(mask[:, :, None], 3, axis=2)
    else:
        mask_rgb = mask

    # Upscale for visibility
    rgb_up = cv2.resize(rgb, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    mask_up = cv2.resize(mask_rgb, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)

    # pygame.surfarray.make_surface expects (w,h,3)
    surf_cam = pygame.surfarray.make_surface(np.transpose(rgb_up, (1, 0, 2)))
    surf_mask = pygame.surfarray.make_surface(np.transpose(mask_up, (1, 0, 2)))

    x0 = margin
    y0 = margin
    screen.blit(surf_cam, (x0, y0))
    screen.blit(surf_mask, (x0 + surf_cam.get_width() + margin, y0))

    # Centroid marker on the camera overlay
    if center is not None:
        cx, cy = center
        px = x0 + int(cx * scale)
        py = y0 + int(cy * scale)
        pygame.draw.circle(screen, (255, 0, 0), (px, py), 4)
        pygame.draw.circle(screen, (255, 255, 255), (px, py), 6, width=1)

    # Text label
    try:
        font = pygame.font.SysFont(None, 20)
        label = f"mode={mode} conf={conf:.2f}"
        text = font.render(label, True, (20, 20, 20))
        screen.blit(text, (x0, y0 + surf_cam.get_height() + 6))
    except Exception:
        # Font init can fail in some headless setups; overlay remains visual-only.
        pass

def get_camera_frame_bgr(cfg: SimConfig, screen: pygame.Surface, world: World) -> np.ndarray:
    """Robot-centric camera frame synthesized from geometry.

    This avoids the instability of cropping the full rendered screen (which can
    cause the target to drop out of view due to crop choices).
    """

    # Background: light gray (in BGR)
    frame = np.full((cfg.cam_h, cfg.cam_w, 3), 245, dtype=np.uint8)

    def world_to_robot(dx: float, dy: float, theta: float) -> tuple[float, float]:
        c = math.cos(theta)
        s = math.sin(theta)
        # Rotate by -theta into robot frame
        x_r = c * dx + s * dy
        y_r = -s * dx + c * dy
        return x_r, y_r

    def project(x_r: float, y_r: float) -> tuple[int, int, float] | None:
        if x_r <= 1.0:
            return None
        bearing = math.atan2(y_r, x_r)
        half_fov = cfg.cam_fov * 0.5
        if abs(bearing) > half_fov:
            return None

        # Horizontal pixel from bearing
        u = (bearing / half_fov) * (cfg.cam_w / 2.0) + (cfg.cam_w / 2.0)

        # Vertical pixel from range (closer -> lower). Use forward distance as range proxy.
        max_range = float(cfg.width)  # px in world space
        v = cfg.cam_h - (x_r / max_range) * cfg.cam_h

        u_i = int(np.clip(round(u), 0, cfg.cam_w - 1))
        v_i = int(np.clip(round(v), 0, cfg.cam_h - 1))
        return u_i, v_i, x_r

    rx, ry, th = float(world.robot.x), float(world.robot.y), float(world.robot.theta)

    # Render target
    tgt = world.current_target()
    if tgt is not None:
        dx = float(tgt[0]) - rx
        dy = float(tgt[1]) - ry
        x_r, y_r = world_to_robot(dx, dy, th)
        p = project(x_r, y_r)
        if p is not None:
            u, v, rng = p
            radius = int(np.clip(12.0 * (140.0 / (rng + 1.0)), 2, 12))
            cv2.circle(frame, (u, v), radius, TARGET_BGR, thickness=-1)

    # Render distractors (same hue)
    for ox, oy, _ in _get_distractors(cfg):
        dx = float(ox) - rx
        dy = float(oy) - ry
        x_r, y_r = world_to_robot(dx, dy, th)
        p = project(x_r, y_r)
        if p is None:
            continue
        u, v, rng = p
        radius = int(np.clip(10.0 * (140.0 / (rng + 1.0)), 2, 10))
        cv2.circle(frame, (u, v), radius, TARGET_BGR, thickness=-1)

    # Render obstacles (gray)
    if hasattr(world, "obstacles"):
        for ox, oy, r in world.obstacles:
            dx = float(ox) - rx
            dy = float(oy) - ry
            x_r, y_r = world_to_robot(dx, dy, th)
            p = project(x_r, y_r)
            if p is None:
                continue
            u, v, rng = p
            radius = int(np.clip(float(r) * (120.0 / (rng + 1.0)), 2, 14))
            cv2.circle(frame, (u, v), radius, (140, 140, 140), thickness=-1)

    return frame


def get_camera_frame_bgr_global(cfg: SimConfig, screen: pygame.Surface) -> np.ndarray:
    """Global camera (baseline): downsample the full screen into a small frame."""
    surf_small = pygame.transform.smoothscale(screen, (cfg.cam_w, cfg.cam_h))
    rgb = pygame.surfarray.array3d(surf_small)  # (w,h,3) RGB
    rgb = np.transpose(rgb, (1, 0, 2))          # -> (h,w,3)
    bgr = rgb[:, :, ::-1].copy()
    return bgr
