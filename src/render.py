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

    # Target
    pygame.draw.circle(screen, TARGET_RGB, (int(world.target.x), int(world.target.y)), cfg.target_radius)

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
    """Robot-centric camera: rotate world to align heading, crop ahead, resize to cam size."""

    # Full RGB frame (pygame gives (w,h,3); transpose to (h,w,3) for cv2)
    rgb_full = pygame.surfarray.array3d(screen)
    rgb_full = np.transpose(rgb_full, (1, 0, 2))

    H, W, _ = rgb_full.shape
    cx, cy = float(world.robot.x), float(world.robot.y)

    # Rotate so robot heading points to +x in the rotated frame
    heading_deg = -math.degrees(world.robot.theta)
    rot_mat = cv2.getRotationMatrix2D((cx, cy), heading_deg, 1.0)
    rotated = cv2.warpAffine(
        rgb_full,
        rot_mat,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Define a square view region centered slightly ahead of the robot
    view_span = max(cfg.cam_w, cfg.cam_h) * 1.8
    half = view_span * 0.5
    look_ahead = view_span * 0.3

    center_x = cx + look_ahead
    center_y = cy

    x0 = int(round(center_x - half))
    x1 = int(round(center_x + half))
    y0 = int(round(center_y - half))
    y1 = int(round(center_y + half))

    x0c, x1c = max(0, x0), min(W, x1)
    y0c, y1c = max(0, y0), min(H, y1)

    crop = rotated[y0c:y1c, x0c:x1c]
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        crop = np.zeros((cfg.cam_h, cfg.cam_w, 3), dtype=np.uint8)
    else:
        crop = cv2.resize(crop, (cfg.cam_w, cfg.cam_h), interpolation=cv2.INTER_LINEAR)

    bgr = crop[:, :, ::-1].copy()
    return bgr
