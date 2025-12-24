# run_demo.py
"""Run a single robotics perception+control episode, with optional logging and GIF capture."""
from __future__ import annotations

import argparse
import csv
import os
import random
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import pygame

from config import SimConfig
from control import ControlMode, HeadingController, StateMachineController, clamp
from estimation import ExponentialSmoother
from perception import detect_target_center_bgr
from render import draw_debug_overlay, draw_world, get_camera_frame_bgr, init_pygame
from sim import World


GOAL_THRESHOLD = 40.0  # px


def run_episode(
    cfg: SimConfig,
    steps: int,
    headless: bool,
    log_path: Optional[str] = None,
    save_gif: bool = False,
    gif_path: Optional[str] = None,
    seed: Optional[int] = None,
    perception_latency: int = 0,
    meas_noise_px: float = 0.0,
    smooth_alpha: float = 0.35,
    draw_distractors: bool = True,
    draw_obstacles: bool = True,
    debug_overlay: bool = False,
) -> Dict[str, float]:
    """Execute one episode and return aggregate metrics."""

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs("outputs", exist_ok=True)
    init_pygame(headless)
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    clock = pygame.time.Clock()

    world = World(cfg, with_obstacles=draw_obstacles)
    ctrl = HeadingController(kp=3.0, kd=0.35)
    sm_ctrl = StateMachineController(
        heading_ctrl=ctrl,
        v_max=cfg.v_max,
        w_max=cfg.w_max,
        approach_radius=140.0,
        lost_frames_for_search=8,
    )
    smoother = ExponentialSmoother(alpha=smooth_alpha)

    meas_queue: Deque[Tuple[Optional[Tuple[int, int]], float]] = deque()

    frames: List[np.ndarray] = []
    log_rows: List[Dict[str, float]] = []
    detections = 0
    v_cmd_hist: List[float] = []
    w_cmd_hist: List[float] = []

    done = False
    initial_distance = world.distance_to_target()
    start_wall = time.perf_counter()

    for t in range(steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        if done:
            break

        draw_world(screen, cfg, world, draw_distractors=draw_distractors, draw_obstacles=draw_obstacles)
        frame_bgr = get_camera_frame_bgr(cfg, screen, world)

        # Perception with latency and noise
        center_raw, mask, conf_raw = detect_target_center_bgr(frame_bgr)
        meas_queue.append((center_raw, conf_raw))
        delayed_center: Optional[Tuple[int, int]] = None
        delayed_conf: float = 0.0
        if len(meas_queue) > perception_latency:
            delayed_center, delayed_conf = meas_queue.popleft()

        center = None
        conf = 0.0
        if delayed_center is not None:
            cx_noisy = delayed_center[0] + np.random.normal(0.0, meas_noise_px)
            cy_noisy = delayed_center[1] + np.random.normal(0.0, meas_noise_px)
            cx_noisy = float(np.clip(cx_noisy, 0, cfg.cam_w - 1))
            cy_noisy = float(np.clip(cy_noisy, 0, cfg.cam_h - 1))
            center = (int(round(cx_noisy)), int(round(cy_noisy)))
            conf = delayed_conf

        if center is not None:
            cx, cy = center
            err = (cx - (cfg.cam_w / 2.0)) / (cfg.cam_w / 2.0)  # normalized [-1,1]
            filt_err = smoother.update(err)
            detections += 1
        else:
            smoother.reset()
            cx, cy = None, None
            err = None
            filt_err = None

        v_cmd, w_cmd, mode = sm_ctrl.step(
            detected=center is not None,
            err=filt_err if center is not None else None,
            dist=world.distance_to_target(),
            dt=cfg.dt,
        )

        if debug_overlay:
            draw_debug_overlay(
                screen,
                cfg,
                frame_bgr=frame_bgr,
                mask=mask,
                center=center,
                mode=mode.value,
                conf=conf,
            )

        v_cmd_hist.append(v_cmd)
        w_cmd_hist.append(w_cmd)
        world.step(v=v_cmd, w=w_cmd)

        pygame.display.flip()
        clock.tick(int(1.0 / cfg.dt))

        if headless and save_gif and t % 2 == 0:
            rgb = np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))
            frames.append(rgb)

        if log_path is not None:
            log_rows.append(
                {
                    "step": t,
                    "sim_time_s": t * cfg.dt,
                    "robot_x": world.robot.x,
                    "robot_y": world.robot.y,
                    "robot_theta": world.robot.theta,
                    "target_x": world.target.x,
                    "target_y": world.target.y,
                    "distance_to_target": world.distance_to_target(),
                    "v_cmd": v_cmd,
                    "w_cmd": w_cmd,
                    "detected": center is not None,
                    "detected_cx": cx if center is not None else "",
                    "detected_cy": cy if center is not None else "",
                    "detect_conf": conf if center is not None else 0.0,
                    "err_norm": err if err is not None else "",
                    "err_norm_filtered": filt_err if filt_err is not None else "",
                    "mode": mode.value if center is not None else sm_ctrl.mode.value,
                }
            )

        if world.distance_to_target() < GOAL_THRESHOLD:
            break

    total_steps = len(v_cmd_hist)
    sim_time = total_steps * cfg.dt
    elapsed_wall = time.perf_counter() - start_wall
    success = world.distance_to_target() < GOAL_THRESHOLD

    if log_path is not None and log_rows:
        fieldnames = list(log_rows[0].keys())
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_rows)

    if headless and save_gif and frames:
        out_path = gif_path or os.path.join("outputs", "demo.gif")
        imageio.mimsave(out_path, frames, fps=30)
        print(f"Saved GIF: {out_path}")

    pygame.quit()

    metrics: Dict[str, float] = {
        "success": float(success),
        "steps": total_steps,
        "sim_time_s": sim_time,
        "distance_final": world.distance_to_target(),
        "distance_initial": initial_distance,
        "detection_rate": detections / total_steps if total_steps else 0.0,
        "avg_v_cmd": float(np.mean(v_cmd_hist)) if v_cmd_hist else 0.0,
        "avg_w_cmd": float(np.mean(w_cmd_hist)) if w_cmd_hist else 0.0,
        "runtime_s": elapsed_wall,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Use dummy video driver and disable window.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--log-csv", type=str, default=None, help="Write per-step log to CSV at this path.")
    parser.add_argument("--save-gif", action="store_true", help="Capture GIF when running headless.")
    parser.add_argument("--gif-path", type=str, default=None, help="Optional GIF output path.")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic seed for numpy/random.")
    parser.add_argument("--perception-latency", type=int, default=0, help="Latency in frames to delay perception output.")
    parser.add_argument("--meas-noise-px", type=float, default=0.0, help="Stddev of pixel noise added to detections.")
    parser.add_argument("--smooth-alpha", type=float, default=0.35, help="EMA alpha for heading error smoothing.")
    parser.add_argument("--no-distractors", action="store_true", help="Disable cyan distractors for cleaner demos.")
    parser.add_argument("--no-obstacles", action="store_true", help="Disable obstacles for cleaner demos.")
    parser.add_argument(
        "--debug-overlay",
        action="store_true",
        help="Draw camera+mask+centroid+mode overlay in the main window.",
    )
    args = parser.parse_args()

    cfg = SimConfig()
    run_episode(
        cfg=cfg,
        steps=args.steps,
        headless=args.headless,
        log_path=args.log_csv,
        save_gif=args.save_gif,
        gif_path=args.gif_path,
        seed=args.seed,
        perception_latency=args.perception_latency,
        meas_noise_px=args.meas_noise_px,
        smooth_alpha=args.smooth_alpha,
        draw_distractors=not args.no_distractors,
        draw_obstacles=not args.no_obstacles,
        debug_overlay=args.debug_overlay,
    )


if __name__ == "__main__":
    main()
