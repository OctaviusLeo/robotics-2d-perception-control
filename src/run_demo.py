from __future__ import annotations
import argparse
import os
import numpy as np
import pygame
import imageio.v2 as imageio

from config import SimConfig
from sim import World
from render import init_pygame, draw_world, get_camera_frame_bgr, TARGET_BGR
from perception import detect_target_center_bgr
from control import HeadingController, clamp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    cfg = SimConfig()
    os.makedirs("outputs", exist_ok=True)

    init_pygame(args.headless)
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    clock = pygame.time.Clock()

    world = World(cfg)
    ctrl = HeadingController(kp=3.0, kd=0.35)

    frames = []
    done = False

    for t in range(args.steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        if done:
            break

        draw_world(screen, cfg, world)
        frame_bgr = get_camera_frame_bgr(cfg, screen)

        center, mask = detect_target_center_bgr(frame_bgr)

        # Control: if we see target, steer toward it
        if center is not None:
            cx, cy = center
            err = (cx - (cfg.cam_w / 2.0)) / (cfg.cam_w / 2.0)  # normalized [-1,1]
            w_cmd = clamp(-ctrl.compute(err, cfg.dt), -cfg.w_max, cfg.w_max)
            v_cmd = cfg.v_max * 0.75
        else:
            # Search turn
            v_cmd = cfg.v_max * 0.2
            w_cmd = cfg.w_max * 0.6

        world.step(v=v_cmd, w=w_cmd)

        pygame.display.flip()
        clock.tick(int(1.0 / cfg.dt))

        if args.headless and t % 2 == 0:
            # capture a reduced frame for gif
            rgb = np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))
            frames.append(rgb)

        # Stop if close enough
        if world.distance_to_target() < 40:
            break

    if args.headless and len(frames) > 0:
        out_path = os.path.join("outputs", "demo.gif")
        imageio.mimsave(out_path, frames, fps=30)
        print(f"Saved: {out_path}")

    pygame.quit()


if __name__ == "__main__":
    main()
