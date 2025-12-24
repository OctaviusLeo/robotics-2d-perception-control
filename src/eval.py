# eval.py
"""Batch evaluation harness that writes aggregated metrics to outputs/metrics.csv."""
from __future__ import annotations

import argparse
import csv
import os
import statistics
from typing import Dict, List

from config import SimConfig
from run_demo import run_episode


DEFAULT_METRICS_PATH = os.path.join("outputs", "metrics.csv")


def write_metrics(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return

    field_order = [
        "episode",
        "seed",
        "success",
        "steps",
        "sim_time_s",
        "distance_initial",
        "distance_final",
        "detection_rate",
        "avg_v_cmd",
        "avg_w_cmd",
        "runtime_s",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run.")
    parser.add_argument("--steps", type=int, default=600, help="Steps per episode.")
    parser.add_argument("--seed", type=int, default=123, help="Base seed; increments per episode.")
    parser.add_argument("--log-dir", type=str, default=None, help="Optional per-episode log directory.")
    parser.add_argument("--metrics-csv", type=str, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--gui", action="store_true", help="Render to a window (slower, for debugging).")
    args = parser.parse_args()

    cfg = SimConfig()
    os.makedirs("outputs", exist_ok=True)
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    rows: List[Dict[str, float]] = []
    for ep in range(args.episodes):
        ep_seed = args.seed + ep if args.seed is not None else None
        log_path = None
        if args.log_dir:
            log_path = os.path.join(args.log_dir, f"episode_{ep:03d}.csv")

        metrics = run_episode(
            cfg=cfg,
            steps=args.steps,
            headless=not args.gui,
            log_path=log_path,
            save_gif=False,
            gif_path=None,
            seed=ep_seed,
            draw_distractors=True,
            draw_obstacles=True,
            debug_overlay=False,
        )
        metrics.update({"episode": ep, "seed": ep_seed})
        rows.append(metrics)

        print(
            f"Episode {ep}: success={bool(metrics['success'])} "
            f"time={metrics['sim_time_s']:.2f}s dist={metrics['distance_final']:.2f}"
        )

    write_metrics(args.metrics_csv, rows)

    if rows:
        success_rate = sum(r["success"] for r in rows) / len(rows)
        mean_time = statistics.mean(r["sim_time_s"] for r in rows)
        print(
            f"Saved metrics to {args.metrics_csv} | "
            f"success_rate={success_rate:.2f} mean_time={mean_time:.2f}s"
        )


if __name__ == "__main__":
    main()
