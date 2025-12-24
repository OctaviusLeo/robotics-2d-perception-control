# Robotics 2D — Perception + Control Loop (Demo)
Purely virtual 2D differential-drive robot with a minimal perception -> control -> act loop.

Pipeline: **sense (robot-centric camera) → perceive (color detection) → control (heading FSM) → act (update state)**

- Robot-centric camera is now geometry-based (projects target/distractors/obstacles with FOV) for stable detection.
- Perception: HSV thresholding with morphology, contour filtering, and a confidence score.
- Control: SEARCH/TRACK/APPROACH FSM with speed scheduling; latency + noise + smoothing options.
- Environment: three targets visited in order (each disappears when reached), optional cyan distractors, optional static obstacles (rendered).
- Outputs: per-step CSV logs, GIF capture, batch eval to `outputs/metrics.csv`.
- Larger 1200x750 window so FOV overlays leave room to see the robot path.

## Recreate Locally
```bash
git clone https://github.com/yourname/robotics-2d-perception-control.git
cd robotics-2d-perception-control

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run Modes
When distracted:
```bash
python src/run_demo.py 
```

The important demo showed:
```bash
python src/run_demo.py --steps 2000 --seed 0 --no-distractors --no-obstacles --camera-mode robot --debug-overlay
```

One-shot clean GIF (shortcut):
```bash
python src/run_demo.py --clean-gif
```

Headless + GIF (600 sim steps ≈ 10 seconds at 60 Hz):
```bash
python src/run_demo.py --headless --steps 600 --log-csv outputs/run1.csv --save-gif --gif-path outputs/run1.gif
```

Latency/noise stress + smoothing:
```bash
python src/run_demo.py --headless --steps 600 --perception-latency 3 --meas-noise-px 2.0 --smooth-alpha 0.4 --log-csv outputs/run_latency.csv
```

Batch evaluation (defaults to robot camera):
```bash
python src/eval.py --episodes 10 --steps 600 --camera-mode robot --no-distractors --no-obstacles --metrics-csv outputs/metrics.csv
```

## Key CLI Flags
- `--headless` use dummy video driver (no window) and enable GIF capture.
- `--steps` simulation steps to run (60 steps ≈ 1 s).
- `--log-csv` write per-step telemetry (pose, cmds, detection, errors, mode).
- `--save-gif` / `--gif-path` capture a GIF of the run when headless.
- `--perception-latency` delay detections by N frames.
- `--meas-noise-px` Gaussian pixel noise added to detections.
- `--smooth-alpha` exponential smoother for heading error.
- `--no-distractors` disable cyan distractors.
- `--no-obstacles` disable obstacles.
- `--debug-overlay` draw camera+mask+centroid+mode overlay.
- `--camera-mode` choose `robot` (geometry-based) or `global` (full-scene downsample).

## Future Features
- Learned detector in place of color thresholding (with measured latency).
- Obstacle-aware control (reactive avoidance or simple planner) tied to perception.
- Trajectory task: follow waypoints while keeping the target in view.
