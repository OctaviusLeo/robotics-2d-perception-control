# Robotics 2D — Perception + Control Loop (Demo)
Purely virtual 2D differential-drive robot with a minimal perception -> control -> act loop.

Pipeline: **sense (robot-centric camera) → perceive (color detection) → control (heading FSM) → act (update state)**

- Robot-centric camera crop/warp (pose + heading) into a small frame.
- Perception: HSV thresholding with morphology, contour filtering, and a confidence score.
- Control: SEARCH/TRACK/APPROACH FSM with speed scheduling on heading error and distance; latency + noise + smoothing options.
- Environment: static target, cyan distractors, optional static obstacles (rendered).
- Outputs: per-step CSV logs, GIF capture, batch eval to `outputs/metrics.csv`.

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
- Interactive window: 
	```bash
	python src/run_demo.py
	```

- Headless with logging/GIF (600 sim steps ≈ 10 seconds at 60 Hz):
	```bash
	python src/run_demo.py --headless --steps 600 --log-csv outputs/run1.csv --save-gif --gif-path outputs/run1.gif
	```

- Stress test latency/noise + smoothing:
	```bash
	python src/run_demo.py --headless --steps 600 \
		--perception-latency 3 --meas-noise-px 2.0 --smooth-alpha 0.4 \
		--log-csv outputs/run_latency.csv
	```

- Batch evaluation (10 episodes, each 600 steps):
	```bash
	python src/eval.py --episodes 10 --steps 600 --log-dir outputs/logs --metrics-csv outputs/metrics.csv
	```

## Key CLI Flags
- `--headless` use dummy video driver (no window) and enable GIF capture.
- `--steps` simulation steps to run (60 steps ≈ 1 s).
- `--log-csv` write per-step telemetry (pose, cmds, detection, errors, mode).
- `--save-gif` / `--gif-path` capture a GIF of the run when headless.
- `--perception-latency` delay detections by N frames.
- `--meas-noise-px` Gaussian pixel noise added to detections.
- `--smooth-alpha` exponential smoother for heading error.

## Future Features
- Replace color thresholding with a tiny detector and measure latency.
- Add noise models + robustness tests.
- Add a second task (avoid obstacles with reactive policy, or follow a path / waypoints).
