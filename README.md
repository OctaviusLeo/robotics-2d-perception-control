# Robotics 2D — Perception + Control Loop (Demo)
This is purely virtual. There is no hardware involved however if flashed onto an MCU with everything set-up, hopefully it'll work.

A minimal robotics-style loop:
**sense (camera frame) -> perceive (detect target) -> control (drive robot) -> act (update state)**

- 2D differential-drive robot in a simple world.
- Synthetic camera frame rendered as an image (numpy array).
- Perception uses OpenCV color thresholding to find the target.
- Controller steers toward target with a PID-like heading controller.

Demo: 

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Run interactive window
python src/run_demo.py

# Headless run with per-step logging and optional GIF
python src/run_demo.py --headless --steps 600 --log-csv outputs/run1.csv --save-gif --gif-path outputs/run1.gif

# Batch evaluation (10 episodes, saves outputs/metrics.csv)
python src/eval.py --episodes 10 --steps 600 --log-dir outputs/logs --metrics-csv outputs/metrics.csv
```

## Future Features
- Replace color thresholding with a small detector (even a tiny CNN) + latency measurements
- Add noise models + robustness tests
- Add a second task (avoid obstacles, follow a path)
