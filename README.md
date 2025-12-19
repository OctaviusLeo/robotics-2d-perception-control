# Robotics 2D — Perception + Control Loop (Demo)

A minimal robotics-style loop:
**sense (camera frame) -> perceive (detect target) -> control (drive robot) -> act (update state)**

- 2D differential-drive robot in a simple world.
- Synthetic camera frame rendered as an image (numpy array).
- Perception uses OpenCV color thresholding to find the target.
- Controller steers toward target with a PID-like heading controller.

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Run interactive window
python src/run_demo.py

# Headless mode (saves a GIF to outputs/)
python src/run_demo.py --headless --steps 600
```

## What to improve next (resume upgrades)
- Replace color thresholding with a small detector (even a tiny CNN) + latency measurements
- Add noise models + robustness tests
- Add a second task (avoid obstacles, follow a path)
