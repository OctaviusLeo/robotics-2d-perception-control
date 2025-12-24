# perception.py
# Perception module for detecting the target center in a BGR image.
from __future__ import annotations
import numpy as np
import cv2

# Target is rendered as a bright cyan-ish circle in BGR.
# This perception module detects it via HSV thresholding, morphology, and contour filtering.


def detect_target_center_bgr(frame_bgr: np.ndarray):
    """Return (center, mask, confidence) for the detected target; center None if not found."""

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Tuned for cyan target
    lower = np.array([70, 80, 80], dtype=np.uint8)
    upper = np.array([100, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Clean up noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask, 0.0

    # Keep the best contour by a confidence score
    best = None
    best_score = 0.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 40:  # filter small blobs
            continue

        peri = cv2.arcLength(c, True)
        circularity = 0.0 if peri == 0 else 4.0 * np.pi * area / (peri * peri)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = 0.0 if hull_area == 0 else area / hull_area

        # Score blends size and shape quality
        size_score = min(area / 4000.0, 1.0)
        score = 0.4 * size_score + 0.3 * circularity + 0.3 * solidity
        if score > best_score:
            best_score = score
            best = c

    if best is None:
        return None, mask, 0.0

    M = cv2.moments(best)
    if M.get("m00", 0) == 0:
        return None, mask, 0.0

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask, float(best_score)
