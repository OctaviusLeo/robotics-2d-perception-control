# perception.py
# Perception module for detecting the target center in a BGR image.
from __future__ import annotations
import numpy as np
import cv2

# Target is rendered as a bright cyan-ish circle in BGR.
# This perception module detects it via HSV thresholding.

def detect_target_center_bgr(frame_bgr: np.ndarray):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Tuned for cyan target
    lower = np.array([70, 80, 80], dtype=np.uint8)
    upper = np.array([100, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask

    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 20:
        return None, mask

    M = cv2.moments(c)
    if M.get("m00", 0) == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask
