"""
Trinetra — Drawing utilities
Shared drawing helpers used by all modules.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


# Class colours (BGR)
CLASS_COLORS = {
    'car':           (255, 200,   0),
    'bus':           (  0, 200, 255),
    'bike':          (  0, 255, 100),
    'truck':         (255,  80,   0),
    'auto_rickshaw': (180,   0, 255),
    'pedestrian':    (220, 220, 220),
}

VIOLATION_COLORS = {
    'no_helmet':      (0,   0, 255),
    'no_seatbelt':    (0,  80, 255),
    'phone_use':      (0, 165, 255),
    'wrong_side':     (0,   0, 200),
    'tampered_plate': (50,  0, 200),
}


def draw_box(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int] = (255, 200, 0),
    thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_hud(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 30),
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    font_scale: float = 0.65,
) -> np.ndarray:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    x, y = position
    cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + 4), bg_color, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
    return frame


def draw_violation_badge(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    violation: str,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    color = VIOLATION_COLORS.get(violation, (0, 0, 255))
    # Red border for violations
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    label = f"⚠ {violation.replace('_', ' ').upper()}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y2), (x1 + tw + 6, y2 + th + 8), color, -1)
    cv2.putText(frame, label, (x1 + 3, y2 + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_density_heatmap(
    frame: np.ndarray,
    density_map: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a density heatmap on the frame."""
    h, w = frame.shape[:2]
    heatmap = cv2.resize(density_map, (w, h))
    heatmap_norm = np.zeros_like(heatmap, dtype=np.uint8)
    cv2.normalize(heatmap, heatmap_norm, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)


def put_stats_panel(
    frame: np.ndarray,
    stats: dict,
    x: int = 10,
    y: int = 60,
) -> np.ndarray:
    """Draw a semi-transparent stats panel on the frame."""
    line_h = 22
    panel_h = len(stats) * line_h + 16
    panel_w = 220
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, (k, v) in enumerate(stats.items()):
        text = f"{k}: {v}"
        cv2.putText(frame, text, (x + 8, y + 14 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 255, 200), 1, cv2.LINE_AA)
    return frame
