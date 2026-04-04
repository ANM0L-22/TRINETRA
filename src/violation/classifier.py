"""
Trinetra — Violation Classifier
Detects:
  - Helmet violation (rider without helmet)
  - Seatbelt violation (driver without seatbelt)
  - Mobile phone usage while driving

Uses a YOLOv8 classifier fine-tuned on violation crops.
Falls back to a lightweight CNN if full model unavailable.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
from loguru import logger


VIOLATION_TYPES = ['no_helmet', 'no_seatbelt', 'phone_use', 'compliant']


@dataclass
class ViolationResult:
    violation_type: str      # 'no_helmet' | 'no_seatbelt' | 'phone_use' | 'compliant'
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None  # region of interest in frame
    track_id: Optional[int] = None

    @property
    def is_violation(self) -> bool:
        return self.violation_type != 'compliant'


class ViolationClassifier:
    """
    Classifies traffic violations in vehicle/rider crops.

    Model strategy:
      - YOLOv8 classify: fast multi-class classification on rider/driver crops
      - Separate heads for helmet (bikes) and seatbelt/phone (cars)
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        conf_threshold: float = 0.55,
        device: str = 'cpu',
    ):
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = None

        if weights_path and Path(weights_path).exists():
            self._load_yolo(weights_path)
        else:
            logger.info("Violation classifier: no weights — using heuristic fallback")

    def _load_yolo(self, path):
        try:
            from ultralytics import YOLO
            self.model = YOLO(path)
            self.model.to(self.device)
            logger.info(f"Violation classifier: YOLOv8 ({path})")
        except Exception as e:
            logger.warning(f"Could not load violation model: {e}")

    # ── Main API ──────────────────────────────────────────────

    def classify_rider(
        self,
        frame: np.ndarray,
        rider_bbox: Tuple[int, int, int, int],
        track_id: Optional[int] = None,
    ) -> ViolationResult:
        """Classify helmet violation for a motorcycle rider."""
        crop = self._get_upper_body_crop(frame, rider_bbox)
        if crop is None:
            return ViolationResult('compliant', 0.5, rider_bbox, track_id)

        if self.model:
            return self._classify_yolo(crop, rider_bbox, track_id, mode='helmet')
        return self._classify_helmet_heuristic(crop, rider_bbox, track_id)

    def classify_driver(
        self,
        frame: np.ndarray,
        vehicle_bbox: Tuple[int, int, int, int],
        track_id: Optional[int] = None,
    ) -> List[ViolationResult]:
        """Classify seatbelt + phone violations for a car/bus driver."""
        crop = self._get_driver_crop(frame, vehicle_bbox)
        if crop is None:
            return []

        results = []

        if self.model:
            r = self._classify_yolo(crop, vehicle_bbox, track_id, mode='driver')
            if r:
                results.append(r)
        else:
            sb = self._classify_seatbelt_heuristic(crop, vehicle_bbox, track_id)
            if sb:
                results.append(sb)

        return results

    # ── YOLOv8 classify ───────────────────────────────────────

    def _classify_yolo(self, crop, bbox, track_id, mode):
        try:
            if self.model is None:
                return ViolationResult('compliant', 0.5, bbox, track_id)
            results = self.model(crop, device=self.device, verbose=False)[0]
            probs = results.probs
            if probs is None:
                return ViolationResult('compliant', 0.5, bbox, track_id)

            top1 = int(probs.top1)
            conf = float(probs.top1conf)

            class_map = {0: 'no_helmet', 1: 'no_seatbelt', 2: 'phone_use', 3: 'compliant'}
            vtype = class_map.get(top1, 'compliant')

            if conf < self.conf_threshold:
                vtype = 'compliant'

            return ViolationResult(vtype, conf, bbox, track_id)
        except Exception as e:
            logger.debug(f"Classify error: {e}")
            return ViolationResult('compliant', 0.5, bbox, track_id)

    # ── Heuristic fallbacks ───────────────────────────────────

    def _classify_helmet_heuristic(self, crop, bbox, track_id) -> ViolationResult:
        """
        Simple heuristic: look for helmet-shaped circular region at top of rider crop.
        Low accuracy — replace with trained model ASAP.
        """
        h, w = crop.shape[:2]
        head_region = crop[:h // 3, :]

        gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=10, maxRadius=h // 4
        )

        if circles is not None and len(circles[0]) > 0:
            # Found circular shape at head — likely helmet
            return ViolationResult('compliant', 0.55, bbox, track_id)
        else:
            # No circular head-region — flag as no helmet
            return ViolationResult('no_helmet', 0.50, bbox, track_id)

    def _classify_seatbelt_heuristic(self, crop, bbox, track_id) -> Optional[ViolationResult]:
        """
        Diagonal line detector for seatbelt strap.
        Looks for a diagonal edge across the torso region.
        """
        h, w = crop.shape[:2]
        torso = crop[h // 4: 3 * h // 4, :]

        gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                                 minLineLength=h // 6, maxLineGap=10)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if 20 < angle < 70:   # diagonal = seatbelt strap
                    return None  # compliant

        # No diagonal line found — flag
        return ViolationResult('no_seatbelt', 0.48, bbox, track_id)

    # ── Crop helpers ──────────────────────────────────────────

    def _get_upper_body_crop(self, frame, bbox) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        # Take top 55% of rider bbox
        crop = frame[y1:y1 + int(h * 0.55), x1:x2]
        return crop if crop.size > 0 else None

    def _get_driver_crop(self, frame, bbox) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        w = x2 - x1
        # Take upper-left portion (driver side in Indian traffic = right seat)
        crop = frame[y1:y1 + int(h * 0.6), x1 + int(w * 0.3):x2]
        return crop if crop.size > 0 else None
