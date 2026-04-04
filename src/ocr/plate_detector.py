"""
Trinetra — Number Plate Detector
Detects and localises license plate regions in vehicle crops.
Uses a dedicated YOLOv8n model fine-tuned for Indian number plates.
Falls back to contour-based classical CV if model not available.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
from loguru import logger


@dataclass
class PlateRegion:
    """A detected license plate region."""
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2 in original frame coords
    confidence: float
    crop: Optional[np.ndarray] = None  # cropped plate image
    is_tampered: bool = False


class PlateDetector:
    """
    Detects license plate bounding boxes.

    Priority:
      1. YOLOv8 plate detection model (if weights exist)
      2. Classical contour fallback
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        tamper_cls_weights_path: Optional[str] = None,
        conf_threshold: float = 0.40,
        device: str = 'cpu',
    ):
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = None
        self.tamper_cls_model = None

        if weights_path and Path(weights_path).exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(weights_path)
                self.model.to(device)
                logger.info(f"Plate detector: YOLOv8 ({weights_path})")
            except Exception as e:
                logger.warning(f"Could not load plate model: {e}. Using classical CV.")
        else:
            logger.info("Plate detector: classical CV (no plate model weights found)")

        if tamper_cls_weights_path and Path(tamper_cls_weights_path).exists():
            try:
                from ultralytics import YOLO
                self.tamper_cls_model = YOLO(tamper_cls_weights_path)
                self.tamper_cls_model.to(device)
                logger.info(f"Tamper classifier: YOLOv8-cls ({tamper_cls_weights_path})")
            except Exception as e:
                logger.warning(f"Could not load tamper classifier: {e}. Using heuristic tampering checks.")

    def detect(self, frame: np.ndarray) -> List[PlateRegion]:
        """Detect all license plates in a frame."""
        if self.model:
            return self._detect_yolo(frame)
        return self._detect_classical(frame)

    def detect_in_vehicle_crop(
        self,
        frame: np.ndarray,
        vehicle_bbox: Tuple[int, int, int, int],
    ) -> List[PlateRegion]:
        """Detect plates within a specific vehicle bounding box."""
        x1, y1, x2, y2 = vehicle_bbox
        # Add small margin
        mx, my = max(0, x1 - 5), max(0, y1 - 5)
        crop = frame[my:min(frame.shape[0], y2 + 5), mx:min(frame.shape[1], x2 + 5)]
        if crop.size == 0:
            return []

        plates = self.detect(crop)
        # Translate coords back to full frame space
        for p in plates:
            px1, py1, px2, py2 = p.bbox
            p.bbox = (px1 + mx, py1 + my, px2 + mx, py2 + my)
            p.crop = frame[p.bbox[1]:p.bbox[3], p.bbox[0]:p.bbox[2]]

        return plates

    # ── YOLOv8 detection ─────────────────────────────────────

    def _detect_yolo(self, frame: np.ndarray) -> List[PlateRegion]:
        if self.model is None:
            return []
        results = self.model(frame, device=self.device, verbose=False)[0]
        plates = []
        if results.boxes is None:
            return plates
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            crop = frame[y1:y2, x1:x2]
            tampered = self._check_tampering(crop) if crop.size > 0 else False
            plates.append(PlateRegion(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                crop=crop,
                is_tampered=tampered,
            ))
        return plates

    # ── Classical CV detection ────────────────────────────────

    def _detect_classical(self, frame: np.ndarray) -> List[PlateRegion]:
        """
        Contour-based plate localisation.
        Works best on clear, well-lit Indian yellow/white plates.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blur, 30, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

        plates = []
        h, w = frame.shape[:2]

        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.018 * perimeter, True)

            if len(approx) == 4:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = bw / max(bh, 1)
                area_ratio = (bw * bh) / (w * h)

                # Indian plate: roughly 4:1 to 5:1 aspect ratio
                if 2.5 <= aspect <= 6.0 and 0.005 <= area_ratio <= 0.15:
                    x1, y1, x2, y2 = x, y, x + bw, y + bh
                    crop = frame[y1:y2, x1:x2]
                    tampered = self._check_tampering(crop) if crop.size > 0 else False
                    plates.append(PlateRegion(
                        bbox=(x1, y1, x2, y2),
                        confidence=0.6,
                        crop=crop,
                        is_tampered=tampered,
                    ))
                    if len(plates) >= 3:
                        break

        return plates

    # ── Tampering detection ───────────────────────────────────

    def _check_tampering(self, plate_crop: np.ndarray) -> bool:
        """
        Heuristic check for blackened/tampered plates.
        Flags plate as tampered if:
          - Mean brightness is very low (blackened)
          - Variance is very low (uniform paint)
          - Unnatural saturation
        """
        if plate_crop.size == 0:
            return False

        # If a classifier is available, use it as primary source.
        if self.tamper_cls_model is not None:
            try:
                res = self.tamper_cls_model.predict(plate_crop, verbose=False, device=self.device)
                if res and len(res) > 0 and getattr(res[0], "probs", None) is not None:
                    probs = res[0].probs
                    top_idx = int(probs.top1)
                    top_name = (res[0].names or {}).get(top_idx, str(top_idx)).lower()
                    top_conf = float(probs.top1conf)
                    if "tamper" in top_name and top_conf >= 0.5:
                        return True
                    if "clean" in top_name and top_conf >= 0.5:
                        return False
            except Exception as e:
                logger.debug(f"Tamper classifier inference failed: {e}")

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        variance = float(np.var(gray))
        # Blackened: dark AND uniform
        if mean_brightness < 40 and variance < 300:
            return True
        # Check if plate is overly saturated (spray paint)
        hsv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1]))
        if mean_sat > 180:
            return True
        return False
