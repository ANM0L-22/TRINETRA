"""
Trinetra — ANPR Pipeline
Combines plate detection + OCR into one clean interface.
Also maintains a plate history cache for tracking plates over time.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from .plate_detector import PlateDetector, PlateRegion
from .ocr_reader import OCRReader, PlateText


@dataclass
class ANPRResult:
    """Combined ANPR result for one vehicle detection."""
    vehicle_bbox: Tuple[int, int, int, int]
    track_id: Optional[int]
    plate_region: Optional[PlateRegion]
    plate_text: Optional[PlateText]
    timestamp: float = field(default_factory=time.time)

    @property
    def plate_number(self) -> str:
        if self.plate_text:
            return self.plate_text.cleaned_text
        return "UNKNOWN"

    @property
    def is_tampered(self) -> bool:
        return self.plate_region.is_tampered if self.plate_region else False

    @property
    def confidence(self) -> float:
        if self.plate_text:
            return self.plate_text.confidence
        return 0.0


class PlateHistoryCache:
    """
    Maintains a rolling history of plate readings per track ID.
    Uses majority vote over recent readings for stable output.
    """

    def __init__(self, max_history: int = 10):
        self.history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.confirmed: Dict[int, str] = {}

    def update(self, track_id: int, plate_text: Optional[PlateText]):
        if plate_text and plate_text.cleaned_text:
            self.history[track_id].append(plate_text.cleaned_text)
            self._recompute_confirmed(track_id)

    def get_best(self, track_id: int) -> Optional[str]:
        return self.confirmed.get(track_id)

    def _recompute_confirmed(self, track_id: int):
        readings = list(self.history[track_id])
        if not readings:
            return
        # Majority vote
        from collections import Counter
        most_common, count = Counter(readings).most_common(1)[0]
        if count >= 2 or len(readings) == 1:
            self.confirmed[track_id] = most_common

    def clear_old(self, active_track_ids: List[int]):
        for tid in list(self.history.keys()):
            if tid not in active_track_ids:
                del self.history[tid]
                self.confirmed.pop(tid, None)


class ANPRPipeline:
    """
    Full ANPR pipeline:
      Detection crops → Plate localisation → OCR → Text cleaning → History smoothing
    """

    def __init__(
        self,
        plate_weights: Optional[str] = None,
        use_gpu: bool = False,
        conf_threshold: float = 0.40,
    ):
        self.detector = PlateDetector(
            weights_path=plate_weights,
            conf_threshold=conf_threshold,
            device='cuda' if use_gpu else 'cpu',
        )
        self.reader = OCRReader(use_gpu=use_gpu)
        self.cache = PlateHistoryCache()
        logger.info("ANPR pipeline initialised")

    def process_vehicle(
        self,
        frame: np.ndarray,
        vehicle_bbox: Tuple[int, int, int, int],
        track_id: Optional[int] = None,
    ) -> ANPRResult:
        """
        Run ANPR on a single detected vehicle region.
        Returns ANPRResult with plate number and tamper status.
        """
        plates = self.detector.detect_in_vehicle_crop(frame, vehicle_bbox)

        if not plates:
            return ANPRResult(
                vehicle_bbox=vehicle_bbox,
                track_id=track_id,
                plate_region=None,
                plate_text=None,
            )

        # Use highest-confidence plate
        best_plate = max(plates, key=lambda p: p.confidence)
        plate_text = None

        if best_plate.crop is not None and best_plate.crop.size > 0:
            plate_text = self.reader.read_plate_multi(best_plate.crop)

        # Update cache
        if track_id is not None:
            self.cache.update(track_id, plate_text)
            # Use cached stable reading if available
            stable = self.cache.get_best(track_id)
            if stable and plate_text:
                plate_text.cleaned_text = stable

        return ANPRResult(
            vehicle_bbox=vehicle_bbox,
            track_id=track_id,
            plate_region=best_plate,
            plate_text=plate_text,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        vehicle_detections: list,  # List of Detection objects
    ) -> List[ANPRResult]:
        """
        Process all vehicles in a frame.
        vehicle_detections: list of Detection objects with .bbox and .track_id
        """
        results = []
        active_ids = [d.track_id for d in vehicle_detections if d.track_id is not None]
        self.cache.clear_old(active_ids)

        for det in vehicle_detections:
            if det.class_name == 'pedestrian':
                continue
            result = self.process_vehicle(frame, det.bbox, det.track_id)
            results.append(result)

        return results

    def draw_results(self, frame: np.ndarray, results: List[ANPRResult]) -> np.ndarray:
        """Draw plate bounding boxes and OCR text on frame."""
        for r in results:
            if r.plate_region is None:
                continue

            x1, y1, x2, y2 = r.plate_region.bbox
            color = (0, 0, 220) if r.is_tampered else (0, 220, 120)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = r.plate_number
            if r.is_tampered:
                label = "⚠ TAMPERED"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return frame
