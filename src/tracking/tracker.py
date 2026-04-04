"""
Trinetra — Vehicle Tracker
Multi-object tracking using DeepSORT / ByteTrack.
Assigns stable track IDs to vehicles across frames.
Also computes per-track trajectory and speed estimation.
"""

import time
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
from loguru import logger

from src.detection.detector import Detection


@dataclass
class Track:
    """A single tracked vehicle over time."""
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    trajectory: deque = field(default_factory=lambda: deque(maxlen=60))
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    frame_count: int = 0

    # Speed estimation (pixels/frame → converted to km/h with calibration)
    speed_px_per_frame: float = 0.0
    speed_kmh: Optional[float] = None

    # Direction
    direction: Optional[str] = None   # N/S/E/W/NE/NW/SE/SW

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.first_seen

    def update(self, bbox, confidence, frame_id):
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = time.time()
        self.frame_count += 1
        self.trajectory.append(self.center)
        self._update_speed_and_direction()

    def _update_speed_and_direction(self):
        if len(self.trajectory) < 2:
            return

        pts = list(self.trajectory)
        lookback = min(5, len(pts))
        dx = pts[-1][0] - pts[-lookback][0]
        dy = pts[-1][1] - pts[-lookback][1]
        self.speed_px_per_frame = np.sqrt(dx**2 + dy**2) / float(lookback)

        # Cardinal direction
        angle = np.degrees(np.arctan2(-dy, dx))  # -dy because y increases downward
        directions = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
        idx = int((angle + 202.5) % 360 / 45)
        self.direction = directions[idx % 8]

    def set_speed_kmh(self, meters_per_pixel: float, fps: float):
        """Convert pixel speed to km/h given calibration."""
        if fps > 0 and meters_per_pixel > 0:
            mps = self.speed_px_per_frame * meters_per_pixel * fps
            self.speed_kmh = mps * 3.6


class VehicleTracker:
    """
    Wraps DeepSORT for vehicle tracking with Trinetra-specific
    trajectory and speed analysis.

    Falls back to a simple IoU-based tracker if deep-sort is unavailable.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        meters_per_pixel: float = 0.05,  # calibration: adjust per camera
        fps: float = 25.0,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.meters_per_pixel = meters_per_pixel
        self.fps = fps

        self.tracks: Dict[int, Track] = {}
        self._tracker = None
        self._frame_id = 0

        self._init_tracker()

    def _init_tracker(self):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self._tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.min_hits,
                nms_max_overlap=1.0,
                max_iou_distance=self.iou_threshold,
                embedder='mobilenet',
                half=False,
                bgr=True,
            )
            logger.info("Tracker: DeepSORT")
        except ImportError:
            logger.warning("deep-sort-realtime not installed — using IoU tracker fallback")
            self._tracker = None

    def update(
        self,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> List[Detection]:
        """
        Update tracker with new detections.
        Returns detections with track_id assigned.
        """
        if not detections:
            self._frame_id += 1
            return detections

        if self._tracker:
            return self._update_deepsort(frame, detections)
        else:
            return self._update_iou(detections)

    # ── DeepSORT ─────────────────────────────────────────────

    def _update_deepsort(self, frame, detections):
        # Convert to DeepSORT format: [[x1,y1,w,h], conf, class]
        raw = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            raw.append(([x1, y1, x2 - x1, y2 - y1], det.confidence, det.class_name))

        tracked = self._tracker.update_tracks(raw, frame=frame)  # type: ignore

        # Map track_ids back to detections by IoU
        for track in tracked:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            best_det = self._match_detection(detections, (x1, y1, x2, y2))
            if best_det:
                best_det.track_id = tid
                self._update_track_record(tid, best_det.bbox, best_det.confidence,
                                          best_det.class_name)

        self._frame_id += 1
        return detections

    # ── IoU tracker fallback ─────────────────────────────────

    def _update_iou(self, detections):
        """Simple IoU-based tracker (no appearance features)."""
        if not self.tracks:
            for i, det in enumerate(detections):
                det.track_id = i + 1
                self.tracks[det.track_id] = Track(
                    track_id=det.track_id,
                    class_name=det.class_name,
                    bbox=det.bbox,
                    confidence=det.confidence,
                )
            self._next_id = len(detections) + 1
            return detections

        if not hasattr(self, '_next_id'):
            self._next_id = max(self.tracks.keys()) + 1

        assigned = set()
        unmatched_dets = list(range(len(detections)))

        for track_id, track in list(self.tracks.items()):
            best_iou = self.iou_threshold
            best_det_idx = None

            for i in unmatched_dets:
                iou = self._iou(track.bbox, detections[i].bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i

            if best_det_idx is not None:
                detections[best_det_idx].track_id = track_id
                track.update(detections[best_det_idx].bbox,
                              detections[best_det_idx].confidence, self._frame_id)
                track.set_speed_kmh(self.meters_per_pixel, self.fps)
                assigned.add(best_det_idx)
                unmatched_dets.remove(best_det_idx)

        # New tracks for unmatched detections
        for i in unmatched_dets:
            tid = self._next_id
            self._next_id += 1
            detections[i].track_id = tid
            self.tracks[tid] = Track(
                track_id=tid,
                class_name=detections[i].class_name,
                bbox=detections[i].bbox,
                confidence=detections[i].confidence,
            )

        self._frame_id += 1
        return detections

    # ── Helpers ───────────────────────────────────────────────

    def _update_track_record(self, tid, bbox, conf, class_name):
        if tid not in self.tracks:
            self.tracks[tid] = Track(
                track_id=tid, class_name=class_name, bbox=bbox, confidence=conf
            )
        else:
            self.tracks[tid].update(bbox, conf, self._frame_id)
            self.tracks[tid].set_speed_kmh(self.meters_per_pixel, self.fps)

    def _match_detection(self, detections, tracked_bbox):
        best, best_iou = None, 0.3
        for det in detections:
            iou = self._iou(det.bbox, tracked_bbox)
            if iou > best_iou:
                best_iou = iou
                best = det
        return best

    @staticmethod
    def _iou(b1, b2) -> float:
        xa = max(b1[0], b2[0]); ya = max(b1[1], b2[1])
        xb = min(b1[2], b2[2]); yb = min(b1[3], b2[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / float(a1 + a2 - inter)

    def get_track(self, track_id: int) -> Optional[Track]:
        return self.tracks.get(track_id)

    def draw_trajectories(self, frame: np.ndarray, min_points: int = 5) -> np.ndarray:
        """Draw vehicle movement trails on frame."""
        for track in self.tracks.values():
            pts = list(track.trajectory)
            if len(pts) < min_points:
                continue
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                color = (int(255 * alpha), int(180 * alpha), 50)
                cv2.line(frame, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)
        return frame

    def draw_speed(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw speed label above each tracked vehicle."""
        for det in detections:
            if det.track_id is None:
                continue
            track = self.get_track(det.track_id)
            if track and track.speed_kmh is not None:
                x1, y1 = det.bbox[0], det.bbox[1]
                label = f"{track.speed_kmh:.0f} km/h"
                cv2.putText(frame, label, (x1, y1 - 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        return frame
