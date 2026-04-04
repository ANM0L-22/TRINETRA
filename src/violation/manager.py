"""
Trinetra — Violation Manager
Aggregates helmet, seatbelt, phone, wrong-side, and tampered plate violations.
Generates violation events with evidence frames for reporting.
"""

import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger

from .classifier import ViolationClassifier, ViolationResult
from .wrong_side import WrongSideDetector


@dataclass
class ViolationEvent:
    """A recorded traffic violation."""
    event_id: str
    violation_type: str
    track_id: Optional[int]
    plate_number: str
    timestamp: float
    frame_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    evidence_frame: Optional[np.ndarray] = None  # saved separately

    def to_dict(self) -> dict:
        return {
            'event_id': self.event_id,
            'violation_type': self.violation_type,
            'track_id': self.track_id,
            'plate_number': self.plate_number,
            'timestamp': self.timestamp,
            'frame_id': self.frame_id,
            'confidence': round(self.confidence, 3),
        }


class ViolationManager:
    """
    Central violation tracking system.

    - Calls ViolationClassifier per vehicle crop
    - Calls WrongSideDetector on trajectories
    - Deduplicates events (same vehicle, same violation within cooldown window)
    - Stores evidence frames
    - Generates ViolationEvent records
    """

    COOLDOWN_SECONDS = 10.0   # min gap between same violation for same track

    def __init__(
        self,
        violation_weights: Optional[str] = None,
        evidence_dir: str = 'results/violations',
        conf_threshold: float = 0.55,
        device: str = 'cpu',
        save_evidence: bool = True,
    ):
        self.classifier = ViolationClassifier(
            weights_path=violation_weights,
            conf_threshold=conf_threshold,
            device=device,
        )
        self.wrong_side = WrongSideDetector()
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.save_evidence = save_evidence

        self.events: List[ViolationEvent] = []
        self._last_event_time: Dict[Tuple, float] = {}   # (track_id, type) -> timestamp
        self._event_counter = 0

    # ── Main update ───────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        detections: list,   # List[Detection]
        tracks: dict,        # track_id -> Track
        anpr_results: list,  # List[ANPRResult]
    ) -> List[ViolationEvent]:
        """
        Process one frame for all violations.
        Returns new ViolationEvent objects generated this frame.
        """
        new_events = []

        # Build plate lookup
        plate_lookup = {r.track_id: r.plate_number for r in anpr_results if r.track_id}

        # Update wrong-side detector
        active_ids = [d.track_id for d in detections if d.track_id is not None]
        for tid in active_ids:
            if tid in tracks:
                self.wrong_side.update(tid, tracks[tid].trajectory)
        wrong_side_ids = set(self.wrong_side.get_wrong_side_tracks(active_ids))
        self.wrong_side.cleanup(active_ids)

        for det in detections:
            plate = plate_lookup.get(det.track_id, 'UNKNOWN')

            # ── Helmet (bikes) ──
            if det.class_name == 'bike':
                r = self.classifier.classify_rider(frame, det.bbox, det.track_id)
                if r.is_violation:
                    ev = self._make_event(r.violation_type, det.track_id, plate,
                                          frame_id, det.bbox, r.confidence, frame)
                    if ev:
                        new_events.append(ev)

            # ── Seatbelt + phone (cars/buses/trucks) ──
            elif det.class_name in ('car', 'bus', 'truck', 'auto_rickshaw'):
                violations = self.classifier.classify_driver(frame, det.bbox, det.track_id)
                for r in violations:
                    if r.is_violation:
                        ev = self._make_event(r.violation_type, det.track_id, plate,
                                               frame_id, det.bbox, r.confidence, frame)
                        if ev:
                            new_events.append(ev)

            # ── Wrong-side ──
            if det.track_id in wrong_side_ids:
                ev = self._make_event('wrong_side', det.track_id, plate,
                                       frame_id, det.bbox, 0.75, frame)
                if ev:
                    new_events.append(ev)

        # ── Tampered plates (from ANPR results) ──
        for anpr in anpr_results:
            if anpr.is_tampered and anpr.track_id is not None:
                ev = self._make_event('tampered_plate', anpr.track_id, anpr.plate_number,
                                       frame_id, anpr.vehicle_bbox, 0.8, frame)
                if ev:
                    new_events.append(ev)

        self.events.extend(new_events)
        return new_events

    # ── Drawing ───────────────────────────────────────────────

    def draw_violations(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw violation overlays on frame."""
        from src.utils.drawing import draw_violation_badge
        for det in detections:
            recent = self._get_active_violation(det.track_id)
            if recent:
                frame = draw_violation_badge(frame, det.bbox, recent)
        return frame

    # ── Summary ───────────────────────────────────────────────

    def get_summary(self) -> dict:
        from collections import Counter
        types = Counter(e.violation_type for e in self.events)
        return {
            'total': len(self.events),
            'by_type': dict(types),
            'recent_plate': self.events[-1].plate_number if self.events else None,
        }

    def export_csv(self, path: str):
        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'event_id', 'violation_type', 'track_id',
                'plate_number', 'timestamp', 'frame_id', 'confidence'
            ])
            writer.writeheader()
            for ev in self.events:
                writer.writerow(ev.to_dict())
        logger.success(f"Violations exported: {path}")

    # ── Internals ─────────────────────────────────────────────

    def _make_event(
        self, vtype, track_id, plate, frame_id, bbox, conf, frame
    ) -> Optional[ViolationEvent]:
        key = (track_id, vtype)
        now = time.time()
        if now - self._last_event_time.get(key, 0) < self.COOLDOWN_SECONDS:
            return None

        self._last_event_time[key] = now
        self._event_counter += 1
        eid = f"TRN-{self._event_counter:05d}"

        ev = ViolationEvent(
            event_id=eid,
            violation_type=vtype,
            track_id=track_id,
            plate_number=plate,
            timestamp=now,
            frame_id=frame_id,
            bbox=bbox,
            confidence=conf,
        )

        if self.save_evidence and frame is not None:
            self._save_evidence(ev, frame)

        logger.warning(f"VIOLATION [{eid}] {vtype} | plate={plate} | conf={conf:.2f}")
        return ev

    def _save_evidence(self, ev: ViolationEvent, frame: np.ndarray):
        x1, y1, x2, y2 = ev.bbox
        pad = 20
        h, w = frame.shape[:2]
        crop = frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
        path = self.evidence_dir / f"{ev.event_id}_{ev.violation_type}.jpg"
        cv2.imwrite(str(path), crop)
        ev.evidence_frame = None  # don't store full frame in memory

    def _get_active_violation(self, track_id) -> Optional[str]:
        """Return most recent active violation type for a track."""
        if track_id is None:
            return None
        now = time.time()
        for (tid, vtype), t in self._last_event_time.items():
            if tid == track_id and now - t < 3.0:
                return vtype
        return None
