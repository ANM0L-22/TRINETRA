"""
Trinetra — Wrong-Side Driving Detector
Detects vehicles moving against traffic flow using trajectory analysis.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from loguru import logger
from typing import Sequence, Iterable

class WrongSideDetector:
    """
    Detects wrong-side driving by:
      1. Estimating dominant traffic flow direction from all track trajectories
      2. Flagging tracks whose direction deviates significantly from dominant flow
    
    Works best on straight roads with clear lane markings.
    Configure road_axis to match your camera orientation.
    """

    def __init__(
        self,
        road_axis: str = 'horizontal',  # 'horizontal' or 'vertical'
        min_track_frames: int = 10,
        deviation_threshold_deg: float = 120.0,
    ):
        """
        Args:
            road_axis: Primary direction of travel in camera view.
            min_track_frames: Minimum frames before evaluating direction.
            deviation_threshold_deg: Angle deviation to flag as wrong-side.
        """
        self.road_axis = road_axis
        self.min_frames = min_track_frames
        self.deviation_threshold = deviation_threshold_deg

        # track_id -> list of movement vectors
        self.movement_vectors: Dict[int, list] = defaultdict(list)

    
    def update(self, track_id: int, trajectory_pts: Sequence[Tuple[int, int]]):
        """Feed trajectory points for a track."""
        if len(trajectory_pts) < 3:
            return
        pts = list(trajectory_pts)
        # Compute displacement vector over last N points
        n = min(len(pts), 10)
        dx = pts[-1][0] - pts[-n][0]
        dy = pts[-1][1] - pts[-n][1]
        if abs(dx) > 2 or abs(dy) > 2:  # filter noise
            self.movement_vectors[track_id].append((dx, dy))

    def get_dominant_direction(self) -> Optional[float]:
        """Compute mean movement angle across all tracks."""
        all_vecs = []
        for vecs in self.movement_vectors.values():
            all_vecs.extend(vecs)
        if len(all_vecs) < 3:
            return None
        angles = [np.degrees(np.arctan2(dy, dx)) for dx, dy in all_vecs]
        return float(np.median(angles))

    def is_wrong_side(self, track_id: int) -> bool:
        """Check if a specific track is going the wrong way."""
        vecs = self.movement_vectors.get(track_id, [])
        if len(vecs) < self.min_frames:
            return False

        dominant = self.get_dominant_direction()
        if dominant is None:
            return False

        # Track's mean direction
        angles = [np.degrees(np.arctan2(dy, dx)) for dx, dy in vecs[-10:]]
        track_angle = float(np.median(angles))

        # Angular difference (wrapped to [-180, 180])
        diff = abs(((track_angle - dominant) + 180) % 360 - 180)
        return diff > self.deviation_threshold

    def get_wrong_side_tracks(self, active_track_ids: List[int]) -> List[int]:
        """Return list of track IDs currently going wrong-side."""
        return [tid for tid in active_track_ids if self.is_wrong_side(tid)]

    def cleanup(self, active_track_ids: List[int]):
        """Remove stale tracks."""
        for tid in list(self.movement_vectors.keys()):
            if tid not in active_track_ids:
                del self.movement_vectors[tid]
