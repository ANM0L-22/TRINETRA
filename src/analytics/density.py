"""
Trinetra — Traffic Density & Congestion Analytics
Computes real-time traffic density, congestion score, flow metrics,
and generates heatmaps for dashboard display.
"""

import time
import numpy as np
import cv2
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from loguru import logger


CONGESTION_LEVELS = {
    (0.0,  0.2):  ('FREE',     (0, 220,   0)),
    (0.2,  0.5):  ('MODERATE', (0, 220, 180)),
    (0.5,  0.75): ('HEAVY',    (0, 165, 255)),
    (0.75, 1.0):  ('GRIDLOCK', (0,   0, 255)),
}


def get_congestion_level(score: float) -> Tuple[str, Tuple]:
    for (lo, hi), (label, color) in CONGESTION_LEVELS.items():
        if lo <= score < hi:
            return label, color
    return 'GRIDLOCK', (0, 0, 255)


@dataclass
class DensitySnapshot:
    """Traffic state at one point in time."""
    timestamp: float
    frame_id: int
    total_vehicles: int
    count_by_class: Dict[str, int]
    congestion_score: float      # 0.0 = free flow, 1.0 = gridlock
    congestion_level: str
    avg_speed_kmh: Optional[float]
    vehicles_per_lane: Dict[int, int] = field(default_factory=dict)


class DensityAnalyzer:
    """
    Real-time traffic density and congestion analysis.

    Congestion score is computed from:
      - Vehicle count relative to road capacity
      - Average vehicle speed (if tracking enabled)
      - Spatial density (occupancy ratio)
    """

    def __init__(
        self,
        road_capacity: int = 20,        # max vehicles per frame before gridlock
        frame_width: int = 1280,
        frame_height: int = 720,
        history_seconds: float = 60.0,  # rolling window for averaging
        fps: float = 25.0,
        num_lanes: int = 2,
    ):
        self.road_capacity = road_capacity
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.num_lanes = num_lanes
        history_frames = int(history_seconds * fps)

        self.snapshots: deque = deque(maxlen=history_frames)
        self.density_map = np.zeros((frame_height // 8, frame_width // 8), dtype=np.float32)

    def update(
        self,
        frame_id: int,
        detections: list,   # List[Detection]
        tracks: dict,        # track_id -> Track
    ) -> DensitySnapshot:
        vehicles = [d for d in detections if d.class_name != 'pedestrian']
        count = len(vehicles)
        count_by_class = {}
        for d in detections:
            count_by_class[d.class_name] = count_by_class.get(d.class_name, 0) + 1

        # Occupancy: sum of vehicle bbox areas / frame area
        frame_area = self.frame_width * self.frame_height
        occupied = sum(
            (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
            for d in vehicles
        )
        occupancy_ratio = min(occupied / max(frame_area, 1), 1.0)

        # Count ratio
        count_ratio = min(count / max(self.road_capacity, 1), 1.0)

        # Speed factor (lower speed = higher congestion)
        if count == 0:
            speed_factor = 0.0
        else:
            speed_factor = 0.5  # default when no tracking
        speeds = []
        for d in vehicles:
            if d.track_id and d.track_id in tracks:
                t = tracks[d.track_id]
                if t.speed_kmh is not None:
                    speeds.append(t.speed_kmh)
        if speeds:
            avg_speed = float(np.mean(speeds))
            # Assume free flow at 50 km/h; normalize
            speed_factor = max(0.0, 1.0 - avg_speed / 50.0)
        else:
            avg_speed = None

        # Weighted congestion score
        score = (0.4 * count_ratio) + (0.35 * occupancy_ratio) + (0.25 * speed_factor)
        score = float(np.clip(score, 0.0, 1.0))

        level, _ = get_congestion_level(score)

        # Update density map
        self._update_density_map(vehicles)

        # Per-lane count
        lane_counts = self._count_per_lane(vehicles)

        snap = DensitySnapshot(
            timestamp=time.time(),
            frame_id=frame_id,
            total_vehicles=count,
            count_by_class=count_by_class,
            congestion_score=score,
            congestion_level=level,
            avg_speed_kmh=avg_speed,
            vehicles_per_lane=lane_counts,
        )
        self.snapshots.append(snap)
        return snap

    def get_rolling_avg(self, window_seconds: float = 30.0) -> Optional[DensitySnapshot]:
        """Average stats over last N seconds."""
        cutoff = time.time() - window_seconds
        recent = [s for s in self.snapshots if s.timestamp >= cutoff]
        if not recent:
            return None
        avg_score = float(np.mean([s.congestion_score for s in recent]))
        avg_count = float(np.mean([s.total_vehicles for s in recent]))
        level, _ = get_congestion_level(avg_score)
        speeds = [s.avg_speed_kmh for s in recent if s.avg_speed_kmh is not None]
        return DensitySnapshot(
            timestamp=time.time(),
            frame_id=recent[-1].frame_id,
            total_vehicles=int(avg_count),
            count_by_class={},
            congestion_score=avg_score,
            congestion_level=level,
            avg_speed_kmh=float(np.mean(speeds)) if speeds else None,
        )

    def get_heatmap(self, target_size: Tuple[int, int] | None = None) -> np.ndarray:
        """Return colour heatmap of vehicle density."""
        dm = self.density_map.copy()
        # Decay
        self.density_map *= 0.95
        if dm.max() > 0:
            dm = cv2.normalize(dm, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore
            dm = dm.astype(np.uint8) if dm is not None else np.zeros_like(dm, dtype=np.uint8)
        else:
            dm = np.zeros_like(dm, dtype=np.uint8)
        heatmap = cv2.applyColorMap(dm, cv2.COLORMAP_JET)
        if target_size:
            heatmap = cv2.resize(heatmap, target_size)
        return heatmap

    def draw_overlay(self, frame: np.ndarray, snapshot: DensitySnapshot) -> np.ndarray:
        """Draw congestion overlay and stats panel on frame."""
        _, color = get_congestion_level(snapshot.congestion_score)

        # Congestion bar at top
        bar_w = int(snapshot.congestion_score * frame.shape[1])
        cv2.rectangle(frame, (0, 0), (bar_w, 6), color, -1)

        # Stats panel
        stats = {
            'Vehicles': snapshot.total_vehicles,
            'Congestion': f"{snapshot.congestion_level} ({snapshot.congestion_score:.2f})",
        }
        if snapshot.avg_speed_kmh is not None:
            stats['Avg speed'] = f"{snapshot.avg_speed_kmh:.1f} km/h"
        for cls, cnt in snapshot.count_by_class.items():
            if cnt > 0:
                stats[cls] = cnt

        from src.utils.drawing import put_stats_panel
        frame = put_stats_panel(frame, stats, x=10, y=40)

        return frame

    def _update_density_map(self, vehicles):
        mh, mw = self.density_map.shape
        for d in vehicles:
            cx = int(((d.bbox[0] + d.bbox[2]) / 2) / self.frame_width * mw)
            cy = int(((d.bbox[1] + d.bbox[3]) / 2) / self.frame_height * mh)
            cx = np.clip(cx, 0, mw - 1)
            cy = np.clip(cy, 0, mh - 1)
            cv2.circle(self.density_map, (cx, cy), 3, 1.0, -1)

    def _count_per_lane(self, vehicles) -> Dict[int, int]:
        lane_w = self.frame_width / self.num_lanes
        counts = defaultdict(int)
        for d in vehicles:
            cx = (d.bbox[0] + d.bbox[2]) / 2
            lane = min(int(cx / lane_w), self.num_lanes - 1)
            counts[lane] += 1
        return dict(counts)
