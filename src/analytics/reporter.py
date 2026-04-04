"""
Trinetra — Report Generator
Generates structured traffic analytics reports:
  - Violation summary (per type, per plate)
  - Traffic flow metrics (hourly, per-lane)
  - Congestion timeline
  - PDF / CSV / JSON export
"""

import json
import csv
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import numpy as np
from loguru import logger


class ReportGenerator:
    """
    Aggregates data from ViolationManager and DensityAnalyzer
    and generates exportable reports.
    """

    def __init__(self, report_dir: str = 'results/reports'):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._session_start = time.time()

    def generate_session_report(
        self,
        violation_events: list,
        density_snapshots: list,
        session_name: Optional[str] = None,
    ) -> dict:
        """
        Build a full session report dictionary.
        """
        if not session_name:
            session_name = datetime.now().strftime('%Y%m%d_%H%M%S')

        duration = time.time() - self._session_start

        # ── Violations ───────────────────────────────────────
        vtype_counts = Counter(e.violation_type for e in violation_events)
        plate_violations = defaultdict(list)
        for e in violation_events:
            plate_violations[e.plate_number].append(e.violation_type)

        top_offenders = sorted(
            plate_violations.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:10]

        # ── Traffic flow ─────────────────────────────────────
        if density_snapshots:
            counts = [s.total_vehicles for s in density_snapshots]
            scores = [s.congestion_score for s in density_snapshots]
            speeds = [s.avg_speed_kmh for s in density_snapshots
                      if s.avg_speed_kmh is not None]
        else:
            counts, scores, speeds = [], [], []

        report = {
            'session': {
                'name': session_name,
                'start': datetime.fromtimestamp(self._session_start).isoformat(),
                'duration_seconds': round(duration, 1),
                'frames_processed': len(density_snapshots),
            },
            'violations': {
                'total': len(violation_events),
                'by_type': dict(vtype_counts),
                'top_offenders': [
                    {'plate': p, 'count': len(v), 'types': list(set(v))}
                    for p, v in top_offenders
                ],
            },
            'traffic_flow': {
                'avg_vehicles_per_frame': round(float(np.mean(counts)), 2) if counts else 0,
                'peak_vehicles': max(counts) if counts else 0,
                'avg_congestion_score': round(float(np.mean(scores)), 3) if scores else 0,
                'avg_speed_kmh': round(float(np.mean(speeds)), 1) if speeds else None,
                'congestion_distribution': self._congestion_distribution(scores),
            },
        }

        return report

    def save_json(self, report: dict, name: Optional[str] = None) -> str:
        fname = name or f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.report_dir / fname
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.success(f"Report saved: {path}")
        return str(path)

    def save_violations_csv(self, violation_events: list, name: Optional[str] = None) -> str:
        fname = name or f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = self.report_dir / fname
        if not violation_events:
            logger.warning("No violations to export")
            return str(path)
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'event_id', 'violation_type', 'track_id',
                'plate_number', 'timestamp', 'frame_id', 'confidence'
            ])
            writer.writeheader()
            for ev in violation_events:
                writer.writerow(ev.to_dict())
        logger.success(f"Violations CSV saved: {path}")
        return str(path)

    def save_density_csv(self, density_snapshots: list, name: Optional[str] = None) -> str:
        fname = name or f"density_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = self.report_dir / fname
        if not density_snapshots:
            return str(path)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_id', 'timestamp', 'total_vehicles',
                              'congestion_score', 'congestion_level', 'avg_speed_kmh'])
            for s in density_snapshots:
                writer.writerow([
                    s.frame_id, round(s.timestamp, 2),
                    s.total_vehicles, round(s.congestion_score, 4),
                    s.congestion_level, s.avg_speed_kmh
                ])
        logger.success(f"Density CSV saved: {path}")
        return str(path)

    def _congestion_distribution(self, scores: list) -> dict:
        if not scores:
            return {}
        arr = np.array(scores)
        return {
            'FREE':     int(np.sum(arr < 0.2)),
            'MODERATE': int(np.sum((arr >= 0.2) & (arr < 0.5))),
            'HEAVY':    int(np.sum((arr >= 0.5) & (arr < 0.75))),
            'GRIDLOCK': int(np.sum(arr >= 0.75)),
        }
