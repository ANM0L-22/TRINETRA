"""
Trinetra — Main Pipeline
Orchestrates all modules: detection → tracking → ANPR → violation → analytics.
This is the single entry point for running the full system.

Usage:
    python -m src.pipeline --source path/to/video.mp4
    python -m src.pipeline --source 0         # webcam
    python -m src.pipeline --source video.mp4 --output out.mp4 --no-show
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from src.detection.detector import VehicleDetector
from src.tracking.tracker import VehicleTracker
from src.ocr.anpr_pipeline import ANPRPipeline
from src.violation.manager import ViolationManager
from src.analytics.density import DensityAnalyzer
from src.analytics.reporter import ReportGenerator
from src.utils.config import load_config, setup_logger
from src.utils.video_io import VideoReader, VideoWriter
from src.utils.drawing import draw_hud, CLASS_COLORS


class TrinetraPipeline:
    """Full Trinetra inference pipeline."""

    def __init__(self, config_path: str = 'configs/config.yaml'):
        self.cfg = load_config(config_path)
        setup_logger('results/logs')

        weights = self._find_weights()
        device  = self.cfg['inference']['device']

        logger.info("Initialising Trinetra pipeline...")

        self.detector  = VehicleDetector(weights, config_path)
        self.tracker   = VehicleTracker(
            max_age=self.cfg['tracking']['max_age'],
            min_hits=self.cfg['tracking']['min_hits'],
            iou_threshold=self.cfg['tracking']['iou_threshold'],
        )
        self.anpr      = ANPRPipeline(use_gpu=device == 'cuda')
        self.violation = ViolationManager(
            evidence_dir=self.cfg['results']['violations_dir'],
            device=device,
        )
        self.density   = DensityAnalyzer()
        self.reporter  = ReportGenerator(self.cfg['results']['reports_dir'])

        logger.success("Pipeline ready.")

    # ── Main run ──────────────────────────────────────────────

    def run(
        self,
        source,
        output_path: Optional[str] = None,
        show: bool = True,
        max_frames: Optional[int] = None,
    ):
        """
        Run the full pipeline on a video source.

        Args:
            source:      File path or camera index.
            output_path: Where to save annotated video.
            show:        Display live window.
            max_frames:  Cap number of frames (None = all).
        """
        all_density_snaps = []

        with VideoReader(source) as reader:
            writer = None
            if output_path:
                writer = VideoWriter(output_path, reader.fps, reader.width, reader.height)

            logger.info(f"Processing: {source} ({reader.total} frames)")
            t_start = time.time()

            for frame_id, frame in reader:
                if max_frames and frame_id >= max_frames:
                    break

                annotated = self.process_frame(frame, frame_id)
                all_density_snaps.append(self._last_density_snap)

                if writer:
                    writer.write(annotated)

                if show:
                    cv2.imshow('Trinetra', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User quit")
                        break

                if frame_id % 100 == 0 and frame_id > 0:
                    elapsed = time.time() - t_start
                    fps_actual = frame_id / elapsed
                    logger.info(
                        f"Frame {frame_id} | "
                        f"{fps_actual:.1f} fps | "
                        f"violations: {len(self.violation.events)}"
                    )

            cv2.destroyAllWindows()
            if writer:
                writer.release()

        self._finalize(all_density_snaps)

    def process_frame(self, frame: np.ndarray, frame_id: int) -> np.ndarray:
        """
        Run one frame through the full pipeline.
        Returns annotated frame.
        """
        # 1. Detect
        det_result = self.detector.detect_frame(frame, frame_id)

        # 2. Track
        tracked_dets = self.tracker.update(frame, det_result.detections)

        # 3. ANPR (every 5th frame to save CPU)
        anpr_results = []
        if frame_id % 5 == 0:
            anpr_results = self.anpr.process_frame(frame, tracked_dets)

        # 4. Violations
        new_violations = self.violation.process_frame(
            frame, frame_id, tracked_dets, self.tracker.tracks, anpr_results
        )

        # 5. Density
        snap = self.density.update(frame_id, tracked_dets, self.tracker.tracks)
        self._last_density_snap = snap

        # 6. Draw everything
        annotated = self._draw_frame(frame.copy(), frame_id, tracked_dets,
                                      anpr_results, snap, new_violations)
        return annotated

    # ── Drawing ───────────────────────────────────────────────

    def _draw_frame(self, frame, frame_id, detections, anpr_results, snap, new_violations):
        # Vehicle boxes
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = CLASS_COLORS.get(det.class_name, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = det.class_name
            if det.track_id is not None:
                label = f"#{det.track_id} {label}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Trajectories
        frame = self.tracker.draw_trajectories(frame)

        # ANPR
        frame = self.anpr.draw_results(frame, anpr_results)

        # Violations
        frame = self.violation.draw_violations(frame, detections)

        # Density overlay
        frame = self.density.draw_overlay(frame, snap)

        # Violation alerts (flash for new ones)
        if new_violations:
            for v in new_violations[:2]:
                alert = f"! {v.violation_type.replace('_', ' ').upper()} — {v.plate_number}"
                h = frame.shape[0]
                cv2.putText(frame, alert, (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

        # Frame counter HUD
        frame = draw_hud(frame, f"Frame {frame_id}", position=(10, frame.shape[0] - 50))

        return frame

    # ── Finalise ──────────────────────────────────────────────

    def _finalize(self, density_snaps):
        logger.info("Generating session report...")
        report = self.reporter.generate_session_report(
            self.violation.events, density_snaps
        )
        self.reporter.save_json(report)
        self.reporter.save_violations_csv(self.violation.events)
        self.reporter.save_density_csv(density_snaps)

        print("\n" + "=" * 52)
        print("  TRINETRA — SESSION SUMMARY")
        print("=" * 52)
        print(f"  Frames processed : {report['session']['frames_processed']}")
        print(f"  Duration         : {report['session']['duration_seconds']:.1f}s")
        print(f"  Total violations : {report['violations']['total']}")
        for vtype, count in report['violations']['by_type'].items():
            print(f"    {vtype:20s}: {count}")
        tf = report['traffic_flow']
        print(f"  Avg vehicles     : {tf['avg_vehicles_per_frame']}")
        print(f"  Avg congestion   : {tf['avg_congestion_score']:.2f}")
        if tf['avg_speed_kmh']:
            print(f"  Avg speed        : {tf['avg_speed_kmh']} km/h")
        print("=" * 52)

    def _find_weights(self) -> str:
        candidates = [
            Path('models/weights/trinetra_best.pt'),
            Path('runs/detect/trinetra_v1/weights/best.pt'),
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        logger.warning("No custom weights found — using YOLOv8m pretrained on COCO")
        return 'yolov8m.pt'


# ── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Trinetra — Traffic Surveillance')
    parser.add_argument('--source', required=True,
                        help='Video path or camera index (0, 1, ...)')
    parser.add_argument('--output', default=None,
                        help='Output annotated video path')
    parser.add_argument('--no-show', action='store_true',
                        help='Suppress display window')
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    pipeline = TrinetraPipeline(args.config)
    pipeline.run(
        source=source,
        output_path=args.output,
        show=not args.no_show,
        max_frames=args.max_frames,
    )


if __name__ == '__main__':
    main()
