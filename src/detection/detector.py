"""
Trinetra — Vehicle Detector
Wraps YOLOv8 for vehicle detection and classification.
"""

import cv2
import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ultralytics import YOLO
from loguru import logger


# ── Data structures ──────────────────────────────────────────

@dataclass
class Detection:
    """Single vehicle detection result."""
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


@dataclass
class FrameResult:
    """All detections for one frame."""
    frame_id: int
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0

    @property
    def count_by_class(self) -> dict:
        counts = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts

    @property
    def total_vehicles(self) -> int:
        return sum(1 for d in self.detections if d.class_name != 'pedestrian')


# ── Detector ─────────────────────────────────────────────────

class VehicleDetector:
    """
    YOLOv8-based vehicle detector for Trinetra.

    Usage:
        detector = VehicleDetector('models/weights/trinetra_v1.pt')
        results = detector.detect_frame(frame)
    """

    # Colour palette per class (BGR for OpenCV)
    CLASS_COLORS = {
        'car':           (255, 200, 0),
        'bus':           (0, 200, 255),
        'bike':          (0, 255, 100),
        'truck':         (255, 80, 0),
        'auto_rickshaw': (180, 0, 255),
        'pedestrian':    (255, 255, 255),
    }

    def __init__(
        self,
        weights_path: str,
        config_path: str = 'configs/config.yaml',
    ):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.conf_threshold = cfg['inference']['conf_threshold']
        self.iou_threshold  = cfg['inference']['iou_threshold']
        self.device         = cfg['inference']['device']
        self.image_size     = cfg['inference']['image_size']
        self.class_names    = cfg['classes']

        logger.info(f"Loading model: {weights_path}")
        self.model = YOLO(weights_path)
        self.model.to(self.device)
        logger.success(f"Model loaded — device: {self.device}")

    # ── Core inference ───────────────────────────────────────

    def detect_frame(self, frame: np.ndarray, frame_id: int = 0) -> FrameResult:
        """Run detection on a single BGR frame."""
        import time
        t0 = time.perf_counter()

        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        detections = self._parse_results(results)

        return FrameResult(
            frame_id=frame_id,
            detections=detections,
            inference_time_ms=elapsed_ms,
        )

    def detect_image(self, image_path: str) -> FrameResult:
        """Detect vehicles in a single image file."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.detect_frame(frame)

    def detect_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show: bool = False,
        max_frames: Optional[int] = None,
    ) -> List[FrameResult]:
        """
        Run detection on a full video.

        Args:
            video_path:  Input video file path.
            output_path: If set, saves annotated video here.
            show:        Display live preview window.
            max_frames:  Limit number of frames (None = all).

        Returns:
            List of FrameResult, one per frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video not found: {video_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video: {width}x{height} @ {fps:.1f}fps — {total} frames")

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # type: ignore
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_results = []
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_id >= max_frames:
                break

            result = self.detect_frame(frame, frame_id)
            all_results.append(result)

            annotated = self.draw_detections(frame, result)
            if annotated is not None and writer is not None:
                writer.write(annotated)
            if show:
                cv2.imshow('Trinetra — Vehicle Detection', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_id += 1
            if frame_id % 30 == 0:
                logger.info(
                    f"Frame {frame_id}/{total} — "
                    f"{result.total_vehicles} vehicles — "
                    f"{result.inference_time_ms:.1f}ms"
                )

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        logger.success(f"Done — {len(all_results)} frames processed")
        return all_results

    # ── Drawing ──────────────────────────────────────────────

    def draw_detections(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        out = frame.copy()

        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            color = self.CLASS_COLORS.get(det.class_name, (200, 200, 200))

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label background
            label = f"{det.class_name} {det.confidence:.2f}"
            if det.track_id is not None:
                label = f"#{det.track_id} {label}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)

            # Label text
            cv2.putText(
                out, label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 1, cv2.LINE_AA,
            )

        # HUD overlay
        hud = f"Frame {result.frame_id} | Vehicles: {result.total_vehicles} | {result.inference_time_ms:.0f}ms"
        cv2.putText(out, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        return out

    # ── Internal ─────────────────────────────────────────────

    def _parse_results(self, results) -> List[Detection]:
        detections = []
        if results.boxes is None:
            return detections

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id   = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = (
                self.class_names[class_id]
                if class_id < len(self.class_names)
                else f"class_{class_id}"
            )
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
            ))

        return detections


# ── Quick test helper ────────────────────────────────────────

def quick_test(image_path: str, weights: str = 'yolov8m.pt'):
    """
    Quickly test detector on an image using pretrained COCO weights.
    Useful before your custom model is trained.

    Run:  python -m src.detection.detector
    """
    detector = VehicleDetector(weights)
    result = detector.detect_image(image_path)

    print(f"\nDetected {len(result.detections)} objects:")
    for d in result.detections:
        print(f"  {d.class_name:15s}  conf={d.confidence:.3f}  bbox={d.bbox}")

    frame = cv2.imread(image_path)
    if frame is not None:
        annotated = detector.draw_detections(frame, result)
        out_path = 'results/detections/test_output.jpg'
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, annotated)
        print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/sample.jpg'
    quick_test(img)
