"""
Trinetra — Unit Tests
Run: pytest tests/ -v
"""

import sys
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Detection tests ───────────────────────────────────────────

class TestDetection:
    def test_detection_center(self):
        from src.detection.detector import Detection
        d = Detection(bbox=(100, 200, 300, 400), class_id=0,
                      class_name='car', confidence=0.9)
        assert d.center == (200, 300)

    def test_detection_area(self):
        from src.detection.detector import Detection
        d = Detection(bbox=(0, 0, 100, 50), class_id=0,
                      class_name='car', confidence=0.8)
        assert d.area == 5000

    def test_frame_result_count(self):
        from src.detection.detector import Detection, FrameResult
        dets = [
            Detection((0,0,100,100), 0, 'car', 0.9),
            Detection((0,0,100,100), 2, 'bike', 0.8),
            Detection((0,0,100,100), 5, 'pedestrian', 0.7),
        ]
        fr = FrameResult(frame_id=1, detections=dets)
        assert fr.total_vehicles == 2  # pedestrian excluded
        assert fr.count_by_class['car'] == 1
        assert fr.count_by_class['pedestrian'] == 1


# ── ANPR tests ────────────────────────────────────────────────

class TestOCR:
    def test_plate_text_validation(self):
        from src.ocr.ocr_reader import OCRReader
        reader = OCRReader.__new__(OCRReader)
        reader.ocr = None
        reader._pytesseract = None

        # Valid Indian plates
        result = reader._build_plate_text("MH12AB1234", 0.9)
        assert result.is_valid_format
        assert result.state_code == "MH"
        assert result.district_code == "12"

    def test_plate_cleaning(self):
        from src.ocr.ocr_reader import OCRReader
        reader = OCRReader.__new__(OCRReader)
        cleaned = reader._clean_text("mh 12 ab 1234")
        assert cleaned == "MH12AB1234"

    def test_invalid_plate(self):
        from src.ocr.ocr_reader import OCRReader
        reader = OCRReader.__new__(OCRReader)
        result = reader._build_plate_text("XXXYYY", 0.5)
        assert not result.is_valid_format


# ── Analytics tests ───────────────────────────────────────────

class TestDensityAnalyzer:
    def setup_method(self):
        from src.analytics.density import DensityAnalyzer
        self.analyzer = DensityAnalyzer(
            road_capacity=10, frame_width=640, frame_height=480, fps=25.0
        )

    def test_zero_vehicles(self):
        snap = self.analyzer.update(0, [], {})
        assert snap.total_vehicles == 0
        assert snap.congestion_score < 0.1

    def test_congestion_score_range(self):
        from src.detection.detector import Detection
        dets = [
            Detection((i*50, 100, i*50+40, 140), 0, 'car', 0.9)
            for i in range(8)
        ]
        snap = self.analyzer.update(1, dets, {})
        assert 0.0 <= snap.congestion_score <= 1.0

    def test_congestion_level_gridlock(self):
        from src.detection.detector import Detection
        from src.analytics.density import get_congestion_level
        label, _ = get_congestion_level(0.9)
        assert label == 'GRIDLOCK'

    def test_congestion_level_free(self):
        from src.analytics.density import get_congestion_level
        label, _ = get_congestion_level(0.1)
        assert label == 'FREE'


# ── Tracker tests ─────────────────────────────────────────────

class TestTracker:
    def test_iou_zero(self):
        from src.tracking.tracker import VehicleTracker
        iou = VehicleTracker._iou((0,0,10,10), (20,20,30,30))
        assert iou == 0.0

    def test_iou_perfect(self):
        from src.tracking.tracker import VehicleTracker
        iou = VehicleTracker._iou((0,0,10,10), (0,0,10,10))
        assert abs(iou - 1.0) < 1e-6

    def test_iou_partial(self):
        from src.tracking.tracker import VehicleTracker
        iou = VehicleTracker._iou((0,0,10,10), (5,5,15,15))
        assert 0.0 < iou < 1.0

    def test_track_center(self):
        from src.tracking.tracker import Track
        t = Track(track_id=1, class_name='car', bbox=(100,200,300,400), confidence=0.9)
        assert t.center == (200, 300)


# ── Violation tests ───────────────────────────────────────────

class TestWrongSide:
    def test_insufficient_data(self):
        from src.violation.wrong_side import WrongSideDetector
        ws = WrongSideDetector()
        from collections import deque
        pts = deque([(i, 100) for i in range(3)])  # too few
        ws.update(1, pts)
        assert not ws.is_wrong_side(1)

    def test_dominant_direction(self):
        from src.violation.wrong_side import WrongSideDetector
        ws = WrongSideDetector()
        from collections import deque
        # 3 tracks moving right
        for tid in range(1, 4):
            pts = deque([(i*5, 100) for i in range(20)])
            ws.update(tid, pts)
        dominant = ws.get_dominant_direction()
        assert dominant is not None
        # Should be close to 0 degrees (rightward)
        assert abs(dominant) < 45


# ── Drawing tests ─────────────────────────────────────────────

class TestDrawing:
    def test_draw_box(self):
        from src.utils.drawing import draw_box
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_box(frame, (10, 10, 200, 100), "car 0.9")
        assert result.shape == (480, 640, 3)

    def test_put_stats_panel(self):
        from src.utils.drawing import put_stats_panel
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = put_stats_panel(frame, {'vehicles': 5, 'congestion': 'FREE'})
        assert result.shape == (480, 640, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
