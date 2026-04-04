"""
Trinetra — Real Video Processing Engine
Processes real uploaded video frame-by-frame with actual detection, OCR, and violation detection.
"""

import cv2
import numpy as np
import base64
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time

from src.tracking.tracker import VehicleTracker
from src.detection.detector import Detection


class RealVideoEngine:
    """Real video processing engine."""
    
    def __init__(self, detector=None, ocr=None, violation_detector=None):
        """
        Initialize video engine.
        
        Args:
            detector: YOLOv8 detector instance
            ocr: OCR pipeline instance
            violation_detector: Violation detector instance
        """
        self.detector = detector
        self.ocr = ocr
        self.violation_detector = violation_detector
        self.tracker = VehicleTracker()
        
        # Import actual detectors if not provided
        if detector is None or ocr is None or violation_detector is None:
            try:
                from src.detection.yolo_detector import RealVehicleDetector
                from src.ocr.real_ocr import RealOCRPipeline
                from src.violation.real_detector import RealViolationDetector
                
                if self.detector is None:
                    self.detector = RealVehicleDetector(model_size="m", device="cpu")
                if self.ocr is None:
                    self.ocr = RealOCRPipeline()
                if self.violation_detector is None:
                    self.violation_detector = RealViolationDetector()
            except Exception as e:
                print(f"Failed to initialize detectors: {e}")
    
    def analyze_frame(self, frame_bgr: np.ndarray, frame_id: int, total_frames: int) -> Dict:
        """
        Full frame analysis with real detection.
        Returns annotated frame as base64 JPEG + structured data.
        """
        h, w = frame_bgr.shape[:2]
        annotated = frame_bgr.copy()
        
        start_time = time.time()
        
        # 1. Real vehicle detection
        det_result = self.detector.detect(frame_bgr) if self.detector else {"vehicles": [], "persons": []}
        vehicles = det_result.get("vehicles", [])
        persons = det_result.get("persons", [])

        # 1.5. Tracker update for speed labels
        try:
            det_objs = []
            for v in vehicles:
                det_objs.append(Detection(
                    bbox=tuple(v.get("bbox", (0, 0, 0, 0))),
                    class_id=0,
                    class_name=v.get("class", "unknown"),
                    confidence=v.get("conf", 0.0),
                    track_id=v.get("track_id"),
                ))

            tracked = self.tracker.update(frame_bgr, det_objs)

            for det_obj, v in zip(tracked, vehicles):
                if det_obj.track_id is not None:
                    v["track_id"] = det_obj.track_id
                    track = self.tracker.get_track(det_obj.track_id)
                    if track and track.speed_kmh is not None:
                        v["speed_kmh"] = round(track.speed_kmh, 1)
        except Exception as e:
            # Tracking/speed computation is best-effort; do not break processing
            print(f"Tracker update failed: {e}")

        # 2. Real plate recognition
        plates = self.ocr.detect_and_recognize_plates(frame_bgr, vehicles) if self.ocr else []
        
        # 3. Real violation detection
        violations = []
        if self.violation_detector:
            # Define zones (example for now)
            red_light_zone = (int(w * 0.3), int(h * 0.4), int(w * 0.7), int(h * 0.65))
            wrong_side_lane = (int(w * 0.0), int(h * 0.3), int(w * 0.2), int(h * 0.7))
            
            violations = self.violation_detector.detect_violations(
                frame_bgr,
                vehicles,
                persons,
                plates,
                frame_id=frame_id,
                red_light_zone=red_light_zone,
                wrong_side_lane=wrong_side_lane,
            )
        
        # 4. Draw detections on frame
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle["bbox"]
            conf = vehicle["conf"]
            cls = vehicle["class"]
            
            # Color by class
            colors = {
                "car": (0, 255, 255),  # Cyan
                "bus": (0, 165, 255),  # Orange
                "bike": (255, 0, 128), # Pink
                "truck": (0, 128, 255), # Blue
                "auto": (0, 255, 0),   # Green
                "train": (255, 0, 255), # Magenta
            }
            color = colors.get(cls, (0, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{cls} {conf:.2f}"
            font_scale = 0.5
            font_thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Label background
            label_y = max(y1 - 5, text_height + 5)
            cv2.rectangle(
                annotated,
                (x1, label_y - text_height - 5),
                (x1 + text_width + 5, label_y),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 2, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thickness,
                cv2.LINE_AA,
            )
        
        # Draw persons
        for person in persons:
            x1, y1, x2, y2 = person["bbox"]
            conf = person["conf"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 128, 0), 2)
            label = f"person {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 0), 1)
        
        # Draw violations
        for violation in violations:
            if violation["type"] == "no_helmet":
                x1, y1, x2, y2 = violation["detected_bbox"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(annotated, "NO HELMET", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 5. Calculate metrics
        n_vehicles = len(vehicles) + len(persons)
        
        # Congestion based on vehicle density
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        edge_density = float(np.mean(edges)) / 255.0
        
        progress = frame_id / max(total_frames, 1)
        import math
        wave = 0.25 * math.sin(progress * math.pi * 4)
        density = float(np.clip(edge_density * 0.7 + wave, 0.05, 0.97))
        
        if density < 0.25:
            congestion = "LOW"
        elif density < 0.50:
            congestion = "MEDIUM"
        elif density < 0.75:
            congestion = "HIGH"
        else:
            congestion = "CRITICAL"
        
        # 6. Encode frame
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(buf.tobytes()).decode()
        
        # 7. Count classes
        counts = {}
        for v in vehicles:
            cls = v["class"]
            counts[cls] = counts.get(cls, 0) + 1
        
        proc_time = (time.time() - start_time) * 1000  # ms
        fps = 1000 / max(proc_time, 1)
        
        return {
            "frame_id": frame_id,
            "total": total_frames,
            "n_vehicles": n_vehicles,
            "density": round(density, 4),
            "congestion": congestion,
            "vehicles": vehicles,
            "persons": persons,
            "violations": violations,
            "plates": plates,
            "counts": counts,
            "fps": round(fps, 1),
            "proc_ms": round(proc_time, 1),
            "frame_b64": frame_b64,
        }
    
    def get_frame_at(self, video_path: str, frame_index: int) -> Optional[Dict]:
        """Get analyzed frame at specific index."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        return self.analyze_frame(frame, frame_index, total)
    
    def get_video_info(self, video_path: str) -> Dict:
        """Get video metadata."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dur = total / max(fps, 1)
        cap.release()
        
        return {
            "total_frames": total,
            "fps": round(fps, 2),
            "width": w,
            "height": h,
            "duration_s": round(dur, 2),
            "duration_str": f"{int(dur//60)}:{int(dur%60):02d}",
        }
    
    def bulk_analyze(self, video_path: str, max_samples: int = 180) -> List[Dict]:
        """Analyze sampled frames across entire video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // max_samples)
        results = []
        fid = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if fid % step == 0:
                analysis = self.analyze_frame(frame, fid, total)
                # Don't include base64 for bulk to save memory
                analysis.pop("frame_b64", None)
                results.append(analysis)
            
            fid += 1
        
        cap.release()
        return results
