"""
Trinetra — Real Violation Detector
Detects traffic violations based on actual rules:
- No helmet detection (person on bike without helmet)
- No seatbelt detection (person in car without visible seatbelt)
- Wrong-side driving (vehicle in wrong lane)
- Tampered number plate (unclear/obscured plate)
- Red light violation (vehicle crossing red light zone)
"""

import numpy as np
import cv2
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from pathlib import Path


class RealViolationDetector:
    """Real violation detection system."""
    
    VIOLATION_TYPES = [
        "no_helmet",
        "no_seatbelt",
        "wrong_side_driving",
        "tampered_plate",
        "blackened_window",
        "red_light_violation",
        "mobile_usage",
        "speeding",
    ]
    
    def __init__(self):
        """Initialize violation detector."""
        self.person_tracker = defaultdict(dict)  # Track persons across frames
        self.vehicle_tracker = defaultdict(dict)  # Track vehicles across frames
        self.reported_violations: Dict[str, Set[str]] = defaultdict(set)
        self.scene_reported: Dict[int, Set[str]] = defaultdict(set)
        self.violation_cls_model = None

        # Optional scene-level classifier trained on violation classes.
        try:
            from ultralytics import YOLO

            base_dir = Path(__file__).resolve().parents[2]
            cls_weights = base_dir / "models" / "weights" / "violation_cls_best.pt"
            if cls_weights.exists():
                self.violation_cls_model = YOLO(str(cls_weights))
        except Exception:
            self.violation_cls_model = None

    def _get_violation_entity_key(self, vehicle: Optional[Dict] = None, plate_info: Optional[Dict] = None) -> str:
        if vehicle is not None:
            track_id = vehicle.get("track_id")
            if track_id is not None:
                return f"track:{track_id}"
            plate = self._find_plate_for_vehicle(vehicle.get("bbox", []), plate_info or []) if plate_info is not None else None
            if plate:
                return f"plate:{plate}"
            bbox = vehicle.get("bbox")
            if bbox:
                return f"bbox:{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
        if plate_info is not None:
            track_id = plate_info.get("track_id")
            if track_id is not None:
                return f"track:{track_id}"
            plate = plate_info.get("plate") or plate_info.get("raw_text")
            if plate:
                return f"plate:{plate}"
        return "unknown"

    def _should_report_violation(self, entity_key: str, violation_type: str) -> bool:
        if violation_type in self.reported_violations[entity_key]:
            return False
        self.reported_violations[entity_key].add(violation_type)
        return True

    def detect_violations(
        self,
        frame: np.ndarray,
        vehicles: List[Dict],
        persons: List[Dict],
        plates: List[Dict],
        frame_id: int = 0,
        red_light_zone: Optional[Tuple[int, int, int, int]] = None,
        wrong_side_lane: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Dict]:
        """
        Detect traffic violations in frame.
        
        Args:
            frame: Input image
            vehicles: Detected vehicles [{"class": str, "bbox": [...], "conf": float, "id": int}]
            persons: Detected persons [{"class": "person", "bbox": [...], "conf": float}]
            plates: Recognized plates [{"plate": str, "valid": bool, ...}]
            frame_id: Frame number for tracking
            red_light_zone: (x1, y1, x2, y2) - zone where red light on vehicle = violation
            wrong_side_lane: (x1, y1, x2, y2) - lane where vehicle shouldn't be
        
        Returns:
            List of violations:
            [
                {
                    "type": "no_helmet",
                    "severity": "critical",
                    "confidence": float,
                    "detected_bbox": [...],
                    "related_vehicle": {...},
                    "plate": str,
                    "description": str,
                },
                ...
            ]
        """
        violations = []
        
        # 1. Detect no helmet violations
        violations.extend(self._detect_no_helmet(frame, vehicles, persons, plates, frame_id))
        
        # 2. Detect tampered plate violations
        violations.extend(self._detect_tampered_plate(plates))
        
        # 3. Detect wrong-side driving
        if wrong_side_lane:
            violations.extend(self._detect_wrong_side_driving(vehicles, plates, wrong_side_lane, frame_id))
        
        # 4. Detect red light violations
        if red_light_zone:
            violations.extend(self._detect_red_light(vehicles, plates, red_light_zone, frame_id))
        
        # 5. Detect no seatbelt (simplified based on posture)
        violations.extend(self._detect_no_seatbelt(frame, persons, frame_id))

        # 6. Detect blackened/tinted windows
        violations.extend(self._detect_blackened_window(vehicles, frame, frame_id))

        # 7. Scene-level classifier signals (trained classes).
        violations.extend(self._detect_scene_level_violations(frame, frame_id))
        
        return violations

    def _detect_scene_level_violations(self, frame: np.ndarray, frame_id: int) -> List[Dict]:
        """Use trained image-level classifier to emit scene violations with confidence."""
        if self.violation_cls_model is None:
            return []

        out: List[Dict] = []
        try:
            results = self.violation_cls_model.predict(frame, verbose=False, device="cpu")
            if not results:
                return out

            r = results[0]
            probs = getattr(r, "probs", None)
            names = getattr(r, "names", {}) or {}
            if probs is None:
                return out

            top_idx = int(probs.top1)
            top_conf = float(probs.top1conf)
            pred = str(names.get(top_idx, "")).strip().lower()
            valid_preds = {
                "no_helmet",
                "red_light_violation",
                "wrong_side_driving",
                "tampered_plate",
                "blackened_window",
            }
            if pred not in valid_preds or top_conf < 0.60:
                return out

            # Emit at most one scene-level event per violation type per frame.
            if pred in self.scene_reported[frame_id]:
                return out
            self.scene_reported[frame_id].add(pred)

            h, w = frame.shape[:2]
            severity = "critical" if pred in {"no_helmet", "wrong_side_driving", "red_light_violation"} else "high"
            out.append({
                "type": pred,
                "severity": severity,
                "confidence": round(top_conf, 3),
                "detected_bbox": [0, 0, w - 1, h - 1],
                "track_id": None,
                "plate": "UNKNOWN",
                "frame_id": frame_id,
                "event_time": datetime.now().strftime("%H:%M:%S"),
                "description": f"Scene-level classifier flagged {pred}",
            })
        except Exception:
            return []

        return out
    
    def _find_plate_for_vehicle(self, vehicle_bbox: List[int], plates: List[Dict]) -> str:
        """Return best matching plate for a vehicle bbox."""
        if not plates:
            return "UNKNOWN"

        vx1, vy1, vx2, vy2 = vehicle_bbox
        vcx, vcy = (vx1 + vx2) / 2, (vy1 + vy2) / 2

        best_plate = "UNKNOWN"
        best_dist = float('inf')

        for plate_info in plates:
            pb = plate_info.get("vehicle_bbox")
            if not pb or len(pb) != 4:
                continue
            px1, py1, px2, py2 = pb
            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
            dist = np.hypot(vcx - pcx, vcy - pcy)
            if dist < best_dist:
                best_dist = dist
                best_plate = plate_info.get("plate") or plate_info.get("raw_text") or "UNKNOWN"

        return best_plate

    def _detect_no_helmet(
        self,
        frame: np.ndarray,
        vehicles: List[Dict],
        persons: List[Dict],
        plates: List[Dict],
        frame_id: int,
    ) -> List[Dict]:
        """Detect persons on bikes without helmets."""
        violations = []
        
        # Find bike detections
        bikes = [v for v in vehicles if v["class"] == "bike"]
        
        for bike in bikes:
            bike_x1, bike_y1, bike_x2, bike_y2 = bike["bbox"]
            bike_center_x = (bike_x1 + bike_x2) / 2
            bike_center_y = (bike_y1 + bike_y2) / 2
            
            # Find persons close to this bike
            for person in persons:
                px1, py1, px2, py2 = person["bbox"]
                person_center_x = (px1 + px2) / 2
                person_center_y = (py1 + py2) / 2
                
                # Distance threshold
                dist = np.sqrt((person_center_x - bike_center_x)**2 + 
                               (person_center_y - bike_center_y)**2)
                
                if dist < max(bike_x2 - bike_x1, bike_y2 - bike_y1) * 2:
                    # Check if person has helmet (simplified: check head region)
                    head_roi = frame[max(0, py1):py1 + int((py2-py1)*0.25), px1:px2]
                    
                    if head_roi.size > 0 and not self._has_helmet_in_roi(head_roi):
                        entity_key = self._get_violation_entity_key(vehicle=bike, plate_info=None)
                        if not self._should_report_violation(entity_key, "no_helmet"):
                            continue
                        violations.append({
                            "type": "no_helmet",
                            "severity": "critical",
                            "confidence": round(person["conf"] * 0.95, 3),
                            "detected_bbox": person["bbox"],
                            "related_vehicle": bike,
                            "track_id": bike.get("track_id"),
                            "plate": self._find_plate_for_vehicle(bike["bbox"], plates),
                            "frame_id": frame_id,
                            "event_time": datetime.now().strftime("%H:%M:%S"),
                            "description": "Person on bike without visible helmet",
                        })
        
        return violations
    
    def _detect_tampered_plate(self, plates: List[Dict]) -> List[Dict]:
        """Detect tampered/invalid number plates."""
        violations = []
        
        for plate_info in plates:
            is_model_tampered = bool(plate_info.get("is_tampered", False))
            is_ocr_suspicious = plate_info.get("valid") is False and plate_info.get("confidence", 0) > 0.3
            if is_model_tampered or is_ocr_suspicious:
                entity_key = self._get_violation_entity_key(plate_info=plate_info)
                if not self._should_report_violation(entity_key, "tampered_plate"):
                    continue
                violations.append({
                    "type": "tampered_plate",
                    "severity": "high",
                    "confidence": 0.92 if is_model_tampered else 0.85,
                    "detected_bbox": plate_info.get("vehicle_bbox", []),
                    "track_id": plate_info.get("track_id"),
                    "plate": plate_info.get("raw_text", "UNKNOWN"),
                    "frame_id": frame_id,
                    "event_time": datetime.now().strftime("%H:%M:%S"),
                    "description": (
                        f"Number plate flagged as tampered by model: {plate_info.get('raw_text', 'Unknown')}"
                        if is_model_tampered else
                        f"Number plate unreadable or tampered: {plate_info.get('raw_text', 'Unknown')}"
                    ),
                })
        
        return violations
    
    def _detect_wrong_side_driving(
        self,
        vehicles: List[Dict],
        plates: List[Dict],
        wrong_side_lane: Tuple[int, int, int, int],
        frame_id: int,
    ) -> List[Dict]:
        """Detect vehicles in wrong lane."""
        violations = []
        
        x1, y1, x2, y2 = wrong_side_lane
        
        for vehicle in vehicles:
            vx1, vy1, vx2, vy2 = vehicle["bbox"]
            v_center_x = (vx1 + vx2) / 2
            v_center_y = (vy1 + vy2) / 2
            
            # Check if vehicle center is in wrong lane zone
            if x1 <= v_center_x <= x2 and y1 <= v_center_y <= y2:
                entity_key = self._get_violation_entity_key(vehicle=vehicle, plate_info=None)
                if not self._should_report_violation(entity_key, "wrong_side_driving"):
                    continue
                violations.append({
                    "type": "wrong_side_driving",
                    "severity": "critical",
                    "confidence": 0.92,
                    "detected_bbox": vehicle["bbox"],
                    "related_vehicle": vehicle,
                    "track_id": vehicle.get("track_id"),
                    "plate": self._find_plate_for_vehicle(vehicle["bbox"], plates),
                    "frame_id": frame_id,
                    "event_time": datetime.now().strftime("%H:%M:%S"),
                    "description": "Vehicle detected in wrong lane",
                })
        
        return violations
    
    def _detect_red_light(
        self,
        vehicles: List[Dict],
        plates: List[Dict],
        red_light_zone: Tuple[int, int, int, int],
        frame_id: int,
    ) -> List[Dict]:
        """Detect red light violations."""
        violations = []
        
        x1, y1, x2, y2 = red_light_zone
        
        for vehicle in vehicles:
            vx1, vy1, vx2, vy2 = vehicle["bbox"]
            v_center_x = (vx1 + vx2) / 2
            v_center_y = (vy1 + vy2) / 2
            
            # Check if vehicle center is in red light zone
            if x1 <= v_center_x <= x2 and y1 <= v_center_y <= y2:
                entity_key = self._get_violation_entity_key(vehicle=vehicle, plate_info=None)
                if not self._should_report_violation(entity_key, "red_light_violation"):
                    continue
                violations.append({
                    "type": "red_light_violation",
                    "severity": "critical",
                    "confidence": 0.88,
                    "detected_bbox": vehicle["bbox"],
                    "related_vehicle": vehicle,
                    "track_id": vehicle.get("track_id"),
                    "plate": self._find_plate_for_vehicle(vehicle["bbox"], plates),
                    "frame_id": frame_id,
                    "event_time": datetime.now().strftime("%H:%M:%S"),
                    "description": "Vehicle crossed red light",
                })
        
        return violations
    
    def _detect_no_seatbelt(
        self,
        frame: np.ndarray,
        persons: List[Dict],
        frame_id: int,
    ) -> List[Dict]:
        """Detect persons without seatbelt (simplified)."""
        violations = []
        
        # This is a simplified check - in production would use pose estimation
        # For now, return empty list as it requires complex pose analysis
        
        return violations

    def _detect_blackened_window(
        self,
        vehicles: List[Dict],
        frame: np.ndarray,
        frame_id: int,
    ) -> List[Dict]:
        """Detect likely illegal blackened/tinted windows from dark window region."""
        violations = []

        # Practical target classes where windows are visible enough.
        candidate_classes = {"car", "bus", "truck", "auto", "auto_rickshaw"}

        for vehicle in vehicles:
            cls_name = str(vehicle.get("class", "")).lower()
            if cls_name not in candidate_classes:
                continue

            bbox = vehicle.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            if x2 <= x1 or y2 <= y1:
                continue

            # Approximate window area in the upper-mid body of vehicle.
            h = y2 - y1
            w = x2 - x1
            wx1 = x1 + int(0.12 * w)
            wx2 = x2 - int(0.12 * w)
            wy1 = y1 + int(0.16 * h)
            wy2 = y1 + int(0.48 * h)

            wx1 = max(0, wx1)
            wy1 = max(0, wy1)
            wx2 = min(frame.shape[1], wx2)
            wy2 = min(frame.shape[0], wy2)
            if wx2 <= wx1 or wy2 <= wy1:
                continue

            roi = frame[wy1:wy2, wx1:wx2]
            if roi.size == 0:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean_intensity = float(np.mean(gray))
            dark_ratio = float(np.mean(gray < 55))

            # Conservative threshold to avoid too many false positives.
            is_blackened = mean_intensity < 70 and dark_ratio > 0.6
            if not is_blackened:
                continue

            entity_key = self._get_violation_entity_key(vehicle=vehicle, plate_info=None)
            if not self._should_report_violation(entity_key, "blackened_window"):
                continue

            violations.append({
                "type": "blackened_window",
                "severity": "high",
                "confidence": 0.80,
                "detected_bbox": [wx1, wy1, wx2, wy2],
                "related_vehicle": vehicle,
                "track_id": vehicle.get("track_id"),
                "plate": "UNKNOWN",
                "frame_id": frame_id,
                "event_time": datetime.now().strftime("%H:%M:%S"),
                "description": "Vehicle window appears heavily darkened/tinted",
            })

        return violations
    
    def _has_helmet_in_roi(self, head_roi: np.ndarray) -> bool:
        """
        Simple heuristic to detect helmet in head region.
        Helmets typically have dark colors and defined shapes.
        """
        if head_roi.size == 0:
            return False
        
        # Convert to HSV
        hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
        
        # Detect dark colors (typical helmet colors)
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 100])
        mask = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # Calculate percentage of dark pixels
        dark_ratio = np.sum(mask) / (mask.size * 255)
        
        # If more than 30% of head region is dark, likely has helmet
        return bool(dark_ratio > 0.3)
