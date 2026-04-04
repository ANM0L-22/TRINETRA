"""
Trinetra — Real OCR Neural Number Plate Recognition
Detects and recognizes Indian number plates using PaddleOCR + YOLO
"""

import numpy as np
import cv2
import re
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from src.ocr.plate_detector import PlateDetector


class RealOCRPipeline:
    """Real OCR pipeline for Indian number plates."""
    
    INDIAN_STATES = [
        "AN", "AP", "AR", "AS", "BR", "CG", "CH", "CT", "DA", "DD", "DL",
        "GA", "GJ", "HR", "HP", "JK", "JH", "KA", "KL", "LD", "LA", "LE",
        "MH", "ML", "MN", "MI", "MZ", "NL", "OD", "OL", "PB", "PY", "RJ",
        "SK", "TG", "TN", "TR", "TT", "UK", "UP", "UR", "WB",
    ]
    
    def __init__(self, use_paddleocr: bool = True):
        """Initialize OCR pipeline."""
        self.use_paddleocr = use_paddleocr
        self.ocr = None
        self.plate_detector = None
        self._ocr_fail_count = 0

        base_dir = Path(__file__).resolve().parents[2]
        plate_det_weights = base_dir / "models" / "weights" / "plate_detector_best.pt"
        tamper_cls_weights = base_dir / "models" / "weights" / "tamper_cls_best.pt"
        self.plate_detector = PlateDetector(
            weights_path=str(plate_det_weights) if plate_det_weights.exists() else None,
            tamper_cls_weights_path=str(tamper_cls_weights) if tamper_cls_weights.exists() else None,
            conf_threshold=0.35,
            device="cpu",
        )
        
        if use_paddleocr:
            try:
                # Reduce Paddle/oneDNN info log spam in terminal.
                os.environ.setdefault("GLOG_minloglevel", "2")
                os.environ.setdefault("FLAGS_logtostderr", "1")
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(use_textline_orientation=True, lang='en')
            except ImportError:
                print("PaddleOCR not available, will use fallback")
                self.use_paddleocr = False

    @staticmethod
    def _parse_ocr_item(item) -> Tuple[str, float]:
        """Parse PaddleOCR output item across multiple output formats safely."""
        text = ""
        conf = 0.0

        # Common format: [points, (text, conf)]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            payload = item[1]
            if isinstance(payload, (list, tuple)) and len(payload) >= 1:
                if isinstance(payload[0], str):
                    text = payload[0]
                if len(payload) >= 2:
                    try:
                        conf = float(payload[1])
                    except Exception:
                        conf = 0.0
            elif isinstance(payload, str):
                text = payload

        # Alternative format: {'text': 'XX', 'score': 0.9}
        if not text and isinstance(item, dict):
            text = str(item.get("text", "") or "")
            try:
                conf = float(item.get("score", item.get("confidence", 0.0)) or 0.0)
            except Exception:
                conf = 0.0

        return text.strip(), conf
    
    def extract_plate_from_roi(self, roi: np.ndarray, debug: bool = False) -> Dict:
        """
        Extract number plate from ROI.
        
        Returns:
            {
                "plate": "MH02AB1234",
                "confidence": float,
                "raw_text": str,
                "valid": bool,
            }
        """
        if roi is None or roi.size == 0:
            return {
                "plate": None,
                "confidence": 0.0,
                "raw_text": "",
                "valid": False,
            }
        
        # Normalize ROI channels and filter unusable crops early.
        try:
            if len(roi.shape) == 2:
                gray = roi
                roi_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            elif len(roi.shape) == 3 and roi.shape[2] == 3:
                roi_color = roi
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                return {
                    "plate": None,
                    "confidence": 0.0,
                    "raw_text": "",
                    "valid": False,
                }
        except Exception:
            return {
                "plate": None,
                "confidence": 0.0,
                "raw_text": "",
                "valid": False,
            }

        h, w = gray.shape[:2]
        if h < 14 or w < 40:
            return {
                "plate": None,
                "confidence": 0.0,
                "raw_text": "",
                "valid": False,
            }

        # Contrast enhancement + thresholding for clearer OCR.
        gray = cv2.equalizeHist(gray)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            10,
        )
        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        if self.use_paddleocr and self.ocr:
            try:
                # Try multiple inputs because PaddleOCR behavior varies by crop quality/version.
                result = None
                for candidate in (thresh_bgr, roi_color):
                    try:
                        result = self.ocr.ocr(candidate, cls=True)
                        if result:
                            break
                    except Exception:
                        result = None
                texts = []
                conf_sum = 0
                
                if result and result[0]:
                    for item in result[0]:
                        text, conf = self._parse_ocr_item(item)
                        if text:
                            texts.append(text)
                            conf_sum += conf
                
                raw_text = "".join(texts).strip()
                avg_conf = conf_sum / len(texts) if texts else 0.0
                
            except Exception as e:
                self._ocr_fail_count += 1
                # Avoid terminal spam: print first few failures and then periodic updates.
                if self._ocr_fail_count <= 3 or self._ocr_fail_count % 50 == 0:
                    print(f"OCR error (recoverable): {type(e).__name__}: {e}")
                raw_text = ""
                avg_conf = 0.0
        else:
            raw_text = ""
            avg_conf = 0.0
        
        # Validate and normalize
        plate, valid = self._validate_indian_plate(raw_text)
        
        return {
            "plate": plate,
            "confidence": round(avg_conf, 3),
            "raw_text": raw_text,
            "valid": valid,
        }
    
    def _validate_indian_plate(self, text: str) -> Tuple[Optional[str], bool]:
        """
        Validate Indian number plate format.
        Format: STATE(2) + DISTRICT(2) + LETTERS(2) + NUMBERS(4)
        Example: MH02AB1234
        """
        # Remove spaces and special chars
        text = re.sub(r'[^A-Za-z0-9]', '', text.upper())
        if not text:
            return None, False

        # Common OCR confusion maps.
        to_digit = str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8"})
        to_alpha = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"})

        # Strict canonical first: SSDDLLNNNN
        match = re.match(r'^([A-Z]{2})(\d{2})([A-Z]{2})(\d{4})$', text)
        if match:
            state, district, letters, numbers = match.groups()
            if state in self.INDIAN_STATES:
                return f"{state}{district}{letters}{numbers}", True

        # Tolerant parse with position-aware normalization.
        if len(text) >= 8:
            state = text[0:2].translate(to_alpha)
            district = text[2:4].translate(to_digit)
            series = text[4:6].translate(to_alpha)
            numbers = text[6:].translate(to_digit)
            numbers = re.sub(r'\D', '', numbers)

            if state in self.INDIAN_STATES and district.isdigit() and series.isalpha() and 3 <= len(numbers) <= 4:
                # Left-pad to 4 digits for normalized storage.
                normalized_num = numbers.zfill(4)
                return f"{state}{district}{series}{normalized_num}", True
        
        return None, False
    
    def detect_and_recognize_plates(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Given frame and vehicle detections, extract and recognize plates.
        
        Args:
            frame: Input image
            detections: List of vehicle detections with bbox
        
        Returns:
            List of detections with plate info added
        """
        plates = []
        
        for det in detections:
            if det["class"] == "person":  # Skip persons
                continue
            
            x1, y1, x2, y2 = det["bbox"]
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(frame.shape[1], int(x2))
            y2 = min(frame.shape[0], int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Use dedicated plate detector first; fallback to heuristic ROIs.
            plate_regions = []
            if self.plate_detector is not None:
                try:
                    plate_regions = self.plate_detector.detect_in_vehicle_crop(frame, (x1, y1, x2, y2))
                except Exception:
                    plate_regions = []

            best = {
                "plate": None,
                "confidence": 0.0,
                "raw_text": "",
                "valid": False,
                "is_tampered": False,
            }

            if plate_regions:
                # Run OCR on best detected plate regions.
                plate_regions = sorted(plate_regions, key=lambda p: p.confidence, reverse=True)[:3]
                for region in plate_regions:
                    crop = region.crop
                    if crop is None or crop.size == 0:
                        continue
                    plate_info = self.extract_plate_from_roi(crop)
                    plate_info["is_tampered"] = bool(region.is_tampered)
                    if plate_info.get("valid"):
                        best = plate_info
                        break
                    if plate_info.get("confidence", 0.0) > best.get("confidence", 0.0):
                        best = plate_info
            else:
                # Heuristic fallback when detector doesn't find plate region.
                w_box = x2 - x1
                h_box = y2 - y1
                if w_box < 40 or h_box < 24:
                    continue

                candidates = []
                lower_y1 = y1 + int(h_box * 0.55)
                lower_y2 = y1 + int(h_box * 0.92)
                candidates.append(frame[lower_y1:lower_y2, x1:x2])
                candidates.append(frame[y1:y2, x1:x2])
                mid_y1 = y1 + int(h_box * 0.45)
                mid_y2 = y1 + int(h_box * 0.78)
                candidates.append(frame[mid_y1:mid_y2, x1:x2])

                for candidate in candidates:
                    if candidate is None or candidate.size == 0:
                        continue
                    plate_info = self.extract_plate_from_roi(candidate)
                    if plate_info.get("valid"):
                        best = plate_info
                        break
                    if plate_info.get("confidence", 0.0) > best.get("confidence", 0.0):
                        best = plate_info

            plates.append({
                "vehicle_bbox": det["bbox"],
                "vehicle_class": det["class"],
                "vehicle_conf": det["conf"],
                **best,
            })
        
        return plates
