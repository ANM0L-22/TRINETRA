"""
Trinetra — Real OCR Neural Number Plate Recognition
Detects and recognizes Indian number plates using PaddleOCR + YOLO
"""

import numpy as np
import cv2
import re
from typing import List, Dict, Tuple, Optional


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
        
        if use_paddleocr:
            try:
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
            except ImportError:
                print("PaddleOCR not available, will use fallback")
                self.use_paddleocr = False
    
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
        
        # Preprocess
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        
        if self.use_paddleocr and self.ocr:
            try:
                result = self.ocr.ocr(thresh, cls=False)
                texts = []
                conf_sum = 0
                
                if result and result[0]:
                    for item in result[0]:
                        text = item[1][0] if isinstance(item[1], tuple) else item[1]
                        conf = item[1][1] if isinstance(item[1], tuple) else 0.9
                        texts.append(text)
                        conf_sum += conf
                
                raw_text = "".join(texts).strip()
                avg_conf = conf_sum / len(texts) if texts else 0.0
                
            except Exception as e:
                print(f"OCR error: {e}")
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
        
        # Match format: 2 letters, 2 digits, 2 letters, 4 digits
        match = re.match(r'^([A-Z]{2})(\d{2})([A-Z]{2})(\d{4})$', text)
        
        if match:
            state, district, letters, numbers = match.groups()
            if state in self.INDIAN_STATES:
                plate = f"{state}{district}{letters}{numbers}"
                return plate, True
        
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
            
            # Plate is typically inside the lower region of the vehicle bbox
            h_box = max(1, y2 - y1)
            plate_y1 = max(y1, y2 - int(h_box * 0.25))
            plate_y2 = y2
            plate_x1 = max(0, x1 - int((x2 - x1) * 0.05))
            plate_x2 = min(frame.shape[1], x2 + int((x2 - x1) * 0.05))
            
            if plate_y1 < plate_y2 and plate_x1 < plate_x2:
                plate_roi = frame[plate_y1:plate_y2, plate_x1:plate_x2]
                
                plate_info = self.extract_plate_from_roi(plate_roi)
                plates.append({
                    "vehicle_bbox": det["bbox"],
                    "vehicle_class": det["class"],
                    "vehicle_conf": det["conf"],
                    "track_id": det.get("track_id"),
                    **plate_info,
                })
        
        return plates
