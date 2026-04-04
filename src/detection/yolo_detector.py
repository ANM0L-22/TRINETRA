"""
Trinetra — Real YOLOv8 Vehicle Detector
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class RealVehicleDetector:
    """Real vehicle detection using YOLOv8."""
    
    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        0: "person",
        6: "train",
        1: "bicycle",
    }
    
    AUTO_RICKSHAW_KEYWORDS = ["auto", "tuk", "rickshaw"]
    
    def __init__(self, model_size: str = "m", device: str = "cpu"):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_size: 'n', 's', 'm', 'l', 'x'
            device: 'cpu', 'cuda', '0', '1', etc.
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        self.yolo_available = YOLO_AVAILABLE
        
        if self.yolo_available:
            try:
                model_name = f"yolov8{model_size}.pt"
                self.model = YOLO(model_name)
                self.model.to(device)
            except Exception as e:
                print(f"YOLOv8 load failed: {e}")
                self.yolo_available = False
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.45) -> Dict:
        """
        Detect vehicles in frame.
        
        Returns:
            {
                "vehicles": [
                    {"class": str, "bbox": [x1, y1, x2, y2], "conf": float, "id": int},
                    ...
                ],
                "persons": [...],
                "raw_results": results,
            }
        """
        if not self.yolo_available or self.model is None:
            return {"vehicles": [], "persons": [], "raw_results": None}
        
        try:
            results = self.model(frame, conf=conf_threshold, verbose=False)
            
            vehicles = []
            persons = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    class_name = self.VEHICLE_CLASSES.get(class_id, "unknown")
                    
                    if class_name == "person":
                        persons.append({
                            "class": "person",
                            "bbox": [x1, y1, x2, y2],
                            "conf": round(conf, 3),
                            "track_id": 0,
                        })
                    elif class_name in ["car", "bus", "truck", "motorcycle", "bicycle", "train"]:
                        # Normalize motorcycle -> bike
                        if class_name == "motorcycle":
                            class_name = "bike"
                        
                        vehicles.append({
                            "class": class_name,
                            "bbox": [x1, y1, x2, y2],
                            "conf": round(conf, 3),
                            "track_id": 0,
                        })
            
            return {
                "vehicles": vehicles,
                "persons": persons,
                "raw_results": results,
            }
        except Exception as e:
            print(f"Detection error: {e}")
            return {"vehicles": [], "persons": [], "raw_results": None}
    
    def detect_batch(self, frames: List[np.ndarray], conf_threshold: float = 0.45) -> List[Dict]:
        """Detect vehicles in batch of frames."""
        return [self.detect(frame, conf_threshold) for frame in frames]
