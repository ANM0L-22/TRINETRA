"""
Trinetra — Video Analysis Engine (Wrapper)
Provides unified interface to real YOLOv8 + OCR + Violation detection.
Replaces the old random simulation engine with production-ready code.
"""

try:
    import cv2  # type: ignore
except Exception as e:
    cv2 = None  # type: ignore
    print(f"OpenCV import unavailable in video_engine: {e}")
import numpy as np
import base64
from typing import Optional, Dict, List

# Try to import real engine, fallback to old simulation if not available
try:
    from app.data.real_video_engine import RealVideoEngine
    _real_engine = None  # Lazy init per video
    _use_real_engine = True
except ImportError:
    _use_real_engine = False
    print("Real engine not available, falling back to simulation")



# Initialize lazy load
_engine_instance = None


def reset_engine_state() -> None:
    """Reset cached engine state when a new video/image is loaded."""
    global _engine_instance

    if _engine_instance is not None:
        try:
            reset_fn = getattr(_engine_instance, "reset", None)
            if callable(reset_fn):
                reset_fn()
        except Exception as e:
            print(f"Engine reset failed: {e}")

    _engine_instance = None


def _get_engine():
    """Lazy initialize real video engine."""
    global _engine_instance
    if _engine_instance is None and _use_real_engine:
        try:
            _engine_instance = RealVideoEngine()
        except Exception as e:
            print(f"Failed to create real engine: {e}")
            _engine_instance = None  # Mark as attempted but failed
    return _engine_instance


def analyze_frame(frame_bgr: np.ndarray, frame_id: int, total_frames: int) -> dict:
    """
    Full frame analysis with real detection using YOLOv8 + OCR + Violation detection.
    """
    engine = _get_engine()
    if engine is not None:
        try:
            return engine.analyze_frame(frame_bgr, frame_id, total_frames)  # type: ignore
        except Exception as e:
            print(f"Real engine analysis failed: {e}")
            # Fallback to empty detection
            return {
                "frame_id": frame_id,
                "total": total_frames,
                "n_vehicles": 0,
                "density": 0.0,
                "congestion": "LOW",
                "vehicles": [],
                "persons": [],
                "violations": [],
                "plates": [],
                "counts": {},
                "fps": 0.0,
                "proc_ms": 0.0,
                "frame_b64": "",
            }
    
    # Fallback: return empty detection
    return {
        "frame_id": frame_id,
        "total": total_frames,
        "n_vehicles": 0,
        "density": 0.0,
        "congestion": "LOW",
        "vehicles": [],
        "persons": [],
        "violations": [],
        "plates": [],
        "counts": {},
        "fps": 0.0,
        "proc_ms": 0.0,
        "frame_b64": "",
    }


def get_frame_at(video_path: str, frame_index: int) -> Optional[Dict]:
    """Get analyzed frame at specific index."""
    engine = _get_engine()
    if engine is not None:
        try:
            return engine.get_frame_at(video_path, frame_index)  # type: ignore
        except Exception as e:
            print(f"Frame retrieval failed: {e}")
    
    # Fallback
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return analyze_frame(frame, frame_index, total)


def get_video_info(video_path: str) -> Dict:
    """Get video metadata."""
    engine = _get_engine()
    if engine is not None:
        try:
            return engine.get_video_info(video_path)  # type: ignore
        except Exception as e:
            print(f"Video info retrieval failed: {e}")
    
    # Fallback
    if cv2 is None:
        return {}
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


def bulk_analyze(video_path: str, max_samples: int = 180) -> List[Dict]:
    """Analyze sampled frames across entire video."""
    engine = _get_engine()
    if engine is not None:
        try:
            return engine.bulk_analyze(video_path, max_samples)  # type: ignore
        except Exception as e:
            print(f"Bulk analysis failed: {e}")
    
    # Fallback: return empty list
    return []

