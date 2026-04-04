"""
Trinetra — OCR Reader
Reads text from license plate crops.
Uses PaddleOCR as primary engine with regex post-processing
for Indian plate formats.
"""

import re
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from loguru import logger


# ── Indian plate format patterns ─────────────────────────────
# Format: XX 00 XX 0000  (state code, district, series, number)
PLATE_PATTERNS = [
    r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$',     # Standard: MH12AB1234
    r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{1,4}$',    # Partial match
    r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$',  # Older format
    r'^\d{2}BH\d{4}[A-Z]{2}$',              # BH (Bharat) series
]

COMPILED_PATTERNS = [re.compile(p) for p in PLATE_PATTERNS]

# Common OCR misread corrections
OCR_CORRECTIONS = {
    '0': 'O', 'O': '0',   # context-dependent, handled in post-process
    '1': 'I', 'I': '1',
    '8': 'B', 'B': '8',
    '5': 'S', 'S': '5',
    '6': 'G', 'G': '6',
}


@dataclass
class PlateText:
    """OCR result for a single plate."""
    raw_text: str
    cleaned_text: str
    confidence: float
    is_valid_format: bool
    state_code: Optional[str] = None
    district_code: Optional[str] = None

    @property
    def display(self) -> str:
        if len(self.cleaned_text) >= 4:
            t = self.cleaned_text
            return f"{t[:2]} {t[2:4]} {t[4:6]} {t[6:]}" if len(t) > 6 else t
        return self.cleaned_text


class OCRReader:
    """
    Extracts license plate text from cropped plate images.

    Uses PaddleOCR when available, falls back to Tesseract or
    simple template matching.
    """

    def __init__(self, use_gpu: bool = False):
        self.ocr = None
        self._init_paddleocr(use_gpu)

    def _init_paddleocr(self, use_gpu: bool):
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=use_gpu,
                show_log=False,
                rec_algorithm='SVTR_LCNet',
            )
            logger.info("OCR engine: PaddleOCR")
        except ImportError:
            logger.warning("PaddleOCR not installed. pip install paddleocr paddlepaddle")
            self._init_tesseract()

    def _init_tesseract(self):
        try:
            import pytesseract  # type: ignore
            self._pytesseract = pytesseract
            logger.info("OCR engine: Tesseract (fallback)")
        except ImportError:
            logger.warning("No OCR engine available. Install paddleocr or pytesseract.")
            self._pytesseract = None

    # ── Public API ────────────────────────────────────────────

    def read_plate(self, plate_crop: np.ndarray) -> Optional[PlateText]:
        """Extract text from a plate crop image."""
        if plate_crop is None or plate_crop.size == 0:
            return None

        preprocessed = self._preprocess(plate_crop)

        if self.ocr:
            return self._read_paddle(preprocessed)
        elif hasattr(self, '_pytesseract') and self._pytesseract:
            return self._read_tesseract(preprocessed)
        else:
            return None

    def read_plate_multi(self, plate_crop: np.ndarray, n_attempts: int = 3) -> Optional[PlateText]:
        """
        Read plate with multiple preprocessing variants and pick best result.
        More robust than single read.
        """
        variants = self._get_preprocessing_variants(plate_crop)
        results = []

        for img in variants[:n_attempts]:
            result = self.read_plate(img)
            if result and result.confidence > 0.3:
                results.append(result)

        if not results:
            return None

        # Return highest confidence valid result, or highest conf overall
        valid = [r for r in results if r.is_valid_format]
        if valid:
            return max(valid, key=lambda r: r.confidence)
        return max(results, key=lambda r: r.confidence)

    # ── PaddleOCR ────────────────────────────────────────────

    def _read_paddle(self, img: np.ndarray) -> Optional[PlateText]:
        try:
            if self.ocr is None:
                return None
            result = self.ocr.ocr(img, cls=True)
            if not result or not result[0]:
                return None

            texts = []
            confs = []
            for line in result[0]:
                if line and len(line) >= 2:
                    text_info = line[1]
                    if text_info:
                        texts.append(text_info[0])
                        confs.append(float(text_info[1]))

            if not texts:
                return None

            raw = ' '.join(texts).strip()
            conf = float(np.mean(confs)) if confs else 0.0
            return self._build_plate_text(raw, conf)

        except Exception as e:
            logger.debug(f"PaddleOCR error: {e}")
            return None

    # ── Tesseract fallback ───────────────────────────────────

    def _read_tesseract(self, img: np.ndarray) -> Optional[PlateText]:
        try:
            if not hasattr(self, '_pytesseract') or self._pytesseract is None:
                return None
            config = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            raw = self._pytesseract.image_to_string(img, config=config).strip()
            data = self._pytesseract.image_to_data(
                img, config=config, output_type=self._pytesseract.Output.DICT
            )
            confs = [float(c) for c in data['conf'] if str(c) != '-1']
            conf = float(np.mean(confs)) / 100.0 if confs else 0.5
            return self._build_plate_text(raw, conf)
        except Exception as e:
            logger.debug(f"Tesseract error: {e}")
            return None

    # ── Preprocessing ─────────────────────────────────────────

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Standard plate preprocessing pipeline."""
        # Resize to standard height
        h, w = img.shape[:2]
        target_h = 64
        scale = target_h / max(h, 1)
        resized = cv2.resize(img, (int(w * scale), target_h))

        # Grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        # Otsu threshold
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert back to BGR for PaddleOCR
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    def _get_preprocessing_variants(self, img: np.ndarray) -> List[np.ndarray]:
        """Generate multiple preprocessing variants for robust reading."""
        variants = [self._preprocess(img)]

        h, w = img.shape[:2]
        resized = cv2.resize(img, (max(w * 2, 200), max(h * 2, 100)))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Variant 2: adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        variants.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))

        # Variant 3: inverted (for dark plates)
        inverted = cv2.bitwise_not(adaptive)
        variants.append(cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR))

        return variants

    # ── Post-processing ───────────────────────────────────────

    def _build_plate_text(self, raw: str, confidence: float) -> PlateText:
        cleaned = self._clean_text(raw)
        valid = self._validate_format(cleaned)
        state = cleaned[:2] if len(cleaned) >= 2 else None

        district = None
        if len(cleaned) >= 4 and cleaned[2:4].isdigit():
            district = cleaned[2:4]

        return PlateText(
            raw_text=raw,
            cleaned_text=cleaned,
            confidence=confidence,
            is_valid_format=valid,
            state_code=state,
            district_code=district,
        )

    def _clean_text(self, text: str) -> str:
        # Remove spaces, special chars
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())

        # Apply position-aware corrections
        if len(cleaned) >= 2:
            # First 2 chars should be letters (state code)
            chars = list(cleaned)
            for i in range(min(2, len(chars))):
                if chars[i].isdigit():
                    chars[i] = OCR_CORRECTIONS.get(chars[i], chars[i])
            # Chars 3-4 should be digits (district)
            for i in range(2, min(4, len(chars))):
                if chars[i].isalpha():
                    chars[i] = OCR_CORRECTIONS.get(chars[i], chars[i])
            cleaned = ''.join(chars)

        return cleaned

    def _validate_format(self, text: str) -> bool:
        return any(p.match(text) for p in COMPILED_PATTERNS)
