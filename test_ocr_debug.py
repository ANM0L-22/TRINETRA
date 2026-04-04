import os
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import cv2
from paddleocr import PaddleOCR

# Initialize
print("Initializing PaddleOCR...")
ocr = PaddleOCR(use_textline_orientation=True, lang='en')

# Load and crop region
img = cv2.imread('static/uploads/upload_no_helmet1.jpeg')
roi = img[156:442, 226:464]

print("ROI shape:", roi.shape)
print("ROI dtype:", roi.dtype)

# Test OCR
print("Running OCR...")
result = ocr.ocr(roi)

print("\nResult type:", type(result))
print("Result length:", len(result) if result else 0)

if result and len(result) > 0:
    print("First result type:", type(result[0]))
    print("First result:", result[0])
    if result[0]:
        print("Number of items in first region:", len(result[0]))
        for i, item in enumerate(result[0][:5]):
            print(f"\nItem {i}:")
            print(f"  Type: {type(item)}")
            if isinstance(item, (list, tuple)):
                print(f"  Length: {len(item)}")
                if len(item) >= 2:
                    print(f"  Box: {item[0]}")
                    print(f"  Text/conf: {item[1]}")
