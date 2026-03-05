import cv2
import time
import os
import easyocr
from ultralytics import YOLO

# Vehicle model + Plate model
vehicle_model = YOLO("yolov8n.pt")
plate_model = YOLO("lp_detector.pt")

reader = easyocr.Reader(['en'], gpu=False)

video_path = "traffic.mp4"
cap = cv2.VideoCapture(video_path)

os.makedirs("plates", exist_ok=True)

def enhance_for_ocr(img):
    """Improve plate crop for OCR readability."""
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # smooth noise
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    # ✅ Detect number plates directly
    plate_results = plate_model(frame, conf=0.4)

    annotated = frame.copy()

    for r in plate_results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            processed = enhance_for_ocr(plate_crop)

            ocr = reader.readtext(processed)
            plate_text = "NA"
            if len(ocr) > 0:
                plate_text = max(ocr, key=lambda x: x[2])[1].replace(" ", "")

            # Save crops
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"plates/plate_{ts}.jpg", plate_crop)

            # Draw
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, plate_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("TRINETRA - Plate Detection + OCR", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
