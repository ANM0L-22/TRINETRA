import cv2
import time
import os
import numpy as np
import pandas as pd
import easyocr
from ultralytics import YOLO

# ---------------- CONFIG ----------------
VIDEO_PATH = "video1.mp4"
MODEL_PATH = "yolov8n.pt"
TARGET_CLASSES = {"car", "bus", "truck", "motorcycle"}

# Road direction line (adjust if needed)
LINE_Y = 360   # for 720p frame, middle line

# Save folders
os.makedirs("violations", exist_ok=True)
os.makedirs("plates", exist_ok=True)

# OCR
reader = easyocr.Reader(['en'], gpu=False)

# ---------------- HELPERS ----------------
def enhance_plate_for_ocr(img):
    """Improve plate crop quality for OCR."""
    if img is None or img.size == 0:
        return None
    img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)

    th = cv2.adaptiveThreshold(sharp, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    return th

def run_ocr(crop):
    """Returns OCR text."""
    processed = enhance_plate_for_ocr(crop)
    if processed is None:
        return "NA"

    results = reader.readtext(processed)
    if len(results) == 0:
        return "NA"

    best = max(results, key=lambda x: x[2])
    text = best[1].replace(" ", "")
    return text

# ---------------- MAIN ----------------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

track_history = {}        # track_id -> previous center
wrong_side_ids = set()    # violated IDs
logs = []                # store CSV logs

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    # Track objects
    results = model.track(frame, conf=0.4, persist=True, tracker="bytetrack.yaml")
    annotated = results[0].plot()
    boxes = results[0].boxes

    # Draw wrong-side line
    cv2.line(annotated, (0, LINE_Y), (1280, LINE_Y), (0, 255, 255), 3)
    cv2.putText(annotated, "TRINETRA - Wrong Side + OCR", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    if boxes.id is not None:
        for i in range(len(boxes)):
            track_id = int(boxes.id[i].item())
            cls_id = int(boxes.cls[i].item())
            class_name = model.names[cls_id]

            if class_name not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # save previous center for direction
            if track_id not in track_history:
                track_history[track_id] = (cx, cy)
                continue

            prev_cx, prev_cy = track_history[track_id]
            track_history[track_id] = (cx, cy)

            dy = cy - prev_cy

            # Crossing check (above->below or below->above)
            crossed = (prev_cy > LINE_Y and cy < LINE_Y)

            # ✅ Wrong-side rule (Demo):
            # if moving UP (dy < 0) AND crossed line => wrong side
            if dy < -2 and crossed and track_id not in wrong_side_ids:
                wrong_side_ids.add(track_id)

                timestamp = time.strftime("%Y%m%d_%H%M%S")

                # Save evidence full frame
                evidence_path = f"violations/wrongside_{class_name}_id{track_id}_{timestamp}.jpg"
                cv2.imwrite(evidence_path, frame)

                # Plate crop (lower part of vehicle bbox)
                plate_y1 = int(y1 + 0.65 * (y2 - y1))
                plate_y2 = y2
                plate_x1 = x1
                plate_x2 = x2

                plate_crop = frame[plate_y1:plate_y2, plate_x1:plate_x2]

                plate_text = run_ocr(plate_crop)

                # Save plate crop
                plate_img_path = f"plates/plate_{class_name}_id{track_id}_{timestamp}.jpg"
                if plate_crop is not None and plate_crop.size != 0:
                    cv2.imwrite(plate_img_path, plate_crop)
                else:
                    plate_img_path = "NA"

                # Log
                logs.append({
                    "time": timestamp,
                    "vehicle_type": class_name,
                    "track_id": track_id,
                    "violation": "WRONG_SIDE",
                    "plate_text": plate_text,
                    "evidence_img": evidence_path,
                    "plate_img": plate_img_path
                })

                print(f"[VIOLATION] Wrong-side ID={track_id} | Plate={plate_text}")

            # Show warning on screen
            if track_id in wrong_side_ids:
                cv2.putText(annotated, "WRONG SIDE!", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # info panel
    cv2.rectangle(annotated, (10, 60), (520, 140), (0, 0, 0), -1)
    cv2.putText(annotated, f"Violations Logged: {len(logs)}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(annotated, f"FPS: {int(fps)}", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    cv2.imshow("TRINETRA - WrongSide + OCR", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Save CSV log file
if len(logs) > 0:
    df = pd.DataFrame(logs)
    df.to_csv("violation_logs.csv", index=False)
    print("✅ Saved: violation_logs.csv")
else:
    print("⚠️ No violations detected, no CSV saved.")

