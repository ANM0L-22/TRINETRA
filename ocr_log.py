import cv2
import time
import os
import numpy as np

import pandas as pd
from ultralytics import YOLO
import easyocr
from collections import defaultdict


model = YOLO("yolov8n.pt")


reader = easyocr.Reader(['en'], gpu=False)

video_path = "video1.mp4"
cap = cv2.VideoCapture(video_path)

TARGET_CLASSES = {"car", "bus", "truck", "motorcycle"}

track_history = {}
wrong_side_ids = set()


os.makedirs("violations", exist_ok=True)
os.makedirs("plates", exist_ok=True)

LINE_Y = 360


logs = []

def run_ocr_on_crop(crop):
    if crop is None or crop.size == 0:
        return "NA"

   
    crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

   
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)

  
    th = cv2.adaptiveThreshold(sharp, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)

    
    results = reader.readtext(th)
    if len(results) == 0:
        return "NA"

    best = max(results, key=lambda x: x[2])
    return best[1].replace(" ", "")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    results = model.track(frame, conf=0.4, persist=True, tracker="bytetrack.yaml")
    annotated = results[0].plot()
    boxes = results[0].boxes


    cv2.line(annotated, (0, LINE_Y), (1280, LINE_Y), (0, 255, 255), 3)

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

            
            if track_id not in track_history:
                track_history[track_id] = (cx, cy)
                continue

            prev_cx, prev_cy = track_history[track_id]
            track_history[track_id] = (cx, cy)

            dy = cy - prev_cy
            crossed_line = (prev_cy > LINE_Y and cy < LINE_Y)

           
            if dy < -2 and crossed_line and track_id not in wrong_side_ids:
                wrong_side_ids.add(track_id)

                timestamp = time.strftime("%Y%m%d_%H%M%S")

               
                evidence_path = f"violations/wrongside_{class_name}_id{track_id}_{timestamp}.jpg"
                cv2.imwrite(evidence_path, frame)

               
                plate_y1 = int(y1 + 0.65 * (y2 - y1))
                plate_y2 = y2
                plate_x1 = x1
                plate_x2 = x2

                plate_crop = frame[plate_y1:plate_y2, plate_x1:plate_x2]
                plate_text = run_ocr_on_crop(plate_crop)

                plate_img_path = f"plates/plate_{class_name}_id{track_id}_{timestamp}.jpg"
                cv2.imwrite(plate_img_path, plate_crop)

               
                logs.append({
                    "time": timestamp,
                    "vehicle_type": class_name,
                    "track_id": track_id,
                    "violation": "WRONG_SIDE",
                    "plate_text": plate_text,
                    "evidence_img": evidence_path,
                    "plate_img": plate_img_path
                })

                print(f"[LOG] Wrong-side ID {track_id} Plate: {plate_text}")

          
            if track_id in wrong_side_ids:
                cv2.putText(annotated, "WRONG SIDE!", (cx - 60, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    cv2.rectangle(annotated, (10, 10), (500, 110), (0, 0, 0), -1)
    cv2.putText(annotated, "TRINETRA - OCR + VIOLATION LOG", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated, f"Violations Logged: {len(logs)}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("TRINETRA - OCR Logging", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


if len(logs) > 0:
    df = pd.DataFrame(logs)
    df.to_csv("violation_logs.csv", index=False)
    print("\n✅ Saved: violation_logs.csv")
else:
    print("\n⚠️ No violations logged, so no CSV saved.")
