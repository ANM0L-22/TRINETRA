import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict


model = YOLO("yolov8n.pt")

video_path = "traffic recording.mp4"   
cap = cv2.VideoCapture(video_path)


TARGET_CLASSES = ["car", "bus", "motorcycle", "qtruck", "person"]


LOW_TH = 10
MED_TH = 25

while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    frame = cv2.resize(frame, (1280, 720))

   
    results = model(frame, conf=0.4, imgsz=640)


    counts = defaultdict(int)
    total_vehicles = 0

    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]

        if class_name in TARGET_CLASSES:
            counts[class_name] += 1

            
            if class_name != "person":
                total_vehicles += 1

    
    if total_vehicles < LOW_TH:
        congestion = "LOW"
    elif total_vehicles < MED_TH:
        congestion = "MEDIUM"
    else:
        congestion = "HIGH"

    
    annotated = results[0].plot()

    
    cv2.rectangle(annotated, (10, 10), (380, 220), (0, 0, 0), -1)

    cv2.putText(annotated, "TRINETRA - LIVE ANALYTICS", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    y = 75
    for k in ["car", "bus", "motorcycle", "truck", "person"]:
        cv2.putText(annotated, f"{k.upper()}: {counts[k]}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 28

    cv2.putText(annotated, f"TOTAL VEHICLES: {total_vehicles}", (20, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(annotated, f"CONGESTION: {congestion}", (20, 225),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if congestion == "LOW"
                else (0, 255, 255) if congestion == "MEDIUM"
                else (0, 0, 255), 2)

    cv2.imshow("TRINETRA - Counting + Congestion", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
