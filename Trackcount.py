import cv2
import time
from ultralytics import YOLO
from collections import defaultdict


model = YOLO("yolov8n.pt")

video_path = "traffic recording.mp4"
cap = cv2.VideoCapture(video_path)


TARGET_CLASSES = {"car", "bus", "truck", "motorcycle"}

seen_ids = defaultdict(set)


prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    frame = cv2.resize(frame, (1280, 720))

    
    results = model.track(frame, conf=0.4, persist=True, tracker="bytetrack.yaml")

    
    annotated = results[0].plot()

   
    boxes = results[0].boxes

    if boxes.id is not None:
        for i in range(len(boxes)):
            track_id = int(boxes.id[i].item())
            cls_id = int(boxes.cls[i].item())
            class_name = model.names[cls_id]

            if class_name in TARGET_CLASSES:
                seen_ids[class_name].add(track_id)

    
    car_count = len(seen_ids["car"])
    bus_count = len(seen_ids["bus"])
    truck_count = len(seen_ids["truck"])
    moto_count = len(seen_ids["motorcycle"])

    total_unique = car_count + bus_count + truck_count + moto_count

    
    if total_unique < 15:
        congestion = "LOW"
        cong_color = (0, 255, 0)
    elif total_unique < 35:
        congestion = "MEDIUM"
        cong_color = (0, 255, 255)
    else:
        congestion = "HIGH"
        cong_color = (0, 0, 255)

   
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    
    cv2.rectangle(annotated, (10, 10), (420, 230), (0, 0, 0), -1)

    cv2.putText(annotated, "TRINETRA - UNIQUE TRACKING", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(annotated, f"CAR (Unique): {car_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, f"BUS (Unique): {bus_count}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, f"TRUCK (Unique): {truck_count}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, f"BIKE (Unique): {moto_count}", (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(annotated, f"TOTAL UNIQUE: {total_unique}", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    cv2.putText(annotated, f"CONGESTION: {congestion}", (20, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, cong_color, 2)

    cv2.putText(annotated, f"FPS: {int(fps)}", (320, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("TRINETRA - Tracking + Unique Count", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
