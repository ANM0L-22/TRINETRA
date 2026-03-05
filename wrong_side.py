import cv2
import time
from ultralytics import YOLO
from collections import defaultdict
import os

# Load YOLO model
model = YOLO("yolov8n.pt")

video_path = "helmet_detection.mp4"
cap = cv2.VideoCapture(video_path)

# classes to track
TARGET_CLASSES = {"car", "bus", "truck", "motorcycle"}

# store last center positions of each track id
track_history = {}

# store which IDs already violated (avoid repeated alerts)
wrong_side_ids = set()

# folder for saving evidence
os.makedirs("violations", exist_ok=True)

# define a virtual line for direction checking (tune for your video)
# (x1,y1) -> (x2,y2)
LINE_Y = 360  # horizontal line (middle of screen)
# For your video 720 height, 360 is center

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

    # Draw direction line
    cv2.line(annotated, (0, LINE_Y), (1280, LINE_Y), (0, 255, 255), 3)
    cv2.putText(annotated, "DIRECTION CHECK LINE", (20, LINE_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if boxes.id is not None:
        for i in range(len(boxes)):
            track_id = int(boxes.id[i].item())
            cls_id = int(boxes.cls[i].item())
            class_name = model.names[cls_id]

            if class_name not in TARGET_CLASSES:
                continue

            # Get bounding box coordinates
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()

            # center of object
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # store previous center
            if track_id not in track_history:
                track_history[track_id] = (cx, cy)
                continue

            prev_cx, prev_cy = track_history[track_id]
            track_history[track_id] = (cx, cy)

            # movement direction (delta y)
            dy = cy - prev_cy

            """
            Wrong-side logic (demo version):
            Assume correct flow should be moving DOWN (dy > 0)
            If vehicle is moving UP (dy < 0) and crosses the line => wrong side
            """

            # check crossing
            crossed_line = (prev_cy > LINE_Y and cy < LINE_Y)

            if dy < -2 and crossed_line and track_id not in wrong_side_ids:
                wrong_side_ids.add(track_id)

                # Save evidence screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"violations/wrongside_{class_name}_id{track_id}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)

                print(f"[VIOLATION] Wrong-side detected -> {filename}")

            # If violated show alert on box
            if track_id in wrong_side_ids:
                cv2.putText(annotated, "WRONG SIDE!", (cx - 50, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Info Panel
    cv2.rectangle(annotated, (10, 10), (450, 130), (0, 0, 0), -1)
    cv2.putText(annotated, "TRINETRA - WRONG SIDE DETECTION", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.putText(annotated, f"Violations Detected: {len(wrong_side_ids)}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(annotated, f"FPS: {int(fps)}", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow("TRINETRA - Wrong Side Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
