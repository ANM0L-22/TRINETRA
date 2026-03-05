import cv2
import time
import numpy as np
from ultralytics import YOLO
import os

# -----------------------------
# CONFIG
# -----------------------------
VIDEO_PATH = "video1.mp4"
MODEL_PATH = "yolov8n.pt"

TARGET_CLASSES = {"car", "bus", "truck", "motorcycle"}  # track vehicles only
os.makedirs("violations", exist_ok=True)

# -----------------------------
# GLOBALS for mouse clicks
# -----------------------------
line_points = []   # will store [(x1,y1),(x2,y2)]
drawing_done = False

def mouse_callback(event, x, y, flags, param):
    global line_points, drawing_done
    if event == cv2.EVENT_LBUTTONDOWN and not drawing_done:
        line_points.append((x, y))
        print(f"Point selected: {x}, {y}")
        if len(line_points) == 2:
            drawing_done = True
            print("✅ Road line fixed! Starting wrong-side detection...")

def side_of_line(P, A, B):
    """
    Returns which side point P lies on relative to directed line A->B.
    >0 : left side
    <0 : right side
     0 : on the line
    """
    return (B[0] - A[0]) * (P[1] - A[1]) - (B[1] - A[1]) * (P[0] - A[0])

# -----------------------------
# MAIN
# -----------------------------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# Read 1st frame for line selection
ret, first_frame = cap.read()
if not ret:
    print("❌ Could not read video.")
    exit()

first_frame = cv2.resize(first_frame, (1280, 720))
clone = first_frame.copy()

cv2.namedWindow("Select Road Line")
cv2.setMouseCallback("Select Road Line", mouse_callback)

print("\n✅ INSTRUCTIONS:")
print("1) Click TWO points on the ROAD DIVIDER line (lane direction boundary).")
print("2) Press 'S' to start detection after selecting points.\n")

while True:
    temp = clone.copy()

    # show clicked points
    for p in line_points:
        cv2.circle(temp, p, 6, (0, 255, 255), -1)

    # draw line if 2 points selected
    if len(line_points) == 2:
        cv2.line(temp, line_points[0], line_points[1], (0, 255, 255), 3)

    cv2.imshow("Select Road Line", temp)

    key = cv2.waitKey(1) & 0xFF

    # Reset points
    if key == ord('r'):
        line_points = []
        drawing_done = False
        print("🔄 Reset line selection.")

    # Start detection
    if key == ord('s') and len(line_points) == 2:
        break

cv2.destroyWindow("Select Road Line")

A, B = line_points[0], line_points[1]

# Tracking memory
track_history = {}     # track_id -> previous center (x,y)
wrong_side_ids = set() # track IDs already violated

# Decide the "correct direction side"
# We'll take the side of the first detected vehicle movement as correct baseline.
correct_side = None

# Re-open the video from beginning
cap.release()
cap = cv2.VideoCapture(VIDEO_PATH)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    # Track using ByteTrack
    results = model.track(frame, conf=0.4, persist=True, tracker="bytetrack.yaml")
    annotated = results[0].plot()
    boxes = results[0].boxes

    # Draw road-aligned line
    cv2.line(annotated, A, B, (0, 255, 255), 3)
    cv2.putText(annotated, "ROAD DIRECTION LINE", (20, 40),
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
            P = (cx, cy)

            # Save prev
            if track_id not in track_history:
                track_history[track_id] = P
                continue

            prevP = track_history[track_id]
            track_history[track_id] = P

            # Find side of line now & before
            prev_side = side_of_line(prevP, A, B)
            curr_side = side_of_line(P, A, B)

            # Establish correct side dynamically (first stable vehicle)
            if correct_side is None and abs(curr_side) > 200:
                correct_side = 1 if curr_side > 0 else -1
                print("✅ Correct side fixed:", "LEFT" if correct_side == 1 else "RIGHT")

            if correct_side is None:
                continue

            # Crossing happened if sign changes
            crossed = (prev_side * curr_side) < 0

            # Movement direction check (optional)
            dy = P[1] - prevP[1]
            movement_up = dy < -2
            movement_down = dy > 2

            # WRONG-SIDE rule:
            # If vehicle crosses the road line AND ends up on opposite side of correct flow → violation
            ends_on_wrong_side = (1 if curr_side > 0 else -1) != correct_side

            if crossed and ends_on_wrong_side and track_id not in wrong_side_ids:
                wrong_side_ids.add(track_id)

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                evidence_path = f"violations/wrongside_{class_name}_id{track_id}_{timestamp}.jpg"
                cv2.imwrite(evidence_path, frame)

                print(f"[VIOLATION] Wrong-side vehicle -> {evidence_path}")

            # Show tag if wrong-side
            if track_id in wrong_side_ids:
                cv2.putText(annotated, "WRONG SIDE!", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            # Draw center point
            cv2.circle(annotated, (cx, cy), 4, (255, 255, 0), -1)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Info box
    cv2.rectangle(annotated, (10, 60), (430, 150), (0, 0, 0), -1)
    cv2.putText(annotated, f"Wrong-side detected: {len(wrong_side_ids)}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(annotated, f"FPS: {int(fps)}", (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("TRINETRA - Road Based Wrong Side", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
