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
TARGET_CLASSES = {"car", "bus", "truck", "motorcycle"}

os.makedirs("violations", exist_ok=True)

# -----------------------------
# Lane Detection Helpers
# -----------------------------
def region_of_interest(img):
    """Mask only road area (bottom half triangle)"""
    h, w = img.shape[:2]
    mask = np.zeros_like(img)

    polygon = np.array([[
        (int(0.05*w), h),
        (int(0.45*w), int(0.55*h)),
        (int(0.55*w), int(0.55*h)),
        (int(0.95*w), h)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def average_slope_intercept(lines):
    """
    Separate left and right lane lines and return averaged line coordinates.
    """
    left = []
    right = []
    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)

        # filter out near horizontal noise
        if abs(slope) < 0.4:
            continue

        intercept = y1 - slope * x1
        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))

    left_lane = np.mean(left, axis=0) if len(left) > 0 else None
    right_lane = np.mean(right, axis=0) if len(right) > 0 else None
    return left_lane, right_lane

def make_line_points(y1, y2, line):
    """Convert slope-intercept to pixel points"""
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return (x1, y1), (x2, y2)

def detect_lane_lines(frame):
    """
    Returns lane line points: left_line, right_line
    Each is ((x1,y1),(x2,y2))
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    cropped_edges = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=60,
        maxLineGap=120
    )

    left_lane, right_lane = average_slope_intercept(lines)

    h, w = frame.shape[:2]
    y1 = h
    y2 = int(h * 0.6)

    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

    return left_line, right_line

def point_in_polygon(point, polygon):
    """Check if a point is inside polygon"""
    return cv2.pointPolygonTest(polygon, point, False) >= 0


# -----------------------------
# MAIN
# -----------------------------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

track_history = {}
wrong_side_ids = set()

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    h, w = frame.shape[:2]

    # ✅ Detect lanes
    left_line, right_line = detect_lane_lines(frame)

    # Build lane polygons (approx)
    lane_left_poly = None
    lane_right_poly = None

    if left_line and right_line:
        (lx1, ly1), (lx2, ly2) = left_line
        (rx1, ry1), (rx2, ry2) = right_line

        # Center line midpoints
        cx_bottom = int((lx1 + rx1) / 2)
        cx_top = int((lx2 + rx2) / 2)

        # Left lane polygon
        lane_left_poly = np.array([
            (lx1, ly1),
            (lx2, ly2),
            (cx_top, ly2),
            (cx_bottom, ly1)
        ], dtype=np.int32)

        # Right lane polygon
        lane_right_poly = np.array([
            (cx_bottom, ly1),
            (cx_top, ly2),
            (rx2, ry2),
            (rx1, ry1)
        ], dtype=np.int32)

    # ✅ Track vehicles
    results = model.track(frame, conf=0.4, persist=True, tracker="bytetrack.yaml")
    annotated = results[0].plot()
    boxes = results[0].boxes

    # Draw lane overlays
    if lane_left_poly is not None:
        cv2.polylines(annotated, [lane_left_poly], True, (0, 255, 255), 3)
        cv2.putText(annotated, "LEFT LANE", (lane_left_poly[1][0], lane_left_poly[1][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if lane_right_poly is not None:
        cv2.polylines(annotated, [lane_right_poly], True, (255, 255, 0), 3)
        cv2.putText(annotated, "RIGHT LANE", (lane_right_poly[2][0], lane_right_poly[2][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # If lanes not detected, show message
    if lane_left_poly is None or lane_right_poly is None:
        cv2.putText(annotated, "Lane detection weak - try clearer road video!",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if boxes.id is not None and lane_left_poly is not None and lane_right_poly is not None:
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

            if track_id not in track_history:
                track_history[track_id] = P
                continue

            prevP = track_history[track_id]
            track_history[track_id] = P

            dy = P[1] - prevP[1]  # +ve means moving down

            # Determine lane side
            in_left_lane = point_in_polygon(P, lane_left_poly)
            in_right_lane = point_in_polygon(P, lane_right_poly)

            # ✅ Define expected direction:
            # Left lane vehicles should move DOWN (+dy)
            # Right lane vehicles should move UP (-dy)
            wrong = False

            if in_left_lane and dy < -2:   # left lane but moving UP
                wrong = True
            if in_right_lane and dy > 2:  # right lane but moving DOWN
                wrong = True

            if wrong and track_id not in wrong_side_ids:
                wrong_side_ids.add(track_id)

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                evidence_path = f"violations/wrongside_{class_name}_id{track_id}_{timestamp}.jpg"
                cv2.imwrite(evidence_path, frame)
                print(f"[VIOLATION] Wrong-side -> {evidence_path}")

            if track_id in wrong_side_ids:
                cv2.putText(annotated, "WRONG SIDE!", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            cv2.circle(annotated, P, 4, (0, 255, 0), -1)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Info overlay
    cv2.rectangle(annotated, (10, 10), (470, 90), (0, 0, 0), -1)
    cv2.putText(annotated, "TRINETRA - Lane Based Wrong Side", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated, f"Wrong-side Count: {len(wrong_side_ids)} | FPS: {int(fps)}",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("TRINETRA - Lane Wrong Side", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
