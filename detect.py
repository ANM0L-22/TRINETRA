import cv2
from ultralytics import YOLO


model = YOLO("yolov8n.pt")

video_path = "traffic recording.mp4"  
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame, conf=0.4)

    
    annotated_frame = results[0].plot()

    cv2.imshow("TRINETRA - Vehicle Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
