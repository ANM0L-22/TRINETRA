"""
Trinetra — Evaluate & Run Inference
Test the trained model on images or video.

Usage:
    # Evaluate on val set (metrics)
    python scripts/evaluate.py --mode eval

    # Run on a single image
    python scripts/evaluate.py --mode image --input path/to/image.jpg

    # Run on a video
    python scripts/evaluate.py --mode video --input path/to/video.mp4 --output results/detections/out.mp4
"""

import argparse
import sys
from pathlib import Path

import cv2
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from loguru import logger
from ultralytics import YOLO

BASE_DIR    = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / 'configs' / 'config.yaml'
DATASET_YML = BASE_DIR / 'configs' / 'dataset.yaml'
WEIGHTS_DIR = BASE_DIR / 'models' / 'weights'


def get_weights(weights_arg: str) -> str:
    if weights_arg and Path(weights_arg).exists():
        return weights_arg

    # Auto-detect best available weights
    candidates = [
        WEIGHTS_DIR / 'trinetra_best.pt',
        BASE_DIR / 'runs' / 'detect' / 'trinetra_v1' / 'weights' / 'best.pt',
    ]
    for c in candidates:
        if c.exists():
            logger.info(f"Using weights: {c}")
            return str(c)

    # Fallback: pretrained COCO (useful for quick testing before custom training)
    logger.warning("No custom weights found — using pretrained YOLOv8m (COCO classes)")
    logger.warning("Train first with: python scripts/train.py")
    return 'yolov8m.pt'


def run_eval(model, cfg):
    """Run validation metrics on the val split."""
    logger.info("Running validation evaluation...")
    metrics = model.val(
        data=str(DATASET_YML),
        imgsz=cfg['inference']['image_size'],
        conf=cfg['inference']['conf_threshold'],
        iou=cfg['inference']['iou_threshold'],
        device=cfg['inference']['device'],
        plots=True,
        save_json=True,
    )

    print("\n" + "=" * 50)
    print("TRINETRA — Evaluation Results")
    print("=" * 50)
    print(f"  mAP50         : {metrics.box.map50:.4f}")
    print(f"  mAP50-95      : {metrics.box.map:.4f}")
    print(f"  Precision     : {metrics.box.mp:.4f}")
    print(f"  Recall        : {metrics.box.mr:.4f}")
    print("=" * 50)

    # Per-class breakdown
    class_names = cfg['classes']
    print("\nPer-class AP50:")
    for i, (name, ap) in enumerate(zip(class_names, metrics.box.ap50)):
        bar = "█" * int(ap * 30)
        print(f"  {name:15s} {ap:.4f}  {bar}")


def run_image(model, image_path: str, cfg, output_path: str | None = None):
    """Detect on a single image and display/save result."""
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error(f"Cannot open image: {image_path}")
        sys.exit(1)

    results = model(
        frame,
        conf=cfg['inference']['conf_threshold'],
        iou=cfg['inference']['iou_threshold'],
        device=cfg['inference']['device'],
    )[0]

    annotated = results.plot()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, annotated)
        logger.success(f"Saved: {output_path}")
    else:
        out = BASE_DIR / 'results' / 'detections' / 'last_image.jpg'
        cv2.imwrite(str(out), annotated)
        logger.success(f"Saved: {out}")

    # Print detections
    boxes = results.boxes
    if boxes is not None:
        print(f"\nDetected {len(boxes)} objects:")
        for box in boxes:
            cls_id  = int(box.cls[0])
            conf    = float(box.conf[0])
            name    = cfg['classes'][cls_id] if cls_id < len(cfg['classes']) else f'cls_{cls_id}'
            print(f"  {name:15s}  confidence={conf:.3f}")


def run_video(model, video_path: str, cfg, output_path: str | None = None, show: bool = False):
    """Run detection on a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not output_path:
        output_path = str(BASE_DIR / 'results' / 'detections' / 'output_video.mp4')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'MJPG'),  # type: ignore
        fps, (width, height)
    )

    logger.info(f"Processing {total} frames...")
    frame_id = 0
    total_vehicles = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(
            frame,
            conf=cfg['inference']['conf_threshold'],
            iou=cfg['inference']['iou_threshold'],
            device=cfg['inference']['device'],
            verbose=False,
        )[0]

        annotated = results.plot()

        n_det = len(results.boxes) if results.boxes is not None else 0
        total_vehicles += n_det

        cv2.putText(annotated, f"Frame: {frame_id} | Detections: {n_det}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(annotated)

        if show:
            cv2.imshow("Trinetra", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_id += 1
        if frame_id % 50 == 0:
            logger.info(f"Frame {frame_id}/{total}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    logger.success(f"Done — {frame_id} frames | avg {total_vehicles/max(frame_id,1):.1f} vehicles/frame")
    logger.success(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Trinetra evaluation & inference')
    parser.add_argument('--mode', choices=['eval', 'image', 'video'], default='image')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--input',  type=str, default=None, help='Image or video path')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    parser.add_argument('--show',   action='store_true',   help='Show live window')
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    weights = get_weights(args.weights)
    model   = YOLO(weights)

    if args.mode == 'eval':
        run_eval(model, cfg)
    elif args.mode == 'image':
        if not args.input:
            logger.error("--input required for image mode")
            sys.exit(1)
        run_image(model, args.input, cfg, args.output)
    elif args.mode == 'video':
        if not args.input:
            logger.error("--input required for video mode")
            sys.exit(1)
        run_video(model, args.input, cfg, args.output, args.show)


if __name__ == '__main__':
    main()
