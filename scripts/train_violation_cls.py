"""
Train multi-class violation classifier (YOLOv8 classification).

Classes expected under data/violation_cls/{train,val}/:
- no_helmet
- red_light_violation
- wrong_side_driving
- tampered_plate
- blackened_window

Output:
  models/weights/violation_cls_best.pt
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "violation_cls"
PROJECT_DIR = BASE_DIR / "runs" / "classify"
OUT_WEIGHTS = BASE_DIR / "models" / "weights" / "violation_cls_best.pt"


REQUIRED_CLASSES = [
    "no_helmet",
    "red_light_violation",
    "wrong_side_driving",
    "tampered_plate",
    "blackened_window",
]


def validate_dataset(root: Path) -> None:
    required = []
    for split in ["train", "val"]:
        for cls in REQUIRED_CLASSES:
            required.append(root / split / cls)
    missing = [p for p in required if not p.exists()]
    if missing:
        missing_text = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing dataset folders:\n{missing_text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train violation classifier")
    parser.add_argument("--data", type=str, default=str(DATA_DIR))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--name", type=str, default="violation_cls_v1")
    args = parser.parse_args()

    dataset = Path(args.data)
    validate_dataset(dataset)

    model = YOLO("yolov8n-cls.pt")
    model.train(
        data=str(dataset),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(PROJECT_DIR),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    best = PROJECT_DIR / args.name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt not found at {best}")

    OUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, OUT_WEIGHTS)
    print(f"Saved model: {OUT_WEIGHTS}")


if __name__ == "__main__":
    main()
