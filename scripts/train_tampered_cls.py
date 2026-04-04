"""
Train tampered-number-plate classifier (YOLOv8 classification).

Expected dataset layout:
    data/tamper_cls/
      train/
        tampered/
          img1.jpg
        clean/
          img2.jpg
      val/
        tampered/
        clean/

Usage:
    python scripts/train_tampered_cls.py --epochs 30 --imgsz 224 --device cpu

Output:
    models/weights/tamper_cls_best.pt
"""

import argparse
import shutil
from pathlib import Path

from loguru import logger
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = BASE_DIR / "data" / "tamper_cls"
DEFAULT_PROJECT = BASE_DIR / "runs" / "classify"
DEFAULT_OUT = BASE_DIR / "models" / "weights" / "tamper_cls_best.pt"


def validate_dataset(dataset_dir: Path) -> None:
    required = [
        dataset_dir / "train" / "tampered",
        dataset_dir / "train" / "clean",
        dataset_dir / "val" / "tampered",
        dataset_dir / "val" / "clean",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        msg = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing dataset folders:\n{msg}")


def train(dataset_dir: Path, epochs: int, imgsz: int, batch: int, device: str, name: str) -> Path:
    validate_dataset(dataset_dir)

    logger.info("=" * 50)
    logger.info("TRINETRA — Tampered Plate Classifier Training")
    logger.info("=" * 50)
    logger.info(f"Dataset  : {dataset_dir}")
    logger.info(f"Epochs   : {epochs}")
    logger.info(f"Image sz : {imgsz}")
    logger.info(f"Batch    : {batch}")
    logger.info(f"Device   : {device}")
    logger.info("=" * 50)

    model = YOLO("yolov8n-cls.pt")
    model.train(
        data=str(dataset_dir),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(DEFAULT_PROJECT),
        name=name,
        exist_ok=True,
        verbose=True,
    )

    best_pt = DEFAULT_PROJECT / name / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Training completed but best.pt not found: {best_pt}")

    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_pt, DEFAULT_OUT)
    logger.success(f"Saved classifier weights: {DEFAULT_OUT}")
    return DEFAULT_OUT


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tampered number plate classifier")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATASET), help="Path to tamper_cls dataset root")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--name", type=str, default="tamper_cls_v1")
    args = parser.parse_args()

    out = train(
        dataset_dir=Path(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
    )
    logger.info(f"Done. Use weights at: {out}")


if __name__ == "__main__":
    main()
