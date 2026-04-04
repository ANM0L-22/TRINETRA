"""
Trinetra — Training Script
Fine-tunes YOLOv8 on the Trinetra vehicle dataset.

Run on Colab (GPU):
    !python scripts/train.py

Run locally (slow, CPU only — for quick sanity check):
    python scripts/train.py --epochs 2 --batch 4 --device cpu
"""

import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger
from ultralytics import YOLO


BASE_DIR    = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / 'configs' / 'config.yaml'
DATASET_YML = BASE_DIR / 'configs' / 'dataset.yaml'


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def patch_dataset_yaml_for_colab(cfg):
    """
    When running on Colab, update dataset.yaml 'path' to absolute path.
    Detects Colab by checking for /content directory.
    """
    import os
    if os.path.exists('/content'):
        # We're on Colab — update path to absolute
        abs_data = str(BASE_DIR / cfg['data']['processed_dir'])
        with open(DATASET_YML) as f:
            d = yaml.safe_load(f)
        d['path'] = abs_data
        with open(DATASET_YML, 'w') as f:
            yaml.dump(d, f)
        logger.info(f"Colab detected — dataset path set to: {abs_data}")


def train(args):
    cfg = load_config()
    patch_dataset_yaml_for_colab(cfg)

    t = cfg['training']

    # Allow CLI overrides
    epochs     = args.epochs     or t['epochs']
    batch      = args.batch      or t['batch_size']
    device     = args.device     or t['device']
    image_size = args.imgsz      or t['image_size']
    name       = args.name       or t['name']

    # Backbone: yolov8m.pt downloads pretrained COCO weights automatically
    backbone = f"{cfg['model']['backbone']}.pt"

    logger.info("=" * 50)
    logger.info("TRINETRA — Vehicle Detection Training")
    logger.info("=" * 50)
    logger.info(f"Backbone    : {backbone}")
    logger.info(f"Dataset     : {DATASET_YML}")
    logger.info(f"Epochs      : {epochs}")
    logger.info(f"Batch size  : {batch}")
    logger.info(f"Image size  : {image_size}")
    logger.info(f"Device      : {device}")
    logger.info(f"Run name    : {name}")
    logger.info("=" * 50)

    model = YOLO(backbone)

    results = model.train(
        data=str(DATASET_YML),
        epochs=epochs,
        batch=batch,
        imgsz=image_size,
        device=device,
        lr0=t['lr0'],
        lrf=t['lrf'],
        momentum=t['momentum'],
        weight_decay=t['weight_decay'],
        patience=t['patience'],
        optimizer=t['optimizer'],
        workers=t['workers'],
        augment=t['augment'],
        project=str(BASE_DIR / 'runs' / 'detect'),
        name=name,
        exist_ok=True,
        plots=True,      # saves training curves
        save=True,
        save_period=10,  # checkpoint every 10 epochs
        verbose=True,
    )

    # Copy best weights to models/weights/
    best_pt = BASE_DIR / 'runs' / 'detect' / name / 'weights' / 'best.pt'
    dest = BASE_DIR / 'models' / 'weights' / 'trinetra_best.pt'
    if best_pt.exists():
        import shutil
        shutil.copy(best_pt, dest)
        logger.success(f"Best weights saved to: {dest}")

    logger.success("Training complete!")
    logger.info(f"Results in: runs/detect/{name}/")
    logger.info("Next step: python scripts/evaluate.py")

    return results


def main():
    parser = argparse.ArgumentParser(description='Trinetra training script')
    parser.add_argument('--epochs',  type=int,   default=None)
    parser.add_argument('--batch',   type=int,   default=None)
    parser.add_argument('--device',  type=str,   default=None, help='cuda device or cpu')
    parser.add_argument('--imgsz',   type=int,   default=None)
    parser.add_argument('--name',    type=str,   default=None)
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
