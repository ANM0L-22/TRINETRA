"""
Trinetra — Dataset Preparation
Downloads and prepares training data for vehicle detection.
Run this script ONCE before training.

Usage:
    python scripts/prepare_dataset.py --source roboflow --api-key YOUR_KEY
    python scripts/prepare_dataset.py --source sample   # download tiny sample
"""

import os
import sys
import argparse
import shutil
import random
import urllib.request
from pathlib import Path

import yaml
import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / 'data'
RAW_DIR   = DATA_DIR / 'raw'
PROC_DIR  = DATA_DIR / 'processed'

SPLITS = ['train', 'val', 'test']
SPLIT_RATIOS = (0.75, 0.15, 0.10)   # train / val / test


def setup_dirs():
    for split in SPLITS:
        (PROC_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (PROC_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Directory structure ready")


# ── Source: Roboflow ─────────────────────────────────────────

def download_roboflow(api_key: str):
    """
    Download Indian vehicle dataset from Roboflow.
    Dataset: vehicles-q0x2v (public, YOLO format)
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("Install roboflow: pip install roboflow")
        sys.exit(1)

    logger.info("Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-100").project("vehicles-q0x2v")
    dataset = project.version(2).download("yolov8", location=str(RAW_DIR / "roboflow"))

    logger.success(f"Downloaded to: {dataset.location}")
    reorganize_to_trinetra(Path(dataset.location))


# ── Source: Sample (no API key) ──────────────────────────────

def create_sample_dataset(n_images: int = 50):
    """
    Creates a tiny synthetic dataset for testing the pipeline.
    Uses solid-colour placeholder images with random YOLO-format labels.
    Replace with real data before training.
    """
    logger.warning("Creating SYNTHETIC sample dataset — for pipeline testing only!")
    logger.warning("Replace with real annotated data before actual training.")

    CLASS_NAMES = ['car', 'bus', 'bike', 'truck', 'auto_rickshaw', 'pedestrian']
    NUM_CLASSES  = len(CLASS_NAMES)

    colors = [
        (60, 120, 200), (220, 80, 40), (40, 180, 80),
        (200, 160, 20), (140, 40, 200), (200, 200, 200),
    ]

    all_images = []
    all_labels = []
    sample_dir = RAW_DIR / 'sample'
    sample_dir.mkdir(exist_ok=True)

    for i in tqdm(range(n_images), desc="Generating sample images"):
        h, w = 480, 640
        img = np.ones((h, w, 3), dtype=np.uint8) * 50   # dark background

        # Random road markings
        cv2.line(img, (w//2, 0), (w//2, h), (80, 80, 80), 2)

        label_lines = []
        n_objects = random.randint(1, 5)

        for _ in range(n_objects):
            cls_id = random.randint(0, NUM_CLASSES - 1)
            bw = random.randint(60, 180)
            bh = random.randint(40, 120)
            bx = random.randint(0, w - bw)
            by = random.randint(0, h - bh)

            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), colors[cls_id], -1)
            cv2.putText(img, CLASS_NAMES[cls_id], (bx + 4, by + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # YOLO format: class cx cy w h (normalised)
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h
            label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        img_path = sample_dir / f"img_{i:04d}.jpg"
        lbl_path = sample_dir / f"img_{i:04d}.txt"

        cv2.imwrite(str(img_path), img)
        lbl_path.write_text("\n".join(label_lines))

        all_images.append(img_path)
        all_labels.append(lbl_path)

    # Split into train / val / test
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)
    n = len(combined)
    tr = int(n * SPLIT_RATIOS[0])
    va = int(n * SPLIT_RATIOS[1])

    splits_data = {
        'train': combined[:tr],
        'val':   combined[tr:tr + va],
        'test':  combined[tr + va:],
    }

    for split, pairs in splits_data.items():
        for img_path, lbl_path in pairs:
            shutil.copy(img_path, PROC_DIR / 'images' / split / img_path.name)
            shutil.copy(lbl_path, PROC_DIR / 'labels' / split / lbl_path.name)

    _print_split_summary(splits_data)
    logger.success("Sample dataset ready in data/processed/")


# ── Reorganize downloaded data ────────────────────────────────

def reorganize_to_trinetra(src_dir: Path):
    """
    Move images + labels from a downloaded dataset into Trinetra's
    data/processed/{images,labels}/{train,val,test} structure.
    Assumes the source already has train/valid/test subdirs.
    """
    logger.info(f"Reorganizing from: {src_dir}")
    split_map = {'train': 'train', 'valid': 'val', 'test': 'test'}

    for src_split, dst_split in split_map.items():
        for kind in ['images', 'labels']:
            src = src_dir / src_split / kind
            dst = PROC_DIR / kind / dst_split
            if not src.exists():
                continue
            files = list(src.iterdir())
            for f in tqdm(files, desc=f"{src_split}/{kind}"):
                shutil.copy(f, dst / f.name)

    logger.success("Reorganization complete")
    _verify_dataset()


def _verify_dataset():
    logger.info("\n── Dataset summary ──")
    for split in SPLITS:
        imgs = list((PROC_DIR / 'images' / split).glob('*.jpg')) + \
               list((PROC_DIR / 'images' / split).glob('*.png'))
        lbls = list((PROC_DIR / 'labels' / split).glob('*.txt'))
        logger.info(f"  {split:5s}: {len(imgs):4d} images  {len(lbls):4d} labels")


def _print_split_summary(splits_data):
    logger.info("\n── Split summary ──")
    for split, pairs in splits_data.items():
        logger.info(f"  {split:5s}: {len(pairs)} samples")


# ── Entry point ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trinetra dataset preparation")
    parser.add_argument('--source', choices=['roboflow', 'sample'], default='sample',
                        help="Data source (default: sample)")
    parser.add_argument('--api-key', default='',
                        help="Roboflow API key (required for --source roboflow)")
    parser.add_argument('--n-samples', type=int, default=200,
                        help="Number of synthetic samples (sample mode only)")
    args = parser.parse_args()

    setup_dirs()

    if args.source == 'roboflow':
        if not args.api_key:
            logger.error("--api-key required for roboflow source")
            sys.exit(1)
        download_roboflow(args.api_key)

    elif args.source == 'sample':
        create_sample_dataset(args.n_samples)

    logger.success("\nDataset preparation complete! Next step: python scripts/train.py")


if __name__ == '__main__':
    main()
