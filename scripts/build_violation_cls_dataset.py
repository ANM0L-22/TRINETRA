"""
Build violation classification dataset from uploaded images using ordered labels.

Default behavior maps the latest 10 image files from static/uploads to the labels
provided by user:
1 no_helmet
2 red_light_violation
3 wrong_side_driving
4 tampered_plate
5 no_helmet
6 no_helmet
7 wrong_side_driving
8 wrong_side_driving
9 blackened_window
10 tampered_plate

Output layout:
  data/violation_cls/train/<class_name>/*.jpg
  data/violation_cls/val/<class_name>/*.jpg
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List
from collections import defaultdict


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUT_DIR = BASE_DIR / "data" / "violation_cls"

DEFAULT_LABELS = [
    "no_helmet",
    "red_light_violation",
    "wrong_side_driving",
    "tampered_plate",
    "no_helmet",
    "no_helmet",
    "wrong_side_driving",
    "wrong_side_driving",
    "blackened_window",
    "tampered_plate",
]

VALID_CLASSES = {
    "no_helmet",
    "red_light_violation",
    "wrong_side_driving",
    "tampered_plate",
    "blackened_window",
}


def latest_images(n: int) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp"}
    files = [p for p in UPLOAD_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:n]


def build(labels: List[str], val_ratio: float = 0.2) -> None:
    for l in labels:
        if l not in VALID_CLASSES:
            raise ValueError(f"Invalid class label: {l}")

    imgs = latest_images(len(labels))
    if len(imgs) < len(labels):
        raise RuntimeError(f"Need {len(labels)} images, found only {len(imgs)} in {UPLOAD_DIR}")

    # Keep chronological mapping: oldest of selected -> label1, newest -> labelN
    imgs = list(reversed(imgs))

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    for split in ["train", "val"]:
        for cls in sorted(VALID_CLASSES):
            (OUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    class_to_items = defaultdict(list)
    for idx, (img, label) in enumerate(zip(imgs, labels), start=1):
        class_to_items[label].append((idx, img))

    random.seed(42)
    for cls, items in class_to_items.items():
        random.shuffle(items)

        # Ensure every class appears in validation.
        if len(items) == 1:
            idx, img = items[0]
            for split in ["train", "val"]:
                dst_name = f"img_{idx:03d}_{img.stem}{img.suffix.lower()}"
                dst = OUT_DIR / split / cls / dst_name
                shutil.copy2(img, dst)
                print(f"[{split}] {img.name} -> {cls}")
            continue

        cls_val_count = max(1, int(len(items) * val_ratio))
        val_items = set(items[:cls_val_count])

        for idx, img in items:
            split = "val" if (idx, img) in val_items else "train"
            dst_name = f"img_{idx:03d}_{img.stem}{img.suffix.lower()}"
            dst = OUT_DIR / split / cls / dst_name
            shutil.copy2(img, dst)
            print(f"[{split}] {img.name} -> {cls}")

    print(f"\nDataset ready at: {OUT_DIR}")


def parse_labels(raw: str) -> List[str]:
    labels = [x.strip() for x in raw.split(",") if x.strip()]
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build violation classification dataset from uploads")
    parser.add_argument(
        "--labels",
        type=str,
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated labels in image order",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    labels = parse_labels(args.labels)
    build(labels=labels, val_ratio=args.val_ratio)


if __name__ == "__main__":
    main()
