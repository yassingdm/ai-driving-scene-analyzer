"""Keep only images that have labels (and optionally sample a subset)."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune dataset to labeled images only")
    parser.add_argument("--images", required=True, help="Input images directory")
    parser.add_argument("--labels", required=True, help="Input labels directory (.txt)")
    parser.add_argument("--out-images", required=True, help="Output images directory")
    parser.add_argument("--out-labels", required=True, help="Output labels directory")
    parser.add_argument("--max", type=int, default=None, help="Limit to N labeled samples")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle labeled samples")
    parser.add_argument("--seed", type=int, default=42, help="Seed for --shuffle")
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    out_images = Path(args.out_images)
    out_labels = Path(args.out_labels)
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    label_files = sorted([p for p in labels_dir.iterdir() if p.suffix == ".txt"])
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(label_files)
    if args.max:
        label_files = label_files[: args.max]

    copied_images = 0
    copied_labels = 0
    missing_images = 0

    for label_file in label_files:
        stem = label_file.stem
        image_path = find_image(images_dir, stem)
        if image_path is None:
            missing_images += 1
            continue
        shutil.copy2(image_path, out_images / image_path.name)
        shutil.copy2(label_file, out_labels / label_file.name)
        copied_images += 1
        copied_labels += 1

    print(f"Images copied : {copied_images}")
    print(f"Labels copied : {copied_labels}")
    print(f"Missing images: {missing_images}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
