"""Tester YOLO sur un dossier d'images."""

from __future__ import annotations

import argparse
from pathlib import Path

from cv.detector import YOLODetector


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def main() -> int:
    parser = argparse.ArgumentParser(description="Tester YOLO sur data/")
    parser.add_argument("--data", default="data", help="Dossier a analyser")
    parser.add_argument("--model", default="yolov8n", help="Nom du modele YOLO")
    parser.add_argument("--max", type=int, default=0, help="Limite d'images (0 = tout)")
    parser.add_argument("--top", type=int, default=5, help="Nb de detections affichees")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"Dossier introuvable: {data_dir}")
        return 1

    detector = YOLODetector(args.model)
    images = iter_images(data_dir)
    if args.max > 0:
        images = images[: args.max]

    if not images:
        print(f"Aucune image dans {data_dir}")
        return 1

    for img in images:
        detections = detector.detect(str(img))
        print(f"\n{img}: {len(detections)} detections")
        for det in detections[: args.top]:
            print(f"  - {det.class_name} {det.confidence:.2f} {det.bbox}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
