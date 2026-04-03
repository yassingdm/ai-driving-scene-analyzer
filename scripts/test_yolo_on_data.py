"""Tester YOLO sur un dossier d'images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv.detector import YOLODetector


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def main() -> int:
    parser = argparse.ArgumentParser(description="Tester YOLO sur data/")
    parser.add_argument("--data", default="data", help="Dossier a analyser")
    parser.add_argument("--model", default="yolov8n", help="Nom du modele YOLO")
    parser.add_argument("--conf", type=float, default=0.25, help="Seuil de confiance")
    parser.add_argument("--iou", type=float, default=0.7, help="Seuil IoU NMS")
    parser.add_argument(
        "--classes",
        default="",
        help="Liste de classes separees par virgule (ex: car,person,truck)",
    )
    parser.add_argument("--out", default="", help="Chemin de sortie JSON")
    parser.add_argument("--max", type=int, default=0, help="Limite d'images (0 = tout)")
    parser.add_argument("--top", type=int, default=5, help="Nb de detections affichees")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"Dossier introuvable: {data_dir}")
        return 1

    detector = YOLODetector(args.model)
    class_filter = [c.strip() for c in args.classes.split(",") if c.strip()]
    images = iter_images(data_dir)
    if args.max > 0:
        images = images[: args.max]

    if not images:
        print(f"Aucune image dans {data_dir}")
        return 1

    output = []
    for img in images:
        detections = detector.detect(str(img), conf=args.conf, iou=args.iou, classes=class_filter or None)
        print(f"\n{img}: {len(detections)} detections")
        for det in detections[: args.top]:
            print(f"  - {det.class_name} {det.confidence:.2f} {det.bbox}")
        if args.out:
            output.append(
                {
                    "image": str(img),
                    "detections": [
                        {
                            "class_id": d.class_id,
                            "class_name": d.class_name,
                            "confidence": d.confidence,
                            "bbox": {
                                "x_min": d.bbox[0],
                                "y_min": d.bbox[1],
                                "x_max": d.bbox[2],
                                "y_max": d.bbox[3],
                            },
                        }
                        for d in detections
                    ],
                }
            )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"results": output}, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
