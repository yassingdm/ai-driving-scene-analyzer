"""Render YOLO detections on images for quick QA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2


def color_for_name(name: str) -> tuple[int, int, int]:
    digest = hashlib.md5(name.encode("utf-8")).digest()
    return int(digest[0]), int(digest[1]), int(digest[2])


def draw_detections(image_path: Path, detections: list[dict], out_path: Path) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        return False

    for det in detections:
        class_name = str(det.get("class_name", ""))
        confidence = float(det.get("confidence", 0.0))
        bbox = det.get("bbox", {}) or {}
        x_min = int(float(bbox.get("x_min", 0)))
        y_min = int(float(bbox.get("y_min", 0)))
        x_max = int(float(bbox.get("x_max", 0)))
        y_max = int(float(bbox.get("y_max", 0)))

        color = color_for_name(class_name)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        label = f"{class_name} {confidence:.2f}".strip()
        cv2.putText(
            image,
            label,
            (x_min, max(0, y_min - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(out_path), image)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize detections on images")
    parser.add_argument("--detections", required=True, help="Path to detections.json")
    parser.add_argument("--out", required=True, help="Output directory for annotated images")
    parser.add_argument("--max", type=int, default=0, help="Limit number of images (0 = all)")
    args = parser.parse_args()

    detections_path = Path(args.detections)
    out_dir = Path(args.out)
    payload = json.loads(detections_path.read_text(encoding="utf-8"))
    results = payload.get("results", []) if isinstance(payload, dict) else []

    if args.max > 0:
        results = results[: args.max]

    written = 0
    skipped = 0
    for item in results:
        image_path = Path(item.get("image", ""))
        if not image_path.exists():
            skipped += 1
            continue
        detections = item.get("detections", []) or []
        out_path = out_dir / image_path.name
        if draw_detections(image_path, detections, out_path):
            written += 1
        else:
            skipped += 1

    print(f"Images ecrites: {written}")
    print(f"Images ignorees: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
