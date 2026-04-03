"""Validate detection JSON output and print quick stats."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


REQUIRED_BBOX_KEYS = {"x_min", "y_min", "x_max", "y_max"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_bbox(bbox: dict) -> list[str]:
    errors: list[str] = []
    if not REQUIRED_BBOX_KEYS.issubset(bbox.keys()):
        missing = REQUIRED_BBOX_KEYS - bbox.keys()
        errors.append(f"missing_bbox_keys={sorted(missing)}")
        return errors

    x_min = float(bbox["x_min"])
    y_min = float(bbox["y_min"])
    x_max = float(bbox["x_max"])
    y_max = float(bbox["y_max"])

    if x_min >= x_max:
        errors.append("x_min>=x_max")
    if y_min >= y_max:
        errors.append("y_min>=y_max")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate detections JSON")
    parser.add_argument("--input", default="outputs/detections.json", help="Fichier JSON a verifier")
    parser.add_argument(
        "--classes",
        default="",
        help="Classes attendues (ex: car,person,truck,traffic light)",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"Fichier introuvable: {path}")
        return 1

    data = load_json(path)
    results = data.get("results", [])

    class_filter = {c.strip() for c in args.classes.split(",") if c.strip()}

    stats = {
        "images": 0,
        "images_with_detections": 0,
        "detections": 0,
        "errors": 0,
    }
    class_counts: Counter[str] = Counter()
    errors: list[str] = []

    for item in results:
        stats["images"] += 1
        detections = item.get("detections", [])
        if detections:
            stats["images_with_detections"] += 1
        for det in detections:
            stats["detections"] += 1
            class_name = str(det.get("class_name", ""))
            if class_name:
                class_counts[class_name] += 1
            if class_filter and class_name not in class_filter:
                errors.append(f"unexpected_class={class_name} image={item.get('image','')}")
            conf = det.get("confidence")
            if conf is None or not (0.0 <= float(conf) <= 1.0):
                errors.append(f"invalid_confidence={conf} image={item.get('image','')}")
            bbox = det.get("bbox", {})
            bbox_errors = validate_bbox(bbox) if isinstance(bbox, dict) else ["bbox_not_dict"]
            for err in bbox_errors:
                errors.append(f"{err} image={item.get('image','')}")

    stats["errors"] = len(errors)

    print("Stats")
    print(f"- images: {stats['images']}")
    print(f"- images_with_detections: {stats['images_with_detections']}")
    print(f"- detections: {stats['detections']}")

    if class_counts:
        print("Top classes")
        for name, count in class_counts.most_common(10):
            print(f"- {name}: {count}")

    if errors:
        print("Errors (first 20)")
        for err in errors[:20]:
            print(f"- {err}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
