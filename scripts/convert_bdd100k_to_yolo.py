"""Convert BDD100K detection labels to YOLO format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cv.bdd_classes import BDD_CLASSES


BDD_ALIASES = {
    "traffic light": "traffic_light",
    "traffic sign": "traffic_sign",
}


def normalize_label(name: str) -> str:
    name = name.strip()
    return BDD_ALIASES.get(name, name)


def load_labels(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_yolo_line(box: dict, class_id: int, img_w: float, img_h: float) -> str:
    x_min = float(box["x1"])
    y_min = float(box["y1"])
    x_max = float(box["x2"])
    y_max = float(box["y2"])

    x_min = max(0.0, min(x_min, img_w))
    x_max = max(0.0, min(x_max, img_w))
    y_min = max(0.0, min(y_min, img_h))
    y_max = max(0.0, min(y_max, img_h))

    w = max(0.0, x_max - x_min)
    h = max(0.0, y_max - y_min)
    cx = x_min + w / 2.0
    cy = y_min + h / 2.0

    if img_w <= 0 or img_h <= 0:
        return ""

    return f"{class_id} {cx / img_w:.6f} {cy / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert BDD100K labels to YOLO")
    parser.add_argument("--labels", required=True, help="Fichier labels BDD100K (json)")
    parser.add_argument("--images", required=True, help="Dossier images pour le split")
    parser.add_argument("--out", required=True, help="Dossier de sortie des labels YOLO")
    parser.add_argument("--write-empty", action="store_true", help="Ecrire des fichiers vides")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    images_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_to_id = {name: idx for idx, name in enumerate(BDD_CLASSES)}

    entries = load_labels(labels_path)
    missing_images = 0
    unknown_labels = 0
    written = 0

    for entry in entries:
        name = entry.get("name", "")
        if not name:
            continue
        image_path = images_dir / name
        if not image_path.exists():
            missing_images += 1
            continue

        image_info = entry.get("imageAttributes", {})
        width = float(entry.get("width", image_info.get("width", 0)) or 0)
        height = float(entry.get("height", image_info.get("height", 0)) or 0)

        yolo_lines: list[str] = []
        for label in entry.get("labels", []) or []:
            category = normalize_label(str(label.get("category", "")))
            if category not in class_to_id:
                unknown_labels += 1
                continue
            box2d = label.get("box2d")
            if not isinstance(box2d, dict):
                continue
            line = to_yolo_line(box2d, class_to_id[category], width, height)
            if line:
                yolo_lines.append(line)

        if yolo_lines or args.write_empty:
            out_path = out_dir / f"{Path(name).stem}.txt"
            out_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            written += 1

    print(f"Labels ecrits: {written}")
    print(f"Images manquantes: {missing_images}")
    print(f"Labels inconnus: {unknown_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
