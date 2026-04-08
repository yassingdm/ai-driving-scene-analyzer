"""Convert BDD100K detection labels to YOLO format."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv.bdd_classes import BDD_CLASSES


BDD_ALIASES = {
    "traffic light": "traffic_light",
    "traffic sign": "traffic_sign",
}


def normalize_label(name: str) -> str:
    name = name.strip()
    return BDD_ALIASES.get(name, name)


def load_labels(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    return []


def iter_objects(entry: dict) -> Iterable[dict]:
    if "frames" in entry:
        frames = entry.get("frames") or []
        if frames:
            objects = frames[0].get("objects") or []
            for obj in objects:
                if isinstance(obj, dict):
                    yield obj
        return
    for label in entry.get("labels", []) or []:
        if isinstance(label, dict):
            yield label


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
    parser.add_argument(
        "--labels",
        required=True,
        help="Fichier labels BDD100K (json) ou dossier contenant des .json",
    )
    parser.add_argument("--images", required=True, help="Dossier images pour le split")
    parser.add_argument("--out", required=True, help="Dossier de sortie des labels YOLO")
    parser.add_argument("--write-empty", action="store_true", help="Ecrire des fichiers vides")
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Limiter au N premiers fichiers (dossier) ou N premieres entrees (fichier)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Tirer des fichiers aleatoirement (dossier uniquement)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed pour --shuffle",
    )
    parser.add_argument(
        "--image-width",
        type=float,
        default=1280.0,
        help="Largeur d'image a utiliser si absente des labels",
    )
    parser.add_argument(
        "--image-height",
        type=float,
        default=720.0,
        help="Hauteur d'image a utiliser si absente des labels",
    )
    args = parser.parse_args()

    labels_path = Path(args.labels)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_to_id = {name: idx for idx, name in enumerate(BDD_CLASSES)}

    if labels_path.is_dir():
        label_files = sorted(labels_path.glob("*.json"))
        if args.shuffle:
            rng = random.Random(args.seed)
            rng.shuffle(label_files)
        if args.max:
            label_files = label_files[: args.max]
        entries: list[dict] = []
        for label_file in label_files:
            entries.extend(load_labels(label_file))
    else:
        entries = load_labels(labels_path)
        if args.max:
            entries = entries[: args.max]
    unknown_labels = 0
    written = 0

    for entry in entries:
        name = entry.get("name", "")
        image_info = entry.get("imageAttributes", {})
        width = float(entry.get("width", image_info.get("width", 0)) or 0)
        height = float(entry.get("height", image_info.get("height", 0)) or 0)
        if width <= 0 or height <= 0:
            width = args.image_width
            height = args.image_height

        yolo_lines: list[str] = []
        for label in iter_objects(entry):
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
            stem = Path(name).stem if name else "label"
            out_path = out_dir / f"{stem}.txt"
            out_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            written += 1

    print(f"Labels ecrits: {written}")
    print(f"Labels inconnus: {unknown_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
