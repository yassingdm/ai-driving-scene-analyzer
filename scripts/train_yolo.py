"""Fine-tune YOLOv8 on BDD100K labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def _allow_ultralytics_weights() -> None:
    try:
        from ultralytics.nn.tasks import DetectionModel
    except Exception:
        return
    torch.serialization.add_safe_globals([DetectionModel])


_allow_ultralytics_weights()


_torch_load = torch.load


def _torch_load_trusted(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_trusted


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on BDD100K")
    parser.add_argument("--data", default="data/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="0", help="Device id or cpu")
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--project", default="runs/train", help="Output project dir")
    parser.add_argument("--name", default="bdd100k", help="Run name")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"data.yaml not found: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
