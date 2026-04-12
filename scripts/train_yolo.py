"""Fine-tune YOLOv8 on BDD100K labels."""

from __future__ import annotations

import argparse
import os
import logging
from pathlib import Path

import torch
from ultralytics import YOLO
import ultralytics.utils as ultralytics_utils


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


def _resolve_data_path(data_arg: str) -> Path:
    p = Path(data_arg)
    if p.exists():
        return p

    script_root = Path(__file__).resolve().parent.parent
    candidates = [
        script_root / p,
        script_root / "data" / "yaml" / p.name,
        script_root / "data" / p.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return p


def _resolve_device(device_arg: str) -> str:
    if device_arg.lower() == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        return device_arg

    print("CUDA indisponible avec ce PyTorch. Bascule automatique sur CPU.")
    return "cpu"


def _set_epoch_only_logging(enabled: bool) -> None:
    if enabled:
        ultralytics_utils.VERBOSE = False
        ultralytics_utils.LOGGER.setLevel(logging.INFO)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on BDD100K")
    parser.add_argument("--data", default="data.yaml", help="Path to data.yaml")
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
    parser.add_argument("--quiet", action="store_true", help="Reduce console logs (disable verbose batch output)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--epoch-only", action="store_true", help="Show only epoch-level logs, not batch progress bars")
    args = parser.parse_args()

    data_path = _resolve_data_path(args.data)
    if not data_path.exists():
        raise SystemExit(f"data.yaml not found: {data_path}")

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"

    if args.quiet or args.epoch_only:
        _set_epoch_only_logging(True)

    device = _resolve_device(args.device)
    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        resume=args.resume,
        verbose=not args.quiet,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
