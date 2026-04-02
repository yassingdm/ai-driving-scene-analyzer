"""Object detection using YOLOv8 for autonomous driving scenes."""

from typing import Any, Dict, List


class ObjectDetector:
    """YOLO-based object detector for driving scenes."""

    DEFAULT_CLASSES = [
        "car",
        "truck",
        "bus",
        "pedestrian",
        "cyclist",
        "motorcycle",
        "traffic light",
        "traffic sign",
        "rider",
        "train",
    ]

    def __init__(self, model_name: str = "yolov8n.pt"):
        """Initialize the detector with YOLO model."""
        self.model_name = model_name
        self.model = None
        self.ready = False

    def load(self) -> None:
        """Prepare model loading step (implementation in next commit)."""
        # TODO: load ultralytics YOLO model here.
        self.ready = True

    def is_ready(self) -> bool:
        """Return True if the detector is initialized and ready."""
        return self.ready

    def _validate_confidence(self, conf: float) -> float:
        """Clamp confidence threshold to [0.0, 1.0]."""
        return max(0.0, min(1.0, float(conf)))

    def _format_detection(
        self,
        bbox: List[int],
        class_name: str,
        confidence: float,
    ) -> Dict[str, Any]:
        """Normalize a detection structure for downstream modules."""
        return {
            "bbox": bbox,
            "class_name": class_name,
            "confidence": round(confidence, 4),
        }

    def detect(self, image_path: str, conf: float = 0.5):
        """Detect objects in an image."""
        conf = self._validate_confidence(conf)
        # TODO: implement file/image loading and YOLO inference.
        return {
            "source": image_path,
            "confidence_threshold": conf,
            "detections": [],
        }

    def detect_from_frame(self, frame, conf: float = 0.5):
        """Detect objects in a video frame."""
        conf = self._validate_confidence(conf)
        # TODO: implement real-time frame inference.
        _ = frame
        return {
            "source": "frame",
            "confidence_threshold": conf,
            "detections": [],
        }

    def filter_allowed_classes(
        self,
        detections: List[Dict[str, Any]],
        allowed_classes: List[str],
    ) -> List[Dict[str, Any]]:
        """Keep only detections in allowed class names."""
        allowed = set(allowed_classes)
        return [d for d in detections if d.get("class_name") in allowed]
