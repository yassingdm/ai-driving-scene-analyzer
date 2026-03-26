"""Object detection using YOLOv8 for autonomous driving scenes."""


class ObjectDetector:
    """YOLO-based object detector for driving scenes."""

    def __init__(self, model_name: str = "yolov8n.pt"):
        """Initialize the detector with YOLO model."""
        self.model_name = model_name

    def detect(self, image_path: str, conf: float = 0.5):
        """Detect objects in an image."""
        pass

    def detect_from_frame(self, frame, conf: float = 0.5):
        """Detect objects in a video frame."""
        pass
