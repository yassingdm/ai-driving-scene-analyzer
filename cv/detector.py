"""Wrapper YOLO pour détection d'objets."""

from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - dependance optionnelle
    YOLO = None


@dataclass
class Detection:
    """Résultat d'une détection."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float,  float, float]  # (x1, y1, x2, y2)


class YOLODetector:
    """Détecteur YOLO simple."""

    def __init__(self, model_name: str = "yolov8n"):
        """
        Initialiser le détecteur.
        
        Args:
            model_name: variante du modèle YOLO
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        if YOLO is None:
            self.model = None
            return
        weights = self.model_name if self.model_name.endswith(".pt") else f"{self.model_name}.pt"
        safe_globals = self._allow_ultralytics_safe_globals()
        try:
            import torch
        except Exception:
            self.model = YOLO(weights)
            return

        serialization = getattr(torch, "serialization", None)
        safe_context = getattr(serialization, "safe_globals", None) if serialization else None

        # Force weights_only=False for trusted checkpoints (Ultralytics).
        original_load = getattr(torch, "load", None)
        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        try:
            if callable(original_load):
                torch.load = _patched_load
            if safe_globals and callable(safe_context):
                with safe_context(safe_globals):
                    self.model = YOLO(weights)
            else:
                self.model = YOLO(weights)
        finally:
            if callable(original_load):
                torch.load = original_load

    def _allow_ultralytics_safe_globals(self) -> list[type]:
        """Allowlist Ultralytics classes for torch.load safety checks."""
        try:
            import torch
            from ultralytics.nn.tasks import DetectionModel
        except Exception:
            return []

        serialization = getattr(torch, "serialization", None)
        if serialization is None:
            return []

        add_safe_globals = getattr(serialization, "add_safe_globals", None)
        if callable(add_safe_globals):
            safe_globals = [DetectionModel]
            try:
                from torch.nn.modules.container import ModuleList, ModuleDict, Sequential
                safe_globals.extend([ModuleList, ModuleDict, Sequential])
            except Exception:
                pass
            for module_path, names in (
                ("ultralytics.nn.modules", ["Conv", "C2f", "C3", "SPPF", "Bottleneck", "Concat", "Detect", "DFL", "DWConv"]),
                ("ultralytics.nn.modules.conv", ["Conv", "DWConv"]),
                ("ultralytics.nn.modules.block", ["C2f", "C3", "SPPF", "Bottleneck", "Concat", "Detect", "DFL"]),
            ):
                try:
                    module = __import__(module_path, fromlist=names)
                except Exception:
                    continue
                for name in names:
                    obj = getattr(module, name, None)
                    if obj is not None:
                        safe_globals.append(obj)
            add_safe_globals(safe_globals)
            return safe_globals

        return []

    def detect(
        self,
        image_path: str,
        *,
        conf: float = 0.25,
        iou: float = 0.7,
        classes: list[int] | list[str] | None = None,
    ) -> list[Detection]:
        """
        Détecter les objets dans l'image.
        
        Args:
            image_path: chemin vers l'image
            
        Returns:
            Liste des détections
        """
        image = Image.open(image_path).convert("RGB")
        frame = np.asarray(image, dtype=np.uint8)

        if self.model is None:
            return self._detect_grid_style(frame)

        results = self.model.predict(source=frame, verbose=False, conf=conf, iou=iou)
        if not results:
            return []

        detections: list[Detection] = []
        result = results[0]
        names = getattr(result, "names", {}) or {}

        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            cls = int(box.cls[0].item()) if box.cls is not None else -1
            name = names.get(cls, "unknown")

            detections.append(
                Detection(
                    class_id=cls,
                    class_name=name,
                    confidence=conf,
                    bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                )
            )

        if classes:
            class_ids = {c for c in classes if isinstance(c, int)}
            class_names = {str(c) for c in classes if isinstance(c, str)}
            detections = [
                det
                for det in detections
                if (not class_ids or det.class_id in class_ids)
                and (not class_names or det.class_name in class_names)
            ]

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _detect_grid_style(self, frame: np.ndarray) -> list[Detection]:
        """Version rudimentaire type YOLO: une passe sur une grille."""
        height, width = frame.shape[:2]

        gray = frame.mean(axis=2)
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        energy = gx + gy

        # On baisse l'impact de la partie haute (souvent ciel) pour limiter les grosses boites.
        vertical_weight = np.linspace(0.35, 1.0, height, dtype=np.float32)[:, None]
        energy = energy * vertical_weight

        rows = 8
        cols = 12
        cell_h = max(1, height // rows)
        cell_w = max(1, width // cols)

        scores = np.zeros((rows, cols), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                y1 = r * cell_h
                y2 = height if r == rows - 1 else (r + 1) * cell_h
                x1 = c * cell_w
                x2 = width if c == cols - 1 else (c + 1) * cell_w
                scores[r, c] = float(energy[y1:y2, x1:x2].mean())

        score_max = float(scores.max()) if float(scores.max()) > 0 else 1.0
        threshold = float(np.percentile(scores, 86))
        active = scores >= threshold

        detections: list[Detection] = []
        visited = np.zeros_like(active, dtype=bool)

        for r in range(rows):
            for c in range(cols):
                if not active[r, c] or visited[r, c]:
                    continue

                # On fusionne les cellules actives voisines pour construire une bbox.
                stack = [(r, c)]
                visited[r, c] = True
                cells = []

                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if active[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))

                min_r = min(rr for rr, _ in cells)
                max_r = max(rr for rr, _ in cells)
                min_c = min(cc for _, cc in cells)
                max_c = max(cc for _, cc in cells)

                x1 = float(min_c * cell_w)
                y1 = float(min_r * cell_h)
                x2 = float(width if max_c == cols - 1 else (max_c + 1) * cell_w)
                y2 = float(height if max_r == rows - 1 else (max_r + 1) * cell_h)

                area = (x2 - x1) * (y2 - y1)
                if area < 0.01 * width * height:
                    continue
                if area > 0.24 * width * height:
                    continue
                if (x2 - x1) > 0.8 * width and (y2 - y1) > 0.45 * height:
                    continue
                if y1 < 0.15 * height and area > 0.10 * width * height:
                    continue

                local_score = max(scores[rr, cc] for rr, cc in cells)
                confidence = min(0.99, max(0.3, float(local_score / score_max)))

                class_id, class_name = self._guess_class((x1, y1, x2, y2), width, height)
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[:8]

    def _guess_class(
        self,
        bbox: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> tuple[int, str]:
        """Heuristique simple pour nommer les boîtes détectées."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        area_ratio = (box_w * box_h) / max(1.0, width * height)

        if cy > 0.60 * height and box_w > box_h:
            return 0, "car"
        if cy < 0.35 * height and box_h >= box_w and area_ratio < 0.03:
            return 2, "traffic_sign"
        if 0.25 * width <= cx <= 0.75 * width and box_h > box_w and area_ratio < 0.06:
            return 1, "pedestrian"
        if cy > 0.50 * height and area_ratio < 0.15:
            return 0, "car"
        return 3, "unknown"
