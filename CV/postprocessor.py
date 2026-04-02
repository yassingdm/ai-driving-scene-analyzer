"""Traitement des résultats de détection."""

from detector import Detection


def filter_by_confidence(
    detections: list[Detection],
    threshold: float = 0.5
) -> list[Detection]:
    """Filtrer les détections par confiance."""
    return [d for d in detections if d.confidence >= threshold]


def format_results(detections: list[Detection]) -> dict:
    """Convertir les détections au format de sortie."""
    
    return {
        "detections": [
            {
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": round(d.confidence, 2),
                "bbox": list(d.bbox),
            }
            for d in detections
        ],
    }
