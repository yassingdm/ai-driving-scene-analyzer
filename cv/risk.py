"""Minimal risk scoring based on CV detections."""

from typing import Dict, List


WEIGHTS = {
    "pedestrian": 0.95,      # Critique - piéton
    "cyclist": 0.85,         # Très élevé - cycliste
    "motorcycle": 0.80,      # Très élevé - motocycliste
    "truck": 0.70,           # Élevé - camion lourd
    "bus": 0.60,             # Moyen-élevé - bus
    "car": 0.40,             # Moyen - voiture
    "rider": 0.75,           # Élevé - rider sur deux-roues
    "traffic_light": 0.15,   # Faible - contexte seulement
    "traffic_sign": 0.10,    # Très faible - contexte seulement
    "train": 0.50,           # Moyen - train (rare en dashcam)
}


def compute_risk_score(detections: List[Dict]) -> float:
    """Compute a simple risk score in [0, 1]."""
    if not detections:
        return 0.0

    score = 0.0
    for det in detections:
        class_name = det.get("class_name", "")
        conf = float(det.get("confidence", 0.0))
        score += WEIGHTS.get(class_name, 0.1) * conf

    return min(1.0, round(score, 4))


def risk_level(score: float) -> str:
    """Map numeric score to coarse risk level."""
    if score >= 0.75:
        return "critical"
    if score >= 0.5:
        return "high"
    if score >= 0.25:
        return "medium"
    if score > 0:
        return "low"
    return "safe"
