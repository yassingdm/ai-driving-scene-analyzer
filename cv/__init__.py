"""Computer Vision module for vehicle and obstacle detection."""

from .detector import ObjectDetector
from .risk import compute_risk_score, risk_level

__all__ = ["ObjectDetector", "compute_risk_score", "risk_level"]
