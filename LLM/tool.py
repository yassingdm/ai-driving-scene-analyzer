import math

def calculDistance(bbox1, bbox2):
    """
    Calcule la distance entre les centres de deux bounding boxes.
    bbox = {x_min, y_min, x_max, y_max}
    """
    x1_center = (bbox1["x_min"] + bbox1["x_max"]) / 2
    y1_center = (bbox1["y_min"] + bbox1["y_max"]) / 2

    x2_center = (bbox2["x_min"] + bbox2["x_max"]) / 2
    y2_center = (bbox2["y_min"] + bbox2["y_max"]) / 2

    distance = math.sqrt((x2_center - x1_center)**2 + (y2_center - y1_center)**2)
    return distance


CalculDistance_tool = {
    "name": "calculDistance",
    "description": "Calcule la distance entre deux objets détectés à partir de leurs bounding boxes.",
    "parameters": {
        "type": "object",
        "properties": {
            "bbox1": {"type": "object"},
            "bbox2": {"type": "object"}
        },
        "required": ["bbox1", "bbox2"]
    },
}