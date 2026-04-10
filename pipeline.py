import json
from LLM.agent import analyze_scene

#on charge le json
with open("outputs/detections.json", "r") as f:
    data = json.load(f)

for entry in data["results"]:
    detections = entry["detections"]
    detections_reduites = [...]
    rapport = analyze_scene(detections_reduites)
    print(rapport)


detections_reduites = [
    {
        "class": d.get("class_name"),
        "confidence": d.get("confidence"),
        "bbox": [
            d["bbox"]["x_min"],
            d["bbox"]["y_min"],
            d["bbox"]["x_max"],
            d["bbox"]["y_max"]
        ]
    }
    for d in detections
]

detections_reduites = detections_reduites[:30]
rapport = analyze_scene(detections_reduites)
print(json.dumps(rapport, indent=4, ensure_ascii=False))
