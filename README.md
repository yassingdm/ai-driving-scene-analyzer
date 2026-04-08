# AI Driving Scene Analyzer

## Description
Application qui analyse des scènes de conduite avec :
- Computer Vision (YOLO)
- LLM (analyse + recommandations)

## Pipeline
Image → Détection → Analyse → Interface

## Installation
```bash
pip install -r requirements.txt
```

## Run
```bash
streamlit run app/app.py
```

## Phase 1: Vision par Ordinateur

### Dataset - BDD100K
Utilise le dataset **BDD100K** (100K images dashcam).

**Structure:**
```
data/
├── data.yaml      (config YOLO)
├── train/         (images JPG)
├── val/           (images JPG)
└── test/          (images JPG)
```

**Validation intégrité:**
```bash
python scripts/validate_dataset.py
```

**Conversion BDD100K -> YOLO (annotations):**
```bash
python scripts/convert_bdd100k_to_yolo.py \
	--labels path/to/bdd100k_labels_images_train.json \
	--images data/train \
	--out data/train/labels \
	--write-empty
```

### Classes (10)
- **Critique:** pedestrian (0.95), cyclist (0.85), motorcycle (0.80)
- **Véhicules:** truck (0.70), bus (0.60), car (0.40)
- **Contexte:** traffic_light, traffic_sign, rider, train

### Détection
- Modèle: **YOLOv8** (pré-entraîné)
- Approche: Fine-tuning sur BDD100K (Phase 2)
- Métriques: mAP@0.5, mAP@0.5:0.95, Precision, Recall

### Phase 2: Test YOLO pré-entraîné
```bash
python scripts/test_yolo_on_data.py --data data --model yolov8n
```

### Phase 3: Fine-tuning YOLO sur BDD100K
```bash
python scripts/train_yolo.py \
	--data data/data.yaml \
	--model yolov8n.pt \
	--epochs 50 \
	--imgsz 640 \
	--batch 16
```

### Sortie détections + validation
```bash
python scripts/test_yolo_on_data.py --data data/test --model yolov8n --max 50 --out outputs/detections.json
python scripts/validate_detections.py --input outputs/detections.json --classes car,person,truck,traffic\ light
```

### Évaluation Risque
Scoring basé sur classes détectées + confiance:
- Piétons/cyclistes = risque critique (0.85-0.95)
- Véhicules lourds = risque moyen-élevé (0.40-0.70)
- Contexte routier = référence seulement

Voir `cv/risk.py` pour détails des poids.
