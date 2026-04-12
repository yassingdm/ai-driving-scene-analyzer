# AI Driving Scene Analyzer

Application de démonstration qui combine:
- CV avec YOLOv8 (détection d'objets routiers)
- LLM (résumé, analyse de risque, recommandations)
- UI Streamlit (upload image + visualisation)

## Statut du projet
- Etat actuel: pipeline fonctionnel avec modèle global entraîné.
- Détection: modèle custom disponible (`best.pt`) + modèle YOLO de base en fallback.
- Entraînement BDD100K: réalisé (train/val globaux générés).

## Pipeline
Image -> YOLO -> JSON de détections -> LLM -> Rapport + audio

## Structure utile
```
app/app.py                  # Interface Streamlit
cv/detector.py              # Wrapper YOLO + fallback heuristique
LLM/agent.py                # Appel LLM + parsing robuste de réponse
scripts/test_yolo_on_data.py
scripts/train_yolo.py
outputs/detections.json
```

## Prérequis
- Python 3.11 (venv local recommandé)
- Accès API Groq (clé API)

## Installation
Depuis la racine du projet:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Configuration
Créer un fichier `.env` à la racine:

```env
API_KEY=<votre_cle_groq>
PROJECT_PATH=<chemin_absolu_dossier_parent_projet>
```

## Lancer l'application (important)
Toujours lancer Streamlit avec le Python du venv du projet:

```powershell
.venv\Scripts\python.exe -m streamlit run app/app.py
```

Ne pas utiliser `streamlit run app/app.py` sans préciser l'interpréteur, sinon Windows/Conda peut prendre un autre Python et YOLO ne sera pas chargé.

## Paramètres CV recommandés (post-entraînement)
Dans la sidebar Streamlit:
- Seuil confiance: 0.40
- Seuil IoU: 0.50
- Sortie brute YOLO: off
- Classes CSV: car,truck,bus,person,bicycle,motorcycle,traffic light,stop sign,train
- Filtre anti-faux positifs dashcam: on
- Mode nuit auto: on

## Utiliser le modèle entraîné (`best.pt`)
Après entraînement, le meilleur poids se trouve dans un dossier `runs/.../weights/best.pt`.

Exemple d'évaluation sur le jeu de test:

```powershell
.venv\Scripts\python.exe scripts/test_yolo_on_data.py --data data/images/test --model runs/detect/runs/train/expert_global_1h_quiet2/weights/best.pt --conf 0.35 --iou 0.5 --max 200
```

## Sortie JSON des détections
```powershell
.venv\Scripts\python.exe scripts/test_yolo_on_data.py --data data/images/test --model runs/detect/runs/train/expert_global_1h_quiet2/weights/best.pt --max 200 --out outputs/detections_expert_global.json
.venv\Scripts\python.exe scripts/validate_detections.py --input outputs/detections_expert_global.json --classes pedestrian,cyclist,motorcycle,truck,bus,car,traffic_light,traffic_sign,rider,train
.venv\Scripts\python.exe scripts/visualize_detections.py --detections outputs/detections_expert_global.json --out outputs/viz_100 --max 100
```

## Entraînement global (déjà utilisé)
Commande de référence pour relancer un entraînement global:

```powershell
.venv\Scripts\python.exe scripts/train_yolo.py --data data.yaml --model yolov8n.pt --epochs 12 --imgsz 448 --batch 16 --workers 8 --patience 4 --name expert_global_1h_quiet --device 0 --epoch-only --no-wandb
```

## Problèmes connus
- Performance variable selon luminosité, densité de scène et angle dashcam (surtout nuit/pluie).
- Certaines classes restent bruyantes (ex: `traffic_sign`) selon le seuil de confiance.
- Le temps d'entraînement peut dépasser 1h selon la taille du split global.

## Dépannage rapide
- `ModuleNotFoundError` sur un package: réinstaller dans le venv avec `.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- Message "mauvais interpréteur Python": relancer avec `.venv\Scripts\python.exe -m streamlit run app/app.py`.
- Message "YOLO non chargé": vérifier que l'app tourne bien dans le venv et cliquer "Recharger le modèle".
