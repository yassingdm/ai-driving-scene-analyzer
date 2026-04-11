# AI Driving Scene Analyzer

Application de démonstration qui combine:
- CV avec YOLOv8 (détection d'objets routiers)
- LLM (résumé, analyse de risque, recommandations)
- UI Streamlit (upload image + visualisation)

## Statut du projet
- Etat actuel: pipeline fonctionnel avant fine-tuning.
- Détection: modèle YOLO pré-entraîné.
- Fine-tuning BDD100K: non démarré dans ce flux de run.

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

## Paramètres CV recommandés (pré-fine-tuning)
Dans la sidebar Streamlit:
- Seuil confiance: 0.40
- Seuil IoU: 0.50
- Sortie brute YOLO: off
- Classes CSV: car,truck,bus,person,bicycle,motorcycle,traffic light,stop sign,train
- Filtre anti-faux positifs dashcam: on
- Mode nuit auto: on

## Tester le module CV seul
```powershell
.venv\Scripts\python.exe scripts/test_yolo_on_data.py --data data/test --model yolov8n --conf 0.4 --iou 0.5 --classes "car,truck,bus,person,bicycle,motorcycle,traffic light,stop sign,train" --max 10 --top 10
```

## Sortie JSON des détections
```powershell
.venv\Scripts\python.exe scripts/test_yolo_on_data.py --data data/test --model yolov8n --max 50 --out outputs/detections.json
.venv\Scripts\python.exe scripts/validate_detections.py --input outputs/detections.json --classes car,person,truck,traffic\ light
```

## Fine-tuning (prochaine étape)
Commande disponible, non utilisée dans le flux courant:

```powershell
.venv\Scripts\python.exe scripts/train_yolo.py --data data/data.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16
```

## Problèmes connus (pré-fine-tuning)
- Performance variable selon luminosité, densité de scène et angle dashcam.
- Certains faux positifs/faux négatifs persistent sans modèle spécialisé dataset.

## Dépannage rapide
- `ModuleNotFoundError` sur un package: réinstaller dans le venv avec `.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- Message "mauvais interpréteur Python": relancer avec `.venv\Scripts\python.exe -m streamlit run app/app.py`.
- Message "YOLO non chargé": vérifier que l'app tourne bien dans le venv et cliquer "Recharger le modèle".
