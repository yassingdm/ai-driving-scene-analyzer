import json
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
DEFAULT_PATH = Path(__file__).resolve().parent.parent
path= os.getenv("PROJECT_PATH", str(DEFAULT_PATH))
OUTPUT_DIR = f'{path}/data/txt' 


def build_tasks():
    tasks = []
    labels_root = Path(path) / "data" / "labels"
    images_root = Path(path) / "data" / "images"

    for split in ("train", "val", "test"):
        labels_dir = labels_root / split
        if labels_dir.exists() and labels_dir.is_dir():
            tasks.append(
                {
                    "nom": split.upper(),
                    "json_dir": str(labels_dir),
                    "image_dir_prefix": str(images_root / split),
                    "fichier_prefix": f"{split}_",
                }
            )

    # Fallback pour conserver la compatibilite avec l'ancien format "gros JSON".
    if not tasks:
        tasks = [
            {
                "nom": "ENTRAINEMENT",
                "json_path": f"{path}/data/det_v2_train_release.json",
                "image_dir_prefix": f"{path}/data/images/train",
                "fichier_prefix": "train_",
            },
            {
                "nom": "VALIDATION",
                "json_path": f"{path}/data/det_v2_val_release.json",
                "image_dir_prefix": f"{path}/data/images/val",
                "fichier_prefix": "val_",
            },
        ]

    return tasks
# ==========================================


def _read_items_from_json_file(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        # Certains jeux ont un objet unique par fichier.
        if "name" in payload:
            return [payload]
        # Cas eventuel d'un wrapper objet.
        for key in ("items", "annotations", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def _empty_scenarios():
    return {
        'urbain': [], 'nuit': [], 'pluie_brouillard': [],
        'autoroute': [], 'parking': [], 'pieton': [], 'scolaire': []
    }


def _resolve_image_path(img_prefix, img_name):
    base_path = os.path.join(img_prefix, img_name)
    _, ext = os.path.splitext(base_path)
    if ext:
        return base_path.replace('\\', '/')

    for candidate_ext in ('.jpg', '.jpeg', '.png', '.webp'):
        candidate = f"{base_path}{candidate_ext}"
        if os.path.exists(candidate):
            return candidate.replace('\\', '/')

    # Fallback if file does not exist yet in local workspace.
    return f"{base_path}.jpg".replace('\\', '/')


def _append_scenario_paths(scenarios, item, img_prefix):
    img_name = item.get('name')
    if not img_name:
        return

    img_path = _resolve_image_path(img_prefix, img_name)

    attrs = item.get('attributes', {})
    scene = attrs.get('scene', '')
    timeofday = attrs.get('timeofday', '')
    weather = attrs.get('weather', '')

    labels = item.get('labels') or []
    objets_presents = [label.get('category') for label in labels if 'category' in label]

    if scene == 'city street':
        scenarios['urbain'].append(img_path)
    if timeofday == 'night':
        scenarios['nuit'].append(img_path)
    if weather in ['rainy', 'foggy']:
        scenarios['pluie_brouillard'].append(img_path)
    if scene == 'highway':
        scenarios['autoroute'].append(img_path)
    if scene == 'parking lot':
        scenarios['parking'].append(img_path)
    if 'pedestrian' in objets_presents:
        scenarios['pieton'].append(img_path)
    if scene == 'residential' and 'traffic sign' in objets_presents:
        scenarios['scolaire'].append(img_path)


def _write_lines(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(f"{line}\n")


def _write_global_file(file_prefix, scenarios):
    global_seen = set()
    global_paths = []
    for liste_chemins in scenarios.values():
        for chemin in liste_chemins:
            if chemin not in global_seen:
                global_seen.add(chemin)
                global_paths.append(chemin)

    global_file = os.path.join(OUTPUT_DIR, f"{file_prefix}_global.txt")
    _write_lines(global_file, global_paths)

def traiter_fichier_json(config):
    nom_tache = config["nom"]
    img_prefix = config["image_dir_prefix"]
    file_prefix = config["fichier_prefix"]

    json_files = []
    if "json_dir" in config:
        json_dir = config["json_dir"]
        if not os.path.exists(json_dir):
            print(f"Le dossier JSON '{json_dir}' est introuvable.")
            return
        json_files = sorted(Path(json_dir).glob("*.json"))
        if not json_files:
            print(f"Aucun fichier JSON trouve dans '{json_dir}'.")
            return
    else:
        json_path = config["json_path"]
        if not os.path.exists(json_path):
            print(f"Le fichier JSON '{json_path}' est introuvable. Veuillez verifier le chemin.")
            return
        json_files = [Path(json_path)]

    scenarios = _empty_scenarios()

    for json_file in json_files:
        data = _read_items_from_json_file(json_file)

        for item in data:
            _append_scenario_paths(scenarios, item, img_prefix)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for nom_scenario, liste_chemins in scenarios.items():
        fichier_txt = os.path.join(OUTPUT_DIR, f"{file_prefix}{nom_scenario}.txt")
        _write_lines(fichier_txt, liste_chemins)

    _write_global_file(nom_tache.lower(), scenarios)

    return scenarios
                

def main():
    for tache in build_tasks():
        traiter_fichier_json(tache)
        

if __name__ == "__main__":
    main()