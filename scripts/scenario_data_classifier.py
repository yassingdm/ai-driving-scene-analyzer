import json
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
DEFAULT_PATH = Path(__file__).resolve().parent.parent
path= os.getenv("PROJECT_PATH", str(DEFAULT_PATH))
OUTPUT_DIR = f'{path}/data/txt' 

TACHES = [
    {
        "nom": "ENTRAÎNEMENT",
        "json_path": f'{path}/data/det_v2_train_release.json',
        "image_dir_prefix": f'{path}/data/images/train/',
        "fichier_prefix": "train_"
    },
    {
        "nom": "VALIDATION",
        "json_path": f'{path}/data/det_v2_val_release.json',
        "image_dir_prefix": f'{path}/data/images/val/',
        "fichier_prefix": "val_"
    }
]
# ==========================================

def traiter_fichier_json(config):
    nom_tache = config["nom"]
    json_path = config["json_path"]
    img_prefix = config["image_dir_prefix"]
    file_prefix = config["fichier_prefix"]
    

    if not os.path.exists(json_path):
        print(f"Le fichier JSON '{json_path}' est introuvable. Veuillez vérifier le chemin.")
        return

    scenarios = {
        'urbain': [], 'nuit': [], 'pluie_brouillard': [],
        'autoroute': [], 'parking': [], 'pieton': [], 'scolaire': []
    }

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        img_name = item.get('name')
        if not img_name:
            continue
            
        img_path = os.path.join(img_prefix, img_name).replace('\\', '/')
        
        attrs = item.get('attributes', {})
        scene = attrs.get('scene', '')
        timeofday = attrs.get('timeofday', '')
        weather = attrs.get('weather', '')
        
        labels = item.get('labels')or []
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

    for nom_scenario, liste_chemins in scenarios.items():
        fichier_txt = os.path.join(OUTPUT_DIR, f"{file_prefix}{nom_scenario}.txt")
        
        with open(fichier_txt, 'w', encoding='utf-8') as f:
            for chemin in liste_chemins:
                f.write(f"{chemin}\n")
                

def main():
    for tache in TACHES:
        traiter_fichier_json(tache)
        

if __name__ == "__main__":
    main()