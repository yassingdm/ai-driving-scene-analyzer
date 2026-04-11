import sys
import os
import yaml
from pathlib import Path

def validate_expert_dataset(yaml_path):
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        print(f"✗ Fichier YAML introuvable : {yaml_path}")
        return False

    # 1. Charger le YAML pour trouver les playlists
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    playlists = {
        "train": config.get('train'),
        "val": config.get('val')
    }

    print(f"🔎 Validation de l'expert : {yaml_path.stem.upper()}")
    
    all_ok = True
    for split, path in playlists.items():
        if not path:
            print(f"✗ Pas de chemin '{split}' défini dans le YAML")
            continue
        
        playlist_file = Path(path)
        if not playlist_file.exists():
            print(f"✗ Playlist {split} introuvable : {playlist_file}")
            all_ok = False
            continue

        # 2. Lire la playlist et vérifier les fichiers
        with open(playlist_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"--- Analyse du split {split} ({len(lines)} images) ---")
        
        missing_images = 0
        missing_labels = 0
        
        for img_path_str in lines:
            img_path = Path(img_path_str)
            
            # Vérifier l'image
            if not img_path.exists():
                missing_images += 1
                continue
            
            # Vérifier le label (YOLO cherche labels/ à la place de images/)
            # Exemple: images/train/001.jpg -> labels/train/001.txt
            label_path = Path(img_path_str.replace('images', 'labels')).with_suffix('.txt')
            
            if not label_path.exists():
                missing_labels += 1

        if missing_images > 0:
            print(f"  ✗ {missing_images} images introuvables au chemin indiqué.")
            all_ok = False
        if missing_labels > 0:
            print(f"  ✗ {missing_labels} labels (.txt) manquants dans le dossier labels/.")
            all_ok = False
        
        if missing_images == 0 and missing_labels == 0:
            print(f"  ✓ Split {split} valide.")

    return all_ok

if __name__ == "__main__":
    # Vous pouvez passer le nom d'un yaml spécifique en argument
    # Exemple: python validate_playlist_dataset.py data/data_nuit.yaml
    if len(sys.argv) > 1:
        target = sys.argv[1]
        success = validate_expert_dataset(target)
    else:
        print("Test de tous les fichiers YAML dans data/...")
        yaml_files = list(Path("data").glob("*.yaml"))
        success = True
        for y in yaml_files:
            if not validate_expert_dataset(y):
                success = False
    
    if success:
        print("\n🏆 Tout est prêt pour l'entraînement !")
    sys.exit(0 if success else 1)