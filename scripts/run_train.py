import subprocess
import sys
import argparse
import os
from pathlib import Path
path="/home/bastos/cours/projIA/ai-driving-scene-analyzer"

def run_step(command):
    print(f"\n--- EXÉCUTION : {command} ---")
    process = subprocess.run(command, shell=True)
    if process.returncode != 0:
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Pipeline complet : Préparation + Entraînement")
    parser.add_argument("--data", required=True, help="Chemin vers le fichier .yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Modèle de base")
    parser.add_argument("--epochs", type=int, default=50, help="Nombre d'époques")
    parser.add_argument("--imgsz", type=int, default=640, help="Taille d'image")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--name", help="Nom de l'entraînement (par défaut extrait du yaml)")
    
    args = parser.parse_args()

    scenario = Path(args.data).stem.replace('data_', '')
    run_name = args.name if args.name else f"expert_{scenario}"

    print(f"Démarrage de l'entrainement pour : {scenario.upper()}")

    if not os.path.exists(f"{path}/data/labels/train") or not os.path.exists(f"{path}/data/labels/val"):
        run_step(f"python3 {path}/scripts/convert_bdd100k_to_yolo.py --labels {path}/data/bdd100kFiles/labels/det_v2_train_release.json --images {path}/data/bdd100kFiles/bdd100k/bdd100k/images/100k/train --out {path}/data/labels/train")
        run_step(f"python3 {path}/scripts/convert_bdd100k_to_yolo.py --labels {path}/data/bdd100kFiles/labels/det_v2_val_release.json --images {path}/data/bdd100kFiles/bdd100k/bdd100k/images/100k/val --out {path}/data/labels/val")
    

    run_step(f"python3 {path}/scripts/scenario_data_classifier.py")

    run_step(f"python3 {path}/scripts/validate_dataset.py")

   
    cmd_train = (f"python3 {path}/scripts/train_yolo.py --data {args.data} --model {args.model} "
                 f"--epochs {args.epochs} --imgsz {args.imgsz} --batch {args.batch} --name {run_name}")
    
    run_step(cmd_train)

    print(f"  Le modèle est sauvegardé dans runs/train/{run_name}/weights/best.pt")

if __name__ == "__main__":
    main()