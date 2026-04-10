"""
Valider l'intégrité du dataset BDD100K.

Structure attendue:
data/
├── data.yaml
├── train/  (images JPG)
├── val/    (images JPG)
└── test/   (images JPG)
"""

import sys
from pathlib import Path


def validate_dataset(data_dir: Path = Path("data")) -> bool:
    """
    Valider structure et intégrité du dataset.
    
    Arguments:
        data_dir: Répertoire data/
    
    Retourne:
        True si structure valide et images présentes
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"Répertoire data/ non trouvé: {data_dir}")
        return False

    # Vérifier data.yaml
    yaml_file = data_dir / "data.yaml"
    if not yaml_file.exists():
        print(f"Configuration manquante: {yaml_file}")
        return False
    print(f"✓ data.yaml trouvé")

    stats = {
        "train": 0,
        "val": 0,
        "test": 0,
        "total_size_mb": 0.0,
        "labels": {"train": 0, "val": 0, "test": 0},
        "missing_labels": {"train": 0, "val": 0, "test": 0},
        "extra_labels": {"train": 0, "val": 0, "test": 0},
    }
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    # Valider chaque split (direct dans data/)
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        
        if not split_dir.exists():
            print(f"✗ Répertoire manquant: data/{split}/")
            return False

        images = sorted([f for f in split_dir.iterdir() if f.suffix in image_extensions])

        if not images:
            print(f"✗ Pas d'images dans data/{split}/")
            return False

        size_mb = sum(f.stat().st_size for f in images) / (1024 ** 2)
        stats[split] = len(images)
        stats["total_size_mb"] += size_mb

        labels_dir = split_dir / "labels"
        if labels_dir.exists():
            label_files = sorted([f for f in labels_dir.iterdir() if f.suffix == ".txt"])
            stats["labels"][split] = len(label_files)
            image_stems = {f.stem for f in images}
            label_stems = {f.stem for f in label_files}
            stats["missing_labels"][split] = len(image_stems - label_stems)
            stats["extra_labels"][split] = len(label_stems - image_stems)
            print(
                f"✓ {split:5} : {len(images):4} images ({size_mb:7.1f} MB) | "
                f"labels: {len(label_files):4}"
            )
        else:
            print(f"✓ {split:5} : {len(images):4} images ({size_mb:7.1f} MB) | labels: none")

    # Résumé
    print("\n" + "=" * 50)
    total = sum(stats[k] for k in ["train", "val", "test"])
    print(f"Total images : {total}")
    print(f"Taille total : {stats['total_size_mb']:.1f} MB")
    print(f"Distribution : {stats['train']}/{stats['val']}/{stats['test']}")
    print(
        "Missing labels : "
        f"{stats['missing_labels']['train']}/"
        f"{stats['missing_labels']['val']}/"
        f"{stats['missing_labels']['test']}"
    )
    print(
        "Extra labels   : "
        f"{stats['extra_labels']['train']}/"
        f"{stats['extra_labels']['val']}/"
        f"{stats['extra_labels']['test']}"
    )
    print("Dataset valide ✓")
    
    return True


if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    success = validate_dataset(data_dir)
    sys.exit(0 if success else 1)
