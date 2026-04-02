"""Point d'entrée du module CV."""

import argparse
import json
from detector import YOLODetector
from postprocessor import filter_by_confidence, format_results


def main() -> None:
    """Détecter les objets dans une image."""
    parser = argparse.ArgumentParser(
        description="Détection YOLO pour scènes routières"
    )
    parser.add_argument("--image", required=True, help="Chemin vers l'image")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Seuil de confiance"
    )
    parser.add_argument(
        "--output",
        help="Fichier de sortie JSON (optionnel)"
    )
    
    args = parser.parse_args()
    
    # Initialiser et lancer détection
    detector = YOLODetector()
    detections = detector.detect(args.image)
    
    # Filtrer par confiance
    filtered = filter_by_confidence(detections, threshold=args.confidence)
    
    # Formater les résultats
    result = format_results(filtered)
    result["image"] = args.image
    
    # Afficher et sauvegarder
    print(json.dumps(result, indent=2))
    
    if args.output:
        # TODO: vérifier que le dossier existe
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Résultats sauvegardés: {args.output}")


if __name__ == "__main__":
    main()
