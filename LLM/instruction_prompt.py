INSTRUCTION_PROMPT = """
Analyse la scène de conduite suivante et produis un JSON strict avec exactement cette structure :

{
  "resume": "",
  "objets_detectes": "",
  "analyse_risques": "",
  "niveau_risque": "",
  "recommandations": ""
}

Exemple de réponse attendue :

{
  "resume": "Une voiture et un piéton sur un passage piéton.",
  "objets_detectes": "car, person",
  "analyse_risques": "Le piéton est proche de la trajectoire de la voiture.",
  "niveau_risque": "Élevé",
  "recommandations": "Ralentir immédiatement et se préparer à s’arrêter."
}

Contraintes :
- Pas de texte avant ou après le JSON.
- Pas de commentaires.
- Pas de markdown.
"""
