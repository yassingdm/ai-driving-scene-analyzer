INSTRUCTION_PROMPT = """
Analyse la scène de conduite suivante et produis un JSON strict avec exactement cette structure :

{
  "resume": "",
  "objets_detectes": "",
  "analyse_risques": "",
  "niveau_risque": "",
  "recommandations": ""
}

Signification des champs :
- "resume" : description synthétique de la scène.
- "objets_detectes" : résumé textuel des objets importants (voitures, camions, piétons, feux, etc.).
- "analyse_risques" : analyse des risques principaux (collision, piétons proches, visibilité, etc.).
- "niveau_risque" : un seul mot parmi : "Faible", "Moyen", "Élevé", "Critique".
- "recommandations" : conseils de conduite concrets et adaptés à la situation.

"""
