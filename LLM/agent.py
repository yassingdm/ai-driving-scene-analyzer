from groq import Groq
import json
from LLM.system_prompt import SYSTEM_PROMPT
from LLM.tool import calculDistance
from LLM.tool import CalculDistance_tool

import os
from dotenv import load_dotenv

load_dotenv() #juste pour charger le.env
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("Il manque la clé.")

client = Groq(api_key=api_key)

def analyze_scene(detections_json):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Voici les détections JSON : {detections_json}"}
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        tools=[CalculDistance_tool], 
        tool_choice="auto"     
    )

    base = response.choices[0].message.content
    debut = base.find("{")
    fin = base.rfind("}") + 1

    cleaned = base[debut:fin]
    cleaned = cleaned.strip() #Supprime les caractères parasites autour
    cleaned = cleaned.replace("```", "").replace("**", "")# Supprime backticks, markdown et étoiles

    if debut == -1 or fin == 0:
        raise ValueError("Le LLM n'a pas renvoyé de JSON. Réponse brute : " + base)

    
    return json.loads(cleaned)