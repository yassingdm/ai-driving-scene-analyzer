from groq import Groq
import json
from LLM.system_prompt import SYSTEM_PROMPT
from LLM.tool import calculDistance

import os
from dotenv import load_dotenv

load_dotenv() #juste pour charger le.env
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY is missing. Please set it in the .env file.")

client = Groq(api_key=api_key)

def analyze_scene(detections_json):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Voici les détections JSON : {detections_json}"}
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    base = response.choices[0].message.content
    debut = base.find("{")
    fin = base.rfind("}") + 1

    if debut == -1 or fin == 0:
        raise ValueError("Le LLM n'a pas renvoyé de JSON. Réponse brute : " + base)

    leJson = base[debut:fin]
    return json.loads(leJson)