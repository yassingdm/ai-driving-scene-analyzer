from groq import Groq
import json
import re
import unicodedata
from LLM.system_prompt import SYSTEM_PROMPT
from LLM.tool import calculDistance

import os
from dotenv import load_dotenv

load_dotenv() #juste pour charger le.env
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY is missing. Please set it in the .env file.")

client = Groq(api_key=api_key)


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )


def _extract_json_payload(raw_text: str) -> dict | None:
    if not raw_text:
        return None

    # Case 1: full response is already JSON.
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Case 2: JSON fenced in markdown blocks.
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", raw_text, flags=re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # Case 3: JSON object embedded in extra text.
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw_text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None

    return None


def _extract_from_structured_text(raw_text: str) -> dict:
    heading_map = {
        "resume": "resume",
        "summary": "resume",
        "objets detectes": "objects",
        "detected objects": "objects",
        "analyse des risques": "risk_analysis",
        "risk analysis": "risk_analysis",
        "niveau de risque": "risk_level",
        "risk level": "risk_level",
        "justification": "justification",
        "recommandations": "recommendations",
        "recommendations": "recommendations",
    }

    buckets = {
        "resume": [],
        "objects": [],
        "risk_analysis": [],
        "risk_level": [],
        "justification": [],
        "recommendations": [],
    }

    current = None
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue

        clean = line.strip("* ")
        if ":" in clean:
            left, right = clean.split(":", 1)
            folded = _strip_accents(left.lower()).strip()
            key = heading_map.get(folded)
            if key:
                current = key
                if right.strip():
                    buckets[key].append(right.strip())
                continue

        if current:
            buckets[current].append(clean)

    risk_analysis_lines = buckets["risk_analysis"][:]
    if buckets["justification"]:
        risk_analysis_lines.append("Justification: " + " ".join(buckets["justification"]))

    return {
        "Résumé": " ".join(buckets["resume"]).strip() or "Analyse indisponible.",
        "Objets détectés": "\n".join(buckets["objects"]).strip() or "Non spécifié.",
        "Analyse des risques": " ".join(risk_analysis_lines).strip() or "Non spécifiée.",
        "Niveau de risque": " ".join(buckets["risk_level"]).strip() or "Non spécifié",
        "Recommandations": "\n".join(buckets["recommendations"]).strip() or "Aucune recommandation fournie.",
    }


def _normalize_report(report: dict, raw_text: str) -> dict:
    def pick(*keys, default=""):
        for key in keys:
            value = report.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return default

    normalized = {
        "Résumé": pick("Résumé", "Resume", "summary", "Summary", default="Analyse indisponible."),
        "Objets détectés": pick("Objets détectés", "Objets detectes", "Detected objects", "Objets"),
        "Analyse des risques": pick("Analyse des risques", "Risk analysis", "Justification", default="Non spécifiée."),
        "Niveau de risque": pick("Niveau de risque", "Risk level", "risk_level", default="Non spécifié"),
        "Recommandations": pick("Recommandations", "Recommendations", default="Aucune recommandation fournie."),
    }

    if not normalized["Résumé"] and raw_text.strip():
        normalized["Résumé"] = raw_text.strip()

    return normalized

def analyze_scene(detections_json):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Voici les détections JSON : "
                f"{detections_json}\n"
                "Return ONLY a valid JSON object with these keys: "
                "'Résumé', 'Objets détectés', 'Analyse des risques', 'Niveau de risque', 'Recommandations'."
            ),
        },
    ]

    request = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "temperature": 0.2,
    }

    try:
        response = client.chat.completions.create(
            **request,
            response_format={"type": "json_object"},
        )
    except Exception:
        response = client.chat.completions.create(**request)

    base = (response.choices[0].message.content or "").strip()
    report = _extract_json_payload(base)
    if report is None:
        report = _extract_from_structured_text(base)

    return _normalize_report(report, base)