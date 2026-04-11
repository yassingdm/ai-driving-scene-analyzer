import streamlit as st
import tempfile
import os
import cv2
from pathlib import Path
from PIL import Image
import json
from gtts import gTTS

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cv.detector import YOLODetector
from huggingface_hub import hf_hub_download
from LLM.agent import analyze_scene
from scripts.visualize_detections import color_for_name

EXPERTS_YOLO = {
    "classique ": "yolov8n",
    "Expert autoroute ": "yolov8n",
    "Expert nuit ": "yolov8n",
    "Expert parking ": "yolov8n",
    "Expert piétons ": "yolov8n",
<<<<<<< HEAD
    "Expert pluie_brouillard ": "yolov8nt",
=======
    "Expert pluie_brouillard ": "yolov8n",
>>>>>>> 577423c (corrections sur l'app streamlit)
    "Expert urbain ": "yolov8n",
}

st.set_page_config(page_title="Analyseur de Scènes de Conduite", layout="wide")
st.title("Analyseur Intelligent de Scènes de Conduite")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

if EXPECTED_VENV_PYTHON.exists():
    current_python = Path(sys.executable).resolve()
    expected_python = EXPECTED_VENV_PYTHON.resolve()
    if current_python != expected_python:
        st.error(
            "Cette app est lancée avec un mauvais interpréteur Python.\n\n"
            f"Python actif: {current_python}\n"
            f"Python attendu: {expected_python}\n\n"
            "Lance avec: .venv/Scripts/python.exe -m streamlit run app/app.py"
        )
        st.stop()

DEFAULT_CLASSES_CSV = "car,truck,bus,person,bicycle,motorcycle,traffic light,stop sign,train"

with st.sidebar:
    st.header("Choix de l'Expert (Modèle)")
    nom_expert = st.selectbox("Sélectionnez le modele :", list(EXPERTS_YOLO.keys()))
    modele_yolo_selectionne = EXPERTS_YOLO[nom_expert]

    st.header("Paramètres CV")
    conf_threshold = st.slider("Seuil confiance", 0.1, 0.9, 0.4, 0.05)
    iou_threshold = st.slider("Seuil IoU", 0.1, 0.9, 0.5, 0.05)
    raw_yolo_mode = st.checkbox("Sortie brute YOLO (sans mapping BDD)", value=False)
    classes_csv = st.text_input("Classes (CSV, vide = toutes)", value=DEFAULT_CLASSES_CSV)
    dashcam_filter = st.checkbox("Filtre anti-faux positifs dashcam", value=True)
    adaptive_low_light = st.checkbox("Mode nuit auto (abaisse le seuil si image sombre)", value=True)
    if st.button("Recharger le modèle"):
        st.cache_resource.clear()
        st.rerun()

@st.cache_resource
def load_model(modelname):
    return YOLODetector(modelname)

detector = load_model(modele_yolo_selectionne)

if detector.model is None:
    st.warning(
        "Le modèle YOLO n'est pas chargé: l'application utilise un fallback heuristique moins précis. "
        "Vérifie l'installation d'ultralytics/torch et recharge le modèle. "
        f"Python actif: {sys.executable}"
    )


def generate_audio(text: str) -> str | None:
    """Génère un fichier audio temporaire à partir du texte."""
    if not text.strip():
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
        audio_path = tmp_audio.name
    tts = gTTS(text=text, lang="fr")
    tts.save(audio_path)
    return audio_path


def post_filter_dashcam(detections, image_width: int, image_height: int):
    """Supprime des faux positifs fréquents (capot, reflets bas d'image)."""
    filtered = []
    area_total = max(1.0, float(image_width * image_height))
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        area_ratio = (box_w * box_h) / area_total

        lower_band = y1 >= 0.68 * image_height
        very_wide = box_w >= 0.55 * image_width
        bottom_touch = y2 >= 0.97 * image_height

        if det.class_name in {"car", "truck", "bus", "vehicle"}:
            if lower_band and very_wide and area_ratio >= 0.08:
                continue
            if bottom_touch and area_ratio >= 0.05:
                continue

        filtered.append(det)

    return filtered


def estimate_brightness(image_bgr) -> float:
    """Retourne une luminance moyenne normalisée [0..1]."""
    if image_bgr is None:
        return 1.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean() / 255.0)

def draw_boxes_on_image(image_path, detections):
    """Dessine les bounding boxes sur l'image avec OpenCV."""
    img = cv2.imread(image_path)
    for det in detections:
        class_name = det.class_name
        conf = det.confidence
        x_min, y_min, x_max, y_max = map(int, det.bbox)
        
        color = color_for_name(class_name)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
        
        label = f"{class_name} {conf:.2f}"
        cv2.putText(img, label, (x_min, max(0, y_min - 6)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

uploaded_file = st.file_uploader("Chargez une image issue d'une dashcam (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_image_path = tmp_file.name

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image Originale")
        st.image(uploaded_file, width="stretch")

    if st.button("Analyser la scène", type="primary"):
        with st.spinner("Traitement en cours (Vision + LLM)..."):
            try:
                class_filter = [c.strip() for c in classes_csv.split(",") if c.strip()]
                image_for_shape = cv2.imread(tmp_image_path)
                brightness = estimate_brightness(image_for_shape)
                effective_conf = conf_threshold

                # Sur scènes nocturnes/sombres, on assouplit le seuil pour récupérer des objets.
                if adaptive_low_light and brightness < 0.33:
                    effective_conf = max(0.2, conf_threshold - 0.15)

                detections = detector.detect(
                    tmp_image_path,
                    conf=effective_conf,
                    iou=iou_threshold,
                    classes=class_filter or None,
                    raw_output=raw_yolo_mode,
                )

                if dashcam_filter and image_for_shape is not None:
                    h, w = image_for_shape.shape[:2]
                    detections_before_filter = detections
                    detections = post_filter_dashcam(detections_before_filter, image_width=w, image_height=h)
                    if not detections and detections_before_filter:
                        detections = detections_before_filter
                        st.info("Le filtre dashcam a été ignoré pour cette image (sinon 0 détection).")
                
                detections_reduites = [
                    {
                        "class": d.class_name,
                        "confidence": d.confidence,
                        "bbox": [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]]
                    }
                    for d in detections
                ][:30] 

                detections_json_str = json.dumps(detections_reduites)
                rapport_llm = analyze_scene(detections_json_str)

                st.success("Analyse terminée !")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    st.subheader("Détections (Computer Vision)")
                    st.caption(
                        f"Mode CV: conf={effective_conf:.2f}, iou={iou_threshold:.2f}, "
                        f"raw={raw_yolo_mode}, "
                        f"lum={brightness:.2f}, "
                        f"classes={'ALL' if not class_filter else ', '.join(class_filter)}"
                    )
                    # Dessin des boîtes
                    annotated_img = draw_boxes_on_image(tmp_image_path, detections)
                    st.image(annotated_img, width="stretch")
                    with st.expander("Voir les données brutes de détection"):
                        st.json(detections_reduites)

                with col4:
                    st.subheader("Rapport de l'Agent IA")
                    niveau_risque = rapport_llm.get("Niveau de risque", "Non spécifié")
                    
                    if "Critique" in niveau_risque:
                        st.error(f" Risque : {niveau_risque}")
                    elif "Élevé" in niveau_risque:
                        st.warning(f" Risque : {niveau_risque}")
                    elif "Moyen" in niveau_risque:
                        st.info(f" Risque : {niveau_risque}")
                    else:
                        st.success(f" Risque : {niveau_risque}")

                    st.write("**Résumé :**", rapport_llm.get("Résumé", "N/A"))
                    st.write("**Analyse des risques :**", rapport_llm.get("Analyse des risques", "N/A"))
                    st.write("**Recommandations :**")
                    st.write(rapport_llm.get("Recommandations", "N/A"))

                    resume = rapport_llm.get("Résumé", "")
                    recommandations = rapport_llm.get("Recommandations", "")
                    texte_vocal = (
                        f"Attention, niveau de risque {niveau_risque}. "
                        f"{resume}. Recommandation : {recommandations}"
                    )
                    try:
                        audio_path = generate_audio(texte_vocal)
                        if audio_path:
                            st.audio(audio_path, format="audio/mp3")
                    except Exception:
                        st.info("Audio indisponible pour cette analyse.")
            
            except Exception as e:
                st.error(f"Une erreur est survenue lors de l'analyse : {e}")
            
            finally:
                if os.path.exists(tmp_image_path):
                    os.remove(tmp_image_path)