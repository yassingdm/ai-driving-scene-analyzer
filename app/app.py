import streamlit as st
import tempfile
import os
import cv2
from PIL import Image
import json


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cv.detector import YOLODetector
from LLM.agent import analyze_scene
from cv.visualize_detections import color_for_name


st.set_page_config(page_title="Analyseur de Scènes de Conduite", layout="wide")
st.title("🚗 Analyseur Intelligent de Scènes de Conduite")

@st.cache_resource
def load_model():
    return YOLODetector("yolov8n")

detector = load_model()

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
        st.image(uploaded_file, use_column_width=True)

    if st.button("Analyser la scène", type="primary"):
        with st.spinner("Traitement en cours (Vision + LLM)..."):
            try:
                detections = detector.detect(tmp_image_path)
                
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
                    # Dessin des boîtes
                    annotated_img = draw_boxes_on_image(tmp_image_path, detections)
                    st.image(annotated_img, use_column_width=True)
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
                    
            except Exception as e:
                st.error(f"Une erreur est survenue lors de l'analyse : {e}")
            
            finally:
                if os.path.exists(tmp_image_path):
                    os.remove(tmp_image_path)