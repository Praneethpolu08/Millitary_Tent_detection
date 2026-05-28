from ultralytics import YOLO
import streamlit as st
import time

from src.config import MODEL_PATH


@st.cache_resource
def load_model():

    return YOLO(MODEL_PATH)


model = load_model()


def detect_tents(image, confidence_threshold):

    start_time = time.time()

    results = model.predict(
        source=image,
        conf=confidence_threshold,
        save=False
    )
    end_time = time.time()

    inference_time = end_time - start_time

    annotated_image = results[0].plot(
    labels=False,
    line_width=1
    )
    detections = []

    boxes = results[0].obb
    if boxes is not None:
        for box in boxes:

            detections.append({

                "class_id": int(box.cls[0]),

                "confidence": float(box.conf[0]),

                "coordinates": (
                    box.xyxy[0].tolist()
                )
            })
    return {
            "annotated_image": annotated_image,
            "detections": detections,
            "inference_time": inference_time
        }