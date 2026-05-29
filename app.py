import streamlit as st
from PIL import Image

from src.config import (
    APP_TITLE,
    DEFAULT_CONFIDENCE
)

from src.image_utils import (
    pil_to_numpy,
    bgr_to_rgb
)

from src.detector import detect_tents

from src.metrics import (
    generate_detection_summary
)

from src.ui_components import (
    render_sidebar,
    render_detection_details
)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Tent Detection",
    page_icon="🏕️",
    layout="wide"
)

# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title(APP_TITLE)

st.markdown(
    """
Upload a satellite image to detect tents using YOLOv8 OBB.
"""
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

confidence_threshold = render_sidebar(
    DEFAULT_CONFIDENCE
)

# -------------------------------------------------
# FILE UPLOADER
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# DETECTION PIPELINE
# -------------------------------------------------

if uploaded_file is not None:

    # LOAD IMAGE
    image = Image.open(uploaded_file).convert("RGB")

    image_np = pil_to_numpy(image)

    # RUN DETECTION
    with st.spinner("Detecting tents..."):

        output = detect_tents(
            image_np,
            confidence_threshold
        )

    # EXTRACT OUTPUTS
    detected_image = output["annotated_image"]

    detections = output["detections"]

    inference_time = output["inference_time"]

    debug_info = output.get("debug_info", {})

    # RGB CONVERSION
    detected_image_rgb = bgr_to_rgb(
        detected_image
    )

    # -------------------------------------------------
    # SIDE BY SIDE IMAGE DISPLAY
    # -------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(
            image,
            width=350
        )

    with col2:

        st.subheader("Detection Result")

        st.image(
            detected_image_rgb,
            width=350
        )

    # -------------------------------------------------
    # DEBUG SECTION
    # -------------------------------------------------

    with st.expander("Debug Information"):

        st.write(
            "Detections Found:",
            len(detections)
        )

        if detections:

            st.write(
                "First Detection:",
                detections[0]
            )

        st.write(
            "Number of OBB Boxes:",
            debug_info.get(
                "num_boxes",
                "N/A"
            )
        )

        st.write(
            "Results Summary:",
            debug_info.get(
                "results_summary",
                "N/A"
            )
        )

        st.write(
            "Class Names:",
            debug_info.get(
                "class_names",
                "N/A"
            )
        )

        st.write(
            "Annotated Image Shape:",
            detected_image.shape
            if hasattr(
                detected_image,
                "shape"
            )
            else "No shape"
        )

    # -------------------------------------------------
    # METRICS
    # -------------------------------------------------

    summary = generate_detection_summary(
        detections
    )

    st.subheader("Detection Metrics")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Total Detections",
            summary["total_detections"]
        )

    with metric2:

        st.metric(
            "Average Confidence",
            f"{summary['average_confidence']:.2f}"
        )

    with metric3:

        st.metric(
            "Inference Time",
            f"{inference_time:.2f} sec"
        )

    # -------------------------------------------------
    # DETECTION DETAILS
    # -------------------------------------------------

    render_detection_details(
        detections
    )

else:

    st.info(
        "Upload an image to begin detection."
    )
