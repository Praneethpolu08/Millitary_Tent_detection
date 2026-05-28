import streamlit as st


def render_sidebar(default_confidence):

    st.sidebar.header("⚙️ Settings")

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=default_confidence,
        step=0.05
    )

    return confidence_threshold


def render_detection_details(detections):

    st.subheader("Detection Details")

    if len(detections) == 0:

        st.warning("No tents detected.")

        return

    for idx, detection in enumerate(detections):

        st.write(
            f"""
            Detection {idx+1}
            - Confidence: {detection['confidence']:.2f}
            """
        )