import sys
import os
import tempfile

import streamlit as st


# =========================
# PROJECT PATH
# =========================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_PATH = os.path.join(
    PROJECT_ROOT,
    "src"
)

sys.path.insert(0, SRC_PATH)


# =========================
# IMPORT PIPELINE
# =========================

from pipeline import run_pipeline


# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="Plant Disease Intelligence System",
    page_icon="🌿",
    layout="wide"
)


# =========================
# HELPER FUNCTION
# =========================

def format_class_name(class_name):

    parts = class_name.split("___")

    if len(parts) == 2:

        plant = parts[0].replace("_", " ")
        disease = parts[1].replace("_", " ")

        return f"{plant} — {disease}"

    return class_name.replace("_", " ")


# =========================
# TITLE
# =========================

st.title(
    "🌿 Plant Disease Intelligence System"
)

st.write(
    "Upload a plant leaf image to get an AI-based disease prediction and explanation."
)


# =========================
# IMAGE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)


# =========================
# DISPLAY IMAGE
# =========================

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded leaf image",
        width=400
    )

    # =========================
    # ANALYZE BUTTON
    # =========================

    if st.button(
        "🔍 Analyze Leaf",
        type="primary"
    ):

        temp_file_path = None

        try:

            # =========================
            # CREATE TEMPORARY FILE
            # =========================

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(
                    uploaded_file.name
                )[1]
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_file_path = temp_file.name

            # =========================
            # RUN PIPELINE
            # =========================

            with st.spinner(
                "Analyzing the leaf image..."
            ):

                result = run_pipeline(
                    temp_file_path
                )

            # =========================
            # PREDICTION
            # =========================

            st.divider()

            st.subheader(
                "🔬 Prediction"
            )

            formatted_prediction = format_class_name(
                result["prediction"]
            )

            st.markdown(
                f"### {formatted_prediction}"
            )

            # =========================
            # CONFIDENCE
            # =========================

            confidence = result["confidence"]

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(
                min(confidence / 100, 1.0)
            )

            # =========================
            # TOP 3
            # =========================

            st.divider()

            st.subheader(
                "📊 Top 3 Predictions"
            )

            for rank, prediction in enumerate(
                result["top_predictions"],
                start=1
            ):

                formatted_class = format_class_name(
                    prediction["class"]
                )

                confidence_value = prediction[
                    "confidence"
                ]

                st.write(
                    f"**{rank}. {formatted_class}**"
                )

                st.progress(
                    min(
                        confidence_value / 100,
                        1.0
                    )
                )

                st.caption(
                    f"{confidence_value:.2f}% confidence"
                )

            # =========================
            # AI EXPLANATION
            # =========================

            st.divider()

            st.subheader(
                "🤖 AI Explanation"
            )

            st.markdown(
                result["explanation"]
            )

        except Exception as e:

            st.error(
                f"An error occurred: {e}"
            )

        finally:

            # =========================
            # CLEAN TEMPORARY FILE
            # =========================

            if (
                temp_file_path
                and os.path.exists(temp_file_path)
            ):

                os.remove(
                    temp_file_path
                )