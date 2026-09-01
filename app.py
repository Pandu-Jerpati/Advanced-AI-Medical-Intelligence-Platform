import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="AI Medical Intelligence Platform",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI Medical Intelligence Platform")

st.write("Upload a chest X-ray for AI-assisted analysis.")

st.divider()

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    if st.button("🔍 Analyze Image"):

        with st.spinner("Analyzing image..."):

            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    files=files
                )

                if response.status_code == 200:

                    result = response.json()

                    st.success("Analysis completed!")

                    # ==============================
                    # PREDICTION
                    # ==============================

                    st.subheader("🧠 AI Analysis")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Prediction",
                            result["prediction"]
                        )

                    with col2:
                        st.metric(
                            "Confidence",
                            f'{result["confidence"]:.2f}%'
                        )

                    with col3:
                        st.metric(
                            "Model",
                            result["model"]
                        )

                    st.divider()

                    # ==============================
                    # ORIGINAL + GRAD-CAM
                    # ==============================

                    st.subheader(
                        "🔬 Explainable AI - Grad-CAM"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.write("### Original X-ray")

                        st.image(
                            image,
                            width="stretch"
                        )

                    with col2:

                        st.write("### Grad-CAM Heatmap")

                        gradcam_path = result.get(
                            "gradcam_path"
                        )

                        if gradcam_path:

                            filename = (
                                gradcam_path
                                .replace("\\", "/")
                                .split("/")[-1]
                            )

                            url = (
                                "http://127.0.0.1:8000/gradcam/"
                                + filename
                            )

                            image_response = requests.get(
                                url
                            )

                            if image_response.status_code == 200:

                                gradcam_image = Image.open(
                                    BytesIO(
                                        image_response.content
                                    )
                                )

                                st.image(
                                    gradcam_image,
                                    caption="Grad-CAM Heatmap",
                                    width="stretch"
                                )

                            else:

                                st.error(
                                    "Grad-CAM could not be loaded."
                                )

                        else:

                            st.warning(
                                "Grad-CAM path not returned by API."
                            )

                    # ==============================
                    # REPORT
                    # ==============================

                    if result.get("report"):

                        st.divider()

                        st.subheader(
                            "📄 AI-Assisted Report"
                        )

                        st.write(
                            result["report"]
                        )

                else:

                    st.error(
                        f"API Error: {response.status_code}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to FastAPI. "
                    "Make sure FastAPI is running."
                )

            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )


st.divider()

st.caption(
    "Medical/Research Disclaimer: This application is for "
    "research and educational purposes only and should not "
    "be used as a substitute for professional medical diagnosis."
)