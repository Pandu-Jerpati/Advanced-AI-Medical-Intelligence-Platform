import streamlit as st
import requests


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Medical Intelligence Platform",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🩺 AI Medical Intelligence Platform")

st.write(
    "AI-assisted chest X-ray analysis using DenseNet121, "
    "Grad-CAM and AI-generated research reports."
)

st.warning(
    "⚠️ This system is for research purposes only. "
    "It is not a substitute for professional medical diagnosis."
)


# ============================================================
# 1. IMAGE UPLOAD
# ============================================================

st.header("1. Upload Chest X-ray")

uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# ANALYZE IMAGE
# ============================================================

if uploaded_file is not None:

    st.subheader("Original Image")

    st.image(
        uploaded_file,
        caption=uploaded_file.name,
        use_container_width=True
    )

    if st.button(
        "🔍 Analyze Image",
        type="primary"
    ):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        try:

            with st.spinner(
                "Analyzing image using DenseNet121..."
            ):

                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=120
                )


            # ==================================================
            # SUCCESS RESPONSE
            # ==================================================

            if response.status_code == 200:

                result = response.json()

                st.success(
                    "Image analysis completed!"
                )


                # ==================================================
                # 2. PREDICTION RESULTS
                # ==================================================

                st.header(
                    "2. Prediction Results"
                )

                col1, col2, col3 = st.columns(3)


                with col1:

                    st.metric(
                        "Prediction",
                        result.get(
                            "prediction",
                            "N/A"
                        )
                    )


                with col2:

                    st.metric(
                        "Confidence",
                        f"{result.get('confidence', 0)}%"
                    )


                with col3:

                    st.metric(
                        "Model",
                        result.get(
                            "model_version",
                            result.get(
                                "model",
                                "N/A"
                            )
                        )
                    )


                # ==================================================
                # 3. GRAD-CAM
                # ==================================================

                st.header(
                    "3. Explainable AI - Grad-CAM"
                )

                col1, col2 = st.columns(2)


                # --------------------------------------------------
                # ORIGINAL X-RAY
                # --------------------------------------------------

                with col1:

                    st.subheader(
                        "Original X-ray"
                    )

                    st.image(
                        uploaded_file,
                        caption="Original X-ray",
                        use_container_width=True
                    )


                # --------------------------------------------------
                # GRAD-CAM HEATMAP
                # --------------------------------------------------

                with col2:

                    st.subheader(
                        "Grad-CAM Heatmap"
                    )

                    gradcam_path = result.get(
                        "gradcam_path"
                    )


                    if gradcam_path:

                        gradcam_filename = (
                            gradcam_path
                            .replace("\\", "/")
                            .split("/")[-1]
                        )


                        try:

                            gradcam_response = requests.get(
                                f"{API_URL}/gradcam/{gradcam_filename}",
                                timeout=30
                            )


                            if gradcam_response.status_code == 200:

                                st.image(
                                    gradcam_response.content,
                                    caption="Grad-CAM Heatmap",
                                    use_container_width=True
                                )

                            else:

                                st.error(
                                    "Grad-CAM image could not be loaded."
                                )


                        except Exception as e:

                            st.error(
                                f"Error loading Grad-CAM: {e}"
                            )


                    else:

                        st.warning(
                            "Grad-CAM path was not returned by FastAPI."
                        )


                # ==================================================
                # 4. AI-ASSISTED REPORT
                # ==================================================

                st.header(
                    "4. AI-Assisted Medical Research Report"
                )


                report = result.get(
                    "report"
                )


                if report:

                    st.markdown(
                        report
                    )

                else:

                    st.info(
                        "AI-assisted report is not available."
                    )


            # ==================================================
            # FASTAPI ERROR
            # ==================================================

            else:

                st.error(
                    f"FastAPI error: {response.text}"
                )


        # ======================================================
        # CONNECTION ERROR
        # ======================================================

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Cannot connect to FastAPI. "
                "Please make sure FastAPI is running."
            )


        # ======================================================
        # TIMEOUT ERROR
        # ======================================================

        except requests.exceptions.Timeout:

            st.error(
                "⏱️ Request timed out. "
                "Please try again."
            )


        # ======================================================
        # OTHER ERROR
        # ======================================================

        except Exception as e:

            st.error(
                f"❌ Error: {e}"
            )


# ============================================================
# 5. PREDICTION HISTORY
# ============================================================

st.header(
    "5. Prediction History"
)


if st.button(
    "🔄 Load Prediction History"
):

    try:

        response = requests.get(
            f"{API_URL}/predictions",
            timeout=30
        )


        if response.status_code == 200:

            history = response.json()


            if history:

                for item in history:

                    prediction = item.get(
                        "prediction",
                        "N/A"
                    )

                    filename = item.get(
                        "image_filename",
                        "N/A"
                    )

                    confidence = item.get(
                        "confidence",
                        0
                    )


                    # ------------------------------------------
                    # HISTORY EXPANDER
                    # ------------------------------------------

                    with st.expander(
                        f"#{item.get('id')} | "
                        f"{filename} | "
                        f"{prediction} | "
                        f"{confidence}%"
                    ):


                        # --------------------------------------
                        # BASIC INFORMATION
                        # --------------------------------------

                        st.write(
                            f"**Image:** {filename}"
                        )

                        st.write(
                            f"**Prediction:** {prediction}"
                        )

                        st.write(
                            f"**Confidence:** {confidence}%"
                        )

                        st.write(
                            f"**Model:** "
                            f"{item.get('model_version', 'N/A')}"
                        )


                        # --------------------------------------
                        # GRAD-CAM
                        # --------------------------------------

                        gradcam_path = item.get(
                            "gradcam_path"
                        )


                        if gradcam_path:

                            gradcam_filename = (
                                gradcam_path
                                .replace("\\", "/")
                                .split("/")[-1]
                            )


                            try:

                                gradcam_response = requests.get(
                                    f"{API_URL}/gradcam/{gradcam_filename}",
                                    timeout=30
                                )


                                if gradcam_response.status_code == 200:

                                    st.image(
                                        gradcam_response.content,
                                        caption="Grad-CAM",
                                        use_container_width=True
                                    )

                                else:

                                    st.warning(
                                        "Grad-CAM image unavailable."
                                    )


                            except Exception:

                                st.warning(
                                    "Could not load Grad-CAM."
                                )


                        else:

                            st.info(
                                "No Grad-CAM available."
                            )


                        # --------------------------------------
                        # AI REPORT
                        # --------------------------------------

                        st.subheader(
                            "AI-Assisted Medical Research Report"
                        )


                        history_report = item.get(
                            "report"
                        )


                        if history_report:

                            st.markdown(
                                history_report
                            )

                        else:

                            st.info(
                                "No AI-assisted report available."
                            )


            else:

                st.info(
                    "No prediction history found."
                )


        else:

            st.error(
                f"Could not load prediction history: "
                f"{response.text}"
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Cannot connect to FastAPI. "
            "Please make sure FastAPI is running."
        )


    except Exception as e:

        st.error(
            f"❌ Error loading history: {e}"
        )