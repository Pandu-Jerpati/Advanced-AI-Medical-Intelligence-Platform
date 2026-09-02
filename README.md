# 🏥 Advanced AI Medical Intelligence Platform

An end-to-end **AI-powered medical image analysis platform** that uses Deep Learning for disease prediction, **Grad-CAM for Explainable AI (XAI)**, **FastAPI for REST APIs**, database storage for prediction history, and **LLM-assisted report generation**.

> ⚠️ **Medical Disclaimer:** This project is developed for educational, research, and demonstration purposes only. It is not intended to replace professional medical diagnosis, treatment, or clinical decision-making.

---

## 📌 Project Overview

The **Advanced AI Medical Intelligence Platform** analyzes chest X-ray images and predicts one of five medical conditions using a trained **DenseNet121** deep learning model.

The platform provides:

* 🩻 Medical image classification
* 🤖 Deep Learning prediction using DenseNet121
* 🔍 Explainable AI using Grad-CAM
* 📊 Prediction confidence score
* 📝 AI-assisted medical report generation
* 🚀 FastAPI REST API
* 🗄️ SQLAlchemy database integration
* 🖥️ Streamlit user interface
* 📚 Swagger API documentation
* 🐳 Docker deployment support
* 🕒 Prediction history

---

## 🎯 Objectives

The main objectives of this project are:

1. Build an AI model for chest X-ray classification.
2. Apply transfer learning using DenseNet121.
3. Provide explainability through Grad-CAM visualizations.
4. Develop REST APIs using FastAPI.
5. Store prediction information using SQLAlchemy.
6. Generate AI-assisted reports from structured prediction results.
7. Provide a simple and interactive Streamlit interface.
8. Create a deployable and scalable AI application.

---

## 🧠 Diseases / Classes

The model classifies chest X-ray images into **5 classes**:

| Class            | Description                               |
| ---------------- | ----------------------------------------- |
| **Normal**       | No targeted abnormality detected          |
| **Pneumonia**    | Pattern associated with pneumonia         |
| **Atelectasis**  | Partial or complete lung collapse pattern |
| **Cardiomegaly** | Enlarged heart pattern                    |
| **Effusion**     | Pleural fluid pattern                     |

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │      User Upload     │
                    │     Chest X-ray      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Streamlit Frontend  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    FastAPI Backend   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Image Preprocessing│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     DenseNet121      │
                    │   Deep Learning Model│
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐        ┌─────────────────┐
        │ Disease         │        │    Grad-CAM     │
        │ Prediction      │        │ Explainability  │
        └────────┬────────┘        └────────┬────────┘
                 │                          │
                 └─────────────┬────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ LLM-Assisted Report  │
                    │     Generation       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     SQL Database     │
                    │ Prediction History   │
                    └──────────────────────┘
```

---

## 🛠️ Technologies Used

### Programming

* Python

### Machine Learning / Deep Learning

* PyTorch
* Torchvision
* DenseNet121
* Transfer Learning
* Deep Learning
* Computer Vision

### Data Processing

* Pandas
* NumPy
* Pillow

### Explainable AI

* Grad-CAM
* Heatmaps
* Model visualization

### Backend

* FastAPI
* Uvicorn
* REST API
* Pydantic

### Database

* SQLAlchemy
* SQLite / relational database

### Frontend

* Streamlit

### AI Report Generation

* Large Language Model (LLM)

### Development / Deployment

* Git
* GitHub
* Docker
* Swagger / OpenAPI

---

## 📂 Project Structure

```text
Advanced-AI-Medical-Intelligence-Platform/
│
├── api/
│   ├── main.py
│   └── ...
│
├── src/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── ...
│
├── data/
│   ├── raw/
│   │   ├── images/
│   │   └── Data_Entry_2017.csv
│   │
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/
│   └── densenet121_medical.pth
│
├── gradcam_outputs/
│   ├── original.jpg
│   ├── heatmap.jpg
│   └── gradcam_overlay.jpg
│
├── database/
│   └── ...
│
├── streamlit_app.py
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## 📊 Dataset

This project uses the **NIH Chest X-ray dataset**.

The original dataset contains a large collection of chest X-ray images. For this project, the data was processed into five target classes:

```text
Atelectasis
Cardiomegaly
Effusion
Normal
Pneumonia
```

### Dataset Split

| Dataset    | Images |
| ---------- | -----: |
| Training   | 53,330 |
| Validation | 11,428 |
| Testing    | 11,429 |

---

## 🔄 Data Preprocessing

The preprocessing pipeline performs:

1. Reading image metadata.
2. Locating X-ray images.
3. Selecting the required disease classes.
4. Organizing images into class folders.
5. Splitting data into training, validation, and testing sets.
6. Applying image transformations.
7. Preparing tensors for model training.

---

## 🧠 Model

### DenseNet121

The project uses **DenseNet121** with transfer learning.

The pretrained network is adapted for the five-class medical image classification problem.

```text
Input Chest X-ray
       ↓
Image Preprocessing
       ↓
DenseNet121
       ↓
Feature Extraction
       ↓
Classification Layer
       ↓
5 Disease Classes
       ↓
Prediction + Confidence
```

### Training Results

The recorded training results include:

| Metric              | Result |
| ------------------- | -----: |
| Training Loss       | 0.8400 |
| Training Accuracy   | 70.91% |
| Validation Loss     | 0.8463 |
| Validation Accuracy | 69.69% |

The trained model is saved as:

```text
models/densenet121_medical.pth
```

---

## 🔍 Explainable AI — Grad-CAM

The platform uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize the regions of an X-ray that contribute to the model's prediction.

The workflow is:

```text
X-ray Image
     ↓
DenseNet121
     ↓
Prediction
     ↓
Gradients
     ↓
Activation Map
     ↓
Heatmap
     ↓
Grad-CAM Overlay
```

Generated files include:

```text
gradcam_outputs/
├── original.jpg
├── heatmap.jpg
└── gradcam_overlay.jpg
```

This improves model transparency by showing the user **where the model focused its attention**.

> Grad-CAM visualization is an explanation of the model's behavior and should not be interpreted as a clinically validated diagnostic localization.

---

## 🚀 FastAPI Backend

The backend is implemented using **FastAPI**.

### Available Endpoints

| Method | Endpoint              | Purpose                         |
| ------ | --------------------- | ------------------------------- |
| GET    | `/`                   | API information                 |
| GET    | `/health`             | Health check                    |
| GET    | `/model-info`         | Model information               |
| POST   | `/predict`            | Upload image and get prediction |
| GET    | `/gradcam/{filename}` | Retrieve Grad-CAM output        |

---

## 📚 Swagger Documentation

After starting the FastAPI server, open:

```text
http://127.0.0.1:8000/docs
```

Swagger provides an interactive interface for testing the API.

### Example Prediction Response

```json
{
    "filename": "original.jpg",
    "prediction": "Normal",
    "confidence": 74.73,
    "model": "DenseNet121"
}
```

---

## 🗄️ Database

SQLAlchemy is used to store prediction history.

The database can store information such as:

```text
Image Reference
Prediction
Confidence
Model Version
Grad-CAM Path
AI Report
Timestamp
```

Example:

```text
Prediction History
        │
        ├── Image
        ├── Prediction
        ├── Confidence
        ├── Model Version
        ├── Grad-CAM
        ├── Report
        └── Timestamp
```

---

## 🤖 AI-Assisted Medical Reports

The platform can generate a structured AI-assisted report using the model output.

The report generation workflow is:

```text
Model Prediction
       +
Confidence
       +
Structured Results
       ↓
      LLM
       ↓
AI-Assisted Report
```

The generated report is intended to help demonstrate how AI-generated explanations can be integrated into a medical AI workflow.

> The generated report is not a medical diagnosis and must not be used for clinical decision-making.

---

## 🖥️ Streamlit Interface

The Streamlit interface provides an easy way to interact with the complete system.

Typical workflow:

```text
1. Open Streamlit application
2. Upload chest X-ray
3. Click Predict
4. Model analyzes image
5. Prediction is displayed
6. Confidence score is displayed
7. Grad-CAM visualization is generated
8. AI-assisted report is displayed
9. Prediction can be stored in history
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Advanced-AI-Medical-Intelligence-Platform.git
```

```bash
cd Advanced-AI-Medical-Intelligence-Platform
```

### 2. Create Virtual Environment

Windows:

```powershell
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## ▶️ Run FastAPI

Start the backend with:

```powershell
uvicorn api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

## ▶️ Run Streamlit

Start the frontend with:

```powershell
streamlit run streamlit_app.py
```

Streamlit will display a local URL in the terminal.

Open the displayed URL in your browser.

---

## 🐳 Run with Docker

Build the Docker image:

```bash
docker build -t medical-ai-platform .
```

Run the container:

```bash
docker run -p 8000:8000 medical-ai-platform
```

Then access:

```text
http://localhost:8000
```

---

## 🧪 Model Evaluation

The model was evaluated using a test dataset and confusion matrix.

Example confusion matrix:

```text
[[ 171    1  388  975    0]
 [  13    2   67  244    0]
 [  81    1  525  619    0]
 [ 226    4  712 7205    0]
 [   9    0   50  136    0]]
```

The evaluation demonstrates the model's performance across the five target classes.

---

## 🔐 Validation & Error Handling

The API includes validation and error-handling mechanisms for:

* Invalid image files
* Missing uploads
* Unsupported file types
* Missing model files
* Prediction failures
* Grad-CAM generation errors
* API request errors

---

## 🔄 Complete Project Workflow

```text
                 ┌─────────────┐
                 │ Upload X-ray│
                 └──────┬──────┘
                        ↓
              ┌──────────────────┐
              │ Image Validation │
              └────────┬─────────┘
                       ↓
             ┌────────────────────┐
             │ Image Preprocessing│
             └─────────┬──────────┘
                       ↓
                ┌─────────────┐
                │ DenseNet121 │
                └──────┬──────┘
                       ↓
             ┌──────────────────┐
             │ Disease Prediction│
             └────────┬─────────┘
                      ↓
              ┌───────────────┐
              │ Confidence    │
              └───────┬───────┘
                      ↓
          ┌───────────────────────┐
          │      Grad-CAM         │
          │ Explainable AI Output │
          └───────────┬───────────┘
                      ↓
             ┌────────────────┐
             │ LLM AI Report  │
             └───────┬────────┘
                     ↓
             ┌────────────────┐
             │ SQLAlchemy DB  │
             └───────┬────────┘
                     ↓
             ┌────────────────┐
             │ Streamlit UI   │
             └────────────────┘
```

---

## 📸 Sample Output

The application produces:

* Original chest X-ray
* Predicted disease
* Confidence score
* Grad-CAM heatmap
* Grad-CAM overlay
* AI-assisted report
* Prediction history

---

## 💡 Key Features

### 1. Deep Learning

Uses DenseNet121 transfer learning for medical image classification.

### 2. Explainable AI

Uses Grad-CAM to visualize model attention.

### 3. REST API

FastAPI provides programmatic access to predictions.

### 4. AI Report Generation

LLM integration converts structured model output into an understandable report.

### 5. Database

Stores prediction history and associated metadata.

### 6. User Interface

Streamlit provides a simple interface for uploading and analyzing images.

### 7. Deployment

Docker support makes the application easier to deploy.

---

## 📈 Future Improvements

Possible future enhancements include:

* Improve class balancing.
* Train for more epochs using GPU acceleration.
* Apply advanced data augmentation.
* Hyperparameter optimization.
* Improve minority-class performance.
* Add ROC-AUC and F1-score analysis.
* Add precision/recall metrics.
* Add authentication and authorization.
* Add cloud deployment.
* Add monitoring and logging.
* Add more medical imaging datasets.
* Support additional diseases.
* Improve LLM report validation.
* Add automated model versioning.

---

## 👨‍💻 Author

**Pandu Jerpati**

Computer Science & Engineering

---

## ⭐ Project Highlights

```text
✔ Deep Learning
✔ DenseNet121
✔ Medical Image Classification
✔ Explainable AI
✔ Grad-CAM
✔ FastAPI
✔ REST API
✔ SQLAlchemy
✔ LLM Integration
✔ Streamlit
✔ Docker
✔ GitHub
```

---

## ⚠️ Disclaimer

This project is intended **only for educational, research, and demonstration purposes**.

The predictions, confidence scores, Grad-CAM visualizations, and AI-generated reports should **not be considered medical advice or a clinical diagnosis**. Always consult a qualified healthcare professional for medical evaluation and treatment decisions.
