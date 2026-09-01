import os
import io

import torch
import torch.nn as nn

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse
from PIL import Image
from torchvision import models, transforms

from database.database import engine, Base, get_db
from database.models import PredictionHistory
from sqlalchemy.orm import Session

from src.gradcam import generate_gradcam
from src.report_generator import generate_report


# ============================================================
# 1. CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Medical Intelligence Platform",
    description="Medical X-ray prediction API using DenseNet121",
    version="1.0.0"
)

Base.metadata.create_all(bind=engine)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 3. MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "densenet121_medical.pth"
)

GRADCAM_DIR = "gradcam_outputs"

os.makedirs(
    GRADCAM_DIR,
    exist_ok=True
)


# ============================================================
# 4. CLASS NAMES
# ============================================================

classes = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Normal",
    "Pneumonia"
]


# ============================================================
# 5. IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 6. LOAD DENSENET121
# ============================================================

print("Loading DenseNet121...")

model = models.densenet121(
    weights=None
)

model.classifier = nn.Linear(
    model.classifier.in_features,
    len(classes)
)


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("DenseNet121 loaded successfully.")


# ============================================================
# 7. HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "AI Medical Intelligence Platform API is running",
        "model": "DenseNet121",
        "classes": classes
    }


# ============================================================
# 8. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device)
    }


# ============================================================
# 9. MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "model": "DenseNet121",
        "framework": "PyTorch",
        "input_size": "224x224",
        "classes": classes
    }


# ============================================================
# 10. PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    # --------------------------------------------------------
    # Check file type
    # --------------------------------------------------------

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg"
    ]:

        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image."
        )


    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    try:

        image_data = await file.read()

        image = Image.open(
            io.BytesIO(image_data)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )


    # --------------------------------------------------------
    # Transform image
    # --------------------------------------------------------

    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)


    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )


    # --------------------------------------------------------
    # Get prediction
    # --------------------------------------------------------

    predicted_index = predicted_class.item()

    predicted_name = classes[
        predicted_index
    ]

    confidence_value = (
        confidence.item() * 100
    )


    # ========================================================
    # GENERATE GRAD-CAM
    # ========================================================

    gradcam_filename = (
        f"{file.filename}_gradcam.jpg"
    )

    gradcam_path = os.path.join(
        GRADCAM_DIR,
        gradcam_filename
    )

    generate_gradcam(
        model=model,
        input_tensor=input_tensor,
        original_image=image,
        predicted_index=predicted_index,
        output_path=gradcam_path
    )


    # ========================================================
    # GENERATE AI REPORT
    # ========================================================

    report = generate_report(
        prediction=predicted_name,
        confidence=confidence_value,
        model_name="DenseNet121"
    )


    # ========================================================
    # SAVE PREDICTION TO DATABASE
    # ========================================================

    prediction_record = PredictionHistory(
        image_filename=file.filename,
        prediction=predicted_name,
        confidence=round(
            confidence_value,
            2
        ),
        model_version="DenseNet121",
        gradcam_path=gradcam_path,
        report=report
    )

    db.add(
        prediction_record
    )

    db.commit()

    db.refresh(
        prediction_record
    )


    # ========================================================
    # RETURN RESPONSE
    # ========================================================

    return {

        "id": prediction_record.id,

        "filename": file.filename,

        "prediction": predicted_name,

        "confidence": round(
            confidence_value,
            2
        ),

        "model": "DenseNet121",

        "model_version": "DenseNet121",

        "gradcam_path": gradcam_path,

        "report": report
    }


# ============================================================
# 11. GRAD-CAM ENDPOINT
# ============================================================

@app.get("/gradcam/{filename}")
def get_gradcam(
    filename: str
):

    file_path = os.path.join(
        GRADCAM_DIR,
        filename
    )


    if not os.path.exists(
        file_path
    ):

        raise HTTPException(
            status_code=404,
            detail="Grad-CAM image not found"
        )


    return FileResponse(
        file_path,
        media_type="image/jpeg"
    )


# ============================================================
# 12. PREDICTION HISTORY
# ============================================================

@app.get("/predictions")
def get_predictions(
    db: Session = Depends(get_db)
):

    predictions = (
        db.query(
            PredictionHistory
        )
        .order_by(
            PredictionHistory.id.desc()
        )
        .all()
    )


    return [

        {
            "id": item.id,

            "image_filename": item.image_filename,

            "prediction": item.prediction,

            "confidence": item.confidence,

            "model_version": item.model_version,

            "gradcam_path": item.gradcam_path,

            "report": item.report
        }

        for item in predictions
    ]