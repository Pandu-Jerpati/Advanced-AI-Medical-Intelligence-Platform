import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# ==========================================
# 1. DEVICE
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ==========================================
# 2. PATHS
# ==========================================

TEST_DIR = r"D:\AI-Medical-Data\data\processed\test"

MODEL_PATH = r"models\densenet121_medical.pth"


# ==========================================
# 3. IMAGE TRANSFORMATION
# ==========================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==========================================
# 4. LOAD TEST DATASET
# ==========================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

classes = test_dataset.classes

print("\nClasses:", classes)
print("Test images:", len(test_dataset))
print("Test batches:", len(test_loader))


# ==========================================
# 5. LOAD DENSENET121
# ==========================================

print("\nLoading DenseNet121...")

model = models.densenet121(weights=None)

num_classes = len(classes)

model.classifier = nn.Linear(
    model.classifier.in_features,
    num_classes
)


# ==========================================
# 6. LOAD TRAINED MODEL
# ==========================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Trained model loaded successfully.")


# ==========================================
# 7. TEST EVALUATION
# ==========================================

all_predictions = []
all_labels = []

print("\nStarting test evaluation...")

with torch.no_grad():

    for batch_idx, (images, labels) in enumerate(test_loader):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Test Batch "
                f"{batch_idx + 1}/{len(test_loader)}"
            )


# ==========================================
# 8. NUMPY ARRAYS
# ==========================================

all_predictions = np.array(all_predictions)

all_labels = np.array(all_labels)


# ==========================================
# 9. TEST ACCURACY
# ==========================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)


# ==========================================
# 10. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=classes,
        digits=4
    )
)


# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")

print(cm)


# ==========================================
# 12. COMPLETED
# ==========================================

print("\n" + "=" * 60)
print("TEST EVALUATION COMPLETED")
print("=" * 60)