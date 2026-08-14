import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# ==========================================
# 1. Configuration
# ==========================================

TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/val"
MODEL_DIR = "models"

BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

NUM_CLASSES = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)


# ==========================================
# 2. Image preprocessing
# ==========================================

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 3. Load datasets
# ==========================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transforms
)

print("\nClasses:")
print(train_dataset.classes)

print("\nTraining images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ==========================================
# 4. Create DataLoaders
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 5. Load pretrained ResNet18
# ==========================================

print("\nLoading pretrained ResNet18...")

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


# ==========================================
# 6. Replace final layer
# ==========================================

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    NUM_CLASSES
)

model = model.to(DEVICE)

print("Model ready.")
print("Number of classes:", NUM_CLASSES)


# ==========================================
# 7. Loss and optimizer
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 8. Training
# ==========================================

best_val_accuracy = 0.0

os.makedirs(MODEL_DIR, exist_ok=True)

for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 50)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print("=" * 50)

    # -------------------------
    # Training
    # -------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Batch {batch_idx + 1}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    print(f"\nTraining Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")


    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")


    # -------------------------
    # Save best model
    # -------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        model_path = os.path.join(
            MODEL_DIR,
            "best_resnet18.pth"
        )

        torch.save(
            model.state_dict(),
            model_path
        )

        print("\nBest model saved!")
        print("Path:", model_path)


print("\n================================")
print("TRAINING COMPLETED!")
print("================================")

print("Best Validation Accuracy:",
      f"{best_val_accuracy:.2f}%")