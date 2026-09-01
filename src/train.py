import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# ==========================================
# 1. DEVICE
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ==========================================
# 2. DATASET PATHS
# ==========================================

TRAIN_DIR = r"D:\AI-Medical-Data\data\processed\train"
VAL_DIR = r"D:\AI-Medical-Data\data\processed\val"


# ==========================================
# 3. IMAGE TRANSFORMS
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==========================================
# 4. LOAD DATASET
# ==========================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)


# ==========================================
# 5. DATA LOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# ==========================================
# 6. LOAD PRETRAINED DENSENET121
# ==========================================

print("\nLoading pretrained DenseNet121...")

model = models.densenet121(
    weights=models.DenseNet121_Weights.DEFAULT
)


# ==========================================
# 7. FREEZE PRETRAINED LAYERS
# ==========================================

for param in model.parameters():
    param.requires_grad = False


# ==========================================
# 8. CHANGE FINAL CLASSIFIER
# ==========================================

num_classes = len(train_dataset.classes)

model.classifier = nn.Linear(
    model.classifier.in_features,
    num_classes
)

model = model.to(device)

print("DenseNet121 model ready.")
print("Number of classes:", num_classes)


# ==========================================
# 9. LOSS FUNCTION
# ==========================================

criterion = nn.CrossEntropyLoss()


# ==========================================
# 10. OPTIMIZER
# ==========================================

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)


# ==========================================
# 11. TRAINING
# ==========================================

num_epochs = 1

for epoch in range(num_epochs):

    print("\n" + "=" * 50)
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("=" * 50)

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # ------------------------------------------
    # Training batches
    # ------------------------------------------

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        # Show progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(
                f"Batch {batch_idx + 1}/{len(train_loader)}"
            )


    # ==========================================
    # TRAINING RESULTS
    # ==========================================

    train_loss = running_loss / len(train_loader)

    train_accuracy = 100 * correct / total


    # ==========================================
    # 12. VALIDATION
    # ==========================================

    print("\nStarting validation...")

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(val_loader):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()

            # Validation progress
            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Validation Batch "
                    f"{batch_idx + 1}/{len(val_loader)}"
                )


    # ==========================================
    # VALIDATION RESULTS
    # ==========================================

    val_loss = val_loss / len(val_loader)

    val_accuracy = 100 * val_correct / val_total


    # ==========================================
    # 13. PRINT RESULTS
    # ==========================================

    print("\n" + "=" * 50)
    print("TRAINING RESULTS")
    print("=" * 50)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")


# ==========================================
# 14. SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/densenet121_medical.pth"
)


# ==========================================
# 15. COMPLETED
# ==========================================

print("\n" + "=" * 50)
print("TRAINING COMPLETED")
print("=" * 50)

print("Model saved:")
print("models/densenet121_medical.pth")