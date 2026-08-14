import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
CSV_PATH = "data/raw/Data_Entry_2017.csv"
IMAGE_DIR = "data/raw/images"
OUTPUT_DIR = "data/processed"

CLASSES = [
    "Normal",
    "Pneumonia",
    "Atelectasis",
    "Cardiomegaly",
    "Effusion"
]

# Read CSV
df = pd.read_csv(CSV_PATH)

print("CSV loaded successfully!")
print("Total records:", len(df))

# Create a dictionary of all images
print("Searching for X-ray images...")

image_paths = {}

for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.lower().endswith(".png"):
            image_paths[file] = os.path.join(root, file)

print("Images found:", len(image_paths))


# Convert NIH labels to our 5 classes
def get_class(label):
    if label == "No Finding":
        return "Normal"

    if "Pneumonia" in label:
        return "Pneumonia"

    if "Atelectasis" in label:
        return "Atelectasis"

    if "Cardiomegaly" in label:
        return "Cardiomegaly"

    if "Effusion" in label:
        return "Effusion"

    return None


# Create class column
df["Class"] = df["Finding Labels"].apply(get_class)

# Remove images that don't belong to our 5 classes
df = df[df["Class"].notna()].copy()

print("\nSelected 5-class dataset:")
print(df["Class"].value_counts())


# Keep only images that actually exist
df["Image_Path"] = df["Image Index"].map(image_paths)

df = df[df["Image_Path"].notna()].copy()

print("\nImages available for processing:", len(df))


# Train / validation / test split
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["Class"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["Class"]
)

print("\nDataset split:")
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))


# Create output directories
for split in ["train", "val", "test"]:
    for class_name in CLASSES:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, class_name),
            exist_ok=True
        )
def copy_images(dataframe, split):
    print(f"\nProcessing {split} images...")

    total = len(dataframe)

    for count, (_, row) in enumerate(dataframe.iterrows(), start=1):

        source = row["Image_Path"]

        destination = os.path.join(
            OUTPUT_DIR,
            split,
            row["Class"],
            row["Image Index"]
        )

        if os.path.exists(source):
            shutil.copy2(source, destination)

        # Show progress every 500 images
        if count % 500 == 0 or count == total:
            print(f"{split}: {count}/{total} images copied")

    print(f"{split} completed.")

copy_images(train_df, "train")
copy_images(val_df, "val")
copy_images(test_df, "test")
print("\n================================")
print("DATASET PREPARATION COMPLETED!")
print("================================")