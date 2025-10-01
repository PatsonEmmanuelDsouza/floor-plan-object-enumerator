import os
import random
import shutil
from pathlib import Path

# Paths
base_dir = Path("latest_images")
images_dir = base_dir / "images"
labels_dir = base_dir / "labels"

# New structure
for split in ["train", "val", "test"]:
    (base_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (base_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

# Get all images
all_images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
random.shuffle(all_images)

# Split ratios
train_split = 0.7
val_split = 0.2
test_split = 0.1

n_total = len(all_images)
n_train = int(n_total * train_split)
n_val = int(n_total * val_split)

train_files = all_images[:n_train]
val_files = all_images[n_train:n_train + n_val]
test_files = all_images[n_train + n_val:]

def move_files(files, split):
    for img in files:
        label = labels_dir / (img.stem + ".txt")
        if not label.exists():
            continue  # skip if no label
        shutil.move(str(img), base_dir / "images" / split / img.name)
        shutil.move(str(label), base_dir / "labels" / split / label.name)

# Move the files
move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print("Dataset split complete!")
