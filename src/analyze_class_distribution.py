from datasets import load_dataset
from collections import Counter


# =========================
# 1. LOAD DATASET
# =========================

dataset = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0",
    split="train"
)

print("Dataset loaded successfully.")


# =========================
# 2. SELECT TRAINING DATA
# =========================

train_data = dataset.filter(
    lambda example: example["split"] == "train"
)

print(
    "\nTraining dataset size:"
)

print(
    len(train_data)
)


# =========================
# 3. COUNT CLASS DISTRIBUTION
# =========================

class_counts = Counter(
    train_data["class_label"]
)


# =========================
# 4. SORT CLASSES
# =========================

sorted_classes = sorted(
    class_counts.items(),
    key=lambda item: item[1]
)


# =========================
# 5. DISPLAY CLASS DISTRIBUTION
# =========================

print(
    "\n--- CLASS DISTRIBUTION ---"
)

for class_name, count in sorted_classes:

    print(
        f"{class_name}: "
        f"{count}"
    )


# =========================
# 6. DISPLAY WEAK CLASSES
# =========================

weak_classes = [
    "Tomato___Early_blight",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Cherry_(including_sour)___healthy",
    "Blueberry___healthy"
]


print(
    "\n--- WEAK CLASS DISTRIBUTION ---"
)

for class_name in weak_classes:

    print(
        f"{class_name}: "
        f"{class_counts[class_name]} training images"
    )


