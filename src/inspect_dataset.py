from datasets import load_dataset
from collections import Counter

dataset = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0",
    split="train"
)

print("Dataset loaded successfully.")

print("\nNumber of examples:", len(dataset))

# Get all class labels
labels = dataset["class_label"]

# Count each class
class_counts = Counter(labels)

print("\nNumber of unique classes:", len(class_counts))

print("\nClass distribution:")

for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count}")


# ==========================================
# MAJORITY-CLASS BASELINE
# ==========================================

majority_class, majority_count = class_counts.most_common(1)[0]

baseline_accuracy = majority_count / len(dataset)

print("\nMajority class:")
print(majority_class)

print("\nMajority class count:")
print(majority_count)

print("\nMajority-class baseline accuracy:")
print(f"{baseline_accuracy:.4f}")

print("\nMajority-class baseline accuracy (%):")
print(f"{baseline_accuracy * 100:.2f}%")


# --------------------------------------------------
# Inspect leaf IDs and grouping
# --------------------------------------------------

leaf_ids = dataset["leaf_id"]
leaf_grouped = dataset["leaf_grouped"]

print("\nNumber of unique leaf IDs:")
print(len(set(leaf_ids)))

print("\nNumber of grouped examples:")
print(sum(leaf_grouped))

print("\nNumber of non-grouped examples:")
print(len(leaf_grouped) - sum(leaf_grouped))

print("\nFirst 10 leaf IDs:")
print(leaf_ids[:10])

print("\nFirst 10 leaf_grouped values:")
print(leaf_grouped[:10])