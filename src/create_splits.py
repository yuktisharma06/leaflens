from datasets import load_dataset, ClassLabel

dataset = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0",
    split="train"
)

# Separate the provided training and test data
train_data = dataset.filter(
    lambda example: example["split"] == "train"
)

test_data = dataset.filter(
    lambda example: example["split"] == "test"
)

print("Original training data:", len(train_data))
print("Original test data:", len(test_data))

# Create a ClassLabel feature for stratified splitting
class_names = sorted(set(dataset["class_label"]))

train_data = train_data.cast_column(
    "class_idx",
    ClassLabel(names=class_names)
)

# Create validation split from training data
train_val_split = train_data.train_test_split(
    test_size=0.10,
    seed=42,
    stratify_by_column="class_idx"
)

final_train = train_val_split["train"]
validation = train_val_split["test"]

print("\nFinal split sizes:")
print("Training:", len(final_train))
print("Validation:", len(validation))
print("Test:", len(test_data))



from collections import Counter

train_counts = Counter(final_train["class_label"])
validation_counts = Counter(validation["class_label"])

print("\nClass distribution check:")

for class_name in sorted(train_counts):
    train_percentage = train_counts[class_name] / len(final_train) * 100
    validation_percentage = validation_counts[class_name] / len(validation) * 100

    print(
        f"{class_name}\n"
        f"  Train: {train_counts[class_name]} ({train_percentage:.2f}%)\n"
        f"  Validation: {validation_counts[class_name]} ({validation_percentage:.2f}%)"
    )

