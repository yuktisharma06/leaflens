from datasets import load_dataset

dataset = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0",
    split="train"
)

# Separate leaf IDs according to the existing split
train_leaf_ids = set(
    dataset["leaf_id"][i]
    for i in range(len(dataset))
    if dataset["split"][i] == "train"
)

test_leaf_ids = set(
    dataset["leaf_id"][i]
    for i in range(len(dataset))
    if dataset["split"][i] == "test"
)

# Find overlap
overlap = train_leaf_ids.intersection(test_leaf_ids)

print("Unique leaf IDs in train:", len(train_leaf_ids))
print("Unique leaf IDs in test:", len(test_leaf_ids))

print("\nNumber of overlapping leaf IDs:", len(overlap))

if len(overlap) == 0:
    print("No leaf-level overlap found.")
else:
    print("Example overlapping leaf IDs:")
    print(list(overlap)[:10])