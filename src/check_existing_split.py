from datasets import load_dataset
from collections import Counter

dataset = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0",
    split="train"
)

# Check existing split values
split_counts = Counter(dataset["split"])

print("Existing split distribution:\n")

for split_name, count in sorted(split_counts.items()):
    print(f"{split_name}: {count}")

print("\nFirst 10 split values:")
print(dataset["split"][:10])