from datasets import load_dataset
from collections import Counter

dataset = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0",
    split="train"
)

print("Dataset loaded successfully.")

print("\nNumber of examples:", len(dataset))

# --------------------------------------------------
# 1. Dataset columns
# --------------------------------------------------

print("\nColumns:")
print(dataset.column_names)


# --------------------------------------------------
# 2. Host / plant distribution
# --------------------------------------------------

hosts = dataset["host"]

print("\nNumber of unique hosts:", len(set(hosts)))

print("\nHost distribution:")

host_counts = Counter(hosts)

for host, count in sorted(host_counts.items()):
    print(f"{host}: {count}")


# --------------------------------------------------
# 3. Disease distribution
# --------------------------------------------------

diseases = dataset["disease"]

print("\nNumber of unique diseases:", len(set(diseases)))

print("\nFirst 20 diseases:")

for disease in sorted(set(diseases))[:20]:
    print(disease)


# --------------------------------------------------
# 4. Leaf grouping information
# --------------------------------------------------

leaf_grouped = dataset["leaf_grouped"]

print("\nLeaf grouped distribution:")

grouped_counts = Counter(leaf_grouped)

for value, count in sorted(grouped_counts.items()):
    print(f"{value}: {count}")


# --------------------------------------------------
# 5. Number of unique leaf IDs
# --------------------------------------------------

leaf_ids = dataset["leaf_id"]

print("\nNumber of unique leaf IDs:", len(set(leaf_ids)))


# --------------------------------------------------
# 6. Dataset split field
# --------------------------------------------------

splits = dataset["split"]

print("\nSplit values:")

split_counts = Counter(splits)

for split_name, count in sorted(split_counts.items()):
    print(f"{split_name}: {count}")