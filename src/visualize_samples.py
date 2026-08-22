from datasets import load_dataset
import matplotlib.pyplot as plt

# Load dataset
dataset = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0",
    split="train"
)

# Classes we want to visually inspect
selected_classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Corn_(maize)___Common_rust_",
    "Grape___Black_rot"
]

# Get the labels once
labels = dataset["class_label"]

# Find one example from each selected class
selected_examples = []

for class_name in selected_classes:

    for i, label in enumerate(labels):

        if label == class_name:
            selected_examples.append((i, class_name))
            break

# Create a 3 x 3 grid
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# Display the selected images
for ax, (index, class_name) in zip(axes.flat, selected_examples):

    image = dataset[index]["image"]

    ax.imshow(image)
    ax.set_title(class_name, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.show()