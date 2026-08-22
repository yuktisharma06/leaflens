from datasets import load_dataset
import matplotlib.pyplot as plt

# Load dataset
dataset = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0",
    split="train"
)

# Select one sample
sample = dataset[0]

# Extract image and label
image = sample["image"]
label = sample["class_label"]

# Print basic information
print("Label:", label)
print("Image size:", image.size)
print("Image mode:", image.mode)

# Display the image
plt.figure(figsize=(6, 6))

plt.imshow(image)
plt.title(label)
plt.axis("off")

plt.show()