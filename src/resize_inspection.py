from datasets import load_dataset
import numpy as np
from PIL import Image

dataset = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0",
    split="train"
)

# Get one image
image = dataset[0]["image"]

print("Original image size:")
print(image.size)

# Resize image
resized_image = image.resize((224, 224))

print("\nResized image size:")
print(resized_image.size)

# Convert both images to NumPy arrays
original_array = np.array(image)
resized_array = np.array(resized_image)

print("\nOriginal array shape:")
print(original_array.shape)

print("\nResized array shape:")
print(resized_array.shape)