from datasets import load_dataset
import numpy as np

dataset = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0",
    split="train"
)

# Get one sample image
sample = dataset[0]

image = sample["image"]

# Convert PIL image to NumPy array
image_array = np.array(image)

print("Class label:")
print(sample["class_label"])

print("\nImage shape:")
print(image_array.shape)

print("\nImage dtype:")
print(image_array.dtype)

print("\nMinimum pixel value:")
print(image_array.min())

print("\nMaximum pixel value:")
print(image_array.max())

print("\nFirst pixel RGB values:")
print(image_array[0, 0])

# Normalize the image
normalized_image = image_array / 255.0

print("\nAfter normalization:")

print("Minimum pixel value:")
print(normalized_image.min())

print("\nMaximum pixel value:")
print(normalized_image.max())

print("\nFirst normalized pixel RGB values:")
print(normalized_image[0, 0])