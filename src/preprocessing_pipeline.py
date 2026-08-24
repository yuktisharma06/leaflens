from datasets import load_dataset
import tensorflow as tf
import numpy as np

# Load dataset
dataset = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0",
    split="train"
)

# Get one image
image = dataset[0]["image"]

# Convert image to NumPy array
image_array = np.array(image)

print("Original image shape:")
print(image_array.shape)

print("\nOriginal dtype:")
print(image_array.dtype)

print("\nOriginal pixel range:")
print(image_array.min(), "to", image_array.max())


# Create preprocessing layers
resize_layer = tf.keras.layers.Resizing(224, 224)

normalize_layer = tf.keras.layers.Rescaling(1.0 / 255)


# Convert NumPy array to TensorFlow tensor
image_tensor = tf.convert_to_tensor(image_array)

print("\nTensor shape:")
print(image_tensor.shape)

print("\nTensor dtype:")
print(image_tensor.dtype)


# Apply resizing
resized_image = resize_layer(image_tensor)

print("\nAfter resizing:")
print("Shape:", resized_image.shape)


# Apply normalization
processed_image = normalize_layer(resized_image)

print("\nAfter normalization:")
print("Minimum:", tf.reduce_min(processed_image).numpy())
print("Maximum:", tf.reduce_max(processed_image).numpy())

print("\nPreprocessing completed successfully.")