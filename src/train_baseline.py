from datasets import load_dataset
import tensorflow as tf
import numpy as np


# =========================
# 1. LOAD DATASET
# =========================

dataset = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0",
    split="train"
)

print("Dataset loaded successfully.")


# =========================
# 2. CREATE TRAIN/VALIDATION/TEST SPLITS
# =========================

train_data = dataset.filter(lambda example: example["split"] == "train")
test_data = dataset.filter(lambda example: example["split"] == "test")

# class_idx must be converted to ClassLabel before stratification
train_data = train_data.class_encode_column("class_idx")

train_val_split = train_data.train_test_split(
    test_size=0.10,
    seed=42,
    stratify_by_column="class_idx"
)

train_data = train_val_split["train"]
validation_data = train_val_split["test"]

print("\nDataset sizes:")
print("Training:", len(train_data))
print("Validation:", len(validation_data))
print("Test:", len(test_data))


# =========================
# 3. CREATE A SMALL SUBSET
# =========================

train_subset = train_data.shuffle(seed=42).select(range(5000))
validation_subset = validation_data.shuffle(seed=42).select(range(1000))

print("\nBaseline subset sizes:")
print("Training:", len(train_subset))
print("Validation:", len(validation_subset))


# =========================
# 4. CONVERT TO TENSORFLOW DATASET
# =========================

def dataset_generator(data):
    for example in data:

        image = np.array(example["image"])

        label = example["class_idx"]

        yield image, label


# TensorFlow output description
output_signature = (
    tf.TensorSpec(
        shape=(256, 256, 3),
        dtype=tf.uint8
    ),
    tf.TensorSpec(
        shape=(),
        dtype=tf.int32
    )
)


# Create TensorFlow datasets
train_tf = tf.data.Dataset.from_generator(
    lambda: dataset_generator(train_subset),
    output_signature=output_signature
)

validation_tf = tf.data.Dataset.from_generator(
    lambda: dataset_generator(validation_subset),
    output_signature=output_signature
)


print("\nTensorFlow datasets created successfully.")


# =========================
# 5. PREPROCESS THE DATA
# =========================

def preprocess(image, label):

    # Convert pixel values to float
    image = tf.cast(image, tf.float32)

    # Resize image
    image = tf.image.resize(image, (224, 224))

    # Normalize pixel values
    image = image / 255.0

    return image, label


# Apply preprocessing to every example
train_tf = train_tf.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_tf = validation_tf.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)


# =========================
# 6. SHUFFLE, BATCH AND PREFETCH
# =========================

BATCH_SIZE = 32

train_tf = train_tf.shuffle(
    buffer_size=1000,
    seed=42
)

train_tf = train_tf.batch(
    BATCH_SIZE
)

validation_tf = validation_tf.batch(
    BATCH_SIZE
)

train_tf = train_tf.prefetch(
    tf.data.AUTOTUNE
)

validation_tf = validation_tf.prefetch(
    tf.data.AUTOTUNE
)

print("\nPreprocessing and batching completed.")


# =========================
# 7. INSPECT ONE BATCH
# =========================

for images, labels in train_tf.take(1):

    print("\nBatch image shape:")
    print(images.shape)

    print("\nBatch label shape:")
    print(labels.shape)

    print("\nImage dtype:")
    print(images.dtype)

    print("\nMinimum pixel value:")
    print(tf.reduce_min(images).numpy())

    print("\nMaximum pixel value:")
    print(tf.reduce_max(images).numpy())


# =========================
# 8. BUILD BASELINE CNN
# =========================

model = tf.keras.Sequential([
    
    tf.keras.layers.Input(shape=(224, 224, 3)),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        38,
        activation="softmax"
    )
])

model.summary()


# =========================
# 9. COMPILE THE MODEL
# =========================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel compiled successfully.")


# =========================
# 10. TRAIN THE MODEL
# =========================

history = model.fit(
    train_tf,
    validation_data=validation_tf,
    epochs=5
)


# =========================
# 11. SAVE BASELINE MODEL
# =========================

model.save("models/baseline_cnn.keras")

print("\nBaseline model saved successfully.")