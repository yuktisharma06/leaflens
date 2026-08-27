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
# 2. CREATE TRAIN / VALIDATION SPLIT
# =========================

train_data = dataset.filter(
    lambda example: example["split"] == "train"
)

train_data = train_data.class_encode_column(
    "class_idx"
)

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


# =========================
# 3. CREATE SMALL SUBSETS
# =========================

train_subset = train_data.shuffle(seed=42).select(
    range(5000)
)

validation_subset = validation_data.shuffle(
    seed=42
).select(
    range(1000)
)

print("\nSubset sizes:")
print("Training:", len(train_subset))
print("Validation:", len(validation_subset))


# =========================
# 4. DATA GENERATOR
# =========================

def dataset_generator(data):

    for example in data:

        image = np.array(example["image"])
        label = example["class_idx"]

        yield image, label


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


# =========================
# 5. CREATE TF DATASETS
# =========================

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
# 6. PREPROCESSING
# =========================

def preprocess(image, label):

    image = tf.cast(
        image,
        tf.float32
    )

    image = tf.image.resize(
        image,
        (224, 224)
    )

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image, label


train_tf = train_tf.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_tf = validation_tf.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)


# =========================
# 7. DATA AUGMENTATION
# =========================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.05
    ),

    tf.keras.layers.RandomZoom(
        0.05
    ),

])


print("\nData augmentation pipeline created successfully.")


# =========================
# 8. APPLY DATA AUGMENTATION
# =========================

def augment(image, label):

    image = data_augmentation(
        image,
        training=True
    )

    return image, label


train_tf = train_tf.map(
    augment,
    num_parallel_calls=tf.data.AUTOTUNE
)

print("\nData augmentation applied to training dataset.")


# =========================
# 9. SHUFFLE AND BATCH
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


# =========================
# 10. PREFETCH DATA
# =========================

train_tf = train_tf.prefetch(
    tf.data.AUTOTUNE
)

validation_tf = validation_tf.prefetch(
    tf.data.AUTOTUNE
)

print("\nShuffling, batching and prefetching completed successfully.")


# =========================
# 11. INSPECT FINAL BATCH
# =========================

for images, labels in train_tf.take(1):

    print("\nFinal training batch image shape:")
    print(images.shape)

    print("\nFinal training batch label shape:")
    print(labels.shape)

    print("\nImage dtype:")
    print(images.dtype)

    print("\nMinimum pixel value:")
    print(tf.reduce_min(images).numpy())

    print("\nMaximum pixel value:")
    print(tf.reduce_max(images).numpy())


# =========================
# 12. VISUALIZE AUGMENTED IMAGES
# =========================

import matplotlib.pyplot as plt


for images, labels in train_tf.take(1):

    plt.figure(figsize=(10, 10))

    for i in range(9):

        plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy())

        plt.title(f"Class: {labels[i].numpy()}")

        plt.axis("off")

    plt.tight_layout()

    plt.show()


# =========================
# 13. LOAD PRETRAINED MODEL
# =========================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

print("\nPretrained MobileNetV2 loaded successfully.")


# =========================
# 14. BUILD TRANSFER LEARNING MODEL
# =========================

inputs = tf.keras.Input(
    shape=(224, 224, 3)
)

x = base_model(
    inputs,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(x)

outputs = tf.keras.layers.Dense(
    38,
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs,
    outputs
)

model.summary()


# =========================
# 15. COMPILE MODEL
# =========================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTransfer learning model compiled successfully.")


# =========================
# 16. TRAIN MODEL
# =========================

history = model.fit(
    train_tf,
    validation_data=validation_tf,
    epochs=5
)


# =========================
# 17. SAVE MODEL
# =========================

model.save(
    "models/transfer_learning_mobilenet.keras"
)

print(
    "\nTransfer learning model saved successfully."
)