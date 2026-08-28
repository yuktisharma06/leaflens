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
# 3. CREATE SUBSETS
# =========================

train_subset = train_data.shuffle(
    seed=42
).select(
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

        image = np.array(
            example["image"]
        )

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
# 8. SHUFFLE AND BATCH
# =========================

BATCH_SIZE = 16

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
# 9. PREFETCH DATA
# =========================

train_tf = train_tf.prefetch(
    tf.data.AUTOTUNE
)

validation_tf = validation_tf.prefetch(
    tf.data.AUTOTUNE
)

print(
    "\nShuffling, batching and prefetching completed successfully."
)


# =========================
# 10. LOAD TRANSFER LEARNING MODEL
# =========================

model = tf.keras.models.load_model(
    "models/transfer_learning_mobilenet.keras"
)

print(
    "\nTransfer learning model loaded successfully."
)


# =========================
# 11. ACCESS BASE MODEL
# =========================

base_model = model.layers[1]

print(
    "\nBase model:",
    base_model.name
)

print(
    "Total layers:",
    len(base_model.layers)
)


# =========================
# 12. CONFIGURE FINE-TUNING
# =========================

base_model.trainable = True


fine_tune_at = len(
    base_model.layers
) - 30


for layer in base_model.layers[:fine_tune_at]:

    layer.trainable = False


print(
    "\nFine-tuning starts from layer:",
    fine_tune_at
)

print(
    "Trainable layers:",
    len(
        base_model.layers[fine_tune_at:]
    )
)


# =========================
# 13. COMPILE FINE-TUNING MODEL
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(
    "\nFine-tuning model compiled successfully."
)


# =========================
# 10. FINE-TUNE MODEL
# =========================

history = model.fit(
    train_tf,
    validation_data=validation_tf,
    epochs=5
)


# =========================
# 11. SAVE FINE-TUNED MODEL
# =========================

model.save(
    "models/fine_tuned_mobilenet.keras"
)

print(
    "\nFine-tuned model saved successfully."
)