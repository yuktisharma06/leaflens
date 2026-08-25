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
# 2. SEPARATE TRAIN AND TEST DATA
# =========================

train_data = dataset.filter(
    lambda example: example["split"] == "train"
)

test_data = dataset.filter(
    lambda example: example["split"] == "test"
)


print("\nTest dataset size:")
print(len(test_data))


# =========================
# 3. CREATE THE SAME LABEL MAPPING
#    USED DURING TRAINING
# =========================

train_data = train_data.class_encode_column(
    "class_idx"
)

class_order = train_data.features[
    "class_idx"
].names

label_mapping = {
    int(original_label): encoded_label
    for encoded_label, original_label
    in enumerate(class_order)
}


print("\nLabel mapping created successfully.")


# =========================
# 4. CREATE TEST DATASET
# =========================

def dataset_generator(data):

    for example in data:

        image = np.array(
            example["image"]
        )

        original_label = example["class_idx"]

        label = label_mapping[
            original_label
        ]

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


test_tf = tf.data.Dataset.from_generator(
    lambda: dataset_generator(test_data),
    output_signature=output_signature
)


# =========================
# 5. PREPROCESS DATA
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

    image = image / 255.0

    return image, label


test_tf = test_tf.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)


# =========================
# 6. BATCH DATA
# =========================

BATCH_SIZE = 32

test_tf = test_tf.batch(
    BATCH_SIZE
)

test_tf = test_tf.prefetch(
    tf.data.AUTOTUNE
)


print("\nTest dataset prepared successfully.")


# =========================
# 7. LOAD SAVED MODEL
# =========================

model = tf.keras.models.load_model(
    "models/augmented_light.keras"
)

print("\nLightly augmented model loaded successfully.")


# =========================
# 8. EVALUATE MODEL
# =========================

test_loss, test_accuracy = model.evaluate(
    test_tf
)


print("\nAugmented Test Results:")

print("Test Loss:", test_loss)

print("Test Accuracy:", test_accuracy)

print(
    "Test Accuracy (%):",
    test_accuracy * 100
)