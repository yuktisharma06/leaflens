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
# 2. SELECT TRAIN AND TEST DATA
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
# 3. CREATE CONSISTENT LABEL MAPPING
# =========================

train_data = train_data.class_encode_column(
    "class_idx"
)

class_names = train_data.features[
    "class_idx"
].names

label_mapping = {
    int(class_name): index
    for index, class_name in enumerate(class_names)
}

print("\nLabel mapping created successfully.")


# =========================
# 4. DATA GENERATOR
# =========================

def dataset_generator(data):

    for example in data:

        image = np.array(
            example["image"]
        )

        original_label = example["class_idx"]

        label = label_mapping[original_label]

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
# 5. CREATE TEST DATASET
# =========================

test_tf = tf.data.Dataset.from_generator(
    lambda: dataset_generator(test_data),
    output_signature=output_signature
)


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


test_tf = test_tf.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)


# =========================
# 7. BATCH AND PREFETCH
# =========================

BATCH_SIZE = 16


test_tf = test_tf.take(
    (len(test_data) + BATCH_SIZE - 1) // BATCH_SIZE
)


test_tf = test_tf.batch(
    BATCH_SIZE
)

test_tf = test_tf.prefetch(
    tf.data.AUTOTUNE
)

print("\nTest dataset prepared successfully.")




# =========================
# 8. LOAD MODEL
# =========================

model = tf.keras.models.load_model(
    "models/transfer_learning_mobilenet.keras"
)

print(
    "\nTransfer learning model loaded successfully."
)



# =========================
# 9. CHECK SAMPLE PREDICTIONS
# =========================

print("\n--- PREDICTION CHECK ---")

sample_indices = [0, 500, 1000, 2000, 4000, 6000, 8000, 10000]

for index in sample_indices:

    example = test_data[index]

    image = np.array(
        example["image"]
    )

    original_label = example["class_idx"]

    label = label_mapping[
        original_label
    ]

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

    image = tf.expand_dims(
        image,
        axis=0
    )

    prediction = model.predict(
        image,
        verbose=0
    )

    predicted_label = tf.argmax(
        prediction[0]
    ).numpy()

    confidence = tf.reduce_max(
        prediction[0]
    ).numpy()

    print(
        f"\nIndex: {index}"
    )

    print(
        "Original label:",
        original_label
    )

    print(
        "Encoded actual label:",
        label
    )

    print(
        "Predicted label:",
        predicted_label
    )

    print(
        "Confidence:",
        confidence
    )


# =========================
# 10. EVALUATE MODEL
# =========================

test_loss, test_accuracy = model.evaluate(
    test_tf
)


# =========================
# 11. PRINT RESULTS
# =========================

print("\nTransfer Learning Test Results:")

print(
    "Test Loss:",
    test_loss
)

print(
    "Test Accuracy:",
    test_accuracy
)

print(
    "Test Accuracy (%):",
    test_accuracy * 100
)