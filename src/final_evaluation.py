from datasets import load_dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)


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
# 2. SELECT TEST DATA
# =========================

test_data = dataset.filter(
    lambda example: example["split"] == "test"
)

print("\nTest dataset size:")
print(len(test_data))


# =========================
# 3. CREATE CONSISTENT LABEL MAPPING
# =========================

train_data = dataset.filter(
    lambda example: example["split"] == "train"
)

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

print("\nNumber of classes:")
print(len(class_names))


# =========================
# 4. CREATE HUMAN-READABLE
#    CLASS NAME MAPPING
# =========================

encoded_class_names = [
    None
] * len(class_names)


for example in train_data:

    original_label = example[
        "class_idx"
    ]

    encoded_label = label_mapping[
        original_label
    ]

    class_name = example[
        "class_label"
    ]

    if encoded_class_names[
        encoded_label
    ] is None:

        encoded_class_names[
            encoded_label
        ] = class_name


print(
    "\nEncoded class names created successfully."
)

print(
    "\nFirst 10 encoded class names:"
)

for i in range(10):

    print(
        i,
        "->",
        encoded_class_names[i]
    )


# =========================
# 5. PREPROCESS IMAGE
# =========================

def preprocess_image(image):

    image = np.array(
        image
    )

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

    return image


# =========================
# 6. LOAD FINAL MODEL
# =========================

model = tf.keras.models.load_model(
    "models/fine_tuned_mobilenet.keras"
)

print(
    "\nFine-tuned model loaded successfully."
)


# =========================
# 7. GENERATE PREDICTIONS
# =========================

BATCH_SIZE = 32

print(
    "\nPreparing test dataset..."
)


def test_generator():

    for example in test_data:

        image = np.array(
            example["image"]
        )

        original_label = example[
            "class_idx"
        ]

        actual_label = label_mapping[
            original_label
        ]

        yield image, actual_label


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
    test_generator,
    output_signature=output_signature
)


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

test_tf = test_tf.batch(
    BATCH_SIZE
)

test_tf = test_tf.prefetch(
    tf.data.AUTOTUNE
)


print(
    "\nGenerating predictions..."
)

y_true = []
y_pred = []
prediction_confidences = []


for image_batch, label_batch in test_tf:

    batch_predictions = model.predict(
        image_batch,
        verbose=0
    )

    batch_predicted_labels = np.argmax(
        batch_predictions,
        axis=1
    )

    batch_confidences = np.max(
        batch_predictions,
        axis=1
    )

    y_true.extend(
        label_batch.numpy()
    )

    y_pred.extend(
        batch_predicted_labels
    )

    prediction_confidences.extend(
        batch_confidences
    )


y_true = np.array(
    y_true
)

y_pred = np.array(
    y_pred
)

prediction_confidences = np.array(
    prediction_confidences
)


print(
    "\nPredictions generated successfully."
)

print(
    "Number of actual labels:",
    len(y_true)
)

print(
    "Number of predicted labels:",
    len(y_pred)
)


# =========================
# 8. CALCULATE EVALUATION METRICS
# =========================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision, recall, f1_score, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )
)


print("\n--- OVERALL EVALUATION RESULTS ---")

print(
    "Accuracy:",
    accuracy
)

print(
    "Accuracy (%):",
    accuracy * 100
)

print(
    "Weighted Precision:",
    precision
)

print(
    "Weighted Recall:",
    recall
)

print(
    "Weighted F1-score:",
    f1_score
)


# =========================
# 8A. SAVE OVERALL METRICS
# =========================

metrics_file = open(
    "results/overall_metrics.txt",
    "w"
)

metrics_file.write(
    "--- OVERALL EVALUATION RESULTS ---\n\n"
)

metrics_file.write(
    f"Accuracy: {accuracy}\n"
)

metrics_file.write(
    f"Accuracy (%): {accuracy * 100:.2f}\n"
)

metrics_file.write(
    f"Weighted Precision: {precision}\n"
)

metrics_file.write(
    f"Weighted Recall: {recall}\n"
)

metrics_file.write(
    f"Weighted F1-score: {f1_score}\n"
)

metrics_file.close()

print(
    "\nOverall metrics saved successfully."
)


# =========================
# 9. CLASSIFICATION REPORT
# =========================

print(
    "\n--- CLASSIFICATION REPORT ---"
)

report = classification_report(
    y_true,
    y_pred,
    labels=list(
        range(
            len(class_names)
        )
    ),
    target_names=encoded_class_names,
    zero_division=0
)

print(report)


# =========================
# 9A. SAVE CLASSIFICATION
#     REPORT
# =========================

report_file = open(
    "results/classification_report.txt",
    "w"
)

report_file.write(
    "--- CLASSIFICATION REPORT ---\n\n"
)

report_file.write(
    report
)

report_file.close()

print(
    "\nClassification report saved successfully."
)


# =========================
# 10. CONFUSION MATRIX
# =========================

conf_matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=list(
        range(
            len(class_names)
        )
    )
)

print(
    "\n--- CONFUSION MATRIX ---"
)

print(conf_matrix)


# =========================
# 10A. SAVE CONFUSION
#      MATRIX VISUALIZATION
# =========================

plt.figure(
    figsize=(20, 16)
)

sns.heatmap(
    conf_matrix,
    cmap="Blues",
    xticklabels=encoded_class_names,
    yticklabels=encoded_class_names
)

plt.title(
    "Fine-Tuned MobileNetV2 Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    rotation=90,
    fontsize=7
)

plt.yticks(
    rotation=0,
    fontsize=7
)

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.close()

print(
    "\nConfusion matrix visualization saved successfully."
)


# =========================
# 11. MOST COMMON CLASS CONFUSIONS
# =========================

confusions = []

for actual_class in range(
    len(class_names)
):

    for predicted_class in range(
        len(class_names)
    ):

        if actual_class != predicted_class:

            count = conf_matrix[
                actual_class,
                predicted_class
            ]

            if count > 0:

                confusions.append(
                    (
                        count,
                        actual_class,
                        predicted_class
                    )
                )


confusions.sort(
    reverse=True
)


print(
    "\n--- TOP 10 CLASS CONFUSIONS ---"
)

for count, actual_class, predicted_class in confusions[:10]:

    print(
        f"Actual: {encoded_class_names[actual_class]} "
        f"Predicted: {encoded_class_names[predicted_class]} "
        f"Count: {count}"
    )


# =========================
# 12. INSPECT INCORRECT
#    PREDICTIONS
# =========================

print(
    "\n--- SAMPLE INCORRECT PREDICTIONS ---"
)

incorrect_count = 0

for index in range(
    len(test_data)
):

    if y_true[index] != y_pred[index]:

        confidence = prediction_confidences[
            index
        ]

        print(
            f"\nImage index: {index}"
        )

        print(
            "Actual:",
            encoded_class_names[
                y_true[index]
            ]
        )

        print(
            "Predicted:",
            encoded_class_names[
                y_pred[index]
            ]
        )

        print(
            "Confidence:",
            confidence
        )

        incorrect_count += 1

        if incorrect_count == 10:

            break


# =========================
# 12A. SAVE SAMPLE
#      INCORRECT PREDICTIONS
# =========================

import os


os.makedirs(
    "results/incorrect_predictions",
    exist_ok=True
)


print(
    "\nSaving sample incorrect predictions..."
)


saved_count = 0


for index in range(
    len(test_data)
):

    if y_true[index] != y_pred[index]:

        image = test_data[index][
            "image"
        ]


        actual_name = encoded_class_names[
            y_true[index]
        ]

        predicted_name = encoded_class_names[
            y_pred[index]
        ]


        confidence = prediction_confidences[
            index
        ]


        plt.figure(
            figsize=(8, 6)
        )


        plt.imshow(
            image
        )


        plt.title(
            f"Actual: {actual_name}\n"
            f"Predicted: {predicted_name}\n"
            f"Confidence: {confidence:.2%}"
        )


        plt.axis(
            "off"
        )


        filename = (
            f"incorrect_{saved_count + 1}.png"
        )


        plt.savefig(
            f"results/incorrect_predictions/{filename}",
            bbox_inches="tight"
        )


        plt.close()


        saved_count += 1


        if saved_count == 20:

            break


print(
    "Sample incorrect predictions saved successfully."
)


# =========================
# 13. MODEL COMPARISON
# =========================

print(
    "\n--- MODEL COMPARISON ---"
)

model_results = {
    "Baseline CNN": 45.34,
    "Lightly Augmented CNN": 46.95,
    "Transfer Learning MobileNetV2": 90.24,
    "Fine-Tuned MobileNetV2": 92.53
}

for model_name, accuracy_value in model_results.items():

    print(
        f"{model_name}: "
        f"{accuracy_value:.2f}%"
    )


print(
    "\nBest model: Fine-Tuned MobileNetV2"
)

print(
    "Final test accuracy: "
    f"{accuracy * 100:.2f}%"
)   


# =========================
# 13A. SAVE MODEL
#      COMPARISON PLOT
# =========================

model_names = list(
    model_results.keys()
)

accuracy_values = list(
    model_results.values()
)


plt.figure(
    figsize=(10, 6)
)

plt.bar(
    model_names,
    accuracy_values
)

plt.title(
    "Model Performance Comparison"
)

plt.xlabel(
    "Models"
)

plt.ylabel(
    "Test Accuracy (%)"
)

plt.xticks(
    rotation=15
)

plt.ylim(
    0,
    100
)


for index, value in enumerate(
    accuracy_values
):

    plt.text(
        index,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )


plt.tight_layout()

plt.savefig(
    "results/model_comparison.png",
    dpi=300
)

plt.close()

print(
    "\nModel comparison plot saved successfully."
)