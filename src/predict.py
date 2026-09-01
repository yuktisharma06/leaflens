from PIL import Image
import numpy as np
import tensorflow as tf

from class_names import CLASS_NAMES


# =========================
# 1. LOAD TRAINED MODEL
# =========================

MODEL_PATH = "models/improved_fine_tuned_mobilenet.keras"

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# =========================
# 2. PREPROCESS IMAGE
# =========================

def preprocess_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image
    ).astype(
        np.float32
    )

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =========================
# 3. PREDICT IMAGE
# =========================

def predict_image(image_path):

    processed_image = preprocess_image(
        image_path
    )

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = np.argmax(
        probabilities
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = probabilities[
        predicted_index
    ] * 100

    top_indices = np.argsort(
        probabilities
    )[-3:][::-1]

    top_predictions = []

    for index in top_indices:

        top_predictions.append({

            "class": CLASS_NAMES[index],

            "confidence": float(
                probabilities[index] * 100
            )

        })

    return {

        "prediction": predicted_class,

        "confidence": float(
            confidence
        ),

        "top_predictions": top_predictions

    }


# =========================
# 4. OPTIONAL DIRECT TEST
# =========================

if __name__ == "__main__":

    IMAGE_PATH = "test_image.png"

    result = predict_image(
        IMAGE_PATH
    )

    print("\n--- PREDICTION RESULT ---")

    print(
        "\nPredicted class:",
        result["prediction"]
    )

    print(
        "Confidence:",
        round(
            result["confidence"],
            2
        ),
        "%"
    )

    print("\nTop 3 predictions:")

    for prediction in result["top_predictions"]:

        print(
            prediction["class"],
            "-",
            round(
                prediction["confidence"],
                2
            ),
            "%"
        )