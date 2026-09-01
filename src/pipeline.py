from predict import predict_image
from ai_explanation import generate_explanation


# =========================
# RUN COMPLETE PIPELINE
# =========================

def run_pipeline(image_path):

    print(
        "\nGenerating disease prediction..."
    )

    prediction_result = predict_image(
        image_path
    )


    print(
        "\n--- PREDICTION RESULT ---"
    )

    print(
        "\nPredicted class:",
        prediction_result["prediction"]
    )

    print(
        "Confidence:",
        round(
            prediction_result["confidence"],
            2
        ),
        "%"
    )


    print(
        "\nTop 3 predictions:"
    )

    for prediction in prediction_result[
        "top_predictions"
    ]:

        print(
            prediction["class"],
            "-",
            round(
                prediction["confidence"],
                2
            ),
            "%"
        )


    print(
        "\nGenerating AI explanation..."
    )

    ai_explanation = generate_explanation(

        prediction=prediction_result[
            "prediction"
        ],

        confidence=prediction_result[
            "confidence"
        ],

        top_predictions=prediction_result[
            "top_predictions"
        ]

    )


    return {

        "prediction": prediction_result[
            "prediction"
        ],

        "confidence": prediction_result[
            "confidence"
        ],

        "top_predictions": prediction_result[
            "top_predictions"
        ],

        "explanation": ai_explanation

    }