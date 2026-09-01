import os

from dotenv import load_dotenv
from google import genai


# =========================
# 1. LOAD API KEY
# =========================

load_dotenv()

api_key = os.getenv(
    "GEMINI_API_KEY"
)

if not api_key:

    raise ValueError(
        "GEMINI_API_KEY not found."
    )


# =========================
# 2. INITIALIZE AI CLIENT
# =========================

client = genai.Client(
    api_key=api_key
)


# =========================
# 3. GENERATE EXPLANATION
# =========================

def generate_explanation(
    prediction,
    confidence,
    top_predictions
):

    top_predictions_text = ""

    for item in top_predictions:

        top_predictions_text += (
            f"- {item['class']}: "
            f"{item['confidence']:.2f}%\n"
        )


    if confidence >= 80:

        confidence_message = (
            "The model has relatively high confidence, "
            "but the result should still be treated as "
            "a visual assessment rather than a guaranteed diagnosis."
        )

    elif confidence >= 50:

        confidence_message = (
            "The model has moderate confidence. "
            "The predicted disease is possible, but another "
            "class may also be relevant."
        )

    else:

        confidence_message = (
            "The model has low confidence. "
            "The prediction is uncertain and should not be "
            "treated as a reliable diagnosis."
        )


    prompt = f"""
You are an AI assistant for a plant disease classification project.

A machine learning model analyzed a plant leaf image.

Predicted class:
{prediction}

Prediction confidence:
{confidence:.2f}%

Top model predictions:
{top_predictions_text}

Important limitation:
{confidence_message}

Provide a concise and practical response using these exact sections:

1. Prediction
2. What it means
3. Common symptoms
4. Basic prevention and management
5. Important limitation

Rules:

- Do not claim the prediction is medically or scientifically certain.
- Clearly state that this is an AI-based visual prediction.
- Do not invent symptoms that are not associated with the predicted disease.
- If the predicted class is healthy, explain that the model found no visual signs of the diseases it was trained to recognize.
- For management advice, give basic safe agricultural guidance.
- Recommend consulting a local agricultural expert when the disease is severe or uncertain.
- Keep the explanation understandable for a general user.
"""


    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )


    return response.text