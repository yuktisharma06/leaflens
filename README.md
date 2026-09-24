# LeafLens 🌿

**AI-Powered Plant Disease Detection**

LeafLens is a deep-learning-based plant disease detection system that classifies plant leaf images into one of **38 plant disease or healthy classes**. It uses **MobileNetV2 transfer learning with fine-tuning** for image classification and **Google Gemini** to generate an understandable explanation of the model's prediction.

> **Note:** LeafLens provides an AI-based visual prediction, not a guaranteed agricultural diagnosis. Predictions should be treated as decision-support information and verified by an agricultural expert when the result is uncertain or the disease is severe.

## ✨ Features

- 38-class plant disease/healthy-leaf classification
- MobileNetV2 transfer learning and fine-tuning
- Top-3 predictions with confidence scores
- AI-generated explanations using Google Gemini
- Practical symptoms and basic management guidance
- Confidence-aware explanations that communicate prediction uncertainty
- Modular training, evaluation, prediction, and explanation pipeline

## 🧠 Model Development

The project was developed iteratively rather than relying on a single model:

1. Explored and inspected the PlantVillage dataset.
2. Built and evaluated a CNN baseline.
3. Evaluated image augmentation and transfer-learning approaches.
4. Used ImageNet-pretrained **MobileNetV2** for transfer learning.
5. Fine-tuned the MobileNetV2 model to improve classification performance.
6. Built the final inference pipeline with class mapping, confidence scores, and Top-3 predictions.

The final model achieved approximately **93% test accuracy** on the PlantVillage test set used for the project.

## 🤖 AI Explanation Layer

After classification, LeafLens sends the model's prediction, confidence, and Top-3 results to Google Gemini.

Gemini generates:

- Prediction summary
- Meaning of the prediction
- Common symptoms
- Basic prevention and management guidance
- An explicit explanation of the model's uncertainty/limitations

Gemini is used for **explanation and guidance**, not for image classification itself.

## 🗂️ Project Structure

```text
leaflens/
├── src/
│   ├── train_baseline.py
│   ├── train_transfer_learning.py
│   ├── fine_tune_mobilenet.py
│   ├── evaluate_baseline.py
│   ├── evaluate_transfer_learning.py
│   ├── evaluate_fine_tuned.py
│   ├── final_evaluation.py
│   ├── predict.py
│   ├── pipeline.py
│   ├── ai_explanation.py
│   └── class_names.py
├── results/
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/yuktisharma06/leaflens.git
cd leaflens
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Do not commit the `.env` file or expose the API key publicly.

### 5. Add the trained model

The trained `.keras` model is excluded from Git because of its size. Place the trained model at:

```text
models/improved_fine_tuned_mobilenet.keras
```

### 6. Run the prediction pipeline

From the project root:

```python
python -c "from src.pipeline import run_pipeline; print(run_pipeline('path/to/leaf_image.jpg'))"
```

## 📊 Dataset

The project uses the **PlantVillage** dataset containing leaf images across 38 plant disease/healthy classes.

The dataset and trained model are not stored directly in this repository because of their size.

## 🔬 Current Status

- [x] Dataset inspection and preprocessing
- [x] CNN baseline
- [x] Transfer learning with MobileNetV2
- [x] MobileNetV2 fine-tuning
- [x] Final evaluation
- [x] Inference pipeline
- [x] Top-3 predictions and confidence scores
- [x] Gemini explanation layer
- [ ] FastAPI REST API
- [ ] Frontend/API integration
- [ ] Deployment

## 🛠️ Tech Stack

**Machine Learning:** Python, TensorFlow, Keras, MobileNetV2, scikit-learn  
**Data & Analysis:** NumPy, Pandas, Matplotlib, Seaborn, Hugging Face Datasets  
**AI:** Google Gemini API  
**Application:** Streamlit (local application)  
**Tools:** Git, GitHub

## ⚠️ Limitations

LeafLens is trained on the PlantVillage dataset and may not generalize perfectly to real-world images with different lighting, backgrounds, plant varieties, or disease presentations. A prediction should therefore not be treated as a definitive diagnosis.

## 👩‍💻 Author

**Yukti Sharma**  
IIIT Lucknow — B.Tech Information Technology

[GitHub](https://github.com/yuktisharma06)
