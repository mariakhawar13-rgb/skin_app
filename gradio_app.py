import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import json

# ===========================
# Load Model
# ===========================
def load_model():
    try:
        model = tf.keras.models.load_model("skin_cancer_model.keras")
        return model, "Model loaded (.keras format)"
    except Exception as e:
        try:
            with open("model_architecture.json", "r") as f:
                model_json = f.read()
            model = tf.keras.models.model_from_json(model_json)
            model.load_weights("model_weights.weights.h5")
            return model, "Model loaded (architecture + weights)"
        except:
            return None, "Model failed to load — demo mode"

model, model_status = load_model()

# ===========================
# Load Class Names
# ===========================
def load_classes():
    try:
        with open("class_names.json") as f:
            return json.load(f)
    except:
        return [
            'basal cell carcinoma', 'seborrheic keratosis', 'dermatofibroma',
            'melanoma', 'nevus', 'vascular lesion',
            'pigmented benign keratosis', 'actinic keratosis',
            'squamous cell carcinoma'
        ]

classes = load_classes()

# ===========================
# Preprocessing (same logic)
# ===========================
def preprocess_image(image):
    img = np.array(image)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ===========================
# Prediction Function
# ===========================
def predict(image):
    if model is None:
        return {
            "status": model_status,
            "message": "Model not loaded. Demo mode.",
            "classes": classes
        }

    preprocessed = preprocess_image(image)
    preds = model.predict(preprocessed, verbose=0)[0]

    top_idx = int(np.argmax(preds))
    top_class = classes[top_idx]
    confidence = float(preds[top_idx])

    # Risk logic
    high_risk_keywords = ["melanoma", "carcinoma", "squamous"]
    is_high_risk = any(k in top_class.lower() for k in high_risk_keywords)

    if is_high_risk and confidence > 0.7:
        risk_status = "🔴 HIGH RISK — Consult a dermatologist immediately."
    elif is_high_risk:
        risk_status = "🟡 Possible concern — Consider a medical check."
    else:
        risk_status = "🟢 Likely benign — Monitor and consult if concerned."

    # Build output
    return {
        "Predicted Class": top_class,
        "Confidence": round(confidence, 3),
        "Risk Assessment": risk_status,
        "All Predictions": {
            classes[i]: float(preds[i]) for i in range(len(classes))
        },
        "Model Info": model_status
    }

# ===========================
# Gradio UI
# ===========================
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.JSON(),
    title="Skin Cancer Classification (Gradio)",
    description="""
Upload a clear image of a skin lesion.

⚠️ **Disclaimer**  
This tool is for **educational use only** and NOT for medical diagnosis.
""",
)

if __name__ == "__main__":
    interface.launch()
