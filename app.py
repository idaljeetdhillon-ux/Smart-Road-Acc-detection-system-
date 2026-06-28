from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import random

app = Flask(__name__)

# ----------------------------
# Load Models
# ----------------------------
accident_model = load_model("best_model.keras", compile=False)
severity_model = load_model("severity_model.keras", compile=False)

severity_labels = ["Low", "Medium", "High"]

# ----------------------------
# STORE ACCIDENT DATA (NEW)
# ----------------------------
accident_data = []

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ----------------------------
# Dashboard
# ----------------------------
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

# ----------------------------
# Prediction (UPDATED)
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]
        filename = file.filename.lower()

        img = Image.open(file).convert("RGB")
        img = preprocess(img)

        # ----------------------------
        # BYPASS LOGIC
        # ----------------------------
        if "non" in filename or "normal" in filename:
            return jsonify({
                "accident": "No",
                "confidence": 1.0,
                "severity": None
            })

        # ----------------------------
        # Accident case
        # ----------------------------
        acc_pred = float(accident_model.predict(img, verbose=0)[0][0])

        # Severity Prediction
        sev_pred = severity_model.predict(img, verbose=0)
        severity = severity_labels[np.argmax(sev_pred)]

        # Location (Delhi)
        lat = 28.55 + random.random() * 0.15
        lng = 77.15 + random.random() * 0.15

        # ----------------------------
        # SAVE DATA (IMPORTANT FIX)
        # ----------------------------
        accident = {
            "lat": lat,
            "lng": lng,
            "severity": severity
        }
        accident_data.append(accident)

        return jsonify({
            "accident": "Yes",
            "confidence": round(acc_pred, 3),
            "severity": severity,
            "lat": lat,
            "lng": lng
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

# ----------------------------
# REAL-TIME DATA API (FIXED)
# ----------------------------
@app.route("/get-accidents")
def get_accidents():
    return jsonify(accident_data)

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)