import os
import webbrowser
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, session
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from thyroid_recommendations import get_recommendation
from thyroid_risk_xai import get_risk_xai_details

app = Flask(__name__)
app.secret_key = "thyroid_ai_secret"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CNN_MODEL_PATH = r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\vgg16_balanced.keras"
STAGE_CSV = r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\stage.csv"

cnn_model = load_model(CNN_MODEL_PATH, compile=False)
stage_df = pd.read_csv(STAGE_CSV)

numeric_features = [
    "Family_History","Radiation_Exposure","BMI","Smoking_History",
    "Alcohol_Consumption","TSH_Receptor_Mutation","BRAF_Mutation",
    "RET_PTC_Mutation","Metastasis","Age","TSH_Level","T3_Level","T4_Level"
]

stage_symptoms = {
    "I": ["Small thyroid nodule", "Usually asymptomatic"],
    "II": ["Neck swelling", "Mild voice change"],
    "III": ["Difficulty swallowing", "Persistent hoarseness", "Enlarged lymph nodes"],
    "IV": ["Breathing difficulty", "Bone pain due to metastasis", "Lung involvement", "Severe weight loss and fatigue"]
}

# ------------------ AI FUNCTIONS (UNCHANGED) ------------------

def find_nearest_stage(cnn_pred, input_features):
    subset = stage_df[stage_df["Diagnosis"] == cnn_pred].copy()
    if subset.empty:
        return "N/A", 0.0
    subset_numeric = subset[numeric_features].values
    input_array = np.array([input_features[f] for f in numeric_features])
    dists = np.linalg.norm(subset_numeric - input_array, axis=1)
    idx = np.argmin(dists)
    stage = subset.iloc[idx]["Stage"]
    max_dist = np.max(dists)
    stage_conf = 100 * (1 - dists[idx] / (max_dist + 1e-6))
    return stage, max(0, min(stage_conf, 100))

def predict_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    probs = cnn_model.predict(img_array, verbose=0)[0]
    labels = ["FTC", "PTC", "Invalid"]
    idx = np.argmax(probs)
    cnn_pred = labels[idx]
    cnn_conf = float(probs[idx] * 100)
    if cnn_pred == "Invalid":
        return None, None
    return cnn_pred, cnn_conf

# ------------------ ROUTES ------------------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload", methods=["POST"])
def upload():
    img_file = request.files.get("image")
    if not img_file:
        return render_template("home.html", error="No image uploaded")

    filename = img_file.filename
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    img_file.save(img_path)

    session["image_path"] = img_path
    session["filename"] = filename

    return redirect(url_for("clinical"))

@app.route("/clinical")
def clinical():
    return render_template("clinical.html", features=numeric_features)

@app.route("/predict", methods=["POST"])
def predict():
    img_path = session.get("image_path")
    filename = session.get("filename")

    clinical_data = {f: float(request.form.get(f, 0)) for f in numeric_features}

    cnn_pred, cnn_conf = predict_image(img_path)
    if cnn_pred is None:
        return render_template("home.html", error="Invalid image detected")

    stage, stage_conf = find_nearest_stage(cnn_pred, clinical_data)
    overall_conf = cnn_conf * (stage_conf / 100)

    rec = get_recommendation(cnn_pred, stage)
    risk_info = get_risk_xai_details(cnn_pred, stage, clinical_data)

    next_stage_map = {"I": "II", "II": "III", "III": "IV"}
    next_stage = next_stage_map.get(stage)
    next_stage_symptoms = stage_symptoms.get(next_stage, []) if next_stage else []

    result = {
        "Diagnosis": cnn_pred,
        "CNN_Confidence": round(cnn_conf),
        "Predicted_Stage": stage,
        "Stage_Confidence": round(stage_conf),
        "Overall_Confidence": round(overall_conf),
        "Recommendations": rec["Recommendation"],
        "Risk_Level": risk_info["risk"],
        "Recurrence": risk_info["recurrence"],
        "XAI_Factors": risk_info["xai_factors"],
        "Next_Stage": next_stage,
        "Next_Stage_Symptoms": next_stage_symptoms
    }

    return render_template("result.html", result=result, filename=filename)

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)
