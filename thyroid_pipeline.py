import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

from thyroid_recommendations import get_recommendation
from thyroid_risk_xai import get_risk_xai_details


# ================================
# Paths
# ================================
CNN_MODEL_PATH = r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\vgg16_balanced.keras"
STAGE_CSV = r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\stage.csv"

# ================================
# Load model and CSV
# ================================
cnn_model = load_model(CNN_MODEL_PATH, compile=False)

stage_df = pd.read_csv(STAGE_CSV)

# Features to consider for stage prediction
numeric_features = [
    "Family_History", "Radiation_Exposure", "BMI", "Smoking_History",
    "Alcohol_Consumption", "TSH_Receptor_Mutation", "BRAF_Mutation",
    "RET_PTC_Mutation", "Metastasis", "Age", "TSH_Level", "T3_Level", "T4_Level"
]

# Stage-wise symptoms (reference only)
stage_symptoms = {
    "I": ["Small thyroid nodule", "Usually asymptomatic"],
    "II": ["Neck swelling", "Mild voice change"],
    "III": ["Difficulty swallowing", "Persistent hoarseness", "Enlarged lymph nodes"],
    "IV": ["Breathing difficulty", "Bone pain due to metastasis", "Lung involvement",
           "Severe weight loss and fatigue"]
}

# ================================
# Find nearest stage function
# ================================
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
    stage_conf = max(0, min(stage_conf, 100))
    return stage, stage_conf

# ================================
# Image-based prediction
# ================================
def predict_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # CNN prediction
    probs = cnn_model.predict(img_array, verbose=0)[0]
    labels = ["FTC", "PTC", "Invalid"]
    idx = np.argmax(probs)
    cnn_pred = labels[idx]
    cnn_conf = float(probs[idx] * 100)
    
    print("\n=================================================")
    print("IMAGE-BASED THYROID CANCER PREDICTION")
    print("=================================================")
    print(f"Predicted Class : {cnn_pred}")
    print(f"Confidence : {cnn_conf:.2f}%")
    
    if cnn_pred == "Invalid":
        print("\n⚠ Invalid image detected. Clinical analysis stopped.")
        return None, None
    
    return cnn_pred, cnn_conf

# ================================
# Collect clinical inputs
# ================================
def collect_clinical_inputs():
    print("\n-------------------------------------------------")
    print("CLINICAL INPUT PARAMETERS")
    print("-------------------------------------------------")
    clinical_data = {
        "Family_History": int(input("Family_History (0/1): ")),
        "Radiation_Exposure": int(input("Radiation_Exposure (0/1): ")),
        "BMI": float(input("BMI: ")),
        "Smoking_History": int(input("Smoking_History (0/1): ")),
        "Alcohol_Consumption": int(input("Alcohol_Consumption (0/1): ")),
        "TSH_Receptor_Mutation": int(input("TSH_Receptor_Mutation (0/1): ")),
        "BRAF_Mutation": int(input("BRAF_Mutation (0/1): ")),
        "RET_PTC_Mutation": int(input("RET_PTC_Mutation (0/1): ")),
        "Metastasis": int(input("Metastasis (0/1): ")),
        "Age": int(input("Age: ")),
        "TSH_Level": float(input("TSH_Level: ")),
        "T3_Level": float(input("T3_Level: ")),
        "T4_Level": float(input("T4_Level: "))
    }
    return clinical_data

# ================================
# Main clinical assessment
# ================================
def clinical_assessment(img_path):
    cnn_pred, cnn_conf = predict_image(img_path)
    if cnn_pred is None:
        return
    
    clinical_data = collect_clinical_inputs()
    
    # Stage prediction
    stage, stage_conf = find_nearest_stage(cnn_pred, clinical_data)
    overall_conf = cnn_conf * (stage_conf / 100)
    
    print("\n-------------------------------------------------")
    print("FINAL CLINICAL ASSESSMENT")
    print("-------------------------------------------------")
    print(f"Diagnosis : {cnn_pred}")
    print(f"Predicted Stage : Stage {stage}")
    print(f"Stage Confidence : {stage_conf:.2f}%")
    print(f"Overall Confidence: {overall_conf:.2f}%")
    
    # Treatment recommendations
    rec = get_recommendation(cnn_pred, stage)
    print("\n-------------------------------------------------")
    print("TREATMENT RECOMMENDATIONS")
    print("-------------------------------------------------")
    for i, step in enumerate(rec["Recommendation"], start=1):
        print(f"{i}. {step}")
    print("\nNote: Final treatment decisions must be made by qualified clinicians.")
    
    # Risk & Explainable AI
    risk_info = get_risk_xai_details(cnn_pred, stage, clinical_data)
    print("\n-------------------------------------------------")
    print("RISK ANALYSIS & EXPLAINABLE AI")
    print("-------------------------------------------------")
    print(f"Risk Level : {risk_info['risk']}")
    print(f"Recurrence Probability : {risk_info['recurrence']}")
    print("\nKey Contributing Factors:")
    for factor in risk_info["xai_factors"]:
        print(f"• {factor}")
    
    # Disease progression warning
    next_stage_map = {"I": "II", "II": "III", "III": "IV"}
    print("\n-------------------------------------------------")
    print("DISEASE PROGRESSION WARNING")
    print("-------------------------------------------------")
    if stage in next_stage_map:
        next_stage = next_stage_map[stage]
        print(f"If appropriate treatment is NOT initiated, "
              f"Stage {stage} {cnn_pred} may progress to Stage {next_stage}.")
        print(f"\nExpected symptoms of Stage {next_stage}:")
        for sym in stage_symptoms[next_stage]:
            print(f"• {sym}")
    else:
        print("The disease is already in an advanced stage and may require palliative care.")
    
    print("\n=================================================")
    print("END OF CLINICAL DECISION SUPPORT REPORT")
    print("=================================================")

# ================================
# Run the program
# ================================
if __name__ == "__main__":
    img_path = input("Enter the path of the thyroid image: ").strip()
    if os.path.exists(img_path):
        clinical_assessment(img_path)
    else:
        print("❌ File not found. Please check the path.")
