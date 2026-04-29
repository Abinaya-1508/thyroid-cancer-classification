import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# Paths
# ================================
STAGE_MODEL_PATH = r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\xgb_stage_model.pkl"
STAGE_LABEL_ENCODER_PATH = r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\stage_label_encoder.pkl"
STAGE_SCALER_PATH = r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\stage_scaler.pkl"

# ================================
# Load Dataset
# ================================
stage_csv_path = r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\stage.csv"
stage_df = pd.read_csv(stage_csv_path)

# ================================
# Define Features
# ================================
feature_cols = [
    'Mean_Intensity','Contrast','Correlation','Energy','Homogeneity','Run_Length','Entropy',
    'Family_History','Radiation_Exposure','BMI','Smoking_History','Alcohol_Consumption',
    'TSH_Receptor_Mutation','BRAF_Mutation','RET_PTC_Mutation','Metastasis',
    'Age','TSH_Level','T3_Level','T4_Level'
]

X = stage_df[feature_cols]
y = stage_df['Stage']

# ================================
# Encode Labels
# ================================
le_stage = LabelEncoder()
y_encoded = le_stage.fit_transform(y)
joblib.dump(le_stage, STAGE_LABEL_ENCODER_PATH)
print("Stage Label Mapping:", dict(zip(le_stage.classes_, le_stage.transform(le_stage.classes_))))

# ================================
# Scale Features
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, STAGE_SCALER_PATH)

# ================================
# Balance Classes (SMOTE)
# ================================
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y_encoded)
print("Balanced classes:", dict(pd.Series(y_res).value_counts()))

# ================================
# Train/Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# ================================
# Train XGBoost Stage Model
# ================================
xgb_stage_model = XGBClassifier(
    n_estimators=700,
    learning_rate=0.03,
    max_depth=10,
    subsample=0.9,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42
)
xgb_stage_model.fit(X_train, y_train)

# ================================
# Evaluate
# ================================
y_pred = xgb_stage_model.predict(X_test)
print("Training Accuracy:", accuracy_score(y_train, xgb_stage_model.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ================================
# Save Model
# ================================
joblib.dump(xgb_stage_model, STAGE_MODEL_PATH)
print("✅ Stage model trained & saved")
