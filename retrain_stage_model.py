import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load your stage features CSV
features_df = pd.read_csv(r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\stage_features.csv")

# Features and labels
X = features_df[['Mean_Intensity','Contrast','Correlation','Energy','Homogeneity','Entropy']]
y = features_df['Stage']  # Stage labels: I, II, III, IV

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest
rf_stage = RandomForestClassifier(n_estimators=100, random_state=42)
rf_stage.fit(X_scaled, y)

# Save the trained model and scaler
joblib.dump(rf_stage, r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\rf_stage_model.pkl")
joblib.dump(scaler, r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\rf_stage_model_scaler.pkl")

print("✅ Random Forest stage model and scaler saved successfully.")
