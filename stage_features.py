import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load your stage features CSV
features_df = pd.read_csv(r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\stage_features.csv")

# Select feature columns (adjust if Run_Length is missing)
feature_columns = ['Mean_Intensity','Contrast','Correlation','Energy','Homogeneity','Entropy']  
X = features_df[feature_columns]

# Fit StandardScaler
scaler = StandardScaler()
scaler.fit(X)

# Save the scaler
joblib.dump(scaler, r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\stage_scaler.pkl")
print("✅ Scaler created and saved successfully!")
