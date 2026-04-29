import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from skimage import io, color
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy

# ---------- Stage CSV ----------
STAGE_CSV = r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\stage.csv"
stage_df = pd.read_csv(STAGE_CSV)

print("\n=== Stage Distribution in stage.csv ===")
print(stage_df["Stage"].value_counts(), "\n")

# ---------- Feature extraction ----------
def compute_image_features(img_path):
    img = io.imread(img_path)
    gray = color.rgb2gray(img)
    gray_u8 = (gray * 255).astype(np.uint8)

    glcm = graycomatrix(gray_u8, distances=[1], angles=[0], symmetric=True, normed=True)

    feats = {
        "Mean_Intensity": float(np.mean(gray_u8)),
        "Contrast": float(graycoprops(glcm, 'contrast')[0,0]),
        "Correlation": float(graycoprops(glcm, 'correlation')[0,0]),
        "Energy": float(graycoprops(glcm, 'energy')[0,0]),
        "Homogeneity": float(graycoprops(glcm, 'homogeneity')[0,0]),
        "Entropy": float(entropy(np.histogram(gray_u8.ravel(), bins=256, range=(0,256), density=True)[0] + 1e-10)),
        "Run_Length": float(np.mean([len(run) for run in np.split(gray_u8.ravel(), np.where(gray_u8.ravel()==0)[0]+1)]))
    }
    return feats

def knn_neighbors(stage_df, features, k=5):
    # Only use numeric columns that overlap
    common_cols = [c for c in features.keys() if c in stage_df.columns]
    df_num = stage_df[common_cols].astype(float).copy()
    query = np.array([float(features[c]) for c in common_cols]).reshape(1,-1)

    sc = StandardScaler()
    Xs = sc.fit_transform(df_num.values)
    qx = sc.transform(query)

    dists = np.linalg.norm(Xs - qx, axis=1)
    idx = np.argsort(dists)[:k]
    neighbors = stage_df.iloc[idx].copy()
    neighbors["__dist"] = dists[idx]
    return neighbors.sort_values("__dist")

# ---------- Test images ----------
TEST_IMAGES = [
    r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\test_images\f.jpg",
    r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\test_images\folli.jpg",
    r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\test_images\pap.jpg",
    r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\test_images\papil.jpg"
]

for img_path in TEST_IMAGES:
    if not os.path.exists(img_path):
        print(f"⚠️ Image not found: {img_path}")
        continue
    feats = compute_image_features(img_path)
    neighbors = knn_neighbors(stage_df, feats, k=5)

    print(f"\n=== Nearest Neighbors for {os.path.basename(img_path)} ===")
    print(neighbors[["Stage", "__dist"] + [c for c in neighbors.columns if c not in ["Stage", "__dist"]]].head(5))
