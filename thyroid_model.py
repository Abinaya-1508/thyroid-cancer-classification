# Thyroid Carcinoma Classification
import os
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # allow partially corrupted images

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color
from scipy.stats import entropy

# ================================
# Paths (Local Windows)
# ================================
BASE_DIR = r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\thyroid_data"  # Image folders
STAGE_CSV_PATH = r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\stage.csv"

VGG16_MODEL_PATH = r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\vgg16_balanced.keras"
XGB_STAGE_MODEL_PATH = r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\xgb_stage_model.pkl"
STAGE_SCALER_PATH = r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\stage_scaler.pkl"
STAGE_LABEL_ENCODER_PATH = r"C:\Users\ABINAYA\Desktop\project\thyroid_dataset\stage_label_encoder.pkl"

# ================================
# Function to clean corrupted images
# ================================
def remove_corrupted_images(folder):
    """
    Walk through all image folders and remove images that PIL cannot open
    """
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            filepath = os.path.join(subdir, file)
            try:
                img = Image.open(filepath)
                img.verify()  # verify if image can be opened
            except Exception:
                print(f"⚠ Removing corrupted image: {filepath}")
                os.remove(filepath)

# Clean dataset before training
remove_corrupted_images(BASE_DIR)

# ================================
# Load stage CSV
# ================================
stage_df = pd.read_csv(STAGE_CSV_PATH)
print(stage_df.head())

# ================================
# CNN Training
# ================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training generator
train_generator = datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Validation generator
val_generator = datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Compute class weights for imbalance
counter = Counter(train_generator.classes)
max_count = float(max(counter.values()))
class_weights = {cls: max_count/count for cls, count in counter.items()}
print("Class Weights:", class_weights)

# Build VGG16 model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
x = Dropout(0.5)(x)
output = Dense(len(train_generator.class_indices), activation="softmax")(x)

cnn_model = Model(inputs=base_model.input, outputs=output)
cnn_model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

checkpoint = ModelCheckpoint(VGG16_MODEL_PATH, save_best_only=True)

# ================================
# Fit the model
# ================================
history = cnn_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    class_weight=class_weights,
    callbacks=[checkpoint]
)
print("✅ CNN Trained & Saved")
