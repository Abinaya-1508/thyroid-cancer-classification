🧠 Thyroid Cancer Classification using Deep Learning

📌 Overview

This project focuses on the automated classification of thyroid cancer using deep learning techniques. It analyzes histopathology images to distinguish between different types of thyroid carcinoma, improving diagnostic support for medical professionals.

The model is trained to classify images into:

* PTC (Papillary Thyroid Carcinoma)
* FTC (Follicular Thyroid Carcinoma)


🎯 Problem Statement

Traditional diagnosis of thyroid cancer relies heavily on manual examination by pathologists, which can be time-consuming and subjective. While many AI models focus only on classification, they often lack robustness in handling irrelevant or noisy inputs.

This project aims to:

* Improve classification accuracy
* Handle invalid/non-medical images
* Provide reliable predictions using deep learning


🧪 Methodology

🔹 Data Collection

* Histopathology image dataset (FTC, PTC)
* Preprocessed and organized into training and validation sets

🔹 Data Preprocessing

* Image resizing and normalization
* Data augmentation (rotation, flipping, zooming)
* Handling corrupted images

🔹 Model Architecture

* Pre-trained VGG16 Convolutional Neural Network
* Transfer learning applied
* Custom layers added:

  * Flatten
  * Dense layers
  * Dropout for regularization

🔹 Training

* Optimizer: Adam
* Loss Function: Categorical Crossentropy
* Metrics: Accuracy


📊 Results

* Achieved high classification accuracy on validation data
* Model effectively distinguishes between FTC, PTC, and Invalid images

📈 Performance Evaluation

* Accuracy and Loss Graphs
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)


🚀 Features

* ✅ Deep learning-based image classification
* ✅ Handles invalid/unrelated images
* ✅ Visual performance metrics
* ✅ Predicts class for new input images
* ✅ Easy to run on Google Colab


🛠️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy, Pandas
* Matplotlib, Seaborn


📂 Project Structure

thyroid-cancer-classification/
│
├── dataset/
│   ├── FTC/
│   ├── PTC/
│   └── Invalid/
│
├── model/
│   └── trained_model.h5
│
├── notebooks/
│   └── training.ipynb
│
├── results/
│   ├── accuracy_plot.png
│   ├── confusion_matrix.png
│
├── README.md
└── requirements.txt


▶️ How to Run the Project

🔹 Step 1: Clone Repository

git clone https://github.com/Abinaya-1508/thyroid-cancer-classification.git


🔹 Step 2: Install Dependencies

pip install -r requirements.txt

🔹 Step 3: Run the Model

* Open Jupyter Notebook / Google Colab
* Load dataset from your local system or Google Drive
* Run all cells

🔹 Step 4: Test Prediction

* Upload a histopathology image
* Model will classify it as FTC / PTC 


📁 Dataset

Dataset used in this project is organized into three categories:

* FTC
* PTC

> Note: Due to size limitations, full dataset is not uploaded. You can use your own dataset or sample images.


🔍 Future Improvements

* Add cancer staging prediction (Stage I–IV)
* Improve accuracy with advanced architectures (ResNet, EfficientNet)
* Deploy as a web application
* Integrate clinical data with image analysis


👤 Author

Abinaya S
B.Tech Information Technology
Passionate about AI, Machine Learning & Healthcare Applications


📬 Contact

For queries or collaboration:

* Email: sabinaya045@gmail.com
* GitHub: https://github.com/Abinaya-1508


⭐ Acknowledgment

This project is developed as part of an academic final year project focused on applying AI in medical diagnosis.


⚠️ Disclaimer

This project is for educational and research purposes only. It is not intended for clinical use.
