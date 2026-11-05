# 🫁 Chest X-Ray Pneumonia Classifier

An end-to-end deep learning pipeline built with **TensorFlow 2 / Keras** to detect **Pneumonia vs. Normal** lungs from chest radiographs.

Medical images are notoriously tricky—they have subtle spatial patterns, high resolution, and severe class imbalance. This project builds a custom 4-block deep CNN from scratch, handles data imbalance with custom class weights, and evaluates performance using real-world clinical metrics (focusing on **high sensitivity** so sick patients aren't missed).

---

## ⚡ Highlights

- **Custom 4-Block CNN Architecture:** 4 Convolutional blocks + Batch Normalization + Global Average Pooling + Dropout (~422k parameters, only ~1.6 MB!).
- **High Sensitivity (~91% Recall on Pneumonia):** Prioritizes minimizing false negatives in clinical triage.
- **Fast GPU Data Pipeline:** Built with `tf.data`, in-graph data augmentation, `.cache()`, and `.prefetch(AUTOTUNE)`.
- **Baseline Benchmark:** Proved why classical ML fails on raw 2D spatial features (Logistic Regression got 63.8%, Random Forest got 58.0% vs. **83.5% CNN**).
- **Single-Image Clinical Inference:** A built-in helper function that takes any raw `.jpeg` X-ray, predicts the diagnosis, and plots confidence scores with an adjustable decision threshold.

---

## 📊 Performance & Results

Evaluated on the unseen test split (**624 images** from the Kaggle dataset):

| Metric | Score | Why It Matters |
| :--- | :---: | :--- |
| **Sensitivity / Recall (Pneumonia)** | **91.03%** | **Most critical metric in medicine** — ensures pneumonia cases aren't missed. |
| **ROC-AUC Score** | **0.9029** | Excellent class separability across different operating thresholds. |
| **Precision (PPV)** | **83.92%** | High reliability when flagging positive pneumonia scans. |
| **Overall Accuracy** | **83.49%** | Solid baseline on a notoriously difficult test distribution. |
| **Specificity** | **70.94%** | Reliable identification of healthy/normal scans. |

### 🔍 Confusion Matrix Breakdown (Test Set)
- **True Positives (Pneumonia detected):** 355
- **True Negatives (Normal detected):** 166
- **False Positives:** 68
- **False Negatives (Missed pneumonia):** Only 35

---

## 🧠 Model Architecture

```text
Input (224x224x3)
  │
  ├── Rescaling (1/255)
  ├── Data Augmentation (RandomFlip, RandomRotation, RandomContrast)
  │
  ├── [Conv2D (32, 3x3)  → BatchNorm → ReLU → MaxPool (2x2)]
  ├── [Conv2D (64, 3x3)  → BatchNorm → ReLU → MaxPool (2x2)]
  ├── [Conv2D (128, 3x3) → BatchNorm → ReLU → MaxPool (2x2)]
  ├── [Conv2D (256, 3x3) → BatchNorm → ReLU → MaxPool (2x2)]
  │
  ├── GlobalAveragePooling2D
  ├── Dense (128, ReLU)
  ├── Dropout (0.4)
  └── Dense (1, Linear Logit)
```

> **Design Choice:** Outputs a raw linear logit paired with `BinaryCrossentropy(from_logits=True)` for numerical stability during backpropagation.

---

## 📁 Dataset

This project uses the benchmark **[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)** dataset:
- **Total Images:** 5,856 radiographs (Train: 5,216 | Val: 16 | Test: 624)
- **Class Imbalance in Train Set:** 74.3% Pneumonia vs. 25.7% Normal
- **Fix:** Handled via balanced gradient weighting during loss calculation:
  $$\text{weight}_c = \frac{\text{Total}}{2 \times \text{Count}_c}$$

---

## 🚀 Quick Start

### 1. Clone the Repo
```bash
git clone https://github.com/Chakrios8/chest-xray-classifier.git
cd chest-xray-classifier
```

### 2. Install Dependencies
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow
```

### 3. Run in Google Colab or Local Jupyter
Open `chestxrayclassifier.ipynb` in [Google Colab](https://colab.research.google.com/) or your local notebook environment:
```bash
jupyter notebook chestxrayclassifier.ipynb
```
*(Recommended: Enable GPU acceleration under **Runtime > Change runtime type > T4 GPU**).*

---

## 🔬 Running Inference on a New Scan

You can pass any chest X-ray image into the inference function to inspect confidence:

```python
from pathlib import Path

# Load your best trained model
best_model = tf.keras.models.load_model("best_medical_cnn.keras")

# Run diagnostic prediction
result = predict_single_xray("path/to/sample_xray.jpeg", model=best_model, threshold=0.50)

print(f"Diagnosis: {result['diagnosis']} with {result['confidence']*100:.1f}% confidence")
```

---

## 🛠️ Tech Stack

- **Deep Learning Framework:** TensorFlow 2.x / Keras
- **Evaluation & Metrics:** Scikit-Learn (ROC-AUC, Confusion Matrix, Classification Report)
- **Data & Visualizations:** NumPy, Matplotlib, Seaborn, Pillow
- **Environment:** Python 3.10+, Google Colab (T4 GPU)

---

## ⚖️ Disclaimer

*This project is for educational, research, and portfolio purposes only. It is not intended to be used as a certified medical diagnostic device.*

---

## 🤝 Contributing & Feedback

Have ideas to improve accuracy (e.g., trying a fine-tuned ResNet/EfficientNet or Grad-CAM heatmaps)? Feel free to open an issue or submit a pull request!
