# 🩺 Breast Cancer Diagnosis Predictor

Predict whether a tumor is **Benign (B)** or **Malignant (M)** using machine learning models with an interactive frontend built in **Gradio**.

---

## 🌟 Features

- **Input Features:** Clinical measurements of breast tumors  
- **Data Preprocessing:**  
  - Skewed features log-transformed  
  - Standardized using `StandardScaler`  
- **Dimensionality Reduction:** `PCA` applied to reduce features while retaining ~99% variance  
- **Models:**  
  - Random Forest Classifier  
  - Logistic Regression with class weighting for imbalanced data  
  - Random Forest with hyperparameter tuning  
- **Interactive UI:** Users can input tumor features and get instant predictions  
- **Prediction Labels:**  
  - `M` → Malignant (Cancer)  
  - `B` → Benign (Non-Cancer)  

---

## 🚀 Demo

Click the link below to try the app online:



## https://huggingface.co/spaces/KishoreR123/Breast-Cancer-Prediction


<img width="1916" height="786" alt="image" src="https://github.com/user-attachments/assets/df076f8f-efcb-4017-bcd2-e73f338d564c" />


---

## 🧰 Technologies Used

- Python 3.12  
- pandas, numpy  
- scikit-learn (PCA, RandomForest, LogisticRegression)  
- Gradio (Frontend)  
- pickle (Model serialization)  

---

## 📊 Model Performance

| Metric          | Random Forest | Logistic Regression | RF with Tuning |
|-----------------|---------------|------------------|----------------|
| Accuracy        | 0.95614       | 0.95614          | 0.964912       |
| Precision (0)   | 0.946667      | 0.985507         | 0.972222       |
| Precision (1)   | 0.974359      | 0.911111         | 0.952381       |
| Recall (0)      | 0.986111      | 0.944444         | 0.972222       |
| Recall (1)      | 0.904762      | 0.97619          | 0.952381       |
| F1-score (0)    | 0.965986      | 0.964539         | 0.972222       |
| F1-score (1)    | 0.938272      | 0.942529         | 0.952381       |
| Confusion Matrix| [[71, 1],[4, 38]] | [[68, 4],[1, 41]] | [[70, 2],[2, 40]] |

---

## ⚙️ Usage

1. Clone the repository:

```
git clone https://github.com/KishoreR2k7/Breast-Cancer-Prediction
cd BreastCancer-Prediction
Install dependencies:


pip install -r requirements.txt
Run the Gradio app:

python app.py
Open the URL displayed in the console to interact with the app.

```

📂 File Structure
```
BreastCancer-Prediction/
│
├── app.py                  # Gradio frontend & prediction code
├── model.pkl               # Trained Logistic Regression model
├── scaler.pkl              # StandardScaler object
├── pca.pkl                 # PCA object
├── model_columns.pkl       # Column order for input features
├── README.md               # Project documentation
└── data/                   # (Optional) Dataset or sample inputs
```
💡 Notes
Ensure input features follow the same scale and order used during model training

Logistic Regression model expects PCA-transformed features

Random Forest model is also saved for benchmarking

🔒 Authentication
Optional: Basic authentication is implemented in the Gradio app (auth=("username","password")) to restrict access.
