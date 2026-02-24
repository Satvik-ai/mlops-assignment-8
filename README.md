# Iris Classification ML Model — Data Poisoning Effect

## 📌 Overview
This project explores how **data poisoning** impacts the performance of a machine learning model trained on the classic IRIS dataset.  

Different levels of synthetic noise (5%, 10%, and 50%) are introduced into the dataset to simulate corrupted or malicious data, and the resulting changes in validation performance are tracked using **MLflow**.

The project demonstrates the importance of **data quality, experiment tracking, and reproducibility** in ML pipelines.

---

## 🎯 Assignment Objectives
- Introduce **data poisoning** using random noise  
- Train models on multiple poisoned datasets  
- Track experiments using **MLflow**  
- Compare validation metrics across poisoning levels  
- Understand how corrupted data affects model reliability  

---

## 📂 Project Structure

```
├── data/
│ └── iris.csv
├── artifacts/
│ └── model.pkl
├── src/
│ └── train.py
├── week8_GA_setup.ipynb
└── README.md
```

---

## 📁 Files Description

### 1️⃣ `data` folder
**Purpose**
- Stores the dataset used for training  

**Contents**
- `iris.csv` → Original IRIS dataset  

---

### 2️⃣ `artifacts` folder
**Purpose**
- Stores trained model outputs  

**Contents**
- Serialized Decision Tree model  

---

### 3️⃣ `src/train.py`
**Purpose**
- Script to train the classification model  

**Key Functions**
- Loads dataset from `data/iris.csv`  
- Applies preprocessing (if any)  
- Trains a **Decision Tree classifier**  
- Saves trained model to `artifacts/`  

---

### 4️⃣ `week8_GA_setup.ipynb`
**Purpose**
- Main workflow notebook (Vertex AI Workbench)  

**Key Tasks**
- Initializes Git repository  
- Configures DVC with GCS remote storage  
- Applies data poisoning at:
  - 5% noise  
  - 10% noise  
  - 50% noise  
- Runs experiments and logs metrics in MLflow  
- Pushes code and data to GitHub  

---

## ⚙️ Methodology

1. Load clean IRIS dataset  
2. Inject random noise into numeric features  
3. Train Decision Tree model on each dataset version  
4. Log experiments in MLflow  
5. Compare validation accuracy and metrics  

---

## 📊 Expected Observations

- **5% noise:** Minor performance drop  
- **10% noise:** Noticeable degradation  
- **50% noise:** Significant loss in predictive power  

👉 This shows how sensitive ML models are to corrupted training data.

---

## 🛠️ Tech Stack

- Python  
- Scikit-learn  
- MLflow  
- DVC  
- Google Cloud Storage  
- Jupyter Notebook / Vertex AI  

---

## 🎥 Video Presentation  
[▶️ Click Here](https://drive.google.com/file/d/1tnVSH7NBh1TWODs54WHooz50QUkv2EFN/view?usp=drive_link)

