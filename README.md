# Iris Classification ML Model Data Poisoning Effect

**MLOps - Week 8 - Assignment - 21f1000344**

## Assignment Objective : 
- Integrate data poisoning for IRIS using randomly generated numbers at various levels(5%,10%,50%) and explain the validation outcomes when trained on such data using MLFlow


---

## Files

### 1. `data` folder
- **Key Utilities:**
  - Stores `iris.csv` data

### 2. `artifacts` folder
- **Key Utilities:**
  - Stores the trained iris classification model

### 3. `src/train.py`
- **Key Utilities:**
  - Loads the `iris.csv` 
  - Trains a `Decision Tree` model

### 4. `week8_GA_setup.ipynb`
- **Key Utilities:**
  - Created in Vertex AI workbench
  - Serves as an interface for performing actions local working directory
  - Setup Git Repository with `main` branch
  - Setup DVC with GCS bucket as remote storage
  - Data poisoning at levels(5%,10%,50%) by adding random noise to numeric features
  - Pushed the local working directory to remote repo on GitHub

---