# src/train.py
"""
Train a Decision Tree classifier using CSV data, log metrics and model to MLflow.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import argparse
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


def main(version: str,poison_percentage: int = 0):
    # --------------------------
    # Setup MLflow
    # --------------------------
    mlflow.set_tracking_uri("http://127.0.0.1:8100")
    mlflow.set_experiment("Iris_DT_Classification_Poisoning_Exp")

    with mlflow.start_run():
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("iris_data_version", version)
        mlflow.log_param("poison_percentage", poison_percentage)

        # --------------------------
        # Load Local Data
        # --------------------------
        data_path = "data/iris.csv"
        print(f"Loading local data from {data_path}...")
        data = pd.read_csv(data_path)

        # --------------------------
        # Prepare Features and Target
        # --------------------------
        X = data.drop("species", axis=1)
        y = data["species"]

        # --------------------------
        # Train/Test Split
        # --------------------------
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --------------------------
        # Train Model
        # --------------------------
        print("Training Decision Tree model...")
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy_score = metrics.accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", accuracy_score)
        print(f"Accuracy: {accuracy_score:.3f}")
        
        precision = metrics.precision_score(y_test, predictions, average='weighted', zero_division=0)
        mlflow.log_metric("precision", precision)
        print(f"Precision: {precision:.3f}")

        recall = metrics.recall_score(y_test, predictions, average='weighted', zero_division=0)
        mlflow.log_metric("recall", recall)
        print(f"Recall: {recall:.3f}")

        f1_score = metrics.f1_score(y_test, predictions, average='weighted', zero_division=0)
        mlflow.log_metric("f1_score", f1_score)
        print(f"F1 Score: {f1_score:.3f}")

        
        mlflow.set_tag("Training Info","Decision Tree Model for IRIS data")
        mlflow.set_tag("Data Poisoning Method","Added random noise to numeric features")

        # --------------------------
        # Save Model
        # --------------------------
        os.makedirs("artifacts", exist_ok=True)
        model_path = "artifacts/model.joblib"
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        # Log model to MLflow
        signature = infer_signature(X_train, model.predict(X_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="iris_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="IRIS-Classifier-dt",
        )

        print("Training and logging complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Decision Tree on Iris dataset with MLflow logging")
    parser.add_argument("--version", type=str, required=True, help="Feature view version, e.g., v1")
    parser.add_argument("--poison_percentage", type=int, default=0)
    args = parser.parse_args()

    main(args.version, args.poison_percentage)