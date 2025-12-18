import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = "namadataset_preprosessing/obesity_preprocessing.csv"
TARGET_COLUMN = "ObesityCategory"

def main():
    # Load data
    df = pd.read_csv(DATASET_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="Advance_CI_Run") as run:
        # Training
        model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluation
        preds = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, average="weighted"),
            "recall": recall_score(y_test, preds, average="weighted")
        }
        
        # Log Metrics & Params
        mlflow.log_params({"n_estimators": 150, "max_depth": 10})
        mlflow.log_metrics(metrics)
        
        # Create & Log Artifacts
        plt.figure(figsize=(10,7))
        sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
        # Log and Register Model 
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="ObesityModel"
        )
        
        print(f"Run Finished. Model Registered as 'ObesityModel'. ID: {run.info.run_id}")

if __name__ == "__main__":
    main()

