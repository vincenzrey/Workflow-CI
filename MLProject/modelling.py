import os
import shutil
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub # Tambahkan ini

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

REPO_OWNER = "vincenzrey"
REPO_NAME = "Eksperimen_SMSML_Vincenz-Reynard-Citro"

dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

DATASET_PATH = "namadataset_preprosessing/obesity_preprocessing.csv"
TARGET_COLUMN = "ObesityCategory"

def read_data(path):
    return pd.read_csv(path)

def split_dataset(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted")
    rec = recall_score(y_test, preds, average="weighted")
    
    plt.figure(figsize=(10,7))
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d')
    plt.savefig("confusion_matrix.png")

    test_results = pd.DataFrame({'actual': y_test, 'predicted': preds})
    test_results.to_csv("test_results.csv", index=False)
    
    return {"accuracy": acc, "precision": prec, "recall": rec}

def main():
    df = read_data(DATASET_PATH)
    X_train, X_test, y_train, y_test = split_dataset(df, TARGET_COLUMN)

    with mlflow.start_run(run_name="RandomForest_Advance_Run") as run:
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params({"n_estimators": 150, "max_depth": 10})
        mlflow.log_metrics(metrics)

        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact("test_results.csv")

        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model",
            registered_model_name="ObesityModel"
        )
        
        print(f"Berhasil logging ke DagsHub. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()
