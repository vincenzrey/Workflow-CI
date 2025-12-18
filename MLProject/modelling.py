import os
import shutil
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

DATASET_PATH = "namadataset_preprosessing/obesity_preprocessing.csv"
TARGET_COLUMN = "ObesityCategory"
RUN_ID_FILE = "run_id.txt"
TEMP_MODEL_DIR = "temp_model"

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Obesity_Classification_Basic")


def read_data(path):
    return pd.read_csv(path)


def split_dataset(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="weighted"),
        "recall": recall_score(y_test, preds, average="weighted"),
    }


def main():
    mlflow.sklearn.autolog()

    df = read_data(DATASET_PATH)
    X_train, X_test, y_train, y_test = split_dataset(df, TARGET_COLUMN)

    if os.path.exists(TEMP_MODEL_DIR):
        shutil.rmtree(TEMP_MODEL_DIR)
    os.makedirs(TEMP_MODEL_DIR)

    with mlflow.start_run(run_name="RandomForest_Obesity_Model") as run:
        model = train_model(X_train, y_train)

        # Simpan model lokal sementara
        local_model_path = os.path.join(TEMP_MODEL_DIR, "model.joblib")
        joblib.dump(model, local_model_path)
        print(f"Model disimpan sementara di: {local_model_path}")

        # Evaluasi & logging
        metrics = evaluate_model(model, X_test, y_test)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
            mlflow.log_metric(k, v)

        # Simpan run_id
        with open(RUN_ID_FILE, "w") as f:
            f.write(run.info.run_id)

        print(f"Run ID disimpan: {run.info.run_id}")

    shutil.rmtree(TEMP_MODEL_DIR)
    print("Folder model sementara dibersihkan")


if __name__ == "__main__":
    main()
