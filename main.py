
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib


DATA_PATH = os.path.join("data", "adult.csv")
MODEL_PATH = os.path.join("artifacts", "model.joblib")

TARGET_COL = "income"

CATEGORICAL_COLS = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",       
    "sex",          
    "native.country"
]

NUMERIC_COLS = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week"
]


def load_data(csv_path: str) -> pd.DataFrame:

    df = pd.read_csv(csv_path)
    return df


def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df.replace(" ?", np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def prepare_features(df: pd.DataFrame):

    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y_raw = df[TARGET_COL].astype(str)

    # binaire : >50K -> 1, sinon 0
    y = y_raw.apply(lambda x: 1 if x.strip() == ">50K" else 0)

    X_encoded = pd.get_dummies(X, drop_first=True)

    return X_encoded, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
 
    model = RandomForestClassifier(
        n_estimators=500,      
        max_depth=None,
        n_jobs=-1,            
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n=== Évaluation du modèle ===")
    print(f"Accuracy : {acc:.4f}\n")
    print("Classification report :")
    print(classification_report(y_test, y_pred))


def save_model(model, feature_columns, path: str):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model,
        "feature_columns": list(feature_columns)
    }
    joblib.dump(payload, path)
    print(f"\nModèle sauvegardé dans : {path}")


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    payload = joblib.load(path)
    return payload["model"], payload["feature_columns"]


def simple_inference(model, feature_columns, df_input: pd.DataFrame) -> np.ndarray:

    X = df_input[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_encoded = X_encoded.reindex(columns=feature_columns, fill_value=0)
    preds = model.predict(X_encoded)
    return preds


def main():
    print("Chargement des données...")
    df = load_data(DATA_PATH)

    print("Nettoyage des données...")
    df_clean = data_cleaning(df)

    print("Préparation des variables explicatives...")
    X, y = prepare_features(df_clean)

    print("Split train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Entraînement du modèle RandomForest...")
    start_train = time.perf_counter()
    model = train_model(X_train, y_train)
    train_duration = time.perf_counter() - start_train
    print(f"Temps d'entraînement (approx.) : {train_duration:.2f} secondes")

    print("Inférence sur le jeu de test...")
    start_infer = time.perf_counter()
    y_pred = model.predict(X_test)
    infer_duration = time.perf_counter() - start_infer
    print(f"Temps d'inférence sur {len(X_test)} échantillons : {infer_duration:.4f} secondes")

    acc = accuracy_score(y_test, y_pred)
    print("\n=== Résultats globaux ===")
    print(f"Accuracy : {acc:.4f}")
    print("\nClassification report :")
    print(classification_report(y_test, y_pred))

    save_model(model, X.columns, MODEL_PATH)

    
if __name__ == "__main__":
    main()
