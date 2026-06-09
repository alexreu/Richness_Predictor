import os
import time

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_PATH = os.path.join(PROJECT_ROOT, "storage", "ml.db")
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model.joblib")
LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "training.log")

TARGET_COL = "income"

CATEGORICAL_COLS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
]

NUMERIC_COLS = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logger.add(LOG_PATH, rotation="1 MB")


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATABASE_PATH):
        raise FileNotFoundError(
            "Base SQLite introuvable. Lancez d'abord: python -m database.init_db"
        )

    logger.info("Connexion a la base SQLite")
    engine = create_engine(DATABASE_URL)
    query = """
    SELECT
        p.age,
        sf.libelle AS marital_status,
        r.libelle AS relationship,
        t.libelle AS workclass,
        m.libelle AS occupation,
        e.niveau AS education,
        e.niveau_num AS education_num,
        emp.heures_par_semaine AS hours_per_week,
        rev.capital_gain,
        rev.capital_loss,
        rev.revenu_superieur_50k AS income
    FROM personne p
    LEFT JOIN emploi emp ON p.id_personne = emp.id_personne
    LEFT JOIN type_travail t ON emp.id_type_travail = t.id_type_travail
    LEFT JOIN metier m ON emp.id_metier = m.id_metier
    LEFT JOIN education e ON p.id_personne = e.id_personne
    LEFT JOIN revenu rev ON p.id_personne = rev.id_personne
    LEFT JOIN situation_familiale sf ON p.id_situation_familiale = sf.id_situation_familiale
    LEFT JOIN relation r ON p.id_relation = r.id_relation
    """
    df = pd.read_sql(query, engine)
    logger.info(f"Donnees chargees : {df.shape}")
    return df


def build_pipeline(model) -> Pipeline:
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ]
    )
    return Pipeline([("preprocessing", preprocessor), ("model", model)])


def train_model(X_train, y_train) -> Pipeline:
    models = {
        "rf": (
            RandomForestClassifier(class_weight="balanced", random_state=42),
            {"model__n_estimators": [100, 300], "model__max_depth": [None, 10]},
        ),
        "logreg": (
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            {"model__C": [0.1, 1, 10]},
        ),
    }

    best_model = None
    best_score = 0.0

    for name, (model, params) in models.items():
        logger.info(f"Training: {name}")
        grid = GridSearchCV(
            build_pipeline(model), param_grid=params, cv=3, n_jobs=-1, scoring="accuracy"
        )
        grid.fit(X_train, y_train)

        logger.info(f"{name} score: {grid.best_score_}")
        logger.info(f"{name} params: {grid.best_params_}")
        mlflow.log_metric(f"{name}_best_cv_accuracy", grid.best_score_)
        mlflow.log_params({f"{name}_{key}": value for key, value in grid.best_params_.items()})

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_

    if best_model is None:
        raise RuntimeError("Aucun modele n'a pu etre entraine")

    return best_model


def evaluate(model, X_test, y_test) -> None:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))
    mlflow.log_metric("accuracy", acc)


def main() -> None:
    mlflow.set_experiment("income_sqlite_pipeline")

    with mlflow.start_run():
        df = load_data()
        missing = set(NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COL]) - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")

        y = df[TARGET_COL].astype(int)
        X = df[NUMERIC_COLS + CATEGORICAL_COLS]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        start = time.time()
        model = train_model(X_train, y_train)
        training_duration = time.time() - start

        logger.info(f"Temps entrainement: {training_duration:.2f}s")
        mlflow.log_metric("training_duration_seconds", training_duration)

        evaluate(model, X_test, y_test)

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        mlflow.sklearn.log_model(model, "model")

        logger.info(f"Modele sauvegarde : {MODEL_PATH}")
        logger.info("Pipeline termine avec succes")


if __name__ == "__main__":
    main()
