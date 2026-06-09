import os
import time
from dataclasses import dataclass

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_PATH = os.path.join(PROJECT_ROOT, "storage", "ml.db")
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model.joblib")
METRICS_CSV_PATH = os.path.join(PROJECT_ROOT, "artifacts", "training_metrics.csv")
METRICS_JSON_PATH = os.path.join(PROJECT_ROOT, "artifacts", "training_metrics.json")
METRICS_HTML_PATH = os.path.join(PROJECT_ROOT, "artifacts", "training_metrics.html")
LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "training.log")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

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


@dataclass(frozen=True)
class PipelineCandidate:
    name: str
    model: object
    params: dict[str, list]
    description: str


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
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ]
    )
    return Pipeline([("preprocessing", preprocessor), ("model", model)])


def get_pipeline_candidates() -> list[PipelineCandidate]:
    return [
        PipelineCandidate(
            name="logistic_regression",
            model=LogisticRegression(max_iter=1000, class_weight="balanced"),
            params={"model__C": [0.1, 1.0, 10.0]},
            description="Baseline machine learning sobre et explicable.",
        ),
        PipelineCandidate(
            name="random_forest_controlled",
            model=RandomForestClassifier(
                class_weight="balanced",
                n_jobs=1,
                random_state=42,
            ),
            params={
                "model__n_estimators": [120],
                "model__max_depth": [10, 14],
                "model__min_samples_leaf": [5],
            },
            description="RandomForest limite en profondeur pour controler cout CPU et surapprentissage.",
        ),
        PipelineCandidate(
            name="neural_network_mlp",
            model=MLPClassifier(
                early_stopping=True,
                max_iter=120,
                random_state=42,
                validation_fraction=0.15,
            ),
            params={
                "model__hidden_layer_sizes": [(32,), (64, 32)],
                "model__alpha": [0.0001, 0.001],
            },
            description="Reseau de neurones dense leger pour comparer avec les modeles tabulaires.",
        ),
    ]


def compute_business_score(metrics: dict[str, float]) -> float:
    return (
        0.50 * metrics["precision_positive"]
        + 0.30 * metrics["recall_positive"]
        + 0.20 * metrics["balanced_accuracy"]
    )


def evaluate_candidate(candidate: PipelineCandidate, model, X_test, y_test) -> dict:
    inference_start = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_duration = time.perf_counter() - inference_start

    y_scores = model.predict_proba(X_test)[:, 1]
    metrics = {
        "pipeline": candidate.name,
        "description": candidate.description,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision_positive": precision_score(y_test, y_pred, zero_division=0),
        "recall_positive": recall_score(y_test, y_pred, zero_division=0),
        "f1_positive": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_scores),
        "inference_duration_seconds": inference_duration,
        "inference_latency_ms_per_sample": inference_duration / len(X_test) * 1000,
    }
    metrics["business_score"] = compute_business_score(metrics)

    logger.info(f"Rapport {candidate.name}:\n{classification_report(y_test, y_pred)}")
    return metrics


def train_and_compare_pipelines(X_train, X_test, y_train, y_test) -> tuple[Pipeline, pd.DataFrame]:
    results = []
    best_model = None
    best_score = -1.0

    for candidate in get_pipeline_candidates():
        logger.info(f"Training pipeline: {candidate.name}")
        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        grid = GridSearchCV(
            build_pipeline(candidate.model),
            param_grid=candidate.params,
            cv=3,
            n_jobs=1,
            scoring="f1",
        )
        grid.fit(X_train, y_train)

        metrics = evaluate_candidate(candidate, grid.best_estimator_, X_test, y_test)
        metrics.update(
            {
                "best_cv_f1": grid.best_score_,
                "training_duration_seconds": time.perf_counter() - wall_start,
                "training_cpu_seconds": time.process_time() - cpu_start,
                "best_params": grid.best_params_,
            }
        )
        results.append(metrics)

        logger.info(f"{candidate.name} metrics: {metrics}")
        mlflow.log_metric(f"{candidate.name}_business_score", metrics["business_score"])
        mlflow.log_metric(f"{candidate.name}_f1_positive", metrics["f1_positive"])
        mlflow.log_metric(
            f"{candidate.name}_precision_positive", metrics["precision_positive"]
        )
        mlflow.log_metric(f"{candidate.name}_recall_positive", metrics["recall_positive"])
        mlflow.log_metric(
            f"{candidate.name}_training_duration_seconds",
            metrics["training_duration_seconds"],
        )
        mlflow.log_metric(
            f"{candidate.name}_inference_latency_ms_per_sample",
            metrics["inference_latency_ms_per_sample"],
        )

        if metrics["business_score"] > best_score:
            best_score = metrics["business_score"]
            best_model = grid.best_estimator_

    if best_model is None:
        raise RuntimeError("Aucun pipeline n'a pu etre entraine")

    results_df = pd.DataFrame(results).sort_values(
        by="business_score", ascending=False
    )
    return best_model, results_df


def save_training_report(results_df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(METRICS_CSV_PATH), exist_ok=True)
    results_df.to_csv(METRICS_CSV_PATH, index=False)
    results_df.to_json(METRICS_JSON_PATH, orient="records", indent=2)
    results_df.to_html(METRICS_HTML_PATH, index=False)
    mlflow.log_artifact(METRICS_CSV_PATH)
    mlflow.log_artifact(METRICS_JSON_PATH)
    mlflow.log_artifact(METRICS_HTML_PATH)
    logger.info(f"Rapport metriques CSV: {METRICS_CSV_PATH}")
    logger.info(f"Rapport metriques JSON: {METRICS_JSON_PATH}")
    logger.info(f"Rapport metriques HTML: {METRICS_HTML_PATH}")


def print_training_report(results_df: pd.DataFrame) -> None:
    display_columns = [
        "pipeline",
        "business_score",
        "precision_positive",
        "recall_positive",
        "f1_positive",
        "balanced_accuracy",
        "training_duration_seconds",
        "inference_latency_ms_per_sample",
    ]
    report = results_df[display_columns].copy()
    numeric_columns = [column for column in display_columns if column != "pipeline"]
    report[numeric_columns] = report[numeric_columns].round(4)

    print("\n=== Comparaison des pipelines ===")
    print(report.to_string(index=False))
    print(f"\nMeilleur pipeline: {results_df.iloc[0]['pipeline']}")
    print(f"Rapport HTML: {METRICS_HTML_PATH}\n")


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
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
        model, results_df = train_and_compare_pipelines(X_train, X_test, y_train, y_test)
        total_training_duration = time.time() - start

        logger.info(f"Temps total comparaison: {total_training_duration:.2f}s")
        logger.info("\n" + results_df.to_string(index=False))
        print_training_report(results_df)
        mlflow.log_metric("total_training_duration_seconds", total_training_duration)
        mlflow.log_metric("best_business_score", results_df.iloc[0]["business_score"])

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        save_training_report(results_df)
        mlflow.sklearn.log_model(model, "model")

        logger.info(f"Meilleur pipeline : {results_df.iloc[0]['pipeline']}")
        logger.info(f"Modele sauvegarde : {MODEL_PATH}")
        logger.info("Pipeline termine avec succes")


if __name__ == "__main__":
    main()
