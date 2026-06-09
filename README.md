# Richness Predictor

Projet de modernisation d'un modele de prediction de revenu base sur le dataset Adult.

## Etat actuel

- Les donnees source sont dans `data/adult.csv`.
- La base SQLite est generee dans `storage/ml.db`.
- L'entrainement lit la base SQLite et compare 3 pipelines ML.
- Le meilleur modele est sauvegarde dans `artifacts/model.joblib`.
- L'API FastAPI expose une route `/predict` cablee au modele local.
- Prometheus, Grafana, Node Exporter et MLflow sont configures avec Docker Compose.

## Architecture

```text
data/adult.csv
  -> database/init_db.py
  -> storage/ml.db
  -> training/train_from_db.py
  -> artifacts/model.joblib
  -> api/main.py
```

## Structure

```text
api/          API FastAPI + schema de prediction
database/     modeles SQLAlchemy + creation SQLite
training/     comparaison des pipelines ML
monitoring/   Prometheus + Grafana dashboards
storage/      base SQLite generee localement
artifacts/    modele genere localement
```

## Commandes

Installer les dependances :

```bash
pip install -r requirements.txt
```

Creer la base SQLite :

```bash
python -m database.init_db
```

Lancer la stack Docker :

```bash
docker compose up --build
```

Entrainer et envoyer les metriques vers MLflow Docker :

```bash
MLFLOW_TRACKING_URI=http://localhost:5001 .venv/bin/python -m training.train_from_db
```

## Pipelines entraines

Le script `training/train_from_db.py` compare :

- `logistic_regression`
- `random_forest_controlled`
- `neural_network_mlp`

Metriques suivies dans MLflow :

- `accuracy`
- `balanced_accuracy`
- `precision_positive`
- `recall_positive`
- `f1_positive`
- `roc_auc`
- `training_duration_seconds`
- `training_cpu_seconds`
- `inference_latency_ms_per_sample`

Le modele retenu est celui qui obtient le meilleur `f1_positive` sur la classe `>50K`.

Ce choix evite une ponderation metier arbitraire et utilise une metrique standard qui equilibre precision et recall.

## API

Endpoints principaux :

| Methode | Endpoint | Role |
|---|---|---|
| `GET` | `/` | Statut de l'API |
| `GET` | `/health` | Healthcheck |
| `POST` | `/predict` | Prediction de revenu |
| `GET` | `/metrics` | Metriques Prometheus |

Exemple de prediction :

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "workclass": "Private",
    "education": "HS-grad",
    "education_num": 9,
    "marital_status": "Widowed",
    "occupation": "Prof-specialty",
    "relationship": "Unmarried",
    "capital_gain": 0,
    "capital_loss": 2824,
    "hours_per_week": 45
  }'
```

Exemple de reponse :

```json
{
  "status": "OK",
  "prediction": ">50K"
}
```

## Monitoring

Services disponibles :

```text
API: http://localhost:8000
Swagger: http://localhost:8000/docs
Prometheus: http://localhost:9090
Grafana: http://localhost:3000
MLflow: http://localhost:5001
Node Exporter: http://localhost:9100/metrics
```

Grafana :

```text
admin / admin
```

Dashboard provisionne :

```text
http://localhost:3000/d/richness-overview-root/richness-predictor-overview
```

Prometheus collecte :

- les metriques FastAPI via `/metrics` ;
- les metriques systeme via Node Exporter ;
- CPU, RAM, charge systeme, debit reseau.

## Choix de donnees

Colonnes exclues de l'entrainement :

- `race` : risque de discrimination directe.
- `sex` : donnee sensible selon le contexte d'usage.
- `native.country` : proxy potentiel d'origine.
- `fnlwgt` : poids statistique du dataset, pas une variable metier.

Colonnes utilisees :

```text
age, workclass, education, education_num, marital_status,
occupation, relationship, capital_gain, capital_loss,
hours_per_week, income
```
