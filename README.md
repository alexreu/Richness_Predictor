# Richness Predictor

Projet de modernisation d'un pipeline IA de prediction de revenu base sur le dataset Adult.

## Architecture active

Le projet utilise maintenant uniquement le flux base de donnees :

```text
data/adult.csv
     -> database/init_db.py
     -> storage/ml.db
     -> training/train_from_db.py
     -> artifacts/model.joblib
```

L'API FastAPI est deja exposee et monitoree, mais le modele n'est pas encore cable a `/predict`.

## Structure

```text
api/                    API FastAPI placeholder + endpoint /metrics
database/               Modeles SQLAlchemy + initialisation SQLite
training/               Entrainement ML depuis SQLite
data/                   Donnees source CSV
storage/                Base SQLite generee localement
monitoring/             Prometheus + provisioning Grafana
```

## Commandes

Creer et remplir la base SQLite :

```bash
python -m database.init_db
```

Entrainer le modele depuis la base :

```bash
python -m training.train_from_db
```

Cette commande compare trois pipelines :

- `logistic_regression` : baseline machine learning sobre et explicable.
- `random_forest_controlled` : RandomForest limite en profondeur pour maitriser le cout CPU.
- `neural_network_mlp` : reseau de neurones dense leger avec early stopping.

Le meilleur pipeline est sauvegarde dans :

```text
artifacts/model.joblib
```

Les metriques de comparaison sont enregistrees dans MLflow.

Metriques comparees :

- `accuracy`
- `balanced_accuracy`
- `precision_positive`
- `recall_positive`
- `f1_positive`
- `roc_auc`
- `training_duration_seconds`
- `training_cpu_seconds`
- `inference_latency_ms_per_sample`
- `business_score`

Le `business_score` privilegie la precision sur la classe `>50K`, car l'usage metier consiste a cibler efficacement des clients pour des offres :

```text
business_score = 0.50 * precision_positive + 0.30 * recall_positive + 0.20 * balanced_accuracy
```

## Resultats du dernier entrainement manuel

Derniere execution :

```bash
.venv/bin/python -m training.train_from_db
```

Resultats obtenus :

| Pipeline | Business score | Precision >50K | Recall >50K | F1 >50K | Balanced accuracy | Temps entrainement | Latence / sample |
|---|---:|---:|---:|---:|---:|---:|---:|
| `neural_network_mlp` | 0.7025 | 0.7212 | 0.6237 | 0.6689 | 0.7736 | 7.53s | 0.0013 ms |
| `random_forest_controlled` | 0.7014 | 0.5561 | 0.8629 | 0.6763 | 0.8222 | 5.99s | 0.0077 ms |
| `logistic_regression` | 0.6920 | 0.5638 | 0.8259 | 0.6701 | 0.8116 | 0.97s | 0.0011 ms |

Lecture metier :

- `neural_network_mlp` est le meilleur selon le `business_score`, car il maximise davantage la precision sur la classe `>50K`.
- `random_forest_controlled` detecte plus de profils `>50K` grace a un meilleur recall, mais il genere plus de faux positifs.
- `logistic_regression` est le pipeline le plus sobre et le plus rapide, avec des performances proches des autres modeles.

Modele retenu automatiquement :

```text
neural_network_mlp
```

Artefact genere localement :

```text
artifacts/model.joblib
```

## Endpoints API

L'API expose actuellement 3 endpoints principaux. Le modele n'est pas encore cable a `/predict`.

| Methode | Endpoint | Role | Reponse actuelle |
|---|---|---|---|
| `GET` | `/` | Verifier que l'API repond | Message `Richness Predictor API is running.` |
| `GET` | `/health` | Healthcheck du service | Message `Service healthy.` |
| `POST` | `/predict` | Point d'entree pour la prediction | Message `{ status: 200, prediction: ">50k" }` |

Exemple de JSON pour la prédiction: 

```json
{
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
}
```


Lancer l'API et le monitoring :

```bash
docker compose up --build
```

Lancer un entrainement en envoyant les metriques vers MLflow Docker :

```bash
MLFLOW_TRACKING_URI=http://localhost:5001 .venv/bin/python -m training.train_from_db
```

Si `MLFLOW_TRACKING_URI` n'est pas defini, le script utilise une base MLflow locale `mlflow.db`.

URLs utiles :

```text
API: http://localhost:8000
Swagger: http://localhost:8000/docs
Prometheus: http://localhost:9090
Grafana: http://localhost:3000
MLflow: http://localhost:5001
Node Exporter: http://localhost:9100/metrics
```

## Monitoring et suivi des experiences

La stack Docker contient :

- `prometheus` : collecte les metriques API et systeme.
- `grafana` : affiche les dashboards de monitoring.
- `node-exporter` : expose les metriques CPU/RAM/disque/reseau de la machine.
- `mlflow` : enregistre les experiences d'entrainement, les hyperparametres, les metriques et les artefacts.

Les donnees sont persistantes via des volumes Docker :

- `prometheus_data` : historique Prometheus.
- `grafana_data` : configuration et etat Grafana.
- `mlflow_data` : base MLflow et artefacts MLflow.

Grafana est provisionne automatiquement avec :

- une datasource Prometheus ;
- un dashboard `Richness Predictor Overview`.

Le dashboard affiche notamment :

- disponibilite de l'API ;
- disponibilite de Node Exporter ;
- CPU usage ;
- memory usage ;
- nombre de requetes API par seconde ;
- charge systeme.

Separation des responsabilites :

- MLflow sert a comparer les pipelines ML et conserver les resultats d'experiences.
- Prometheus sert a collecter les metriques techniques dans le temps.
- Grafana sert a visualiser les metriques Prometheus.

Verifier les targets Prometheus :

```text
http://localhost:9090/targets
```

Ouvrir le dashboard Grafana :

```text
http://localhost:3000
```

Identifiants Grafana par defaut :

```text
admin / admin
```

## Choix de donnees

Les colonnes `race`, `sex`, `native.country` et `fnlwgt` ne sont pas utilisees dans le pipeline d'entrainement actif.

Raisons :

- `race` : origine ethnique, risque de discrimination directe.
- `sex` : donnee personnelle sensible selon le contexte d'usage.
- `native.country` : peut agir comme proxy d'origine.
- `fnlwgt` : poids statistique du dataset, pas une donnee metier exploitable.

Les donnees retenues pour l'entrainement sont :

```text
age
workclass
education
education_num
marital_status
occupation
relationship
hours_per_week
capital_gain
capital_loss
income
```

## Enjeux éthiques
Nous sommes ici sur un modele local qui a été généré par un MACBOOK PRO M2 en 25s pour entrainer 3 modèles : 1 regression RandomForest, 1 regression logisitique et 1 CNN (faible usage GPU)
L'usage via API de ce modèle est faible coté CPU 
-> c'est un "non-sujet" ici selon nous 
