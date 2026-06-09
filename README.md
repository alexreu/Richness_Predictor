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

Lancer l'API et le monitoring :

```bash
docker compose up --build
```

URLs utiles :

```text
API: http://localhost:8000
Swagger: http://localhost:8000/docs
Prometheus: http://localhost:9090
Grafana: http://localhost:3000
Node Exporter: http://localhost:9100/metrics
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
