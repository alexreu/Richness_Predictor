import os

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import (
    Base,
    Education,
    Emploi,
    Metier,
    Pays,
    Personne,
    Relation,
    Revenu,
    SituationFamiliale,
    TypeTravail,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FILE = os.path.join(PROJECT_ROOT, "data", "adult.csv")
DATABASE_PATH = os.path.join(PROJECT_ROOT, "storage", "ml.db")
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"


def get_or_create(session, model, field_name: str, value):
    if pd.isna(value) or value in {"?", " ?"}:
        return None

    instance = session.query(model).filter(getattr(model, field_name) == value).first()
    if instance:
        return instance

    instance = model(**{field_name: value})
    session.add(instance)
    session.flush()
    return instance


def reset_database(engine) -> None:
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def load_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_FILE)
    return df.replace(" ?", pd.NA)


def insert_rows(session, df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        type_travail = get_or_create(session, TypeTravail, "libelle", row["workclass"])
        metier = get_or_create(session, Metier, "libelle", row["occupation"])
        situation = get_or_create(
            session, SituationFamiliale, "libelle", row["marital.status"]
        )
        relation = get_or_create(session, Relation, "libelle", row["relationship"])
        pays = get_or_create(session, Pays, "nom", row["native.country"])

        personne = Personne(
            age=int(row["age"]) if not pd.isna(row["age"]) else None,
            id_situation_familiale=situation.id_situation_familiale
            if situation
            else None,
            id_relation=relation.id_relation if relation else None,
            id_pays=pays.id_pays if pays else None,
        )
        session.add(personne)
        session.flush()

        session.add(
            Emploi(
                id_personne=personne.id_personne,
                id_type_travail=type_travail.id_type_travail if type_travail else None,
                id_metier=metier.id_metier if metier else None,
                heures_par_semaine=int(row["hours.per.week"])
                if not pd.isna(row["hours.per.week"])
                else None,
            )
        )
        session.add(
            Education(
                id_personne=personne.id_personne,
                niveau=row["education"],
                niveau_num=int(row["education.num"])
                if not pd.isna(row["education.num"])
                else None,
            )
        )
        session.add(
            Revenu(
                id_personne=personne.id_personne,
                revenu_superieur_50k=str(row["income"]).strip() == ">50K",
                capital_gain=int(row["capital.gain"])
                if not pd.isna(row["capital.gain"])
                else 0,
                capital_loss=int(row["capital.loss"])
                if not pd.isna(row["capital.loss"])
                else 0,
            )
        )


def main() -> None:
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

    engine = create_engine(DATABASE_URL)
    reset_database(engine)

    session = sessionmaker(bind=engine)()
    try:
        df = load_csv()
        insert_rows(session, df)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    print(f"Base de donnees creee et remplie : {DATABASE_PATH}")


if __name__ == "__main__":
    main()
