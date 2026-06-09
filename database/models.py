from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


class TypeTravail(Base):
    __tablename__ = "type_travail"

    id_type_travail = Column(Integer, primary_key=True)
    libelle = Column(String, nullable=False, unique=True)


class Metier(Base):
    __tablename__ = "metier"

    id_metier = Column(Integer, primary_key=True)
    libelle = Column(String, nullable=False, unique=True)


class SituationFamiliale(Base):
    __tablename__ = "situation_familiale"

    id_situation_familiale = Column(Integer, primary_key=True)
    libelle = Column(String, nullable=False, unique=True)


class Relation(Base):
    __tablename__ = "relation"

    id_relation = Column(Integer, primary_key=True)
    libelle = Column(String, nullable=False, unique=True)


class Pays(Base):
    __tablename__ = "pays"

    id_pays = Column(Integer, primary_key=True)
    nom = Column(String, nullable=False, unique=True)


class Personne(Base):
    __tablename__ = "personne"

    id_personne = Column(Integer, primary_key=True)
    age = Column(Integer)

    id_situation_familiale = Column(
        Integer, ForeignKey("situation_familiale.id_situation_familiale")
    )
    id_relation = Column(Integer, ForeignKey("relation.id_relation"))
    id_pays = Column(Integer, ForeignKey("pays.id_pays"))

    situation_familiale = relationship("SituationFamiliale")
    relation = relationship("Relation")
    pays = relationship("Pays")

    emploi = relationship("Emploi", uselist=False, back_populates="personne")
    education = relationship("Education", uselist=False, back_populates="personne")
    revenu = relationship("Revenu", uselist=False, back_populates="personne")


class Emploi(Base):
    __tablename__ = "emploi"

    id_emploi = Column(Integer, primary_key=True)
    id_personne = Column(Integer, ForeignKey("personne.id_personne"))
    id_type_travail = Column(Integer, ForeignKey("type_travail.id_type_travail"))
    id_metier = Column(Integer, ForeignKey("metier.id_metier"))
    heures_par_semaine = Column(Integer)

    personne = relationship("Personne", back_populates="emploi")
    type_travail = relationship("TypeTravail")
    metier = relationship("Metier")


class Education(Base):
    __tablename__ = "education"

    id_education = Column(Integer, primary_key=True)
    id_personne = Column(Integer, ForeignKey("personne.id_personne"))
    niveau = Column(String)
    niveau_num = Column(Integer)

    personne = relationship("Personne", back_populates="education")


class Revenu(Base):
    __tablename__ = "revenu"

    id_revenu = Column(Integer, primary_key=True)
    id_personne = Column(Integer, ForeignKey("personne.id_personne"))
    revenu_superieur_50k = Column(Boolean)
    capital_gain = Column(Integer)
    capital_loss = Column(Integer)

    personne = relationship("Personne", back_populates="revenu")
