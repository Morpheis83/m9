"""Configuration centrale du projet Retour Emploi.

Ce module regroupe les chemins, constantes de reproductibilité
et listes de variables utilisées dans les différentes étapes
d'analyse et de modélisation.
"""

from pathlib import Path


# ============================================================
# REPRODUCTIBILITE
# ============================================================

SEED = 42


# ============================================================
# CHEMINS
# ============================================================

ROOT = Path(
    __file__
).resolve().parents[1]

DATA_DIR = (
    ROOT
    / "data"
)

OUTPUTS_DIR = (
    ROOT
    / "outputs"
)

DATA_FILE = (
    DATA_DIR
    / (
        "dataset_trajectoire_emploi_Sujet Examen CISIA - "
        "Promo Upskilling Atlas - mai-oct2026 "
        "(Session-00279143).csv"
    )
)


# ============================================================
# CIBLE
# ============================================================

TARGET = "classe_retour_emploi"


# ============================================================
# VARIABLES DISPONIBLES
# ============================================================

FEATURES_DISPONIBLES = [
    "usager_id",
    "age",
    "niveau_diplome",
    "anciennete_poste_ans",
    "code_rome_vise",
    "code_insee_commune",
    "est_allocataire",
    "nationalite_hors_ue",
    "synthese_entretien",
]


# ============================================================
# IDENTIFIANTS
# ============================================================

IDENTIFIANTS = [
    "usager_id",
]


# ============================================================
# VARIABLES QUALITATIVES
# ============================================================

FEATURES_QUALITATIVES = [
    "niveau_diplome",
    "code_rome_vise",
    "est_allocataire",
    "code_insee_commune",
    "nationalite_hors_ue",
    "synthese_entretien",
]


# ============================================================
# VARIABLES NUMERIQUES
# ============================================================

NUMERIC = [
    "age",
    "anciennete_poste_ans",
]

# ============================================================
# REGLES DE COHERENCE METIER - AGE
# ============================================================

AGE_MIN_DEBUT_ACTIVITE = 16

AGE_MIN_PAR_DIPLOME = {
    "Bac": 16,
    "Bac+2": 18,
    "Bac+5": 20,
}