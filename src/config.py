"""Configuration centrale du projet (Module 9).
"""
from pathlib import Path

# --- Reproductibilite -----------------------------------------
SEED = 42

# --- Chemins ---------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
DATA_FILE = DATA_DIR / "dataset_trajectoire_emploi_Sujet Examen CISIA - Promo Upskilling Atlas - mai-oct2026 (Session-00279143).csv"  # placer ici le CSV Kaggle si disponible

# cible du dataset
TARGET = "classe_retour_emploi"

# Ensemble des features du dataset
FEATURES_DISPONIBLES = [
# df.columns.tolist()
 'usager_id',
 'age',
 'niveau_diplome',
 'anciennete_poste_ans',
 'code_rome_vise',
 'code_insee_commune',
 'est_allocataire',
 'nationalite_hors_ue',
 'synthese_entretien'
]


# Ensemble des features relatif à un identifiant du dataset
IDENTIFIANTS = [
# TODO
]

# Ensemble des features de type categorielle
CATEGORICAL = [
# TODO
]

# Ensemble des features de type numerique
NUMERIC = [
# TODO
]
