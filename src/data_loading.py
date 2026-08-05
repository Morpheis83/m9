"""Chargement des données.

`charger_donnees()` lit le jeu trajectoire emploi depuis `data/dataset_trajectoire_emploi_Sujet Examen CISIA - Promo Upskilling Atlas - mai-oct2026 (Session-00279143).csv`. Placez ce
fichier avant d'executer le projet (voir `data/README.md`).
`identification()` réalise une première analyse des données nécessaire à la création du dictionnaire des varaibles.


"""
import pandas as pd

from . import config
from .config import FEATURES_DISPONIBLES


def charger_donnees(verbose: bool = True) -> pd.DataFrame:
    """Charge le jeu trajectoire emploi depuis data/dataset_trajectoire_emploi_Sujet Examen CISIA - Promo Upskilling Atlas - mai-oct2026 (Session-00279143).csv."""
    if not config.DATA_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {config.DATA_FILE}"
        )
    if verbose:
        print(f"Chargement du CSV : {config.DATA_FILE}")
    return pd.read_csv(config.DATA_FILE)

def identification(df: pd.DataFrame) -> dict:
    """Identification des données : types, cardinalité, min, max."""

    resultats = []

    for feature in FEATURES_DISPONIBLES:
        serie = df[feature]

        # Cardinalité avec nunique()
        cardinalite = serie.nunique()

        try:
            valeur_min = serie.min()
            valeur_max = serie.max()
        except TypeError:
            valeur_min = "Non applicable"
            valeur_max = "Non applicable"

        resultats.append({
            "Feature": feature,
            "Type": str(serie.dtype),
            "Cardinalité": cardinalite,
            "Minimum": valeur_min,
            "Maximum": valeur_max,
        })

    return pd.DataFrame(resultats)