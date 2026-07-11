"""Chargement et profilage des donnees (etape 1 du brief).

`charger_donnees()` lit le jeu trajectoire emploi depuis `data/dataset_trajectoire_emploi_Sujet Examen CISIA - Promo Upskilling Atlas - mai-oct2026 (Session-00279143).csv`. Placez ce
fichier avant d'executer le projet (voir `data/README.md`).
"""
import pandas as pd

from . import config


def charger_donnees(verbose: bool = True) -> pd.DataFrame:
    """Charge le jeu trajectoire emploi depuis data/dataset_trajectoire_emploi_Sujet Examen CISIA - Promo Upskilling Atlas - mai-oct2026 (Session-00279143).csv."""
    if not config.DATA_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {config.DATA_FILE}"
        )
    if verbose:
        print(f"Chargement du CSV : {config.DATA_FILE}")
    return pd.read_csv(config.DATA_FILE)
