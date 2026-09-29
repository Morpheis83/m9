"""Chargement et identification du jeu de données Retour Emploi.

Le module centralise la lecture du fichier CSV utilisé par le projet
ainsi que la production d'un résumé descriptif des variables.
"""

import pandas as pd

from src import config
from src.config import FEATURES_DISPONIBLES


# ============================================================
# CHARGEMENT DES DONNEES
# ============================================================


def charger_donnees(
    verbose: bool = True,
) -> pd.DataFrame:
    """Charge le jeu de données configuré dans ``config.DATA_FILE``.

    Args:
        verbose: Affiche le chemin du fichier chargé lorsque la valeur
            est True.

    Returns:
        Le jeu de données sous forme de DataFrame pandas.

    Raises:
        FileNotFoundError: Si le fichier configuré n'existe pas.
    """

    if not config.DATA_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {config.DATA_FILE}"
        )

    if verbose:
        print(
            f"Chargement du CSV : {config.DATA_FILE}"
        )

    return pd.read_csv(
        config.DATA_FILE
    )


# ============================================================
# CONTROLES DE COHERENCE METIER
# ============================================================


def masque_incoherences_age(
    df: pd.DataFrame,
) -> pd.Series:
    """Identifie les observations présentant une incohérence liée à l'âge."""

    incoherence_anciennete = (
        df["age"].notna()
        & df["anciennete_poste_ans"].notna()
        & (
            df["age"]
            - df["anciennete_poste_ans"]
            < config.AGE_MIN_DEBUT_ACTIVITE
        )
    )

    age_min_diplome = (
        df["niveau_diplome"]
        .map(config.AGE_MIN_PAR_DIPLOME)
    )

    incoherence_diplome = (
        df["age"].notna()
        & age_min_diplome.notna()
        & (
            df["age"]
            < age_min_diplome
        )
    )

    return (
        incoherence_anciennete
        | incoherence_diplome
    )


def exclure_incoherences_age(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Exclut les observations présentant une incohérence liée à l'âge."""

    masque = masque_incoherences_age(df)

    return df.loc[
        ~masque
    ].copy()

# ============================================================
# IDENTIFICATION DES VARIABLES
# ============================================================


def identification(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Construit un résumé des variables du jeu de données.

    Pour chaque feature attendue, le résumé contient le type pandas,
    la cardinalité ainsi que les valeurs minimale et maximale lorsque
    ces opérations sont applicables au type de données.
    """

    resultats = []

    for feature in FEATURES_DISPONIBLES:
        serie = df[
            feature
        ]

        cardinalite = serie.nunique()

        try:
            valeur_min = serie.min()
            valeur_max = serie.max()

        except TypeError:
            valeur_min = "Non applicable"
            valeur_max = "Non applicable"

        resultats.append(
            {
                "Feature": feature,
                "Type": str(
                    serie.dtype
                ),
                "Cardinalité": cardinalite,
                "Minimum": valeur_min,
                "Maximum": valeur_max,
            }
        )

    return pd.DataFrame(
        resultats
    )