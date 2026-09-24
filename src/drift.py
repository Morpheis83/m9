"""Calcul et persistance des références de dérive des données.

Le module construit une distribution de référence à partir des données
d'entraînement puis compare les données de production à cette référence
à l'aide du Population Stability Index (PSI).

Les variables à forte cardinalité ou textuelles sont transformées avant
le calcul afin de produire des indicateurs de dérive exploitables.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Evite les divisions par zéro et les logarithmes de zéro
# lors du calcul du PSI.
EPSILON = 1e-6


# ============================================================
# TRANSFORMATIONS COMMUNES
# ============================================================


def famille_rome(
    series: pd.Series,
) -> pd.Series:
    """Réduit un code ROME à sa famille représentée par sa première lettre."""

    def convertir(value):
        if pd.isna(value):
            return "__MISSING__"

        value = str(value).strip()

        if not value:
            return "__MISSING__"

        return value[0]

    return series.map(convertir)


def longueur_texte(
    series: pd.Series,
) -> pd.Series:
    """Convertit une série de textes en longueurs de chaînes."""

    return series.map(
        lambda value: (
            np.nan
            if pd.isna(value)
            else len(str(value))
        )
    )


# ============================================================
# REFERENCE NUMERIQUE
# ============================================================


def construire_reference_numerique(
    series: pd.Series,
    bins: int = 10,
) -> dict:
    """Construit la distribution de référence d'une variable numérique."""

    series = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid = series.dropna()

    if len(valid) == 0:
        cuts = []

    else:
        # Les bornes sont calculées par quantiles afin d'obtenir
        # des classes adaptées à la distribution d'entraînement.
        quantiles = np.linspace(
            0,
            1,
            bins + 1,
        )[1:-1]

        cuts = np.unique(
            valid.quantile(
                quantiles
            ).to_numpy()
        ).tolist()

    bucket_ids = np.digitize(
        valid.to_numpy(),
        cuts,
    )

    counts = np.bincount(
        bucket_ids,
        minlength=len(cuts) + 1,
    ).astype(float)

    # Les valeurs manquantes constituent un bucket dédié afin
    # que leur évolution soit également détectable.
    missing_count = float(
        series.isna().sum()
    )

    counts = np.append(
        counts,
        missing_count,
    )

    total = counts.sum()

    if total == 0:
        proportions = np.zeros(
            len(counts)
        )
    else:
        proportions = counts / total

    return {
        "type": "numeric",
        "cuts": cuts,
        "proportions": proportions.tolist(),
    }


# ============================================================
# REFERENCE CATEGORIELLE
# ============================================================


def construire_reference_categorielle(
    series: pd.Series,
) -> dict:
    """Construit la distribution de référence d'une variable catégorielle."""

    normalized = series.map(
        lambda value: (
            "__MISSING__"
            if (
                pd.isna(value)
                or not str(value).strip()
            )
            else str(value)
        )
    )

    categories = sorted(
        value
        for value in normalized.unique()
        if value != "__MISSING__"
    )

    # __OTHER__ permet de mesurer en production l'apparition
    # de modalités absentes des données d'entraînement.
    buckets = categories + [
        "__OTHER__",
        "__MISSING__",
    ]

    counts = []

    for bucket in buckets:
        if bucket == "__OTHER__":
            count = 0
        else:
            count = int(
                (
                    normalized
                    == bucket
                ).sum()
            )

        counts.append(count)

    counts = np.asarray(
        counts,
        dtype=float,
    )

    proportions = (
        counts
        / counts.sum()
    )

    return {
        "type": "categorical",
        "categories": categories,
        "proportions": proportions.tolist(),
    }


# ============================================================
# REFERENCE COMPLETE
# ============================================================


def construire_reference_drift(
    dataframe: pd.DataFrame,
) -> dict:
    """Construit la référence de drift des variables surveillées."""

    return {
        "version": 1,
        "features": {
            "niveau_diplome": (
                construire_reference_categorielle(
                    dataframe["niveau_diplome"]
                )
            ),
            "anciennete_poste_ans": (
                construire_reference_numerique(
                    dataframe["anciennete_poste_ans"]
                )
            ),

            # Les codes ROME sont trop granulaires pour une petite
            # fenêtre de production : le monitoring porte sur leur famille.
            "famille_rome": (
                construire_reference_categorielle(
                    famille_rome(
                        dataframe["code_rome_vise"]
                    )
                )
            ),

            # Le texte métier n'est jamais envoyé dans Prometheus.
            # Seule sa longueur est utilisée comme indicateur structurel.
            "longueur_synthese": (
                construire_reference_numerique(
                    longueur_texte(
                        dataframe["synthese_entretien"]
                    )
                )
            ),
        },
    }


# ============================================================
# SAUVEGARDE / CHARGEMENT
# ============================================================


def sauvegarder_reference_drift(
    reference: dict,
    path: Path,
) -> None:
    """Sauvegarde une référence de drift au format JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            reference,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def charger_reference_drift(
    path: Path,
) -> dict:
    """Charge une référence de drift depuis un fichier JSON."""

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# DISTRIBUTION PRODUCTION NUMERIQUE
# ============================================================


def distribution_numerique(
    series: pd.Series,
    reference: dict,
) -> np.ndarray:
    """Calcule une distribution numérique selon les buckets de référence."""

    series = pd.to_numeric(
        series,
        errors="coerce",
    )

    cuts = reference["cuts"]
    valid = series.dropna()

    bucket_ids = np.digitize(
        valid.to_numpy(),
        cuts,
    )

    counts = np.bincount(
        bucket_ids,
        minlength=len(cuts) + 1,
    ).astype(float)

    counts = np.append(
        counts,
        float(
            series.isna().sum()
        ),
    )

    if counts.sum() == 0:
        return np.zeros(
            len(counts)
        )

    return (
        counts
        / counts.sum()
    )


# ============================================================
# DISTRIBUTION PRODUCTION CATEGORIELLE
# ============================================================


def distribution_categorielle(
    series: pd.Series,
    reference: dict,
) -> np.ndarray:
    """Calcule une distribution catégorielle selon la référence."""

    categories = reference[
        "categories"
    ]

    known = set(categories)

    def normaliser(value):
        if (
            pd.isna(value)
            or not str(value).strip()
        ):
            return "__MISSING__"

        value = str(value)

        if value in known:
            return value

        return "__OTHER__"

    normalized = series.map(
        normaliser
    )

    buckets = categories + [
        "__OTHER__",
        "__MISSING__",
    ]

    counts = np.asarray(
        [
            (
                normalized
                == bucket
            ).sum()
            for bucket in buckets
        ],
        dtype=float,
    )

    if counts.sum() == 0:
        return np.zeros(
            len(counts)
        )

    return (
        counts
        / counts.sum()
    )


# ============================================================
# PSI
# ============================================================


def calculer_psi(
    expected,
    actual,
) -> float:
    """Calcule le Population Stability Index entre deux distributions."""

    expected = np.asarray(
        expected,
        dtype=float,
    )

    actual = np.asarray(
        actual,
        dtype=float,
    )

    # Les valeurs nulles sont remplacées par une valeur très faible
    # afin que le logarithme reste défini.
    expected = np.where(
        expected <= 0,
        EPSILON,
        expected,
    )

    actual = np.where(
        actual <= 0,
        EPSILON,
        actual,
    )

    psi = np.sum(
        (actual - expected)
        * np.log(
            actual / expected
        )
    )

    return float(psi)


# ============================================================
# CALCUL DU DRIFT
# ============================================================


def calculer_drift(
    dataframe: pd.DataFrame,
    reference: dict,
) -> dict:
    """Calcule le PSI de chaque variable surveillée."""

    production_features = {
        "niveau_diplome": (
            dataframe["niveau_diplome"]
        ),
        "anciennete_poste_ans": (
            dataframe["anciennete_poste_ans"]
        ),
        "famille_rome": famille_rome(
            dataframe["code_rome_vise"]
        ),
        "longueur_synthese": longueur_texte(
            dataframe["synthese_entretien"]
        ),
    }

    resultats = {}

    for feature_name, series in production_features.items():
        ref = reference[
            "features"
        ][
            feature_name
        ]

        if ref["type"] == "numeric":
            production_distribution = (
                distribution_numerique(
                    series,
                    ref,
                )
            )
        else:
            production_distribution = (
                distribution_categorielle(
                    series,
                    ref,
                )
            )

        resultats[
            feature_name
        ] = calculer_psi(
            ref["proportions"],
            production_distribution,
        )

    return resultats