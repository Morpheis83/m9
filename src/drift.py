import json

from pathlib import Path

import numpy as np
import pandas as pd


EPSILON = 1e-6


# ============================================================
# TRANSFORMATIONS COMMUNES
# ============================================================

def famille_rome(
    series: pd.Series
):

    def convertir(
        value
    ):

        if pd.isna(
            value
        ):

            return "__MISSING__"

        value = str(
            value
        ).strip()

        if not value:

            return "__MISSING__"

        return value[0]


    return series.map(
        convertir
    )


def longueur_texte(
    series: pd.Series
):

    return series.map(
        lambda value:
            np.nan
            if pd.isna(value)
            else len(str(value))
    )


# ============================================================
# REFERENCE NUMERIQUE
# ============================================================

def construire_reference_numerique(
    series: pd.Series,
    bins: int = 10
):

    series = pd.to_numeric(
        series,
        errors="coerce"
    )


    valid = series.dropna()


    if len(valid) == 0:

        cuts = []

    else:

        quantiles = np.linspace(
            0,
            1,
            bins + 1
        )[1:-1]

        cuts = np.unique(
            valid.quantile(
                quantiles
            ).to_numpy()
        ).tolist()


    bucket_ids = np.digitize(
        valid.to_numpy(),
        cuts
    )


    counts = np.bincount(
        bucket_ids,
        minlength=len(cuts) + 1
    ).astype(float)


    missing_count = float(
        series.isna().sum()
    )


    counts = np.append(
        counts,
        missing_count
    )


    total = counts.sum()


    if total == 0:

        proportions = np.zeros(
            len(counts)
        )

    else:

        proportions = (
            counts
            / total
        )


    return {

        "type":
            "numeric",

        "cuts":
            cuts,

        "proportions":
            proportions.tolist()
    }


# ============================================================
# REFERENCE CATEGORIELLE
# ============================================================

def construire_reference_categorielle(
    series: pd.Series
):

    normalized = series.map(
        lambda value:
            "__MISSING__"
            if (
                pd.isna(value)
                or not str(value).strip()
            )
            else str(value)
    )


    categories = sorted(
        value

        for value
        in normalized.unique()

        if value
        != "__MISSING__"
    )


    buckets = (
        categories
        + [
            "__OTHER__",
            "__MISSING__"
        ]
    )


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

        counts.append(
            count
        )


    counts = np.asarray(
        counts,
        dtype=float
    )


    proportions = (
        counts
        / counts.sum()
    )


    return {

        "type":
            "categorical",

        "categories":
            categories,

        "proportions":
            proportions.tolist()
    }


# ============================================================
# REFERENCE COMPLETE
# ============================================================

def construire_reference_drift(
    dataframe: pd.DataFrame
):

    return {

        "version":
            1,

        "features": {

            "niveau_diplome":
                construire_reference_categorielle(
                    dataframe[
                        "niveau_diplome"
                    ]
                ),

            "anciennete_poste_ans":
                construire_reference_numerique(
                    dataframe[
                        "anciennete_poste_ans"
                    ]
                ),

            # 50 codes ROME seraient trop fins pour
            # une petite fenêtre de production.
            # On surveille donc leur famille.
            "famille_rome":
                construire_reference_categorielle(
                    famille_rome(
                        dataframe[
                            "code_rome_vise"
                        ]
                    )
                ),

            # On ne pousse jamais le texte lui-même
            # dans Prometheus.
            #
            # On surveille ici une caractéristique
            # structurelle du texte.
            "longueur_synthese":
                construire_reference_numerique(
                    longueur_texte(
                        dataframe[
                            "synthese_entretien"
                        ]
                    )
                )
        }
    }


# ============================================================
# SAUVEGARDE / CHARGEMENT
# ============================================================

def sauvegarder_reference_drift(
    reference: dict,
    path: Path
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        json.dumps(
            reference,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


def charger_reference_drift(
    path: Path
):

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
    reference: dict
):

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    cuts = reference[
        "cuts"
    ]

    valid = series.dropna()


    bucket_ids = np.digitize(
        valid.to_numpy(),
        cuts
    )


    counts = np.bincount(
        bucket_ids,
        minlength=len(cuts) + 1
    ).astype(float)


    counts = np.append(
        counts,
        float(
            series.isna().sum()
        )
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
    reference: dict
):

    categories = reference[
        "categories"
    ]

    known = set(
        categories
    )


    def normaliser(
        value
    ):

        if (
            pd.isna(value)
            or not str(value).strip()
        ):

            return "__MISSING__"

        value = str(
            value
        )

        if value in known:

            return value

        return "__OTHER__"


    normalized = series.map(
        normaliser
    )


    buckets = (
        categories
        + [
            "__OTHER__",
            "__MISSING__"
        ]
    )


    counts = np.asarray(
        [
            (
                normalized
                == bucket
            ).sum()

            for bucket
            in buckets
        ],
        dtype=float
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
    actual
):

    expected = np.asarray(
        expected,
        dtype=float
    )

    actual = np.asarray(
        actual,
        dtype=float
    )


    expected = np.where(
        expected <= 0,
        EPSILON,
        expected
    )

    actual = np.where(
        actual <= 0,
        EPSILON,
        actual
    )


    psi = np.sum(
        (
            actual
            - expected
        )
        * np.log(
            actual
            / expected
        )
    )


    return float(
        psi
    )


# ============================================================
# CALCUL DES PSI
# ============================================================

def calculer_drift(
    dataframe: pd.DataFrame,
    reference: dict
):

    production_features = {

        "niveau_diplome":
            dataframe[
                "niveau_diplome"
            ],

        "anciennete_poste_ans":
            dataframe[
                "anciennete_poste_ans"
            ],

        "famille_rome":
            famille_rome(
                dataframe[
                    "code_rome_vise"
                ]
            ),

        "longueur_synthese":
            longueur_texte(
                dataframe[
                    "synthese_entretien"
                ]
            )
    }


    resultats = {}


    for feature_name, series in (
        production_features.items()
    ):

        ref = reference[
            "features"
        ][
            feature_name
        ]


        if ref[
            "type"
        ] == "numeric":

            production_distribution = (
                distribution_numerique(
                    series,
                    ref
                )
            )

        else:

            production_distribution = (
                distribution_categorielle(
                    series,
                    ref
                )
            )


        resultats[
            feature_name
        ] = calculer_psi(
            ref[
                "proportions"
            ],
            production_distribution
        )


    return resultats