"""Fonctions d'analyse exploratoire des données.

Le module contient les outils utilisés dans le notebook pour profiler
les variables, détecter les valeurs atypiques, évaluer différentes
stratégies d'imputation et étudier les associations entre variables.
"""

import re

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)
from sklearn.model_selection import train_test_split

from src.config import FEATURES_DISPONIBLES


# ============================================================
# VALEURS ABERRANTES
# ============================================================


def detecter_valeurs_aberrantes(
    serie: pd.Series,
) -> tuple[float, int]:
    """Détecte les valeurs atypiques avec la règle de Tukey."""

    q1 = serie.quantile(
        0.25
    )

    q3 = serie.quantile(
        0.75
    )

    iqr = q3 - q1

    borne_basse = (
        q1 - 1.5 * iqr
    )

    borne_haute = (
        q3 + 1.5 * iqr
    )

    masque_aberrant = (
        (serie < borne_basse)
        | (serie > borne_haute)
    )

    nombre_valeurs_aberrantes = (
        masque_aberrant.sum()
    )

    return (
        iqr,
        int(
            nombre_valeurs_aberrantes
        ),
    )


# ============================================================
# PROFILAGE
# ============================================================


def profiler(
    df: pd.DataFrame,
    list_to_process=None,
) -> pd.DataFrame:
    """Produit un profil synthétique des variables sélectionnées.

    Le profil contient le type, la cardinalité, les valeurs manquantes,
    les doublons et, lorsque cela est pertinent, l'IQR et le nombre
    de valeurs considérées comme atypiques.
    """

    resultats = []

    features = (
        FEATURES_DISPONIBLES
        if list_to_process is None
        else list_to_process
    )

    for feature in features:
        serie = df[
            feature
        ]

        # unique() conserve les NaN, contrairement à nunique()
        # utilisé avec son comportement par défaut.
        cardinalite = len(
            serie.unique()
        )

        nombre_manquants = (
            serie.isna().sum()
        )

        # Un code INSEE ne s'analyse pas avec l'IQR même lorsqu'il
        # est représenté numériquement : son format est contrôlé.
        if feature == "code_insee_commune":
            serie_normalisee = (
                serie
                .astype("string")
                .str.strip()
                .str.upper()
            )

            format_valide = (
                serie_normalisee
                .str.fullmatch(
                    r"(?:\d{5}|2[AB]\d{3})",
                    na=False,
                )
            )

            iqr = "N/A"

            nombre_aberrantes = (
                ~format_valide
            ).sum()

        elif pd.api.types.is_numeric_dtype(
            serie
        ):
            (
                iqr,
                nombre_aberrantes,
            ) = detecter_valeurs_aberrantes(
                serie
            )

        else:
            iqr = "N/A"
            nombre_aberrantes = "N/A"

        # duplicated() considère la première occurrence
        # comme la valeur de référence et non comme un doublon.
        nombre_doublons = (
            serie.duplicated().sum()
        )

        resultats.append(
            {
                "Feature": feature,
                "Type": str(
                    serie.dtype
                ),
                "Cardinalité": cardinalite,
                "Valeurs manquantes": nombre_manquants,
                "IQR": iqr,
                "Valeurs aberrantes selon l'IQR": (
                    nombre_aberrantes
                ),
                "Valeurs dupliquées": nombre_doublons,
            }
        )

    return pd.DataFrame(
        resultats
    )


# ============================================================
# VALIDATION D'IMPUTATION PAR GROUPES
# ============================================================


def validation_mae(
    df: pd.DataFrame,
    colonnes_groupe: list[str],
    feature_name: str,
    nombre_iterations: int = 10,
    test_size: float = 0.20,
) -> pd.DataFrame:
    """Évalue une imputation par médiane conditionnelle.

    Les valeurs connues sont séparées plusieurs fois entre un jeu
    de référence et un jeu de validation. Les médianes sont calculées
    uniquement sur le jeu de référence afin d'éviter toute fuite
    d'information vers l'évaluation.
    """

    resultats = []

    feature_name_predict = (
        f"{feature_name}_predit"
    )

    # Seules les valeurs réellement connues peuvent servir
    # à mesurer l'erreur de la stratégie d'imputation.
    df_valeurs_connues = df.loc[
        df[
            feature_name
        ].notna()
    ].copy()

    for seed in range(
        nombre_iterations
    ):
        df_reference, df_validation = train_test_split(
            df_valeurs_connues,
            test_size=test_size,
            random_state=seed,
        )

        medianes_groupes = (
            df_reference
            .groupby(
                colonnes_groupe,
                dropna=False,
                observed=True,
            )[
                feature_name
            ]
            .median()
            .rename(
                feature_name_predict
            )
            .reset_index()
        )

        validation_predite = (
            df_validation.merge(
                medianes_groupes,
                on=colonnes_groupe,
                how="left",
            )
        )

        # Un groupe absent du jeu de référence utilise
        # la médiane globale comme stratégie de repli.
        validation_predite[
            feature_name_predict
        ] = (
            validation_predite[
                feature_name_predict
            ]
            .fillna(
                df_reference[
                    feature_name
                ].median()
            )
            .round()
        )

        valeurs_reelles = (
            validation_predite[
                feature_name
            ]
        )

        valeurs_predites = (
            validation_predite[
                feature_name_predict
            ]
        )

        erreurs_absolues = (
            valeurs_reelles
            - valeurs_predites
        ).abs()

        resultats.append(
            {
                "seed": seed,
                "MAE": mean_absolute_error(
                    valeurs_reelles,
                    valeurs_predites,
                ),
                "Erreur_mediane": (
                    median_absolute_error(
                        valeurs_reelles,
                        valeurs_predites,
                    )
                ),
                "RMSE": np.sqrt(
                    mean_squared_error(
                        valeurs_reelles,
                        valeurs_predites,
                    )
                ),
                f"{feature_name}_exact_%": (
                    erreurs_absolues == 0
                ).mean() * 100,
                "Erreur_de_2_unites_%": (
                    erreurs_absolues <= 2
                ).mean() * 100,
                "Erreur_de_5_unites_%": (
                    erreurs_absolues <= 5
                ).mean() * 100,
                "Erreur_de_10_unites_%": (
                    erreurs_absolues <= 10
                ).mean() * 100,
            }
        )

    return pd.DataFrame(
        resultats
    )


# ============================================================
# VALIDATION D'IMPUTATION PAR MEDIANE GLOBALE
# ============================================================


def validation_mediane_globale(
    df: pd.DataFrame,
    feature_name: str,
    nombre_iterations: int = 10,
    test_size: float = 0.20,
) -> pd.DataFrame:
    """Évalue une stratégie d'imputation par médiane globale."""

    df_valeurs_connues = df.loc[
        df[
            feature_name
        ].notna()
    ].copy()

    resultats = []

    for seed in range(
        nombre_iterations
    ):
        df_reference, df_validation = train_test_split(
            df_valeurs_connues,
            test_size=test_size,
            random_state=seed,
        )

        # La médiane est calculée exclusivement sur le jeu
        # de référence pour éviter la fuite de données.
        mediane_globale = (
            df_reference[
                feature_name
            ].median()
        )

        valeurs_predites = pd.Series(
            mediane_globale,
            index=df_validation.index,
        ).round()

        valeurs_reelles = (
            df_validation[
                feature_name
            ]
        )

        erreurs_absolues = (
            valeurs_reelles
            - valeurs_predites
        ).abs()

        resultats.append(
            {
                "seed": seed,
                "Mediane_globale": mediane_globale,
                "MAE": mean_absolute_error(
                    valeurs_reelles,
                    valeurs_predites,
                ),
                "Erreur_mediane": (
                    median_absolute_error(
                        valeurs_reelles,
                        valeurs_predites,
                    )
                ),
                "RMSE": np.sqrt(
                    mean_squared_error(
                        valeurs_reelles,
                        valeurs_predites,
                    )
                ),
                f"{feature_name}_exact_%": (
                    erreurs_absolues == 0
                ).mean() * 100,
                "Erreur_de_2_unites_%": (
                    erreurs_absolues <= 2
                ).mean() * 100,
                "Erreur_de_5_unites_%": (
                    erreurs_absolues <= 5
                ).mean() * 100,
                "Erreur_de_10_unites_%": (
                    erreurs_absolues <= 10
                ).mean() * 100,
            }
        )

    return pd.DataFrame(
        resultats
    )


# ============================================================
# TRANSFORMATION GEOGRAPHIQUE
# ============================================================


def extraire_departement(
    code_insee,
):
    """Extrait le département à partir d'un code INSEE communal.

    Les départements métropolitains utilisent deux caractères,
    la Corse conserve 2A ou 2B et les codes 97/98 utilisent
    trois caractères.
    """

    if pd.isna(
        code_insee
    ):
        return pd.NA

    code = str(
        code_insee
    ).strip().upper()

    format_valide = re.fullmatch(
        r"(?:\d{5}|2[AB]\d{3})",
        code,
    )

    if format_valide is None:
        return pd.NA

    if code.startswith(
        (
            "97",
            "98",
        )
    ):
        return code[:3]

    return code[:2]


# ============================================================
# ASSOCIATION ENTRE VARIABLES QUALITATIVES
# ============================================================


def cramers_v(
    x,
    y,
) -> float:
    """Calcule le V de Cramér entre deux variables qualitatives.

    Une valeur proche de 0 indique une faible association et une
    valeur proche de 1 une association forte.
    """

    table = pd.crosstab(
        x,
        y,
    )

    chi2 = chi2_contingency(
        table
    )[0]

    n = (
        table
        .to_numpy()
        .sum()
    )

    r, k = table.shape

    denominateur = min(
        r - 1,
        k - 1,
    )

    if denominateur == 0:
        return 0.0

    return float(
        np.sqrt(
            chi2
            / (
                n
                * denominateur
            )
        )
    )