"""Préprocesseurs utilisés par les différents scénarios de modélisation.

Le module centralise les transformations numériques, catégorielles,
géographiques et textuelles afin que l'entraînement et l'inférence
utilisent exactement les mêmes traitements.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

from src.eda import extraire_departement


# ============================================================
# FONCTIONS DE TRANSFORMATION
# ============================================================


def extraire_departement_dataframe(
    X,
) -> pd.DataFrame:
    """Transforme un code INSEE de commune en département.

    ColumnTransformer transmet ici un DataFrame à une colonne.
    Un DataFrame est également retourné afin de rester compatible
    avec les étapes suivantes du pipeline.
    """

    serie = X.iloc[:, 0]

    departements = serie.apply(
        extraire_departement
    )

    return pd.DataFrame(
        {
            "departement": departements,
        },
        index=X.index,
    )


def preparer_texte(
    X,
) -> pd.Series:
    """Prépare une colonne textuelle pour TfidfVectorizer."""

    # TfidfVectorizer attend une structure 1D alors que
    # ColumnTransformer transmet ici une structure 2D.
    return (
        X.iloc[:, 0]
        .fillna("Non renseigné")
        .astype(str)
    )


def convertir_en_str(X):
    """Convertit les valeurs en chaînes pour l'encodage catégoriel."""

    return X.astype(str)


# ============================================================
# PIPELINES ELEMENTAIRES
# ============================================================

# L'âge est traité séparément afin de conserver une information
# indiquant si la valeur était initialement manquante.
age_pipe = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


# Les variables numériques sont imputées par médiane avant
# standardisation afin de rester robustes aux valeurs manquantes.
num_pipe = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


# L'absence d'une valeur catégorielle est conservée comme
# une modalité explicite plutôt que remplacée par la majorité.
cat_pipe = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Non renseigné",
            ),
        ),
        (
            "to_string",
            FunctionTransformer(
                convertir_en_str,
                feature_names_out="one-to-one",
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
        ),
    ]
)


# La commune n'est pas utilisée directement : elle est réduite
# au département afin de limiter la cardinalité géographique.
geo_pipe = Pipeline(
    [
        (
            "departement",
            FunctionTransformer(
                extraire_departement_dataframe,
                validate=False,
            ),
        ),
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Non renseigné",
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
        ),
    ]
)


# Le texte libre est représenté par des unigrammes et bigrammes TF-IDF.
text_pipe = Pipeline(
    [
        (
            "prepare_text",
            FunctionTransformer(
                preparer_texte,
                validate=False,
            ),
        ),
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
            ),
        ),
    ]
)


# ============================================================
# NETTOYAGE ETHIQUE DU TEXTE
# ============================================================

REMPLACEMENTS_ETHIQUES = {
    (
        "Cumul de difficultés. Pas de moyen de transport "
        "et garde d'enfants complexe."
    ): (
        "Cumul de difficultés logistiques pouvant limiter "
        "la disponibilité et la mobilité."
    ),
    (
        "Perte de confiance importante. Risque d'exclusion, "
        "barrière de la langue."
    ): (
        "Besoin d'accompagnement renforcé pour la remobilisation "
        "et la communication professionnelle."
    ),
    (
        "Freins périphériques majeurs. Situation "
        "d'illettrisme numérique constatée."
    ): (
        "Freins périphériques majeurs. Difficultés importantes "
        "dans l'utilisation des outils numériques."
    ),
    (
        "Problème ponctuel de mobilité géographique. "
        "Zone mal desservie."
    ): (
        "Difficulté ponctuelle de mobilité nécessitant "
        "un accompagnement."
    ),
}


def nettoyer_texte_ethique(
    X,
) -> pd.Series:
    """Prépare la synthèse d'entretien avec les remplacements éthiques.

    Les formulations identifiées lors du cadrage sont remplacées par
    des formulations métier moins sensibles tout en conservant
    l'information considérée comme pertinente pour la modélisation.
    """

    serie = (
        X.iloc[:, 0]
        .fillna("Non renseigné")
        .astype(str)
    )

    return serie.replace(
        REMPLACEMENTS_ETHIQUES
    )


text_pipe_ethique = Pipeline(
    [
        (
            "nettoyage_ethique",
            FunctionTransformer(
                nettoyer_texte_ethique,
                validate=False,
            ),
        ),
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
            ),
        ),
    ]
)


# ============================================================
# CONSTRUCTION DES PREPROCESSEURS
# ============================================================


def make_preprocessor(
    age_feature=None,
    num_features=None,
    cat_features=None,
    geo_feature=None,
    text_feature=None,
    text_pipe_custom=None,
) -> ColumnTransformer:
    """Construit le préprocesseur correspondant à un scénario.

    Chaque famille de variables est optionnelle afin que les différents
    scénarios puissent utiliser la même recette tout en sélectionnant
    des périmètres de données différents.
    """

    blocks = []

    if age_feature:
        blocks.append(
            (
                "age",
                age_pipe,
                [age_feature],
            )
        )

    if num_features:
        blocks.append(
            (
                "num",
                num_pipe,
                num_features,
            )
        )

    if cat_features:
        blocks.append(
            (
                "cat",
                cat_pipe,
                cat_features,
            )
        )

    if geo_feature:
        blocks.append(
            (
                "geo",
                geo_pipe,
                [geo_feature],
            )
        )

    if text_feature:
        pipeline_texte = (
            text_pipe_custom
            if text_pipe_custom is not None
            else text_pipe
        )

        blocks.append(
            (
                "text",
                pipeline_texte,
                [text_feature],
            )
        )

    assert blocks, (
        "Aucune feature sélectionnée : "
        "vérifier le périmètre du scénario."
    )

    return ColumnTransformer(
        transformers=blocks,
        remainder="drop",
    )


# ============================================================
# SCENARIOS
# ============================================================

preprocessor_s1 = make_preprocessor(
    age_feature="age",
    num_features=[
        "anciennete_poste_ans",
        "nationalite_hors_ue",
    ],
    cat_features=[
        "niveau_diplome",
        "code_rome_vise",
        "est_allocataire",
    ],
    geo_feature="code_insee_commune",
    text_feature="synthese_entretien",
)


preprocessor_s2 = make_preprocessor(
    cat_features=[
        "niveau_diplome",
        "code_rome_vise",
    ],
    num_features=[
        "anciennete_poste_ans",
    ],
    text_feature="synthese_entretien",
    text_pipe_custom=text_pipe_ethique,
)


preprocessor_s3 = make_preprocessor(
    text_feature="synthese_entretien",
)

preprocessor_s4 = make_preprocessor(
    age_feature="age",
    num_features=[
        "anciennete_poste_ans",
    ],
    cat_features=[
        "niveau_diplome",
    ],
    geo_feature="code_insee_commune",
)