import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer
)
from sklearn.feature_extraction.text import TfidfVectorizer
from src.eda import extraire_departement

# ============================================================
# Fonctions de transformation
# ============================================================

def extraire_departement_dataframe(X):
    """
    Transforme code_insee_commune en département.

    La fonction reçoit un DataFrame contenant une seule colonne
    et retourne un DataFrame afin de rester compatible avec
    ColumnTransformer / OneHotEncoder.
    """

    serie = X.iloc[:, 0]

    departements = serie.apply(
        extraire_departement
    )

    return pd.DataFrame(
        {"departement": departements},
        index=X.index
    )


def preparer_texte(X):
    """
    Prépare la colonne textuelle pour TfidfVectorizer.

    TfidfVectorizer attend une série 1D de chaînes de caractères,
    alors que ColumnTransformer transmet un tableau 2D.
    """

    return (
        X.iloc[:, 0]
        .fillna("Non renseigné")
        .astype(str)
    )

def convertir_en_str(X):
    return X.astype(str)

def make_preprocessor(
    age_feature=None,
    num_features=None,
    cat_features=None,
    geo_feature=None,
    text_feature=None,
    text_pipe_custom=None
):
    """
    Instancie la même recette de preprocessing sur le périmètre
    de données demandé par chaque scénario.

    Chaque bloc est optionnel : un scénario peut donc supprimer
    une feature ou une modalité entière (texte, géographie, etc.)
    sans modifier la logique de traitement.
    """

    blocks = []

    # Age traité séparément pour créer automatiquement
    # l'indicateur age_manquant.
    if age_feature:
        blocks.append(
            (
                "age",
                age_pipe,
                [age_feature]
            )
        )

    # Variables quantitatives classiques
    if num_features:
        blocks.append(
            (
                "num",
                num_pipe,
                num_features
            )
        )

    # Variables catégorielles
    if cat_features:
        blocks.append(
            (
                "cat",
                cat_pipe,
                cat_features
            )
        )

    # Géographie : commune -> département
    if geo_feature:
        blocks.append(
            (
                "geo",
                geo_pipe,
                [geo_feature]
            )
        )

    # Texte libre
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
                [text_feature]
            )
        )       

    assert blocks, (
        "Aucune feature sélectionnée : "
        "vérifier le périmètre du scénario."
    )

    return ColumnTransformer(
        transformers=blocks,
        remainder="drop"
    )


# ============================================================
# Pipelines élémentaires
# ============================================================

# ------------------------------------------------------------
# AGE
#
# Décision EDA :
# - imputation par médiane
# - conservation de l'information de valeur manquante
#
# add_indicator=True crée automatiquement une colonne binaire
# indiquant si l'âge était initialement manquant.
# ------------------------------------------------------------

age_pipe = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median",
            add_indicator=True
        )
    ),
    (
        "scaler",
        StandardScaler()
    )
])


# ------------------------------------------------------------
# VARIABLES NUMÉRIQUES
#
# Exemple : anciennete_poste_ans
#
# Aucune imputation nécessaire actuellement, mais on conserve
# une recette robuste à d'éventuelles valeurs manquantes futures.
# ------------------------------------------------------------

num_pipe = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),
    (
        "scaler",
        StandardScaler()
    )
])


# ------------------------------------------------------------
# VARIABLES CATÉGORIELLES
#
# Décision EDA :
# - ne pas attribuer arbitrairement la modalité majoritaire
# - conserver explicitement l'absence d'information
# ------------------------------------------------------------

cat_pipe = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="constant",
            fill_value="Non renseigné"
        )
    ),
    (
        "to_string",
        FunctionTransformer(
            convertir_en_str, # conversion en string (est_allocataire uniquement => binaire 0 ou 1 avec modalité 'Non renseigné')
            feature_names_out="one-to-one"
        )
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])


# ------------------------------------------------------------
# GÉOGRAPHIE
#
# code_insee_commune n'est pas utilisé directement :
# commune -> département -> OneHotEncoder
# ------------------------------------------------------------

geo_pipe = Pipeline([
    (
        "departement",
        FunctionTransformer(
            extraire_departement_dataframe,
            validate=False
        )
    ),
    (
        "imputer",
        SimpleImputer(
            strategy="constant",
            fill_value="Non renseigné"
        )
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])


# ------------------------------------------------------------
# TEXTE
#
# Décision EDA :
# - NaN -> "Non renseigné"
# - vectorisation TF-IDF
# - unigrammes + bigrammes
# ------------------------------------------------------------

text_pipe = Pipeline([
    (
        "prepare_text",
        FunctionTransformer(
            preparer_texte,
            validate=False
        )
    ),
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2)
        )
    )
])


# ------------------------------------------------------------
# CAS PARTICULIER  Cadrage particulier pour le scénario 2 pour la feature synthese_entretien 
#
# ------------------------------------------------------------

REMPLACEMENTS_ETHIQUES = {
    "Cumul de difficultés. Pas de moyen de transport et garde d'enfants complexe.":
        "Cumul de difficultés logistiques pouvant limiter la disponibilité et la mobilité.",

    "Perte de confiance importante. Risque d'exclusion, barrière de la langue.":
        "Besoin d'accompagnement renforcé pour la remobilisation et la communication professionnelle.",

    "Freins périphériques majeurs. Situation d'illettrisme numérique constatée.":
        "Freins périphériques majeurs. Difficultés importantes dans l'utilisation des outils numériques.",

    "Problème ponctuel de mobilité géographique. Zone mal desservie.":
        "Difficulté ponctuelle de mobilité nécessitant un accompagnement."
}

def nettoyer_texte_ethique(X):
    """
    Nettoyage éthique de synthese_entretien pour le scénario 2.

    - remplace les formulations identifiées comme à risque
    - conserve les autres formulations
    - remplace les valeurs manquantes par 'Non renseigné'
    - retourne une série 1D compatible avec TfidfVectorizer
    """

    serie = (
        X.iloc[:, 0]
        .fillna("Non renseigné")
        .astype(str)
    )

    return serie.replace(REMPLACEMENTS_ETHIQUES)

text_pipe_ethique = Pipeline([
    (
        "nettoyage_ethique",
        FunctionTransformer(
            nettoyer_texte_ethique,
            validate=False
        )
    ),
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2)
        )
    )
])


preprocessor_s1 = make_preprocessor(
    age_feature="age",

    num_features=[
        "anciennete_poste_ans",
        "nationalite_hors_ue"
    ],

    cat_features=[
        "niveau_diplome",
        "code_rome_vise",
        "est_allocataire"
    ],

    geo_feature="code_insee_commune",

    text_feature="synthese_entretien"
)

preprocessor_s2 = make_preprocessor(
    cat_features=[
        "niveau_diplome",
        "code_rome_vise",
    ],
 
    num_features=[
        "anciennete_poste_ans"
    ],

    text_feature="synthese_entretien"
)

preprocessor_s3 = make_preprocessor(
    text_feature="synthese_entretien",
    text_pipe_custom=text_pipe_ethique
)

preprocessor_s4 = make_preprocessor(
    age_feature="age",

    num_features=[
        "anciennete_poste_ans"
    ],

    cat_features=[
        "niveau_diplome",
    ],

    geo_feature="code_insee_commune"
)