"""Fonctions d'analyse exploratoire et de validation des imputations.

Ce module fournit des fonctions permettant :

- de détecter les valeurs aberrantes d'une série numérique avec la méthode
  de l'écart interquartile (IQR) ;
- de produire un profil synthétique des variables d'un DataFrame ;
- d'évaluer une méthode d'imputation conditionnelle par masquage artificiel
  et calcul de métriques d'erreur.

Fonctions
---------
detecter_valeurs_aberrantes
    Calcule l'IQR et dénombre les valeurs situées en dehors des bornes
    définies par la règle de Tukey.

profiler
    Produit un tableau de profilage contenant notamment les types,
    cardinalités, valeurs manquantes, doublons et valeurs aberrantes.

validation_mae
    Évalue une imputation par médiane de groupe sur plusieurs découpages
    apprentissage-validation.
"""

import re
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)
from sklearn.model_selection import train_test_split

from scipy.stats import chi2_contingency

from .config import FEATURES_DISPONIBLES


def detecter_valeurs_aberrantes(
    serie: pd.Series,
) -> tuple[float, int]:
    """Détecter les valeurs aberrantes d'une série numérique avec l'IQR.

    La méthode utilise la règle de Tukey :

    - borne basse = Q1 - 1,5 × IQR ;
    - borne haute = Q3 + 1,5 × IQR.

    Une valeur située en dehors de ces bornes est considérée comme
    statistiquement atypique. Elle n'est toutefois pas nécessairement
    erronée au sens métier.

    Parameters
    ----------
    serie : pd.Series
        Série numérique à analyser.

    Returns
    -------
    tuple[float, int]
        Un tuple contenant :

        - la valeur de l'écart interquartile ;
        - le nombre de valeurs détectées comme aberrantes.
    """

    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)

    iqr = q3 - q1

    borne_basse = q1 - 1.5 * iqr
    borne_haute = q3 + 1.5 * iqr

    masque_aberrant = (
        (serie < borne_basse)
        | (serie > borne_haute)
    )

    nombre_valeurs_aberrantes = masque_aberrant.sum()

    return iqr, int(nombre_valeurs_aberrantes)


def profiler(df: pd.DataFrame, list_to_process = None) -> pd.DataFrame:
    """Produire un profil synthétique des variables du DataFrame.

    Pour chaque variable déclarée dans le paramètre ``list_to_process`` (ou à defaut ``FEATURES_DISPONIBLES``), la fonction
    calcule :

    - le type de données ;
    - la cardinalité, valeurs manquantes comprises ;
    - le nombre de valeurs manquantes ;
    - l'écart interquartile pour les variables numériques ;
    - le nombre de valeurs aberrantes détectées avec l'IQR ; Traitement particulier pour code_insee_commune => vérification format INSEE
    - le nombre de valeurs dupliquées dans la colonne.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à profiler.

    Returns
    -------
    pd.DataFrame
        Tableau récapitulatif contenant une ligne par variable.

    Raises
    ------
    KeyError
        Si une variable de ``FEATURES_DISPONIBLES`` est absente du DataFrame.
    """

    resultats = []

    if list_to_process is None:
        features = FEATURES_DISPONIBLES
    else:
        features = list_to_process

    for feature in features:
        serie = df[feature]

        # La méthode unique() conserve les valeurs NaN dans la cardinalité.
        cardinalite = len(serie.unique())

        # Comptage des valeurs absentes dans la colonne.
        nombre_manquants = serie.isna().sum()

        # L'IQR est pertinent uniquement pour les variables numériques.
        # Traitement du cas particulier code_insee_commune dont le format est standardisé 5 digits ou "2A" suivi de 3 digits ou "2B" suivi de 3 digits
        if (feature == "code_insee_commune"):
            serie_normalisee = (
                serie
                .astype("string")
                .str.strip()
                .str.upper()
            )

            format_valide = serie_normalisee.str.fullmatch(
                r"(?:\d{5}|2[AB]\d{3})",
                na=False
            )

            iqr = "N/A"
            nombre_aberrantes = (~format_valide).sum()
        elif pd.api.types.is_numeric_dtype(serie):
            iqr, nombre_aberrantes = detecter_valeurs_aberrantes(
                serie
            )
        else:
            iqr = "N/A"
            nombre_aberrantes = "N/A"

        # duplicated() considère la première occurrence comme non dupliquée.
        nombre_doublons = serie.duplicated().sum()

        resultats.append({
            "Feature": feature,
            "Type": str(serie.dtype),
            "Cardinalité": cardinalite,
            "Valeurs manquantes": nombre_manquants,
            "IQR": iqr,
            "Valeurs aberrantes selon l'IQR": nombre_aberrantes,
            "Valeurs dupliquées": nombre_doublons,
        })

    return pd.DataFrame(resultats)


def validation_mae(
    df: pd.DataFrame,
    colonnes_groupe: list[str],
    feature_name: str,
    nombre_iterations: int = 10,
    test_size: float = 0.20,
) -> pd.DataFrame:
    """Évaluer une imputation par médiane de groupe.

    La fonction réalise plusieurs découpages aléatoires des observations
    dont la valeur de la feature est connue.

    Pour chaque itération :

    1. un jeu de référence et un jeu de validation sont créés ;
    2. les médianes sont calculées uniquement sur le jeu de référence ;
    3. ces médianes sont appliquées aux lignes du jeu de validation ;
    4. la médiane globale du jeu de référence est utilisée en dernier recours ;
    5. plusieurs métriques comparent les valeurs réelles et prédites.

    Cette séparation évite qu'une valeur réelle participe au calcul de la
    médiane utilisée pour prédire cette même valeur.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant la variable à valider et les variables de groupe.

    colonnes_groupe : list[str]
        Colonnes utilisées pour constituer les groupes d'imputation.

    feature_name : str
        Nom de la variable numérique à prédire et à valider.

    nombre_iterations : int, default=10
        Nombre de découpages aléatoires à effectuer.

    test_size : float, default=0.20
        Proportion des données utilisée pour la validation à chaque itération.

    Returns
    -------
    pd.DataFrame
        Résultats détaillés de chaque itération contenant notamment :

        - la MAE ;
        - l'erreur médiane ;
        - la RMSE ;
        - le pourcentage de prédictions exactes ;
        - les pourcentages d'erreurs inférieures à différents seuils.
    """

    resultats = []

    # Nom de la colonne contenant les valeurs estimées.
    feature_name_predict = f"{feature_name}_predit"

    # La validation nécessite uniquement les lignes dont la valeur est connue.
    df_valeurs_connues = df.loc[
        df[feature_name].notna()
    ].copy()

    for seed in range(nombre_iterations):

        # Le jeu de référence sert au calcul des statistiques d'imputation.
        # Le jeu de validation simule les valeurs qui seraient manquantes.
        df_reference, df_validation = train_test_split(
            df_valeurs_connues,
            test_size=test_size,
            random_state=seed,
        )

        # Calcul des médianes uniquement à partir du jeu de référence.
        medianes_groupes = (
            df_reference
            .groupby(
                colonnes_groupe,
                dropna=False,
                observed=True,
            )[feature_name]
            .median()
            .rename(feature_name_predict)
            .reset_index()
        )

        # Association de chaque ligne de validation à la médiane de son groupe.
        validation_predite = df_validation.merge(
            medianes_groupes,
            on=colonnes_groupe,
            how="left",
        )

        # Lorsque le groupe n'existe pas dans le jeu de référence,
        # utilisation de la médiane globale comme valeur de repli.
        validation_predite[feature_name_predict] = (
            validation_predite[feature_name_predict]
            .fillna(df_reference[feature_name].median())
            .round()
        )

        valeurs_reelles = validation_predite[feature_name]
        valeurs_predites = validation_predite[
            feature_name_predict
        ]

        erreurs_absolues = (
            valeurs_reelles - valeurs_predites
        ).abs()

        resultats.append({
            "seed": seed,
            "MAE": mean_absolute_error(
                valeurs_reelles,
                valeurs_predites,
            ),
            "Erreur_mediane": median_absolute_error(
                valeurs_reelles,
                valeurs_predites,
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
        })

    return pd.DataFrame(resultats)


def validation_mediane_globale(
    df: pd.DataFrame,
    feature_name: str,
    nombre_iterations: int = 10,
    test_size: float = 0.20
) -> pd.DataFrame:
    """Évaluer une imputation par médiane globale.

    À chaque itération, les données connues sont séparées en un jeu de
    référence et un jeu de validation. La médiane est calculée uniquement
    sur le jeu de référence, puis utilisée pour prédire toutes les valeurs
    du jeu de validation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant la variable à évaluer.

    feature_name : str
        Nom de la variable numérique à imputer.

    nombre_iterations : int, default=10
        Nombre de découpages aléatoires réalisés.

    test_size : float, default=0.20
        Proportion des données utilisée pour la validation.

    Returns
    -------
    pd.DataFrame
        Métriques obtenues pour chaque itération.
    """

    # Conserver uniquement les valeurs connues pour pouvoir
    # comparer les valeurs prédites aux valeurs réelles.
    df_valeurs_connues = df.loc[
        df[feature_name].notna()
    ].copy()

    resultats = []

    for seed in range(nombre_iterations):

        df_reference, df_validation = train_test_split(
            df_valeurs_connues,
            test_size=test_size,
            random_state=seed
        )

        # Calcul exclusivement sur le jeu de référence.
        mediane_globale = df_reference[feature_name].median()

        # Toutes les lignes de validation reçoivent la médiane globale.
        valeurs_predites = pd.Series(
            mediane_globale,
            index=df_validation.index
        ).round()

        valeurs_reelles = df_validation[feature_name]

        erreurs_absolues = (
            valeurs_reelles - valeurs_predites
        ).abs()

        resultats.append({
            "seed": seed,
            "Mediane_globale": mediane_globale,
            "MAE": mean_absolute_error(
                valeurs_reelles,
                valeurs_predites
            ),
            "Erreur_mediane": median_absolute_error(
                valeurs_reelles,
                valeurs_predites
            ),
            "RMSE": np.sqrt(
                mean_squared_error(
                    valeurs_reelles,
                    valeurs_predites
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
            ).mean() * 100
        })

    return pd.DataFrame(resultats)

def extraire_departement(code_insee):
    """
    Extraire le département à partir d'un code INSEE communal.

    Règles appliquées :
    - Métropole : les deux premiers caractères ;
    - Corse : conservation de `2A` ou `2B` ;
    - Outre-mer : les trois premiers caractères pour les codes
      commençant par `97` ou `98`.

    Exemples
    --------
    83000  -> 83
    2A004  -> 2A
    2B123  -> 2B
    97105  -> 971
    97411  -> 974

    Parameters
    ----------
    code_insee : str
        Code INSEE communal à cinq caractères.

    Returns
    -------
    str ou pd.NA
        Code du département extrait, ou `pd.NA` lorsque le code
        est manquant ou ne respecte pas le format attendu.
    """

    # Une valeur absente ne permet pas d'extraire un département.
    if pd.isna(code_insee):
        return pd.NA

    # Normalisation du code avant son contrôle.
    code = str(code_insee).strip().upper()

    # Vérification du format :
    # - cinq chiffres ;
    # - ou code corse commençant par 2A ou 2B.
    format_valide = re.fullmatch(
        r"(?:\d{5}|2[AB]\d{3})",
        code
    )

    if format_valide is None:
        return pd.NA

    # Les départements et collectivités ultramarins
    # sont identifiés par trois caractères.
    if code.startswith(("97", "98")):
        return code[:3]

    # Pour les départements métropolitains,
    # les deux premiers chiffres suffisent.
    return code[:2]

def cramers_v(x, y):
    """
    Calcule le V de Cramér entre deux variables qualitatives.

    Retourne une valeur comprise entre 0 et 1 :
    - 0   : aucune association
    - 1   : association parfaite
    """

    # Tableau de contingence
    table = pd.crosstab(x, y)

    # Test du Chi²
    chi2 = chi2_contingency(table)[0]

    # Nombre total d'observations
    n = table.to_numpy().sum()

    # Dimensions du tableau
    r, k = table.shape

    # Éviter une division par zéro
    denominateur = min(r - 1, k - 1)

    if denominateur == 0:
        return 0.0

    # Calcul du V de Cramér
    v = np.sqrt(
        chi2 / (n * denominateur)
    )

    return v