"""Interface Streamlit du service Retour Emploi.

L'interface permet de demander une prédiction à l'API FastAPI,
d'enregistrer la vérité terrain associée à une prédiction et
de consulter l'historique des évaluations.
"""

import os

import pandas as pd
import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
)

CLASS_LABELS = {
    0: "Retour rapide — moins de 6 mois",
    1: "Retour moyen — entre 6 et 12 mois",
    2: "Risque de longue durée — au-delà de 12 mois",
}


# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================

# set_page_config doit être exécuté avant les autres commandes Streamlit.
st.set_page_config(
    page_title="Retour à l'emploi",
    page_icon="📊",
    layout="wide",
)

st.title(
    "Aide à l'évaluation du retour à l'emploi"
)

st.caption(
    "Prototype pédagogique d'aide à la décision"
)

st.info(
    """
    **Interprétation des classes :**

    - **Retour rapide** (classe 0) : moins de 6 mois
    - **Retour moyen** (classe 1) : entre 6 et 12 mois
    - **Risque de longue durée** (classe 2) : au-delà de 12 mois
    """
)


# ============================================================
# OUTILS INTERNES
# ============================================================


def extraire_detail_erreur(
    response: requests.Response,
    exception: Exception,
) -> str:
    """Extrait le message d'erreur retourné par l'API."""

    try:
        return str(
            response.json().get(
                "detail",
                str(exception),
            )
        )
    except Exception:
        return str(exception)


# ============================================================
# ONGLETS
# ============================================================

tab_prediction, tab_feedback, tab_history = st.tabs(
    [
        "Nouvelle évaluation",
        "Résultat réel",
        "Historique",
    ]
)


# ============================================================
# ONGLET 1 - PREDICTION
# ============================================================

with tab_prediction:
    st.subheader(
        "Nouvelle évaluation"
    )

    st.write(
        "Renseignez les informations nécessaires "
        "pour obtenir une estimation du retour à l'emploi."
    )

    with st.form(
        "prediction_form"
    ):
        session_id = st.text_input(
            "Identifiant de session",
            placeholder="Exemple : session-001",
        )

        niveau_diplome = st.selectbox(
            "Niveau de diplôme",
            [
                "Sans diplôme",
                "Bac",
                "Bac+2",
                "Bac+5",
                "Non renseigné",
            ],
        )

        anciennete_poste_ans = st.number_input(
            "Ancienneté dans le poste précédent (années)",
            min_value=0.0,
            max_value=60.0,
            value=2.0,
            step=0.5,
        )

        code_rome_vise = st.text_input(
            "Code ROME visé",
            value="M1805",
            max_chars=5,
        )

        synthese_entretien = st.text_area(
            "Synthèse de l'entretien",
            height=160,
            placeholder=(
                "Exemple : Profil autonome, "
                "recherche active, aucun frein périphérique identifié."
            ),
        )

        submit_prediction = st.form_submit_button(
            "Calculer la prédiction"
        )

    if submit_prediction:
        payload = {
            "session_id": session_id,
            "niveau_diplome": niveau_diplome,
            "anciennete_poste_ans": anciennete_poste_ans,
            "code_rome_vise": code_rome_vise,
            "synthese_entretien": synthese_entretien,
        }

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=10,
            )

            response.raise_for_status()

            result = response.json()

            prediction = result.get(
                "prediction"
            )

            prediction_id = result.get(
                "prediction_id"
            )

            model_version = result.get(
                "model_version",
                "inconnue",
            )

            probabilities = result.get(
                "probabilities",
                {},
            )

            prediction_label = CLASS_LABELS.get(
                prediction,
                f"Classe inconnue ({prediction})",
            )

            st.success(
                f"Résultat : {prediction_label}"
            )

            st.info(
                f"Identifiant de prédiction : {prediction_id}"
            )

            st.caption(
                f"Version du modèle : {model_version}"
            )

            if probabilities:
                st.subheader(
                    "Probabilités par classe"
                )

                df_probabilities = pd.DataFrame(
                    [
                        {
                            "Classe": CLASS_LABELS.get(
                                int(classe),
                                str(classe),
                            ),
                            "Probabilité": float(
                                probabilite
                            ),
                        }
                        for classe, probabilite
                        in probabilities.items()
                    ]
                )

                df_probabilities = (
                    df_probabilities
                    .sort_values(
                        by="Classe"
                    )
                    .reset_index(
                        drop=True
                    )
                )

                st.dataframe(
                    df_probabilities,
                    use_container_width=True,
                    hide_index=True,
                )

                st.bar_chart(
                    df_probabilities.set_index(
                        "Classe"
                    )
                )

        except requests.exceptions.HTTPError as exc:
            detail = extraire_detail_erreur(
                response,
                exc,
            )

            st.error(
                f"Erreur API : {detail}"
            )

        except requests.exceptions.RequestException as exc:
            st.error(
                f"Impossible de contacter l'API : {exc}"
            )


# ============================================================
# ONGLET 2 - FEEDBACK / VERITE TERRAIN
# ============================================================

with tab_feedback:
    st.subheader(
        "Enregistrer le résultat réellement observé"
    )

    st.write(
        "Lorsque la situation réelle de l'usager est connue, "
        "elle peut être rattachée à la prédiction initiale."
    )

    st.info(
        "Cette information servira à mesurer les performances "
        "réelles du modèle et à enrichir les données "
        "pour les futurs réentraînements."
    )

    with st.form(
        "feedback_form"
    ):
        prediction_id_feedback = st.text_input(
            "Identifiant de prédiction",
            placeholder="UUID retourné lors de la prédiction",
        )

        actual_class = st.selectbox(
            "Résultat réellement observé",
            options=[
                0,
                1,
                2,
            ],
            format_func=lambda x: CLASS_LABELS[x],
        )

        submit_feedback = st.form_submit_button(
            "Enregistrer le résultat réel"
        )

    if submit_feedback:
        payload_feedback = {
            "prediction_id": prediction_id_feedback,
            "actual_class": actual_class,
        }

        try:
            response = requests.post(
                f"{API_URL}/feedback",
                json=payload_feedback,
                timeout=10,
            )

            response.raise_for_status()

            st.success(
                "Résultat réel enregistré avec succès."
            )

        except requests.exceptions.HTTPError as exc:
            detail = extraire_detail_erreur(
                response,
                exc,
            )

            st.error(
                f"Erreur API : {detail}"
            )

        except requests.exceptions.RequestException as exc:
            st.error(
                f"Impossible de contacter l'API : {exc}"
            )


# ============================================================
# ONGLET 3 - HISTORIQUE
# ============================================================

with tab_history:
    st.subheader(
        "Historique des prédictions"
    )

    col1, col2 = st.columns(
        [1, 3]
    )

    with col1:
        limit = st.number_input(
            "Nombre maximum de lignes",
            min_value=1,
            max_value=1000,
            value=100,
            step=10,
        )

    with col2:
        st.write("")

    if st.button(
        "Actualiser l'historique"
    ):
        try:
            response = requests.get(
                f"{API_URL}/history",
                params={
                    "limit": int(limit),
                },
                timeout=10,
            )

            response.raise_for_status()

            history = response.json()

            if history:
                df_history = pd.DataFrame(
                    history
                )

                st.dataframe(
                    df_history,
                    use_container_width=True,
                    hide_index=True,
                )

                # Les indicateurs de cet écran sont calculés uniquement
                # sur les lignes d'historique actuellement affichées.
                st.subheader(
                    "Indicateurs"
                )

                nb_predictions = len(
                    df_history
                )

                nb_feedbacks = 0

                if "actual_class" in df_history.columns:
                    nb_feedbacks = int(
                        df_history[
                            "actual_class"
                        ]
                        .notna()
                        .sum()
                    )

                taux_feedback = (
                    nb_feedbacks
                    / nb_predictions
                    * 100
                    if nb_predictions > 0
                    else 0
                )

                metric1, metric2, metric3 = st.columns(
                    3
                )

                metric1.metric(
                    "Prédictions",
                    nb_predictions,
                )

                metric2.metric(
                    "Résultats réels",
                    nb_feedbacks,
                )

                metric3.metric(
                    "Taux d'enrichissement",
                    f"{taux_feedback:.1f} %",
                )

                if (
                    "actual_class" in df_history.columns
                    and "predicted_class" in df_history.columns
                ):
                    df_evaluable = (
                        df_history
                        .dropna(
                            subset=[
                                "actual_class",
                            ]
                        )
                        .copy()
                    )

                    if not df_evaluable.empty:
                        accuracy = (
                            df_evaluable[
                                "actual_class"
                            ].astype(int)
                            == df_evaluable[
                                "predicted_class"
                            ].astype(int)
                        ).mean()

                        st.metric(
                            "Accuracy sur données enrichies",
                            f"{accuracy:.3f}",
                        )

            else:
                st.info(
                    "Aucune prédiction enregistrée."
                )

        except requests.exceptions.HTTPError as exc:
            detail = extraire_detail_erreur(
                response,
                exc,
            )

            st.error(
                f"Erreur API : {detail}"
            )

        except requests.exceptions.RequestException as exc:
            st.error(
                f"Impossible de contacter l'API : {exc}"
            )