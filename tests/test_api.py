import pytest


# ============================================================
# PAYLOAD VALIDE COMMUN
# ============================================================

VALID_PAYLOAD = {

    "session_id":
        "session-test-001",

    "niveau_diplome":
        "Bac+2",

    "anciennete_poste_ans":
        4.5,

    "code_rome_vise":
        "M1805",

    "synthese_entretien":
        "Profil autonome, recherche active."
}


# ============================================================
# T01 - HEALTHCHECK
# ============================================================

def test_health(
    client
):

    response = client.get(
        "/health"
    )

    assert (
        response.status_code
        == 200
    )

    body = response.json()

    assert (
        body["status"]
        == "ok"
    )

    assert (
        body["model_loaded"]
        is True
    )

    assert (
        body["model_version"]
        == "test-1"
    )


# ============================================================
# T02 - PREDICTION VALIDE
# ============================================================

def test_prediction_valide(
    client
):

    response = client.post(
        "/predict",
        json=VALID_PAYLOAD
    )

    assert (
        response.status_code
        == 200
    )

    body = response.json()


    # Prediction ID obligatoire
    assert (
        body["prediction_id"]
        is not None
    )

    assert (
        len(
            body["prediction_id"]
        )
        > 0
    )


    # Classe autorisée
    assert (
        body["prediction"]
        in [
            0,
            1,
            2
        ]
    )


    # Faux modèle => classe 2
    assert (
        body["prediction"]
        == 2
    )


    # Version du modèle
    assert (
        body["model_version"]
        == "test-1"
    )


    # Probabilités
    probabilities = (
        body[
            "probabilities"
        ]
    )

    assert (
        set(
            probabilities.keys()
        )
        == {
            "0",
            "1",
            "2"
        }
    )


    # Somme des probabilités = 1
    assert (
        sum(
            probabilities.values()
        )
        == pytest.approx(
            1.0
        )
    )


    # Score = probabilité
    # de la classe prédite
    assert (
        body["score"]
        == pytest.approx(
            0.70
        )
    )


# ============================================================
# T03 - VARIABLE SENSIBLE INTERDITE
#
# Le scénario éthique interdit notamment
# nationalite_hors_ue.
#
# ConfigDict(extra="forbid") doit produire 422.
# ============================================================

def test_variable_sensible_refusee(
    client
):

    payload = (
        VALID_PAYLOAD.copy()
    )

    payload[
        "nationalite_hors_ue"
    ] = 1


    response = client.post(
        "/predict",
        json=payload
    )


    assert (
        response.status_code
        == 422
    )


# ============================================================
# T04 - CODE ROME INVALIDE
# ============================================================

def test_code_rome_invalide(
    client
):

    payload = (
        VALID_PAYLOAD.copy()
    )

    payload[
        "code_rome_vise"
    ] = "M18"


    response = client.post(
        "/predict",
        json=payload
    )


    assert (
        response.status_code
        == 422
    )


# ============================================================
# T05 - FEEDBACK VALIDE
# ============================================================

def test_feedback_valide(
    client
):

    # --------------------------------------------------------
    # Création d'une prédiction
    # --------------------------------------------------------

    prediction_response = (
        client.post(
            "/predict",
            json=VALID_PAYLOAD
        )
    )

    assert (
        prediction_response.status_code
        == 200
    )


    prediction_id = (
        prediction_response
        .json()[
            "prediction_id"
        ]
    )


    # --------------------------------------------------------
    # Ajout de la vérité terrain
    # --------------------------------------------------------

    response = client.post(
        "/feedback",
        json={
            "prediction_id":
                prediction_id,

            "actual_class":
                1
        }
    )


    assert (
        response.status_code
        == 200
    )


    body = response.json()


    assert (
        body["status"]
        == "ok"
    )

    assert (
        body["prediction_id"]
        == prediction_id
    )

    assert (
        body["actual_class"]
        == 1
    )


# ============================================================
# T06 - HISTORIQUE ENRICHI
#
# Cycle complet :
#
# /predict
#      ↓
# prediction_id
#      ↓
# /feedback
#      ↓
# /history
#
# On vérifie que la prédiction ET
# la vérité terrain sont présentes.
# ============================================================

def test_historique_enrichi(
    client
):

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction_response = (
        client.post(
            "/predict",
            json=VALID_PAYLOAD
        )
    )

    assert (
        prediction_response.status_code
        == 200
    )


    prediction_id = (
        prediction_response
        .json()[
            "prediction_id"
        ]
    )


    # --------------------------------------------------------
    # Feedback
    # --------------------------------------------------------

    feedback_response = (
        client.post(
            "/feedback",
            json={
                "prediction_id":
                    prediction_id,

                "actual_class":
                    1
            }
        )
    )

    assert (
        feedback_response.status_code
        == 200
    )


    # --------------------------------------------------------
    # Historique
    # --------------------------------------------------------

    history_response = (
        client.get(
            "/history"
        )
    )

    assert (
        history_response.status_code
        == 200
    )


    history = (
        history_response.json()
    )

    assert (
        len(history)
        >= 1
    )


    # Recherche de notre prédiction
    record = next(
        item

        for item in history

        if (
            item[
                "prediction_id"
            ]
            == prediction_id
        )
    )


    # FakeModel prédit toujours 2
    assert (
        record[
            "predicted_class"
        ]
        == 2
    )


    # Feedback réellement observé
    assert (
        record[
            "actual_class"
        ]
        == 1
    )


    assert (
        record[
            "model_version"
        ]
        == "test-1"
    )