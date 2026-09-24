"""Tests fonctionnels unitaires de l'API Retour Emploi.

Ces tests vérifient le contrat HTTP, la validation des entrées,
le cycle prédiction-feedback et la persistance dans l'historique.
"""

import pytest


# ============================================================
# DONNEES DE TEST
# ============================================================

VALID_PAYLOAD = {
    "session_id": "session-test-001",
    "niveau_diplome": "Bac+2",
    "anciennete_poste_ans": 4.5,
    "code_rome_vise": "M1805",
    "synthese_entretien": (
        "Profil autonome, recherche active."
    ),
}


# ============================================================
# HEALTHCHECK
# ============================================================


def test_health(client):
    """Vérifie que l'API expose un modèle disponible."""

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "test-1"


# ============================================================
# PREDICTION
# ============================================================


def test_prediction_valide(client):
    """Vérifie le contrat d'une prédiction valide."""

    response = client.post(
        "/predict",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["prediction_id"]
    assert body["prediction"] in [0, 1, 2]

    # Le FakeModel prédit systématiquement la classe 2.
    assert body["prediction"] == 2
    assert body["model_version"] == "test-1"

    probabilities = body[
        "probabilities"
    ]

    # Les clés numériques du dictionnaire Python sont sérialisées
    # en chaînes de caractères dans la réponse JSON.
    assert set(
        probabilities.keys()
    ) == {
        "0",
        "1",
        "2",
    }

    assert sum(
        probabilities.values()
    ) == pytest.approx(
        1.0
    )

    assert body[
        "score"
    ] == pytest.approx(
        0.70
    )


def test_variable_sensible_refusee(client):
    """Vérifie qu'une variable non autorisée est rejetée."""

    payload = VALID_PAYLOAD.copy()

    payload[
        "nationalite_hors_ue"
    ] = 1

    response = client.post(
        "/predict",
        json=payload,
    )

    # PredictionInput utilise extra="forbid".
    assert response.status_code == 422


def test_code_rome_invalide(client):
    """Vérifie le rejet d'un code ROME au format invalide."""

    payload = VALID_PAYLOAD.copy()

    payload[
        "code_rome_vise"
    ] = "M18"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


# ============================================================
# FEEDBACK
# ============================================================


def test_feedback_valide(client):
    """Vérifie l'enregistrement d'une vérité terrain."""

    prediction_response = client.post(
        "/predict",
        json=VALID_PAYLOAD,
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

    response = client.post(
        "/feedback",
        json={
            "prediction_id": prediction_id,
            "actual_class": 1,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert (
        body["prediction_id"]
        == prediction_id
    )
    assert body["actual_class"] == 1


# ============================================================
# HISTORIQUE
# ============================================================


def test_historique_enrichi(client):
    """Vérifie le cycle prédiction, feedback puis historique."""

    prediction_response = client.post(
        "/predict",
        json=VALID_PAYLOAD,
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

    feedback_response = client.post(
        "/feedback",
        json={
            "prediction_id": prediction_id,
            "actual_class": 1,
        },
    )

    assert (
        feedback_response.status_code
        == 200
    )

    history_response = client.get(
        "/history"
    )

    assert (
        history_response.status_code
        == 200
    )

    history = history_response.json()

    assert len(history) >= 1

    record = next(
        item
        for item in history
        if (
            item["prediction_id"]
            == prediction_id
        )
    )

    # FakeModel prédit toujours 2 tandis que le feedback indique 1.
    assert record["predicted_class"] == 2
    assert record["actual_class"] == 1
    assert record["model_version"] == "test-1"