from pathlib import Path

import pytest

from fastapi.testclient import TestClient

import src.api as api
import src.history_store as history_store


# ============================================================
# FAUX MODELE
#
# Permet de tester l'API sans dépendre du vrai artefact ML.
#
# Résultat volontairement déterministe :
#
# classe prédite = 2
#
# probabilités :
# classe 0 = 0.10
# classe 1 = 0.20
# classe 2 = 0.70
# ============================================================

class FakeModel:

    classes_ = [
        0,
        1,
        2
    ]

    def predict(
        self,
        X
    ):

        return [
            2
        ]


    def predict_proba(
        self,
        X
    ):

        return [
            [
                0.10,
                0.20,
                0.70
            ]
        ]


# ============================================================
# FIXTURE CLIENT API
# ============================================================

@pytest.fixture
def client(
    tmp_path,
    monkeypatch
):

    # --------------------------------------------------------
    # Base SQLite temporaire
    # --------------------------------------------------------

    db_path = (
        tmp_path
        / "history_test.db"
    )

    monkeypatch.setattr(
        history_store,
        "DB_PATH",
        db_path
    )


    # --------------------------------------------------------
    # Répertoire runtime temporaire
    # --------------------------------------------------------

    runtime_path = (
        tmp_path
        / "runtime"
    )

    monkeypatch.setattr(
        api,
        "RUNTIME_DIR",
        runtime_path
    )


    # --------------------------------------------------------
    # Faux modèle
    # --------------------------------------------------------

    monkeypatch.setattr(
        api,
        "model",
        FakeModel()
    )

    monkeypatch.setattr(
        api,
        "model_version",
        "test-1"
    )

    monkeypatch.setattr(
        api,
        "model_metadata",
        {
            "version": "test-1",
            "scenario": "test",
            "algorithm": "FakeModel"
        }
    )

    # --------------------------------------------------------
    # TestClient
    #
    # Le context manager déclenche le lifespan FastAPI,
    # donc notamment init_db().
    # --------------------------------------------------------

    with TestClient(
        api.app
    ) as test_client:

        yield test_client