"""Fixtures communes aux tests de l'API.

Les tests utilisent un modèle déterministe et une base SQLite temporaire
afin d'être indépendants du modèle de production et des données locales.
"""

import pytest
from fastapi.testclient import TestClient

import src.api as api
import src.history_store as history_store


# ============================================================
# FAUX MODELE
# ============================================================


class FakeModel:
    """Modèle déterministe utilisé pour isoler les tests de l'API."""

    classes_ = [
        0,
        1,
        2,
    ]

    def predict(self, X):
        """Retourne systématiquement la classe 2."""

        return [2]

    def predict_proba(self, X):
        """Retourne des probabilités fixes pour les trois classes."""

        return [
            [
                0.10,
                0.20,
                0.70,
            ]
        ]


# ============================================================
# FIXTURE CLIENT API
# ============================================================


@pytest.fixture
def client(
    tmp_path,
    monkeypatch,
):
    """Crée un client FastAPI isolé pour chaque test."""

    db_path = (
        tmp_path
        / "history_test.db"
    )

    runtime_path = (
        tmp_path
        / "runtime"
    )

    # Chaque test utilise sa propre base afin d'éviter les interactions
    # avec l'historique de développement ou les autres tests.
    monkeypatch.setattr(
        history_store,
        "DB_PATH",
        db_path,
    )

    monkeypatch.setattr(
        api,
        "RUNTIME_DIR",
        runtime_path,
    )

    # Le vrai artefact ML n'est pas nécessaire pour tester le contrat API.
    monkeypatch.setattr(
        api,
        "model",
        FakeModel(),
    )

    monkeypatch.setattr(
        api,
        "model_version",
        "test-1",
    )

    monkeypatch.setattr(
        api,
        "model_metadata",
        {
            "version": "test-1",
            "scenario": "test",
            "algorithm": "FakeModel",
        },
    )

    # Le context manager déclenche le lifespan FastAPI et notamment init_db().
    with TestClient(
        api.app
    ) as test_client:
        yield test_client