"""Persistance SQLite des prédictions, feedbacks et jobs de réentraînement.

Ce module centralise l'accès à la base d'historique utilisée par l'API,
le monitoring et le worker de réentraînement.
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_DB_PATH = (
    BASE_DIR
    / "runtime"
    / "history.db"
)

DB_PATH = Path(
    os.getenv(
        "HISTORY_DB_PATH",
        str(DEFAULT_DB_PATH),
    )
)

DB_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONNEXION
# ============================================================


def get_connection() -> sqlite3.Connection:
    """Crée une connexion SQLite retournant les lignes par nom de colonne."""

    connection = sqlite3.connect(
        DB_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# INITIALISATION
# ============================================================


def init_db() -> None:
    """Crée les tables nécessaires si elles n'existent pas."""

    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                session_id TEXT,
                niveau_diplome TEXT,
                anciennete_poste_ans REAL,
                code_rome_vise TEXT,
                synthese_entretien TEXT,
                predicted_class INTEGER NOT NULL,
                probability_0 REAL,
                probability_1 REAL,
                probability_2 REAL,
                model_version TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                prediction_id TEXT PRIMARY KEY,
                observed_at TEXT NOT NULL,
                actual_class INTEGER NOT NULL,

                FOREIGN KEY(prediction_id)
                    REFERENCES predictions(prediction_id)
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS retrain_jobs (
                job_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                status TEXT NOT NULL,
                feedback_count INTEGER,
                current_version TEXT,
                candidate_version TEXT,
                promoted INTEGER NOT NULL DEFAULT 0,
                current_f1_macro REAL,
                candidate_f1_macro REAL,
                current_recall_class_2 REAL,
                candidate_recall_class_2 REAL,
                candidate_latency_p95_ms REAL,
                message TEXT
            )
            """
        )

        conn.commit()


# ============================================================
# PREDICTIONS
# ============================================================


def enregistrer_prediction(
    prediction_id,
    session_id,
    niveau_diplome,
    anciennete_poste_ans,
    code_rome_vise,
    synthese_entretien,
    predicted_class,
    probabilities,
    model_version,
) -> None:
    """Persiste une prédiction et les données nécessaires à son suivi."""

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                prediction_id,
                created_at,
                session_id,
                niveau_diplome,
                anciennete_poste_ans,
                code_rome_vise,
                synthese_entretien,
                predicted_class,
                probability_0,
                probability_1,
                probability_2,
                model_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                datetime.now(
                    timezone.utc
                ).isoformat(),
                session_id,
                niveau_diplome,
                anciennete_poste_ans,
                code_rome_vise,
                synthese_entretien,
                predicted_class,
                probabilities.get(0),
                probabilities.get(1),
                probabilities.get(2),
                model_version,
            ),
        )

        conn.commit()


# ============================================================
# FEEDBACK / VERITE TERRAIN
# ============================================================


def enregistrer_feedback(
    prediction_id,
    actual_class,
) -> None:
    """Associe ou met à jour la vérité terrain d'une prédiction."""

    with get_connection() as conn:
        prediction = conn.execute(
            """
            SELECT prediction_id
            FROM predictions
            WHERE prediction_id = ?
            """,
            (
                prediction_id,
            ),
        ).fetchone()

        if prediction is None:
            raise ValueError(
                "Prediction inconnue"
            )

        # L'UPSERT permet de corriger ultérieurement une vérité terrain
        # déjà renseignée sans créer plusieurs feedbacks.
        conn.execute(
            """
            INSERT INTO feedback (
                prediction_id,
                observed_at,
                actual_class
            )
            VALUES (?, ?, ?)

            ON CONFLICT(prediction_id)
            DO UPDATE SET
                observed_at = excluded.observed_at,
                actual_class = excluded.actual_class
            """,
            (
                prediction_id,
                datetime.now(
                    timezone.utc
                ).isoformat(),
                actual_class,
            ),
        )

        conn.commit()


# ============================================================
# HISTORIQUE METIER
# ============================================================


def historique(
    limit: int = 100,
) -> list[dict]:
    """Retourne les dernières prédictions enrichies de leur feedback."""

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                p.prediction_id,
                p.created_at,
                p.session_id,
                p.niveau_diplome,
                p.anciennete_poste_ans,
                p.code_rome_vise,
                p.synthese_entretien,
                p.predicted_class,
                p.probability_0,
                p.probability_1,
                p.probability_2,
                p.model_version,
                f.actual_class,
                f.observed_at

            FROM predictions p

            LEFT JOIN feedback f
                ON p.prediction_id = f.prediction_id

            ORDER BY p.created_at DESC

            LIMIT ?
            """,
            (
                limit,
            ),
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# DONNEES DE MONITORING
# ============================================================


def donnees_monitoring(
    limit: int = 5000,
) -> list[dict]:
    """Retourne les données nécessaires aux métriques ML et de drift."""

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                p.prediction_id,
                p.created_at,
                p.niveau_diplome,
                p.anciennete_poste_ans,
                p.code_rome_vise,
                p.synthese_entretien,
                p.predicted_class,
                p.model_version,
                f.actual_class,
                f.observed_at

            FROM predictions p

            LEFT JOIN feedback f
                ON p.prediction_id = f.prediction_id

            ORDER BY p.created_at DESC

            LIMIT ?
            """,
            (
                limit,
            ),
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# JOBS DE REENTRAINEMENT
# ============================================================


def creer_retrain_job() -> str:
    """Crée un job de réentraînement dans l'état pending."""

    job_id = str(
        uuid4()
    )

    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO retrain_jobs (
                job_id,
                created_at,
                status
            )
            VALUES (?, ?, ?)
            """,
            (
                job_id,
                created_at,
                "pending",
            ),
        )

    return job_id


def recuperer_retrain_job(
    job_id: str,
) -> dict | None:
    """Retourne un job de réentraînement à partir de son identifiant."""

    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM retrain_jobs
            WHERE job_id = ?
            """,
            (
                job_id,
            ),
        ).fetchone()

    if row is None:
        return None

    return dict(row)


def prendre_prochain_retrain_job() -> dict | None:
    """Réserve atomiquement le plus ancien job en attente."""

    with get_connection() as conn:
        # BEGIN IMMEDIATE empêche deux workers de réserver
        # simultanément le même job SQLite.
        conn.execute(
            "BEGIN IMMEDIATE"
        )

        row = conn.execute(
            """
            SELECT *
            FROM retrain_jobs
            WHERE status = 'pending'
            ORDER BY created_at
            LIMIT 1
            """
        ).fetchone()

        if row is None:
            conn.commit()
            return None

        started_at = datetime.now(
            timezone.utc
        ).isoformat()

        conn.execute(
            """
            UPDATE retrain_jobs
            SET
                status = 'running',
                started_at = ?
            WHERE job_id = ?
            """,
            (
                started_at,
                row["job_id"],
            ),
        )

        conn.commit()

        return dict(row)


def terminer_retrain_job(
    job_id: str,
    status: str,
    feedback_count: int | None = None,
    current_version: str | None = None,
    candidate_version: str | None = None,
    promoted: bool = False,
    current_f1_macro: float | None = None,
    candidate_f1_macro: float | None = None,
    current_recall_class_2: float | None = None,
    candidate_recall_class_2: float | None = None,
    candidate_latency_p95_ms: float | None = None,
    message: str | None = None,
) -> None:
    """Enregistre le résultat final d'un job de réentraînement."""

    finished_at = datetime.now(
        timezone.utc
    ).isoformat()

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE retrain_jobs

            SET
                finished_at = ?,
                status = ?,
                feedback_count = ?,
                current_version = ?,
                candidate_version = ?,
                promoted = ?,
                current_f1_macro = ?,
                candidate_f1_macro = ?,
                current_recall_class_2 = ?,
                candidate_recall_class_2 = ?,
                candidate_latency_p95_ms = ?,
                message = ?

            WHERE job_id = ?
            """,
            (
                finished_at,
                status,
                feedback_count,
                current_version,
                candidate_version,
                int(promoted),
                current_f1_macro,
                candidate_f1_macro,
                current_recall_class_2,
                candidate_recall_class_2,
                candidate_latency_p95_ms,
                message,
                job_id,
            ),
        )


# ============================================================
# DONNEES DE REENTRAINEMENT
# ============================================================


def donnees_reentrainement() -> list[dict]:
    """Retourne les prédictions disposant d'une vérité terrain."""

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                p.niveau_diplome,
                p.anciennete_poste_ans,
                p.code_rome_vise,
                p.synthese_entretien,
                f.actual_class

            FROM predictions p

            INNER JOIN feedback f
                ON p.prediction_id = f.prediction_id

            ORDER BY f.observed_at
            """
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]