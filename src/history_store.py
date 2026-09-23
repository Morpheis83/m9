from pathlib import Path
from datetime import datetime, timezone
import os
import sqlite3
from uuid import uuid4

# ============================================================
# Configuration
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
        str(DEFAULT_DB_PATH)
    )
)

DB_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Connexion
# ============================================================

def get_connection():

    connection = sqlite3.connect(
        DB_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# Initialisation
# ============================================================

def init_db():

    with get_connection() as conn:

        conn.execute("""
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
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (

                prediction_id TEXT PRIMARY KEY,

                observed_at TEXT NOT NULL,

                actual_class INTEGER NOT NULL,

                FOREIGN KEY(prediction_id)
                    REFERENCES predictions(prediction_id)
            )
        """)


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
# Enregistrement d'une prédiction
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
    model_version
):

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

                model_version
            )
        )

        conn.commit()


# ============================================================
# Feedback / vérité terrain
# ============================================================

def enregistrer_feedback(
    prediction_id,
    actual_class
):

    with get_connection() as conn:

        prediction = conn.execute(
            """
            SELECT prediction_id
            FROM predictions
            WHERE prediction_id = ?
            """,
            (
                prediction_id,
            )
        ).fetchone()

        if prediction is None:
            raise ValueError(
                "Prediction inconnue"
            )

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

                actual_class
            )
        )

        conn.commit()


# ============================================================
# Historique enrichi
# ============================================================

def historique(
    limit=100
):

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
                ON p.prediction_id =
                   f.prediction_id

            ORDER BY p.created_at DESC

            LIMIT ?
            """,
            (
                limit,
            )
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]

def donnees_monitoring(
    limit: int = 5000
):

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.row_factory = (
            sqlite3.Row
        )

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
            )
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]

def creer_retrain_job():

    job_id = str(
        uuid4()
    )

    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    with sqlite3.connect(
        DB_PATH
    ) as conn:

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
                "pending"
            )
        )

    return job_id


def recuperer_retrain_job(
    job_id: str
):

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.row_factory = sqlite3.Row

        row = conn.execute(
            """
            SELECT *
            FROM retrain_jobs
            WHERE job_id = ?
            """,
            (
                job_id,
            )
        ).fetchone()

    if row is None:

        return None

    return dict(
        row
    )


def prendre_prochain_retrain_job():

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.row_factory = sqlite3.Row

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
                row["job_id"]
            )
        )

        conn.commit()

        return dict(
            row
        )


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
    message: str | None = None
):

    finished_at = datetime.now(
        timezone.utc
    ).isoformat()

    with sqlite3.connect(
        DB_PATH
    ) as conn:

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
                job_id
            )
        )
def donnees_reentrainement():

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.row_factory = sqlite3.Row

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