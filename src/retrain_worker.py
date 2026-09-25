"""Worker de réentraînement et de promotion du modèle.

Le worker consomme les demandes de réentraînement enregistrées en base,
enrichit les données d'entraînement avec les feedbacks de production,
entraîne un modèle candidat puis décide de sa promotion selon des
critères de performance et de latence.

Les expérimentations et versions de modèles sont historisées dans MLflow.
"""

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split

from src import config
from src.data_loading import charger_donnees
from src.drift import (
    construire_reference_drift,
    sauvegarder_reference_drift,
)
from src.history_store import (
    donnees_reentrainement,
    prendre_prochain_retrain_job,
    terminer_retrain_job,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "pipeline_model.joblib"
DRIFT_PATH = BASE_DIR / "model" / "drift_reference.json"

FEATURES = [
    "niveau_diplome",
    "anciennete_poste_ans",
    "code_rome_vise",
    "synthese_entretien",
]

MIN_FEEDBACKS = int(
    os.getenv(
        "MIN_RETRAIN_FEEDBACK",
        "20",
    )
)

MAX_LATENCY_P95_MS = float(
    os.getenv(
        "MAX_RETRAIN_LATENCY_P95_MS",
        "200",
    )
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

logger = logging.getLogger("retrain_worker")


# ============================================================
# EVALUATION
# ============================================================


def evaluer(
    model,
    X,
    y,
) -> tuple[float, float]:
    """Calcule le F1 macro et le recall de la classe 2."""

    predictions = model.predict(X)

    f1 = f1_score(
        y,
        predictions,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )

    recall_2 = recall_score(
        y,
        predictions,
        labels=[2],
        average="macro",
        zero_division=0,
    )

    return float(f1), float(recall_2)


def mesurer_latence(
    model,
    X,
) -> float:
    """Mesure la latence P95 d'une prédiction unitaire en millisecondes."""

    durations = []
    sample_size = min(100, len(X))

    for i in range(sample_size):
        row = X.iloc[[i]]

        start = time.perf_counter()
        model.predict(row)

        durations.append(
            (time.perf_counter() - start) * 1000
        )

    return float(
        np.percentile(
            durations,
            95,
        )
    )


# ============================================================
# TRAITEMENT D'UN JOB
# ============================================================


def traiter_job(job: dict) -> None:
    """Entraîne, évalue et éventuellement promeut un modèle candidat."""

    job_id = job["job_id"]

    logger.info(
        "Début retrain job_id=%s",
        job_id,
    )

    feedback_rows = donnees_reentrainement()
    feedback_count = len(feedback_rows)

    if feedback_count < MIN_FEEDBACKS:
        terminer_retrain_job(
            job_id=job_id,
            status="rejected",
            feedback_count=feedback_count,
            message=(
                f"Feedback insuffisant : "
                f"{feedback_count}/{MIN_FEEDBACKS}"
            ),
        )

        return

    # ========================================================
    # DONNEES ORIGINALES
    # ========================================================

    df = charger_donnees(
        verbose=False
    )

    X = df[FEATURES].copy()

    y = df[
        config.TARGET
    ].astype(int)

    # Le holdout reste constitué uniquement des données originales.
    # Les feedbacks de production enrichissent uniquement le train.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=config.SEED,
    )

    # ========================================================
    # FEEDBACKS DE PRODUCTION
    # ========================================================

    feedback_df = pd.DataFrame(
        feedback_rows
    )

    X_feedback = feedback_df[
        FEATURES
    ].copy()

    y_feedback = feedback_df[
        "actual_class"
    ].astype(int)

    X_train_augmented = pd.concat(
        [
            X_train,
            X_feedback,
        ],
        ignore_index=True,
    )

    y_train_augmented = pd.concat(
        [
            y_train.reset_index(drop=True),
            y_feedback.reset_index(drop=True),
        ],
        ignore_index=True,
    )

    # ========================================================
    # MODELE COURANT
    # ========================================================

    artefact = joblib.load(
        MODEL_PATH
    )

    current_model = artefact[
        "model"
    ]

    current_version = str(
        artefact[
            "metadata"
        ].get(
            "version",
            "unknown",
        )
    )

    current_f1, current_recall_2 = evaluer(
        current_model,
        X_test,
        y_test,
    )

    # ========================================================
    # MODELE CANDIDAT
    # ========================================================

    # Le pipeline courant est cloné afin de conserver exactement
    # la même architecture et les mêmes hyperparamètres.
    candidate_model = clone(
        current_model
    )

    candidate_model.fit(
        X_train_augmented,
        y_train_augmented,
    )

    candidate_f1, candidate_recall_2 = evaluer(
        candidate_model,
        X_test,
        y_test,
    )

    latency_p95_ms = mesurer_latence(
        candidate_model,
        X_test,
    )

    # ========================================================
    # REGLES DE PROMOTION
    # ========================================================

    # Une faible tolérance à la dégradation est autorisée sur une
    # métrique uniquement si l'autre apporte une amélioration.
    non_degradation = (
        candidate_f1 >= current_f1 - 0.005
        and candidate_recall_2 >= current_recall_2 - 0.01
    )

    improvement = (
        candidate_f1 > current_f1 + 0.001
        or candidate_recall_2 > current_recall_2 + 0.005
    )

    latency_ok = (
        latency_p95_ms
        <= MAX_LATENCY_P95_MS
    )

    promote = (
        non_degradation
        and improvement
        and latency_ok
    )

    # ========================================================
    # MLFLOW
    # ========================================================

    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://mlflow:5000",
        )
    )

    mlflow.set_experiment(
        "Retour Emploi"
    )

    with mlflow.start_run(
        run_name=f"retrain_{job_id}"
    ) as run:
        mlflow.log_params(
            {
                "feedback_count": feedback_count,
                "current_version": current_version,
                "promotion_rule": (
                    "non_degradation_and_improvement"
                ),
            }
        )

        mlflow.log_metrics(
            {
                "current_f1_macro": current_f1,
                "candidate_f1_macro": candidate_f1,
                "current_recall_class_2": current_recall_2,
                "candidate_recall_class_2": candidate_recall_2,
                "candidate_latency_p95_ms": latency_p95_ms,
                "promotion_approved": int(promote),
            }
        )

        mlflow.set_tags(
            {
                "retrain_job_id": job_id,
                "model_role": "candidate",
            }
        )

        mlflow.sklearn.log_model(
            sk_model=candidate_model,
            artifact_path="model",
            serialization_format=(
                mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS
            ),
            skops_trusted_types=[
                "numpy.dtype",
                "src.prepa_data.convertir_en_str",
                "src.prepa_data.preparer_texte",
                "src.prepa_data.nettoyer_texte_ethique",
            ],
            code_paths=[
                str(BASE_DIR / "src"),
            ],
        )

        registered = mlflow.register_model(
            model_uri=(
                f"runs:/{run.info.run_id}/model"
            ),
            name="retour_emploi_model",
        )

    candidate_version = str(
        registered.version
    )

    # ========================================================
    # PROMOTION
    # ========================================================

    if promote:
        metadata = {
            "version": candidate_version,
            "registered_model_name": "retour_emploi_model",
            "mlflow_run_id": run.info.run_id,
            "scenario": "Scenario 2 - Ethique",
            "algorithm": "LogisticRegression",
            "trained_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "retrain_job_id": job_id,
            "feedback_count": feedback_count,
            "metrics": {
                "f1_macro": candidate_f1,
                "recall_class_2": candidate_recall_2,
                "latency_p95_ms": latency_p95_ms,
            },
        }

        candidate_artifact = {
            "model": candidate_model,
            "metadata": metadata,
        }

        temp_model_path = MODEL_PATH.with_suffix(
            ".tmp"
        )

        joblib.dump(
            candidate_artifact,
            temp_model_path,
        )

        # os.replace réalise le basculement atomiquement : l'API
        # ne voit jamais un fichier modèle partiellement écrit.
        os.replace(
            temp_model_path,
            MODEL_PATH,
        )

        # La référence de drift doit correspondre aux données
        # utilisées pour entraîner le nouveau modèle promu.
        reference = construire_reference_drift(
            X_train_augmented
        )

        temp_drift_path = DRIFT_PATH.with_suffix(
            ".tmp"
        )

        sauvegarder_reference_drift(
            reference,
            temp_drift_path,
        )

        os.replace(
            temp_drift_path,
            DRIFT_PATH,
        )

        client = mlflow.tracking.MlflowClient()

        client.set_registered_model_alias(
            name="retour_emploi_model",
            alias="champion",
            version=candidate_version,
        )

        status = "promoted"
        message = "Candidat promu en production"

    else:
        status = "rejected"
        message = (
            "Candidat rejeté par les "
            "critères de promotion"
        )

    terminer_retrain_job(
        job_id=job_id,
        status=status,
        feedback_count=feedback_count,
        current_version=current_version,
        candidate_version=candidate_version,
        promoted=promote,
        current_f1_macro=current_f1,
        candidate_f1_macro=candidate_f1,
        current_recall_class_2=current_recall_2,
        candidate_recall_class_2=candidate_recall_2,
        candidate_latency_p95_ms=latency_p95_ms,
        message=message,
    )


# ============================================================
# WORKER
# ============================================================


def main() -> None:
    """Consomme en continu les jobs de réentraînement en attente."""

    logger.info(
        "Retrain worker démarré"
    )

    while True:
        job = prendre_prochain_retrain_job()

        if job is None:
            time.sleep(5)
            continue

        try:
            traiter_job(
                job
            )

        except Exception as exc:
            logger.exception(
                "Échec du réentraînement "
                "job_id=%s",
                job["job_id"],
            )

            terminer_retrain_job(
                job_id=job["job_id"],
                status="failed",
                message=str(exc),
            )


if __name__ == "__main__":
    main()