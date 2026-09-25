"""Initialisation du modèle de production.

Le script entraîne le pipeline retenu, mesure ses performances,
enregistre l'expérimentation dans MLflow, crée la référence de drift
et produit l'artefact Joblib consommé par l'API FastAPI.
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_loading import charger_donnees
from src.drift import (
    construire_reference_drift,
    sauvegarder_reference_drift,
)
from src.prepa_data import preprocessor_s2


# ============================================================
# CONFIGURATION GENERALE
# ============================================================

RANDOM_STATE = 42

EXPERIMENT_NAME = "retour_emploi"

REGISTERED_MODEL_NAME = (
    "retour_emploi_model"
)

BASE_DIR = Path(
    __file__
).resolve().parent.parent

MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = (
    MODEL_DIR
    / "pipeline_model.joblib"
)

DRIFT_REFERENCE_PATH = (
    MODEL_DIR
    / "drift_reference.json"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION MLFLOW
# ============================================================

# Depuis l'hôte, MLflow est exposé sur localhost.
# Docker Compose surcharge cette valeur avec http://mlflow:5000.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# ============================================================
# CHARGEMENT DES DONNEES
# ============================================================

df = charger_donnees()

X = df.drop(
    columns=[
        "classe_retour_emploi",
    ]
)

y = df[
    "classe_retour_emploi"
]


# ============================================================
# SPLIT TRAIN / TEST
# ============================================================

# Le modèle et ses hyperparamètres ayant déjà été sélectionnés,
# ce holdout reste réservé à l'évaluation du modèle final.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)


# ============================================================
# REFERENCE DE DRIFT
# ============================================================

# La référence est construite uniquement sur les données
# d'entraînement afin de représenter la population nominale.
reference_drift = construire_reference_drift(
    X_train
)

sauvegarder_reference_drift(
    reference_drift,
    DRIFT_REFERENCE_PATH,
)


# ============================================================
# PIPELINE FINAL
# ============================================================

# Configuration retenue après l'étape de sélection :
# LogisticRegression, C=1 et surpondération de la classe 2.
pipeline_final = Pipeline(
    [
        (
            "preprocessor",
            preprocessor_s2,
        ),
        (
            "model",
            LogisticRegression(
                C=1,
                class_weight={
                    0: 1,
                    1: 1,
                    2: 3,
                },
                solver="lbfgs",
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


# ============================================================
# RUN MLFLOW
# ============================================================

# Sécurise l'exécution lorsque le script est lancé alors
# qu'un run MLflow est encore actif dans le processus.
if mlflow.active_run() is not None:
    mlflow.end_run()


with mlflow.start_run(
    run_name="scenario_2_logistic_regression"
) as run:

    # La référence de drift est enregistrée dans le même run
    # que le modèle et ses métriques.
    mlflow.log_artifact(
        str(DRIFT_REFERENCE_PATH),
        artifact_path="monitoring",
    )

    # ========================================================
    # ENTRAINEMENT ET EVALUATION
    # ========================================================

    pipeline_final.fit(
        X_train,
        y_train,
    )

    y_pred = pipeline_final.predict(
        X_test
    )

    recall_classe_2 = recall_score(
        y_test,
        y_pred,
        labels=[2],
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro",
    )

    # ========================================================
    # LATENCE D'INFERENCE
    # ========================================================

    temps_inference = []

    X_latence = X_test.iloc[
        :min(
            100,
            len(X_test),
        )
    ]

    # La première inférence est exclue afin de limiter
    # l'impact du warm-up sur la mesure de latence.
    if len(X_latence) > 0:
        pipeline_final.predict(
            X_latence.iloc[[0]]
        )

    for i in range(
        len(X_latence)
    ):
        observation = X_latence.iloc[
            [i]
        ]

        debut = time.perf_counter()

        pipeline_final.predict(
            observation
        )

        fin = time.perf_counter()

        temps_inference.append(
            (fin - debut) * 1000
        )

    inference_ms = float(
        np.mean(
            temps_inference
        )
    )

    inference_p95_ms = float(
        np.percentile(
            temps_inference,
            95,
        )
    )

    # ========================================================
    # TRACKING MLFLOW
    # ========================================================

    mlflow.log_params(
        {
            "scenario": "Scenario 2 - Ethique",
            "algorithm": "LogisticRegression",
            "C": 1,
            "class_weight": "{0:1,1:1,2:3}",
            "solver": "lbfgs",
            "max_iter": 3000,
            "random_state": RANDOM_STATE,
            "test_size": 0.20,
        }
    )

    mlflow.log_metrics(
        {
            "f1_macro": float(f1_macro),
            "recall_classe_2": float(
                recall_classe_2
            ),
            "inference_ms": inference_ms,
            "inference_p95_ms": (
                inference_p95_ms
            ),
        }
    )

    mlflow.set_tags(
        {
            "scenario": "ethique",
            "model_family": "linear",
            "usage": "production_candidate",
            "dataset_size": str(
                len(df)
            ),
        }
    )

    # ========================================================
    # MODELE MLFLOW
    # ========================================================

    # SKOPS est utilisé pour éviter la sérialisation Pickle
    # classique. Les transformations personnalisées utilisées
    # par le pipeline sont déclarées explicitement comme fiables.
    mlflow.sklearn.log_model(
        sk_model=pipeline_final,
        artifact_path="model",
        serialization_format=(
            mlflow.sklearn
            .SERIALIZATION_FORMAT_SKOPS
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

    # ========================================================
    # MODEL REGISTRY
    # ========================================================

    model_uri = (
        f"runs:/{run.info.run_id}/model"
    )

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
        await_registration_for=300,
    )

    model_version = str(
        registered_model.version
    )

    # ========================================================
    # ARTEFACT DE PRODUCTION
    # ========================================================

    # L'artefact contient le pipeline et les métadonnées
    # nécessaires au suivi de la version chargée par FastAPI.
    artefact_modele = {
        "model": pipeline_final,
        "metadata": {
            "version": model_version,
            "registered_model_name": (
                REGISTERED_MODEL_NAME
            ),
            "mlflow_run_id": run.info.run_id,
            "scenario": "Scenario 2 - Ethique",
            "algorithm": "LogisticRegression",
            "trained_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "hyperparameters": {
                "C": 1,
                "class_weight": {
                    0: 1,
                    1: 1,
                    2: 3,
                },
                "solver": "lbfgs",
                "max_iter": 3000,
                "random_state": RANDOM_STATE,
            },
            "metrics": {
                "f1_macro": float(
                    f1_macro
                ),
                "recall_classe_2": float(
                    recall_classe_2
                ),
                "inference_ms": (
                    inference_ms
                ),
                "inference_p95_ms": (
                    inference_p95_ms
                ),
            },
        },
    }

    joblib.dump(
        artefact_modele,
        MODEL_PATH,
    )

    # Conserver le Joblib dans MLflow permet de retrouver
    # exactement l'artefact utilisé par FastAPI.
    mlflow.log_artifact(
        str(MODEL_PATH),
        artifact_path="deployment",
    )


# ============================================================
# RESUME
# ============================================================

print()
print("============================================")
print("MODELE FINAL ENREGISTRE")
print("============================================")
print(f"Modèle local        : {MODEL_PATH}")
print(f"MLflow Tracking URI : {MLFLOW_TRACKING_URI}")
print(f"MLflow Run ID       : {run.info.run_id}")
print(f"Model Registry      : {REGISTERED_MODEL_NAME}")
print(f"Version MLflow      : {model_version}")
print(f"F1 macro            : {f1_macro:.4f}")
print(f"Recall classe 2     : {recall_classe_2:.4f}")
print(f"Latence moyenne     : {inference_ms:.2f} ms")
print(f"Latence P95         : {inference_p95_ms:.2f} ms")
print("============================================")