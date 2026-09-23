from pathlib import Path
from datetime import datetime, timezone
import os
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_loading import charger_donnees
from src.prepa_data import preprocessor_s2
from src.drift import (
    construire_reference_drift,
    sauvegarder_reference_drift
)

# ============================================================
# CONFIGURATION GENERALE
# ============================================================

RANDOM_STATE = 42

EXPERIMENT_NAME = "retour_emploi"

REGISTERED_MODEL_NAME = "retour_emploi_model"


# ============================================================
# CHEMINS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "pipeline_model.joblib"

DRIFT_REFERENCE_PATH = (
    BASE_DIR
    / "model"
    / "drift_reference.json"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# CONFIGURATION MLFLOW
#
# Par défaut :
# - init_model.py exécuté depuis l'hôte
# - MLflow exposé par Docker sur localhost:5000
#
# Si le script est exécuté dans Docker Compose :
# MLFLOW_TRACKING_URI=http://mlflow:5000
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
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
        "classe_retour_emploi"
    ]
)

y = df[
    "classe_retour_emploi"
]


# ============================================================
# SPLIT TRAIN / TEST
#
# Le modèle et les hyperparamètres ont déjà été sélectionnés.
# Le test est utilisé ici pour l'évaluation finale.
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)



reference_drift = (
    construire_reference_drift(
        X_train
    )
)

sauvegarder_reference_drift(
    reference_drift,
    DRIFT_REFERENCE_PATH
)

mlflow.log_artifact(
    str(
        DRIFT_REFERENCE_PATH
    ),
    artifact_path="monitoring"
)


# ============================================================
# PIPELINE FINAL
#
# Scenario 2 - Ethique
#
# Configuration retenue après optimisation :
# C = 1
# class_weight = {0:1, 1:1, 2:3}
# solver = lbfgs
# ============================================================

pipeline_final = Pipeline([
    (
        "preprocessor",
        preprocessor_s2
    ),
    (
        "model",
        LogisticRegression(
            C=1,
            class_weight={
                0: 1,
                1: 1,
                2: 3
            },
            solver="lbfgs",
            max_iter=3000,
            random_state=RANDOM_STATE
        )
    )
])


# ============================================================
# DEMARRAGE DU RUN MLFLOW
# ============================================================
# ============================================================
# SECURISATION DU CYCLE DE VIE MLFLOW
# ============================================================

if mlflow.active_run() is not None:

    mlflow.end_run()
    
with mlflow.start_run(
    run_name="scenario_2_logistic_regression"
) as run:

    # ========================================================
    # ENTRAINEMENT
    # ========================================================

    pipeline_final.fit(
        X_train,
        y_train
    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    y_pred = pipeline_final.predict(
        X_test
    )


    # ========================================================
    # CRITERE C1
    # Recall classe 2
    # ========================================================

    recall_classe_2 = recall_score(
        y_test,
        y_pred,
        labels=[2],
        average="macro",
        zero_division=0
    )


    # ========================================================
    # CRITERE C2
    # F1 macro
    # ========================================================

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro"
    )


    # ========================================================
    # CRITERE C3
    # Temps d'inférence
    #
    # Mesure unitaire sur maximum 100 observations.
    # ========================================================

    temps_inference = []

    X_latence = X_test.iloc[
        :min(
            100,
            len(X_test)
        )
    ]

    # Warm-up : première prédiction non comptabilisée
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
            95
        )
    )


    # ========================================================
    # PARAMETRES MLFLOW
    # ========================================================

    mlflow.log_params({

        "scenario":
            "Scenario 2 - Ethique",

        "algorithm":
            "LogisticRegression",

        "C":
            1,

        "class_weight":
            "{0:1,1:1,2:3}",

        "solver":
            "lbfgs",

        "max_iter":
            3000,

        "random_state":
            RANDOM_STATE,

        "test_size":
            0.20
    })


    # ========================================================
    # METRIQUES MLFLOW
    # ========================================================

    mlflow.log_metrics({

        "f1_macro":
            float(f1_macro),

        "recall_classe_2":
            float(recall_classe_2),

        "inference_ms":
            inference_ms,

        "inference_p95_ms":
            inference_p95_ms
    })


    # ========================================================
    # TAGS MLFLOW
    # ========================================================

    mlflow.set_tags({

        "scenario":
            "ethique",

        "model_family":
            "linear",

        "usage":
            "production_candidate",

        "dataset_size":
            str(len(df))
    })


    # ========================================================
    # ENREGISTREMENT DU PIPELINE DANS MLFLOW
    #
    # skops est utilisé par MLflow pour une sérialisation plus
    # sûre que pickle.
    #
    # Le pipeline contient des FunctionTransformer utilisant
    # des fonctions définies dans src.prepa_data.
    # Elles sont explicitement déclarées comme fiables.
    # ========================================================

    mlflow.sklearn.log_model(

        sk_model=pipeline_final,

        artifact_path="model",

        serialization_format=
            mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,

        skops_trusted_types=[
            "numpy.dtype",
            "src.prepa_data.convertir_en_str",
            "src.prepa_data.preparer_texte"
        ],

        code_paths=[
            str(
                BASE_DIR / "src"
            )
        ]
    )


    # ========================================================
    # MODEL REGISTRY
    #
    # Enregistrement du modèle associé au run courant.
    # MLflow attribue automatiquement une nouvelle version.
    # ========================================================

    model_uri = (
        f"runs:/{run.info.run_id}/model"
    )

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
        await_registration_for=300
    )

    model_version = str(
        registered_model.version
    )


    # ========================================================
    # CREATION DE L'ARTEFACT JOBLIB
    #
    # Cet artefact est actuellement consommé par FastAPI.
    #
    # La version n'est plus définie manuellement :
    # elle provient du Model Registry MLflow.
    # ========================================================

    artefact_modele = {

        "model":
            pipeline_final,

        "metadata": {

            "version":
                model_version,

            "registered_model_name":
                REGISTERED_MODEL_NAME,

            "mlflow_run_id":
                run.info.run_id,

            "scenario":
                "Scenario 2 - Ethique",

            "algorithm":
                "LogisticRegression",

            "trained_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "hyperparameters": {

                "C":
                    1,

                "class_weight": {
                    0: 1,
                    1: 1,
                    2: 3
                },

                "solver":
                    "lbfgs",

                "max_iter":
                    3000,

                "random_state":
                    RANDOM_STATE
            },

            "metrics": {

                "f1_macro":
                    float(
                        f1_macro
                    ),

                "recall_classe_2":
                    float(
                        recall_classe_2
                    ),

                "inference_ms":
                    inference_ms,

                "inference_p95_ms":
                    inference_p95_ms
            }
        }
    }


    # ========================================================
    # SAUVEGARDE JOBLIB POUR FASTAPI
    # ========================================================

    joblib.dump(
        artefact_modele,
        MODEL_PATH
    )


    # ========================================================
    # SAUVEGARDE DU JOBLIB COMME ARTEFACT MLFLOW
    #
    # Permet également de retrouver exactement l'artefact
    # actuellement utilisé pour le déploiement FastAPI.
    # ========================================================

    mlflow.log_artifact(
        str(MODEL_PATH),
        artifact_path="deployment"
    )


# ============================================================
# RESUME
# ============================================================

print()
print(
    "============================================"
)

print(
    "MODELE FINAL ENREGISTRE"
)

print(
    "============================================"
)

print(
    f"Modèle local        : {MODEL_PATH}"
)

print(
    f"MLflow Tracking URI : {MLFLOW_TRACKING_URI}"
)

print(
    f"MLflow Run ID       : {run.info.run_id}"
)

print(
    f"Model Registry      : {REGISTERED_MODEL_NAME}"
)

print(
    f"Version MLflow      : {model_version}"
)

print(
    f"F1 macro            : {f1_macro:.4f}"
)

print(
    f"Recall classe 2     : {recall_classe_2:.4f}"
)

print(
    f"Latence moyenne     : {inference_ms:.2f} ms"
)

print(
    f"Latence P95         : {inference_p95_ms:.2f} ms"
)

print(
    "============================================"
)