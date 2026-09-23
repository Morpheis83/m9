from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from uuid import uuid4

import logging
import os
import sys
import time
import asyncio
import threading

import joblib
import pandas as pd

from fastapi import (
    FastAPI,
    HTTPException,
    Header,
    Request
)

from fastapi.responses import Response

from prometheus_client import (
    generate_latest,
    CONTENT_TYPE_LATEST
)

from src.metrics import (
    API_REQUESTS,
    API_REQUEST_DURATION,
    PREDICTIONS,
    PREDICTION_DURATION,
    MODEL_AVAILABLE,
    FEEDBACKS
)
from src.ml_monitoring import (
    mettre_a_jour_metriques_ml
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field
)

from src.history_store import (
    init_db,
    enregistrer_prediction,
    enregistrer_feedback,
    historique,
    creer_retrain_job,
    recuperer_retrain_job
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RUNTIME_DIR = (
    BASE_DIR
    / "runtime"
)

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "pipeline_model.joblib"
)

DRIFT_REFERENCE_PATH = (
    BASE_DIR
    / "model"
    / "drift_reference.json"
)

ADMIN_TOKEN = os.getenv(
    "ADMIN_TOKEN",
    "change-me"
)


# ============================================================
# LOGGING TECHNIQUE
#
# Les logs techniques sont envoyés sur stdout.
#
# Consultation :
# docker compose logs api
# docker compose logs -f api
#
# Les données métier ne sont PAS stockées dans ces logs.
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s "
        "%(levelname)s "
        "%(name)s - "
        "%(message)s"
    ),
    handlers=[
        logging.StreamHandler(
            sys.stdout
        )
    ]
)

logger = logging.getLogger(
    "retour_emploi_api"
)


# ============================================================
# OUTILS INTERNES
# ============================================================

def verifier_repertoire_ecriture(
    directory: Path
):

    try:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        # Test réel d'écriture
        test_file = (
            directory
            / ".write_test"
        )

        test_file.write_text(
            "test",
            encoding="utf-8"
        )

        test_file.unlink()

    except PermissionError as exc:

        logger.exception(
            "Permissions insuffisantes sur le répertoire %s",
            directory
        )

        raise RuntimeError(
            f"Répertoire non inscriptible : {directory}"
        ) from exc

    except OSError as exc:

        logger.exception(
            "Impossible d'accéder au répertoire %s",
            directory
        )

        raise RuntimeError(
            f"Répertoire indisponible : {directory}"
        ) from exc


# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

model = None
model_metadata = {}
model_version = "unknown"

# Date de dernière modification du fichier chargé
model_file_mtime_ns = None

# Verrou pour remplacer proprement le modèle en mémoire
model_lock = threading.RLock()


MODEL_RELOAD_INTERVAL_SECONDS = float(
    os.getenv(
        "MODEL_RELOAD_INTERVAL_SECONDS",
        "5"
    )
)

def charger_modele():

    global model
    global model_metadata
    global model_version
    global model_file_mtime_ns


    try:

        # ----------------------------------------------------
        # Chargement complet du nouvel artefact
        # AVANT de toucher au modèle actuellement en mémoire
        # ----------------------------------------------------

        artefact = joblib.load(
            MODEL_PATH
        )


        nouveau_model = artefact[
            "model"
        ]


        nouvelles_metadata = artefact[
            "metadata"
        ]


        nouvelle_version = (
            nouvelles_metadata.get(
                "version",
                "unknown"
            )
        )


        nouveau_mtime_ns = (
            MODEL_PATH
            .stat()
            .st_mtime_ns
        )


        # ----------------------------------------------------
        # Remplacement atomique des références
        # ----------------------------------------------------

        with model_lock:

            ancienne_version = (
                model_version
            )

            model = nouveau_model

            model_metadata = (
                nouvelles_metadata
            )

            model_version = (
                nouvelle_version
            )

            model_file_mtime_ns = (
                nouveau_mtime_ns
            )


        MODEL_AVAILABLE.set(
            1
        )


        logger.info(
            "Modèle chargé "
            "ancienne_version=%s "
            "nouvelle_version=%s "
            "scenario=%s "
            "algorithm=%s",
            ancienne_version,
            model_version,
            model_metadata.get(
                "scenario",
                "unknown"
            ),
            model_metadata.get(
                "algorithm",
                "unknown"
            )
        )


    except FileNotFoundError:

        logger.exception(
            "Artefact modèle introuvable : %s",
            MODEL_PATH
        )

        # Un ancien modèle déjà chargé reste utilisable.
        MODEL_AVAILABLE.set(
            1
            if model is not None
            else 0
        )


    except KeyError:

        logger.exception(
            "Structure de l'artefact modèle invalide"
        )

        MODEL_AVAILABLE.set(
            1
            if model is not None
            else 0
        )


    except Exception:

        logger.exception(
            "Erreur lors du chargement du modèle"
        )

        MODEL_AVAILABLE.set(
            1
            if model is not None
            else 0
        )
def modele_a_change():

    try:

        current_mtime_ns = (
            MODEL_PATH
            .stat()
            .st_mtime_ns
        )

    except FileNotFoundError:

        return False


    if model_file_mtime_ns is None:

        return True


    return (
        current_mtime_ns
        != model_file_mtime_ns
    )


async def surveiller_modele():

    logger.info(
        "Surveillance du modèle activée "
        "intervalle=%.1fs",
        MODEL_RELOAD_INTERVAL_SECONDS
    )


    while True:

        try:

            await asyncio.sleep(
                MODEL_RELOAD_INTERVAL_SECONDS
            )


            if modele_a_change():

                logger.info(
                    "Modification de l'artefact "
                    "modèle détectée"
                )


                await asyncio.to_thread(
                    charger_modele
                )


        except asyncio.CancelledError:

            logger.info(
                "Arrêt de la surveillance du modèle"
            )

            raise


        except Exception:

            logger.exception(
                "Erreur pendant la surveillance "
                "du modèle"
            )
# ============================================================
# LIFESPAN FASTAPI
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI
):

    logger.info(
        "Démarrage de l'API"
    )


    verifier_repertoire_ecriture(
        RUNTIME_DIR
    )


    model_watcher_task = None


    try:

        init_db()

        logger.info(
            "Base d'historique initialisée"
        )


        if model is None:

            charger_modele()


        if model is None:

            raise RuntimeError(
                "Impossible de démarrer : "
                "aucun modèle disponible"
            )


        # ----------------------------------------------------
        # Surveillance automatique de l'artefact modèle
        # ----------------------------------------------------

        model_watcher_task = (
            asyncio.create_task(
                surveiller_modele()
            )
        )


    except Exception:

        logger.exception(
            "Impossible d'initialiser l'API"
        )

        raise


    try:

        yield


    finally:

        if model_watcher_task is not None:

            model_watcher_task.cancel()

            try:

                await model_watcher_task

            except asyncio.CancelledError:

                pass


        logger.info(
            "Arrêt de l'API"
        )


# ============================================================
# APPLICATION FASTAPI
# ============================================================

app = FastAPI(
    title="API Retour Emploi",
    version="1.0",
    lifespan=lifespan
)


# ============================================================
# MODELES DE DONNEES
# ============================================================

class PredictionInput(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    niveau_diplome: str | None = Field(
        default=None,
        max_length=50
    )

    anciennete_poste_ans: float = Field(
        ge=0,
        le=67
    )

    code_rome_vise: str | None = Field(
        default=None,
        pattern=r"^[A-Z]\d{4}$"
    )

    synthese_entretien: str | None = Field(
        default=None,
        max_length=3000
    )

    session_id: str | None = Field(
        default=None,
        max_length=100
    )


class PredictionOutput(BaseModel):

    prediction_id: str

    prediction: int

    score: float | None

    probabilities: dict[int, float]

    model_version: str

    timestamp: str


class FeedbackInput(BaseModel):

    prediction_id: str = Field(
        min_length=1,
        max_length=100
    )

    actual_class: int = Field(
        ge=0,
        le=2
    )

# ============================================================
# MONITORING HTTP
# ============================================================

@app.middleware(
    "http"
)
async def monitoring_middleware(
    request: Request,
    call_next
):

    # On ne mesure pas le scraping Prometheus lui-même
    if request.url.path == "/metrics":

        return await call_next(
            request
        )

    start_time = time.perf_counter()

    status = "500"

    try:

        response = await call_next(
            request
        )

        status = str(
            response.status_code
        )

        return response

    finally:

        duration = (
            time.perf_counter()
            - start_time
        )

        API_REQUESTS.labels(
            endpoint=request.url.path,
            method=request.method,
            status=status
        ).inc()

        API_REQUEST_DURATION.labels(
            endpoint=request.url.path
        ).observe(
            duration
        )

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/health"
)
def health():

    if model is None:

        logger.warning(
            "Healthcheck dégradé : modèle non chargé"
        )
        MODEL_AVAILABLE.set(
            0
        )

        raise HTTPException(
            status_code=503,
            detail={
                "status":
                    "degraded",

                "model_loaded":
                    False
            }
        )

    MODEL_AVAILABLE.set(
        1
    )
    return {
        "status":
            "ok",

        "model_loaded":
            True,

        "model_version":
            model_version
    }

# ============================================================
# METRIQUES PROMETHEUS
# ============================================================

@app.get(
    "/metrics",
    include_in_schema=False
)
def metrics():

    try:

        mettre_a_jour_metriques_ml(
            DRIFT_REFERENCE_PATH
        )

    except Exception:

        logger.exception(
            "Impossible de recalculer "
            "les métriques ML"
        )


    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# ============================================================
# PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionOutput
)
def predict(
    data: PredictionInput
):

    if model is None:

        logger.error(
            "Tentative de prédiction sans modèle disponible"
        )

        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible"
        )

    with model_lock:

        current_model = model

        current_model_version = (
            model_version
        )


    # --------------------------------------------------------
    # Identifiant unique
    # --------------------------------------------------------

    prediction_id = str(
        uuid4()
    )

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()


    try:

        # ----------------------------------------------------
        # Préparation des données
        # ----------------------------------------------------

        input_dict = (
            data.model_dump()
        )

        session_id = input_dict.pop(
            "session_id",
            None
        )

        X = pd.DataFrame(
            [
                input_dict
            ]
        )


        # ----------------------------------------------------
        # Prédiction + latence
        # ----------------------------------------------------

        start_time = (
            time.perf_counter()
        )

        prediction = int(
            current_model.predict(
                X
            )[0]
        )


        probabilities = {}

        score = None


        # ----------------------------------------------------
        # Probabilités
        # ----------------------------------------------------

        if hasattr(
            current_model,
            "predict_proba"
        ):

            raw_probabilities = (
                current_model.predict_proba(
                    X
                )[0]
            )

            classes = getattr(
                current_model,
                "classes_",
                range(
                    len(
                        raw_probabilities
                    )
                )
            )

            probabilities = {
                int(classe):
                    float(probability)

                for classe, probability
                in zip(
                    classes,
                    raw_probabilities
                )
            }

            # Score associé à la classe prédite
            score = probabilities.get(
                prediction
            )


        duration_seconds = (
            time.perf_counter()
            - start_time
        )

        duration_ms = (
            duration_seconds
            * 1000
        )


        # ----------------------------------------------------
        # PERSISTANCE METIER
        #
        # SQLite :
        # - features
        # - prédiction
        # - probabilités
        # - version du modèle
        #
        # Ceci permet ensuite d'ajouter la vérité terrain.
        # ----------------------------------------------------

        try:

            enregistrer_prediction(

                prediction_id=
                    prediction_id,

                session_id=
                    session_id,

                niveau_diplome=
                    data.niveau_diplome,

                anciennete_poste_ans=
                    data.anciennete_poste_ans,

                code_rome_vise=
                    data.code_rome_vise,

                synthese_entretien=
                    data.synthese_entretien,

                predicted_class=
                    prediction,

                probabilities=
                    probabilities,

                model_version=
                    current_model_version
            )

            PREDICTIONS.labels(
                predicted_class=str(
                    prediction
                )
            ).inc()

            PREDICTION_DURATION.observe(
                duration_seconds
            )

        except Exception as exc:

            logger.exception(
                "Échec de persistance métier "
                "prediction_id=%s",
                prediction_id
            )

            raise RuntimeError(
                "Impossible d'enregistrer la prédiction"
            ) from exc


        # ----------------------------------------------------
        # LOG TECHNIQUE
        #
        # Pas de données personnelles.
        # Pas de synthèse entretien.
        # ----------------------------------------------------

        logger.info(
            "Prediction réussie "
            "prediction_id=%s "
            "classe=%s "
            "model_version=%s "
            "duration_ms=%.2f",
            prediction_id,
            prediction,
            current_model_version,
            duration_ms
        )


        # ----------------------------------------------------
        # Réponse API
        # ----------------------------------------------------

        return {

            "prediction_id":
                prediction_id,

            "prediction":
                prediction,

            "score":
                score,

            "probabilities":
                probabilities,

            "model_version":
                current_model_version,

            "timestamp":
                timestamp
        }


    # ========================================================
    # ERREURS
    # ========================================================

    except HTTPException:

        raise


    except ValueError:

        logger.warning(
            "Données incompatibles avec le modèle "
            "prediction_id=%s",
            prediction_id,
            exc_info=True
        )

        raise HTTPException(
            status_code=422,
            detail=(
                "Les données fournies sont "
                "incompatibles avec le modèle."
            )
        )


    except RuntimeError:

        logger.exception(
            "Erreur de persistance "
            "prediction_id=%s",
            prediction_id
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "La prédiction a été calculée mais "
                "son enregistrement a échoué."
            )
        )


    except Exception:

        logger.exception(
            "Erreur interne pendant la prédiction "
            "prediction_id=%s",
            prediction_id
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur interne lors de la prédiction."
            )
        )


# ============================================================
# FEEDBACK / VERITE TERRAIN
# ============================================================

@app.post(
    "/feedback"
)
def feedback(
    data: FeedbackInput
):

    try:

        enregistrer_feedback(

            prediction_id=
                data.prediction_id,

            actual_class=
                data.actual_class
        )
        
        FEEDBACKS.labels(
            actual_class=str(
                data.actual_class
            )
        ).inc()


        logger.info(
            "Feedback enregistré "
            "prediction_id=%s "
            "actual_class=%s",
            data.prediction_id,
            data.actual_class
        )


        return {

            "status":
                "ok",

            "prediction_id":
                data.prediction_id,

            "actual_class":
                data.actual_class
        }


    except ValueError:

        logger.warning(
            "Feedback refusé : prédiction inconnue "
            "prediction_id=%s",
            data.prediction_id
        )

        raise HTTPException(
            status_code=404,
            detail="Prédiction inconnue"
        )


    except Exception:

        logger.exception(
            "Erreur lors de l'enregistrement du feedback "
            "prediction_id=%s",
            data.prediction_id
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Impossible d'enregistrer "
                "le résultat réel."
            )
        )


# ============================================================
# HISTORIQUE METIER
# ============================================================

@app.get(
    "/history"
)
def get_history(
    limit: int = 100
):

    if limit < 1:

        raise HTTPException(
            status_code=422,
            detail=(
                "La limite doit être "
                "supérieure à 0."
            )
        )


    limit = min(
        limit,
        1000
    )


    try:

        return historique(
            limit=limit
        )


    except Exception:

        logger.exception(
            "Erreur lors de la lecture de l'historique"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Impossible de lire "
                "l'historique."
            )
        )


# ============================================================
# RETRAIN
# ============================================================

@app.post(
    "/retrain"
)
def retrain(
    x_admin_token: str | None = Header(
        default=None
    )
):

    if (
        x_admin_token
        != ADMIN_TOKEN
    ):

        logger.warning(
            "Tentative de réentraînement non autorisée"
        )

        raise HTTPException(
            status_code=403,
            detail="Accès interdit"
        )


    job_id = creer_retrain_job()


    logger.info(
        "Demande de réentraînement créée "
        "job_id=%s",
        job_id
    )


    return {

        "status":
            "accepted",

        "job_id":
            job_id
    }

@app.get(
    "/retrain/{job_id}"
)
def retrain_status(
    job_id: str,
    x_admin_token: str | None = Header(
        default=None
    )
):

    if (
        x_admin_token
        != ADMIN_TOKEN
    ):

        raise HTTPException(
            status_code=403,
            detail="Accès interdit"
        )


    job = recuperer_retrain_job(
        job_id
    )


    if job is None:

        raise HTTPException(
            status_code=404,
            detail="Job inconnu"
        )


    return job