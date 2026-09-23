#!/usr/bin/env bash

set -euo pipefail


# ============================================================
# CONFIGURATION
# ============================================================

API_URL="http://127.0.0.1:8000"
MLFLOW_URL="http://127.0.0.1:5000"
UI_URL="http://127.0.0.1:8501"

MAX_RETRIES=30
SLEEP_SECONDS=3

RESPONSE_FILE="/tmp/retour_emploi_prediction.json"


# ============================================================
# AFFICHAGE DES LOGS EN CAS D'ERREUR
# ============================================================

show_logs() {

    echo
    echo "============================================"
    echo "LOGS DOCKER"
    echo "============================================"

    docker compose ps -a

    echo
    echo "--- model-init ---"
    docker compose logs model-init --tail=100 || true

    echo
    echo "--- api ---"
    docker compose logs api --tail=100 || true

    echo
    echo "--- ui ---"
    docker compose logs ui --tail=50 || true
}


trap show_logs ERR


# ============================================================
# DEMARRAGE DE LA STACK
# ============================================================

echo "============================================"
echo "Démarrage de la stack"
echo "============================================"

docker compose up -d --build


# ============================================================
# ATTENTE DE L'API
# ============================================================

echo
echo "Attente de l'API..."

API_READY=false

for ((i=1; i<=MAX_RETRIES; i++))
do

    if curl \
        --noproxy "*" \
        --silent \
        --fail \
        "${API_URL}/health" \
        > /dev/null
    then

        API_READY=true
        break
    fi

    echo "Tentative ${i}/${MAX_RETRIES}..."

    sleep "${SLEEP_SECONDS}"

done


if [ "${API_READY}" != "true" ]
then

    echo "ERREUR : API non disponible."
    exit 1

fi


echo "OK - API disponible"


# ============================================================
# TEST MLFLOW
# ============================================================

echo
echo "Test MLflow..."

curl \
    --noproxy "*" \
    --silent \
    --fail \
    "${MLFLOW_URL}/health" \
    > /dev/null

echo "OK - MLflow disponible"


# ============================================================
# TEST IHM STREAMLIT
# ============================================================

echo
echo "Test Streamlit..."

curl \
    --noproxy "*" \
    --silent \
    --fail \
    "${UI_URL}" \
    > /dev/null

echo "OK - IHM disponible"


# ============================================================
# TEST /HEALTH
# ============================================================

echo
echo "Test /health..."

HEALTH_RESPONSE=$(
    curl \
        --noproxy "*" \
        --silent \
        --fail \
        "${API_URL}/health"
)

echo "${HEALTH_RESPONSE}"


# ============================================================
# TEST D'UNE VRAIE PREDICTION
# ============================================================

echo
echo "Test /predict..."

curl \
    --noproxy "*" \
    --silent \
    --fail \
    -X POST \
    "${API_URL}/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "session_id": "smoke-test",
        "niveau_diplome": "Bac+2",
        "anciennete_poste_ans": 4.5,
        "code_rome_vise": "M1805",
        "synthese_entretien": "Profil autonome, recherche active."
    }' \
    > "${RESPONSE_FILE}"


cat "${RESPONSE_FILE}"


# ============================================================
# VALIDATION DE LA REPONSE
# ============================================================

python - "${RESPONSE_FILE}" <<'PY'

import json
import sys


path = sys.argv[1]

with open(
    path,
    "r",
    encoding="utf-8"
) as f:

    result = json.load(f)


# ------------------------------------------------------------
# prediction_id
# ------------------------------------------------------------

assert result.get(
    "prediction_id"
), "prediction_id absent"


# ------------------------------------------------------------
# classe
# ------------------------------------------------------------

assert result.get(
    "prediction"
) in {
    0,
    1,
    2
}, "classe prédite invalide"


# ------------------------------------------------------------
# version modèle
# ------------------------------------------------------------

assert result.get(
    "model_version"
), "model_version absent"


# ------------------------------------------------------------
# probabilités
# ------------------------------------------------------------

probabilities = result.get(
    "probabilities"
)

assert probabilities, (
    "probabilities absentes"
)

assert set(
    probabilities.keys()
) == {
    "0",
    "1",
    "2"
}, "classes de probabilités invalides"


total = sum(
    probabilities.values()
)

assert abs(
    total - 1.0
) < 1e-6, (
    f"Somme des probabilités invalide : {total}"
)


print()
print(
    "Réponse de prédiction valide"
)

print(
    f"prediction_id = {result['prediction_id']}"
)

print(
    f"prediction    = {result['prediction']}"
)

print(
    f"model_version = {result['model_version']}"
)

PY


# ============================================================
# RESULTAT
# ============================================================

echo
echo "============================================"
echo "SMOKE TEST : SUCCES"
echo "============================================"
echo "MLflow      : OK"
echo "FastAPI     : OK"
echo "Streamlit   : OK"
echo "Prediction  : OK"
echo "============================================"