#!/usr/bin/env bash

# Smoke test complet de la stack Docker Retour Emploi.
#
# Le script vérifie :
# - la validité de docker-compose.yml ;
# - le démarrage des services ;
# - l'initialisation du modèle ;
# - les principaux endpoints FastAPI ;
# - la persistance SQLite ;
# - les métriques Prometheus ;
# - le provisioning Grafana ;
# - Streamlit ;
# - le worker de réentraînement ;
# - la protection de l'endpoint /retrain.

set -Eeuo pipefail


# ============================================================
# CONFIGURATION
# ============================================================

API_URL="${API_URL:-http://127.0.0.1:8000}"
MLFLOW_URL="${MLFLOW_URL:-http://127.0.0.1:5000}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://127.0.0.1:9090}"
GRAFANA_URL="${GRAFANA_URL:-http://127.0.0.1:3000}"
STREAMLIT_URL="${STREAMLIT_URL:-http://127.0.0.1:8501}"

GRAFANA_USER="${GRAFANA_USER:-admin}"
GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin}"

MAX_RETRIES="${MAX_RETRIES:-40}"
SLEEP_SECONDS="${SLEEP_SECONDS:-3}"

TMP_DIR="$(mktemp -d)"

PREDICTION_FILE="${TMP_DIR}/prediction.json"
HEALTH_FILE="${TMP_DIR}/health.json"


# ============================================================
# OUTILS
# ============================================================

section() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}


success() {
    echo "OK - $1"
}


cleanup() {
    rm -rf "${TMP_DIR}"
}


show_logs() {
    echo
    echo "============================================================"
    echo "SMOKE TEST EN ECHEC - ETAT DOCKER"
    echo "============================================================"

    docker compose ps -a || true

    echo
    echo "---------------- API ----------------"
    docker compose logs --tail=100 api || true

    echo
    echo "------------- MODEL INIT -------------"
    docker compose logs --tail=100 model-init || true

    echo
    echo "---------- RETRAIN WORKER ------------"
    docker compose logs --tail=100 retrain-worker || true

    echo
    echo "------------- PROMETHEUS -------------"
    docker compose logs --tail=50 prometheus || true

    echo
    echo "--------------- GRAFANA --------------"
    docker compose logs --tail=100 grafana || true
}


wait_http() {
    local name="$1"
    local url="$2"

    echo
    echo "Attente : ${name}"

    for ((i = 1; i <= MAX_RETRIES; i++)); do
        if curl \
            --noproxy "*" \
            --silent \
            --fail \
            --max-time 5 \
            "${url}" \
            > /dev/null
        then
            success "${name}"
            return 0
        fi

        echo "  tentative ${i}/${MAX_RETRIES}"
        sleep "${SLEEP_SECONDS}"
    done

    echo "ERREUR - ${name} indisponible"
    return 1
}


trap cleanup EXIT
trap show_logs ERR


# ============================================================
# 1. VALIDATION DOCKER COMPOSE
# ============================================================

section "1. Validation Docker Compose"

docker compose config --quiet

success "docker-compose.yml valide"


# ============================================================
# 2. DEMARRAGE DE LA STACK
# ============================================================

section "2. Démarrage de la stack"

docker compose up \
    -d \
    --build


# ============================================================
# 3. DISPONIBILITE DES SERVICES
# ============================================================

section "3. Disponibilité des services"

wait_http \
    "MLflow" \
    "${MLFLOW_URL}/health"

wait_http \
    "FastAPI" \
    "${API_URL}/health"

wait_http \
    "Prometheus" \
    "${PROMETHEUS_URL}/-/healthy"

wait_http \
    "Grafana" \
    "${GRAFANA_URL}/api/health"

wait_http \
    "Streamlit" \
    "${STREAMLIT_URL}"


# ============================================================
# 4. MODEL-INIT
# ============================================================

section "4. Vérification model-init"

MODEL_INIT_CONTAINER="$(
    docker compose ps \
        -a \
        -q \
        model-init
)"

if [ -z "${MODEL_INIT_CONTAINER}" ]; then
    echo "ERREUR - conteneur model-init introuvable"
    exit 1
fi

MODEL_INIT_STATUS="$(
    docker inspect \
        "${MODEL_INIT_CONTAINER}" \
        --format '{{.State.Status}}'
)"

MODEL_INIT_EXIT_CODE="$(
    docker inspect \
        "${MODEL_INIT_CONTAINER}" \
        --format '{{.State.ExitCode}}'
)"

echo "Status    : ${MODEL_INIT_STATUS}"
echo "Exit code : ${MODEL_INIT_EXIT_CODE}"

if [ "${MODEL_INIT_STATUS}" != "exited" ] \
    || [ "${MODEL_INIT_EXIT_CODE}" != "0" ]
then
    echo "ERREUR - model-init n'a pas terminé correctement"
    exit 1
fi

success "model-init terminé avec exit code 0"


# ============================================================
# 5. HEALTHCHECK API
# ============================================================

section "5. Healthcheck API"

curl \
    --noproxy "*" \
    --silent \
    --fail \
    --max-time 10 \
    "${API_URL}/health" \
    > "${HEALTH_FILE}"

python - "${HEALTH_FILE}" <<'PY'
import json
import sys

with open(
    sys.argv[1],
    "r",
    encoding="utf-8",
) as file:
    result = json.load(file)

assert result["status"] == "ok"
assert result["model_loaded"] is True

assert result["model_version"] not in (
    None,
    "",
    "unknown",
)

print(
    "OK - modèle chargé "
    f"version={result['model_version']}"
)
PY


# ============================================================
# 6. PREDICTION
# ============================================================

section "6. Test /predict"

curl \
    --noproxy "*" \
    --silent \
    --fail \
    --max-time 10 \
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
    > "${PREDICTION_FILE}"

PREDICTION_ID="$(
    python - "${PREDICTION_FILE}" <<'PY'
import json
import sys

with open(
    sys.argv[1],
    "r",
    encoding="utf-8",
) as file:
    result = json.load(file)

assert result.get("prediction_id")
assert result.get("prediction") in {0, 1, 2}
assert result.get("model_version")

probabilities = result.get("probabilities")

assert probabilities

assert set(
    probabilities.keys()
) == {
    "0",
    "1",
    "2",
}

assert abs(
    sum(probabilities.values()) - 1.0
) < 1e-6

print(result["prediction_id"])
PY
)"

success "prédiction valide prediction_id=${PREDICTION_ID}"


# ============================================================
# 7. FEEDBACK
# ============================================================

section "7. Test /feedback"

FEEDBACK_RESPONSE="$(
    curl \
        --noproxy "*" \
        --silent \
        --fail \
        --max-time 10 \
        -X POST \
        "${API_URL}/feedback" \
        -H "Content-Type: application/json" \
        -d "{
            \"prediction_id\": \"${PREDICTION_ID}\",
            \"actual_class\": 1
        }"
)"

echo "${FEEDBACK_RESPONSE}" | python -c '
import json
import sys

result = json.load(sys.stdin)

assert result["status"] == "ok"
assert result["actual_class"] == 1

print("OK - feedback enregistré")
'


# ============================================================
# 8. HISTORIQUE SQLITE
# ============================================================

section "8. Test historique"

HISTORY_RESPONSE="$(
    curl \
        --noproxy "*" \
        --silent \
        --fail \
        --max-time 10 \
        "${API_URL}/history?limit=100"
)"

PREDICTION_ID="${PREDICTION_ID}" \
    python -c '
import json
import os
import sys

records = json.load(sys.stdin)

prediction_id = os.environ["PREDICTION_ID"]

record = next(
    (
        item
        for item in records
        if item["prediction_id"] == prediction_id
    ),
    None,
)

assert record is not None
assert record["actual_class"] == 1

print(
    "OK - prédiction + feedback présents dans SQLite"
)
' <<< "${HISTORY_RESPONSE}"


# ============================================================
# 9. METRIQUES FASTAPI
# ============================================================

section "9. Test métriques FastAPI"

METRICS="$(
    curl \
        --noproxy "*" \
        --silent \
        --fail \
        --max-time 10 \
        "${API_URL}/metrics"
)"

grep -q \
    "retour_emploi_model_available 1" \
    <<< "${METRICS}"

grep -q \
    "retour_emploi_predictions_total" \
    <<< "${METRICS}"

grep -q \
    "retour_emploi_ml_predictions_persisted" \
    <<< "${METRICS}"

grep -q \
    "retour_emploi_data_drift_ready" \
    <<< "${METRICS}"

success "métriques applicatives et ML exposées"


# ============================================================
# 10. PROMETHEUS
# ============================================================

section "10. Test Prometheus"

# Laisse à Prometheus le temps d'effectuer un scrape
# après la prédiction réalisée précédemment.
sleep 6

PROM_RESULT="$(
    curl \
        --noproxy "*" \
        --silent \
        --fail \
        --max-time 10 \
        --get \
        "${PROMETHEUS_URL}/api/v1/query" \
        --data-urlencode \
        "query=retour_emploi_model_available"
)"

echo "${PROM_RESULT}" | python -c '
import json
import sys

result = json.load(sys.stdin)

assert result["status"] == "success"

values = result["data"]["result"]

assert len(values) >= 1

value = float(
    values[0]["value"][1]
)

assert value == 1.0

print(
    "OK - Prometheus collecte les métriques"
)
'


# ============================================================
# 11. GRAFANA DATASOURCE
# ============================================================

section "11. Test datasource Grafana"

GRAFANA_DS="$(
    curl \
        --noproxy "*" \
        --silent \
        --fail \
        --max-time 10 \
        -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
        "${GRAFANA_URL}/api/datasources/uid/prometheus"
)"

echo "${GRAFANA_DS}" | python -c '
import json
import sys

result = json.load(sys.stdin)

assert result["uid"] == "prometheus"
assert result["type"] == "prometheus"
assert result["url"] == "http://prometheus:9090"

print(
    "OK - datasource Prometheus provisionnée"
)
'


# ============================================================
# 12. GRAFANA DASHBOARD
# ============================================================

section "12. Test dashboard Grafana"

GRAFANA_DASHBOARD="$(
    curl \
        --noproxy "*" \
        --silent \
        --fail \
        --max-time 10 \
        -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
        "${GRAFANA_URL}/api/dashboards/uid/retour-emploi-monitoring"
)"

echo "${GRAFANA_DASHBOARD}" | python -c '
import json
import sys

result = json.load(sys.stdin)

dashboard = result["dashboard"]

assert dashboard["uid"] == "retour-emploi-monitoring"
assert len(dashboard["panels"]) > 0

print(
    "OK - dashboard Retour Emploi provisionné"
)
'


# ============================================================
# 13. RETRAIN WORKER
# ============================================================

section "13. Test retrain-worker"

RETRAIN_CONTAINER="$(
    docker compose ps \
        -q \
        retrain-worker
)"

if [ -z "${RETRAIN_CONTAINER}" ]; then
    echo "ERREUR - retrain-worker introuvable"
    exit 1
fi

RETRAIN_RUNNING="$(
    docker inspect \
        "${RETRAIN_CONTAINER}" \
        --format '{{.State.Running}}'
)"

if [ "${RETRAIN_RUNNING}" != "true" ]; then
    echo "ERREUR - retrain-worker non actif"
    exit 1
fi

success "retrain-worker actif"


# ============================================================
# 14. SECURITE /RETRAIN
# ============================================================

section "14. Test protection /retrain"

HTTP_CODE="$(
    curl \
        --noproxy "*" \
        --silent \
        --output /dev/null \
        --write-out "%{http_code}" \
        --max-time 10 \
        -X POST \
        "${API_URL}/retrain" \
        -H "X-Admin-Token: mauvais-token"
)"

if [ "${HTTP_CODE}" != "403" ]; then
    echo \
        "ERREUR - /retrain devrait retourner 403, reçu ${HTTP_CODE}"
    exit 1
fi

success "/retrain protégé par token"


# ============================================================
# RESULTAT
# ============================================================

section "SMOKE TEST DOCKER : SUCCES"

echo "Docker Compose           : OK"
echo "MLflow                   : OK"
echo "model-init               : OK"
echo "FastAPI                  : OK"
echo "Modèle                   : OK"
echo "Prediction               : OK"
echo "Feedback                 : OK"
echo "SQLite / historique      : OK"
echo "Monitoring API / ML      : OK"
echo "Prometheus               : OK"
echo "Grafana datasource       : OK"
echo "Grafana dashboard        : OK"
echo "Streamlit                : OK"
echo "retrain-worker           : OK"
echo "Sécurité /retrain        : OK"

echo
echo "============================================================"