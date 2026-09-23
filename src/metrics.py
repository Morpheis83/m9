from prometheus_client import (
    Counter,
    Histogram,
    Gauge
)


# ============================================================
# API
# ============================================================

API_REQUESTS = Counter(
    "retour_emploi_api_requests_total",
    "Nombre total de requêtes HTTP",
    [
        "endpoint",
        "method",
        "status"
    ]
)


API_REQUEST_DURATION = Histogram(
    "retour_emploi_api_request_duration_seconds",
    "Durée des requêtes HTTP",
    [
        "endpoint"
    ]
)


# ============================================================
# PREDICTIONS
# ============================================================

PREDICTIONS = Counter(
    "retour_emploi_predictions_total",
    "Nombre de prédictions réalisées",
    [
        "predicted_class"
    ]
)


PREDICTION_DURATION = Histogram(
    "retour_emploi_prediction_duration_seconds",
    "Durée du calcul d'une prédiction"
)


# ============================================================
# MODELE
# ============================================================

MODEL_AVAILABLE = Gauge(
    "retour_emploi_model_available",
    "Disponibilité du modèle : 1 disponible, 0 indisponible"
)


# ============================================================
# FEEDBACK
# ============================================================

FEEDBACKS = Counter(
    "retour_emploi_feedback_total",
    "Nombre de résultats réels enregistrés",
    [
        "actual_class"
    ]
)

# ============================================================
# MONITORING ML
# ============================================================

ML_PREDICTIONS_PERSISTED = Gauge(
    "retour_emploi_ml_predictions_persisted",
    "Nombre de prédictions persistées dans SQLite"
)


ML_LABELED_PREDICTIONS = Gauge(
    "retour_emploi_ml_labeled_predictions",
    "Nombre de prédictions disposant d'une vérité terrain"
)


ML_ENRICHMENT_RATIO = Gauge(
    "retour_emploi_ml_enrichment_ratio",
    "Part des prédictions enrichies par un feedback"
)


ML_ACCURACY = Gauge(
    "retour_emploi_ml_accuracy",
    "Accuracy mesurée sur les prédictions enrichies"
)


ML_F1_MACRO = Gauge(
    "retour_emploi_ml_f1_macro",
    "F1 macro mesuré sur les prédictions enrichies"
)


ML_RECALL_CLASS_2 = Gauge(
    "retour_emploi_ml_recall_class_2",
    "Recall de la classe 2 sur les prédictions enrichies"
)


# ============================================================
# DATA DRIFT
# ============================================================

DATA_DRIFT_PSI = Gauge(
    "retour_emploi_data_drift_psi",
    "Population Stability Index par variable",
    [
        "feature"
    ]
)


DATA_DRIFT_SAMPLE_SIZE = Gauge(
    "retour_emploi_data_drift_sample_size",
    "Nombre d'observations utilisées pour mesurer le drift"
)


DATA_DRIFT_READY = Gauge(
    "retour_emploi_data_drift_ready",
    "1 si suffisamment de données sont disponibles pour mesurer le drift"
)


DATA_DRIFT_MAX_PSI = Gauge(
    "retour_emploi_data_drift_max_psi",
    "PSI maximum observé parmi les variables surveillées"
)


DATA_DRIFT_ALERT_FEATURES = Gauge(
    "retour_emploi_data_drift_alert_features",
    "Nombre de variables dont le PSI dépasse 0.25"
)