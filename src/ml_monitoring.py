"""Calcul des métriques ML et de dérive exposées à Prometheus.

Les métriques de performance utilisent uniquement les prédictions
disposant d'une vérité terrain. Le drift est calculé sur une fenêtre
récente de données de production comparée à la référence d'entraînement.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)

from src.drift import (
    calculer_drift,
    charger_reference_drift,
)
from src.history_store import donnees_monitoring
from src.metrics import (
    DATA_DRIFT_ALERT_FEATURES,
    DATA_DRIFT_MAX_PSI,
    DATA_DRIFT_PSI,
    DATA_DRIFT_READY,
    DATA_DRIFT_SAMPLE_SIZE,
    ML_ACCURACY,
    ML_ENRICHMENT_RATIO,
    ML_F1_MACRO,
    ML_LABELED_PREDICTIONS,
    ML_PREDICTIONS_PERSISTED,
    ML_RECALL_CLASS_2,
)


# ============================================================
# CONFIGURATION
# ============================================================

PERFORMANCE_WINDOW = 5000

DRIFT_WINDOW = 500

MIN_DRIFT_SAMPLES = 50

DRIFT_ALERT_THRESHOLD = 0.25


# ============================================================
# MONITORING ML
# ============================================================


def mettre_a_jour_metriques_ml(
    drift_reference_path: Path,
) -> None:
    """Recalcule les métriques ML et de drift exposées à Prometheus."""

    records = donnees_monitoring(
        limit=PERFORMANCE_WINDOW
    )

    total_predictions = len(
        records
    )

    ML_PREDICTIONS_PERSISTED.set(
        total_predictions
    )

    # ========================================================
    # PERFORMANCE AVEC VERITE TERRAIN
    # ========================================================

    # Les performances du modèle ne peuvent être mesurées que
    # lorsqu'une vérité terrain a été associée à la prédiction.
    labeled_records = [
        record
        for record in records
        if record.get(
            "actual_class"
        ) is not None
    ]

    total_labeled = len(
        labeled_records
    )

    ML_LABELED_PREDICTIONS.set(
        total_labeled
    )

    if total_predictions > 0:
        enrichment_ratio = (
            total_labeled
            / total_predictions
        )
    else:
        enrichment_ratio = 0.0

    ML_ENRICHMENT_RATIO.set(
        enrichment_ratio
    )

    if total_labeled == 0:
        # NaN distingue l'absence de vérité terrain d'une
        # performance réellement égale à zéro.
        ML_ACCURACY.set(
            float("nan")
        )

        ML_F1_MACRO.set(
            float("nan")
        )

        ML_RECALL_CLASS_2.set(
            float("nan")
        )

    else:
        y_true = [
            int(
                record["actual_class"]
            )
            for record in labeled_records
        ]

        y_pred = [
            int(
                record["predicted_class"]
            )
            for record in labeled_records
        ]

        ML_ACCURACY.set(
            accuracy_score(
                y_true,
                y_pred,
            )
        )

        ML_F1_MACRO.set(
            f1_score(
                y_true,
                y_pred,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        )

        # Le recall de la classe 2 n'est interprétable que
        # si cette classe existe dans les vérités terrain.
        if 2 in y_true:
            ML_RECALL_CLASS_2.set(
                recall_score(
                    y_true,
                    y_pred,
                    labels=[2],
                    average="macro",
                    zero_division=0,
                )
            )

        else:
            ML_RECALL_CLASS_2.set(
                float("nan")
            )

    # ========================================================
    # DATA DRIFT
    # ========================================================

    # L'historique étant retourné du plus récent au plus ancien,
    # les premières lignes constituent la fenêtre de production récente.
    drift_records = records[
        :DRIFT_WINDOW
    ]

    drift_sample_size = len(
        drift_records
    )

    DATA_DRIFT_SAMPLE_SIZE.set(
        drift_sample_size
    )

    # Un volume minimal est imposé pour éviter de déclencher
    # des alertes PSI sur un échantillon trop faible.
    if drift_sample_size < MIN_DRIFT_SAMPLES:
        DATA_DRIFT_READY.set(0)

        DATA_DRIFT_MAX_PSI.set(
            float("nan")
        )

        DATA_DRIFT_ALERT_FEATURES.set(
            0
        )

        return

    DATA_DRIFT_READY.set(1)

    dataframe = pd.DataFrame(
        drift_records
    )

    reference = charger_reference_drift(
        drift_reference_path
    )

    psi_values = calculer_drift(
        dataframe,
        reference,
    )

    for feature, psi in psi_values.items():
        DATA_DRIFT_PSI.labels(
            feature=feature
        ).set(
            psi
        )

    max_psi = max(
        psi_values.values()
    )

    DATA_DRIFT_MAX_PSI.set(
        max_psi
    )

    alert_features = sum(
        1
        for psi in psi_values.values()
        if psi >= DRIFT_ALERT_THRESHOLD
    )

    DATA_DRIFT_ALERT_FEATURES.set(
        alert_features
    )