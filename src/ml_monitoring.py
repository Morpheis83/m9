from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score
)

from src.history_store import (
    donnees_monitoring
)

from src.drift import (
    charger_reference_drift,
    calculer_drift
)

from src.metrics import (
    ML_PREDICTIONS_PERSISTED,
    ML_LABELED_PREDICTIONS,
    ML_ENRICHMENT_RATIO,
    ML_ACCURACY,
    ML_F1_MACRO,
    ML_RECALL_CLASS_2,
    DATA_DRIFT_PSI,
    DATA_DRIFT_SAMPLE_SIZE,
    DATA_DRIFT_READY,
    DATA_DRIFT_MAX_PSI,
    DATA_DRIFT_ALERT_FEATURES
)


PERFORMANCE_WINDOW = 5000

DRIFT_WINDOW = 500

MIN_DRIFT_SAMPLES = 50


def mettre_a_jour_metriques_ml(
    drift_reference_path: Path
):

    # ========================================================
    # CHARGEMENT HISTORIQUE
    # ========================================================

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


    # --------------------------------------------------------
    # Pas de feedback :
    # métriques de performance non calculables
    # --------------------------------------------------------

    if total_labeled == 0:

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
                record[
                    "actual_class"
                ]
            )

            for record
            in labeled_records
        ]


        y_pred = [

            int(
                record[
                    "predicted_class"
                ]
            )

            for record
            in labeled_records
        ]


        ML_ACCURACY.set(
            accuracy_score(
                y_true,
                y_pred
            )
        )


        ML_F1_MACRO.set(
            f1_score(
                y_true,
                y_pred,
                labels=[
                    0,
                    1,
                    2
                ],
                average="macro",
                zero_division=0
            )
        )


        # Le recall classe 2 n'a de sens que
        # si une classe 2 existe réellement
        # dans les feedbacks.

        if 2 in y_true:

            ML_RECALL_CLASS_2.set(
                recall_score(
                    y_true,
                    y_pred,
                    labels=[
                        2
                    ],
                    average="macro",
                    zero_division=0
                )
            )

        else:

            ML_RECALL_CLASS_2.set(
                float("nan")
            )


    # ========================================================
    # DATA DRIFT
    # ========================================================

    drift_records = records[
        :DRIFT_WINDOW
    ]


    DATA_DRIFT_SAMPLE_SIZE.set(
        len(
            drift_records
        )
    )


    if (
        len(
            drift_records
        )
        < MIN_DRIFT_SAMPLES
    ):

        DATA_DRIFT_READY.set(
            0
        )

        DATA_DRIFT_MAX_PSI.set(
            float("nan")
        )

        DATA_DRIFT_ALERT_FEATURES.set(
            0
        )

        return


    DATA_DRIFT_READY.set(
        1
    )


    dataframe = pd.DataFrame(
        drift_records
    )


    reference = (
        charger_reference_drift(
            drift_reference_path
        )
    )


    psi_values = calculer_drift(
        dataframe,
        reference
    )


    for feature, psi in (
        psi_values.items()
    ):

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

        for psi
        in psi_values.values()

        if psi >= 0.25
    )


    DATA_DRIFT_ALERT_FEATURES.set(
        alert_features
    )