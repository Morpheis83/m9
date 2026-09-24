"""Génère des feedbacks synthétiques pour tester le réentraînement.

Le script crée des prédictions via l'API puis leur associe des vérités
terrain artificielles. Ces données servent uniquement à valider le workflow
MLOps et ne doivent pas être interprétées comme des observations métier.
"""

import os
import random
import time

import requests


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = (
    "http://127.0.0.1:8000"
)

NB_FEEDBACKS = 20

SESSION_PREFIX = (
    "retrain-test"
)

REQUEST_TIMEOUT = 10


# Les appels vers l'API locale ne doivent pas traverser
# les proxys HTTP éventuellement configurés sur la machine.
os.environ[
    "NO_PROXY"
] = "127.0.0.1,localhost"

os.environ[
    "no_proxy"
] = "127.0.0.1,localhost"


# ============================================================
# DONNEES SYNTHETIQUES
# ============================================================

DIPLOMES = [
    "Sans diplôme",
    "Bac",
    "Bac+2",
    "Bac+5",
]

CODES_ROME = [
    "M1805",
    "D1503",
    "G1302",
    "A1203",
    "N1301",
]

SYNTHESES = [
    "Profil autonome avec une recherche active.",
    "Projet professionnel défini et mobilité possible.",
    (
        "Besoin de formation complémentaire "
        "avant retour à l'emploi."
    ),
    "Difficultés de mobilité freinant la recherche.",
    (
        "Compétences numériques nécessitant "
        "une actualisation."
    ),
    (
        "Expérience professionnelle cohérente "
        "avec le métier recherché."
    ),
    (
        "Projet de reconversion nécessitant "
        "un accompagnement."
    ),
    (
        "Recherche active mais difficultés "
        "à obtenir des entretiens."
    ),
]

# Les classes sont volontairement équilibrées pour tester
# le workflow. Elles ne représentent aucune vérité métier.
ACTUAL_CLASSES = [
    0, 1, 2,
    0, 1, 2,
    0, 1, 2,
    0, 1, 2,
    0, 1, 2,
    0, 1, 2,
    0, 1,
]


# ============================================================
# GENERATION
# ============================================================


def main() -> None:
    """Crée des prédictions puis leur associe des feedbacks."""

    session = requests.Session()

    # Ignore HTTP_PROXY et HTTPS_PROXY de l'environnement
    # pour les appels à l'API locale.
    session.trust_env = False

    success_count = 0

    for i in range(
        NB_FEEDBACKS
    ):
        session_id = (
            f"{SESSION_PREFIX}-{i:02d}"
        )

        payload = {
            "session_id": session_id,
            "niveau_diplome": random.choice(
                DIPLOMES
            ),
            "anciennete_poste_ans": round(
                random.uniform(
                    0,
                    20,
                ),
                1,
            ),
            "code_rome_vise": random.choice(
                CODES_ROME
            ),
            "synthese_entretien": random.choice(
                SYNTHESES
            ),
        }

        # Une prédiction doit être créée avant de pouvoir
        # lui associer une vérité terrain.
        try:
            prediction_response = session.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

        except requests.RequestException as exc:
            print(
                f"[{i + 1:02d}] "
                f"ERREUR /predict : {exc}"
            )
            continue

        if (
            prediction_response.status_code
            != 200
        ):
            print(
                f"[{i + 1:02d}] "
                f"ERREUR /predict : "
                f"{prediction_response.status_code} "
                f"{prediction_response.text}"
            )
            continue

        prediction_data = (
            prediction_response.json()
        )

        prediction_id = (
            prediction_data[
                "prediction_id"
            ]
        )

        predicted_class = (
            prediction_data[
                "prediction"
            ]
        )

        actual_class = (
            ACTUAL_CLASSES[i]
        )

        try:
            feedback_response = session.post(
                f"{API_URL}/feedback",
                json={
                    "prediction_id": prediction_id,
                    "actual_class": actual_class,
                },
                timeout=REQUEST_TIMEOUT,
            )

        except requests.RequestException as exc:
            print(
                f"[{i + 1:02d}] "
                f"ERREUR /feedback : {exc}"
            )
            continue

        if (
            feedback_response.status_code
            != 200
        ):
            print(
                f"[{i + 1:02d}] "
                f"ERREUR /feedback : "
                f"{feedback_response.status_code} "
                f"{feedback_response.text}"
            )
            continue

        success_count += 1

        print(
            f"[{i + 1:02d}/{NB_FEEDBACKS}] "
            f"OK "
            f"session={session_id} "
            f"prediction={predicted_class} "
            f"actual={actual_class} "
            f"id={prediction_id}"
        )

        time.sleep(
            0.1
        )

    print()
    print("============================================")
    print("GENERATION TERMINEE")
    print("============================================")
    print(
        f"Feedbacks créés : "
        f"{success_count}/{NB_FEEDBACKS}"
    )
    print("============================================")


if __name__ == "__main__":
    main()