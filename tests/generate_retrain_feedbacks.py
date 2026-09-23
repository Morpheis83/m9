import random
import time
import os
import requests


session = requests.Session()

# Ne pas utiliser HTTP_PROXY / HTTPS_PROXY de l'environnement
session.trust_env = False


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"

# Désactivation des proxys pour les appels locaux
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

NB_FEEDBACKS = 20

SESSION_PREFIX = "retrain-test"


# ============================================================
# DONNEES SYNTHETIQUES
# ============================================================

diplomes = [
    "Sans diplôme",
    "Bac",
    "Bac+2",
    "Bac+5"
]


codes_rome = [
    "M1805",
    "D1503",
    "G1302",
    "A1203",
    "N1301"
]


syntheses = [
    "Profil autonome avec une recherche active.",
    "Projet professionnel défini et mobilité possible.",
    "Besoin de formation complémentaire avant retour à l'emploi.",
    "Difficultés de mobilité freinant la recherche.",
    "Compétences numériques nécessitant une actualisation.",
    "Expérience professionnelle cohérente avec le métier recherché.",
    "Projet de reconversion nécessitant un accompagnement.",
    "Recherche active mais difficultés à obtenir des entretiens."
]


# ============================================================
# CLASSES REELLES SYNTHETIQUES
#
# On équilibre volontairement 0 / 1 / 2 pour le test.
# Ce jeu ne doit PAS être interprété comme une vérité métier.
# ============================================================

actual_classes = [
    0, 1, 2,
    0, 1, 2,
    0, 1, 2,
    0, 1, 2,
    0, 1, 2,
    0, 1, 2,
    0, 1
]


# ============================================================
# GENERATION
# ============================================================

success_count = 0


for i in range(
    NB_FEEDBACKS
):

    session_id = (
        f"{SESSION_PREFIX}-{i:02d}"
    )


    payload = {

        "session_id":
            session_id,

        "niveau_diplome":
            random.choice(
                diplomes
            ),

        "anciennete_poste_ans":
            round(
                random.uniform(
                    0,
                    20
                ),
                1
            ),

        "code_rome_vise":
            random.choice(
                codes_rome
            ),

        "synthese_entretien":
            random.choice(
                syntheses
            )
    }


    # ========================================================
    # 1. PREDICTION
    # ========================================================

    prediction_response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=10
    )


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


    # ========================================================
    # 2. FEEDBACK
    # ========================================================

    actual_class = (
        actual_classes[
            i
        ]
    )


    feedback_response = requests.post(
        f"{API_URL}/feedback",
        json={
            "prediction_id":
                prediction_id,

            "actual_class":
                actual_class
        },
        timeout=10
    )


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


# ============================================================
# RESUME
# ============================================================

print()
print(
    "============================================"
)

print(
    "GENERATION TERMINEE"
)

print(
    "============================================"
)

print(
    f"Feedbacks créés : "
    f"{success_count}/{NB_FEEDBACKS}"
)

print(
    "============================================"
)