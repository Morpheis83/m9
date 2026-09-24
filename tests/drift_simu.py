"""Génère des prédictions synthétiques pour tester le monitoring du drift.

Ce script alimente l'API avec des observations artificielles. Les données
produites servent uniquement à valider le fonctionnement du monitoring
et ne représentent pas une distribution métier réelle.
"""

import random

import requests


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = (
    "http://127.0.0.1:8000/predict"
)

NB_PREDICTIONS = 50

REQUEST_TIMEOUT = 10


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

TEXTES = [
    "Profil autonome, recherche active.",
    "Mobilité géographique possible.",
    "Besoin de formation complémentaire.",
    "Difficultés de mobilité.",
    "Compétences numériques à actualiser.",
]


# ============================================================
# GENERATION
# ============================================================


def main() -> None:
    """Envoie les observations synthétiques à l'API."""

    for i in range(
        NB_PREDICTIONS
    ):
        payload = {
            "session_id": (
                f"drift-test-{i}"
            ),
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
                TEXTES
            ),
        }

        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            print(
                f"[{i + 1:02d}/{NB_PREDICTIONS}] "
                f"HTTP {response.status_code}"
            )

        except requests.RequestException as exc:
            print(
                f"[{i + 1:02d}/{NB_PREDICTIONS}] "
                f"ERREUR : {exc}"
            )


if __name__ == "__main__":
    main()