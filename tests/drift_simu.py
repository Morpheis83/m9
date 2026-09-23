import random
import requests

API_URL = "http://127.0.0.1:8000/predict"

diplomes = [
    "Sans diplôme",
    "Bac",
    "Bac+2",
    "Bac+5"
]

romes = [
    "M1805",
    "D1503",
    "G1302",
    "A1203",
    "N1301"
]

textes = [
    "Profil autonome, recherche active.",
    "Mobilité géographique possible.",
    "Besoin de formation complémentaire.",
    "Difficultés de mobilité.",
    "Compétences numériques à actualiser."
]

for i in range(50):

    payload = {
        "session_id": f"drift-test-{i}",
        "niveau_diplome": random.choice(diplomes),
        "anciennete_poste_ans": round(
            random.uniform(0, 20),
            1
        ),
        "code_rome_vise": random.choice(romes),
        "synthese_entretien": random.choice(textes)
    }

    response = requests.post(
        API_URL,
        json=payload
    )

    print(
        i,
        response.status_code
    )