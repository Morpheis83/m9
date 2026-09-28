# Retour Emploi — prototype d'aide à l'orientation

Ce projet explore une aide à l'orientation des demandeurs d'emploi à partir de données administratives et d'une synthèse d'entretien. Le modèle prédit une classe de délai de retour à l'emploi parmi trois classes (0, 1 et 2). Sa prédiction sert d'appui au conseiller.

Le dépôt rassemble l'analyse dans [le notebook](notebooks/cas-usage-evrard.ipynb) et un prototype MLOps exécutable avec Docker Compose. Il ne constitue pas une application de production raccordée au système d'information d'une agence.

## Prérequis

- Docker et le plugin Docker Compose ;
- les ports locaux 8000, 8501, 5000, 9090 et 3000 disponibles ;
- une connexion permettant de récupérer les images et les dépendances Python lors du premier démarrage.

Le CSV utilisé par le modèle se trouve dans `data/`. Son chemin est défini dans `src/config.py`. Le lancement de `model-init` échouera si ce fichier n'est pas présent sous le nom attendu.

## Démarrage

Depuis la racine du dépôt :

```bash
# Choisir un jeton pour les requêtes d'administration du prototype.
export ADMIN_TOKEN="choisir-un-jeton-local"

docker compose up -d --build
docker compose ps -a
```

Docker Compose initialise les volumes persistants et entraîne le modèle initial si `pipeline_model.joblib` ou `drift_reference.json` manque dans le volume `model-data`. Cette étape peut prendre plusieurs minutes au premier lancement. Le conteneur `model-init` doit ensuite afficher `Exited (0)` ; `runtime-init` est également une tâche ponctuelle. `retrain-worker` tourne sans serveur HTTP : son état normal est `Up`, sans mention `healthy`.

Pour suivre l'initialisation ou diagnostiquer un démarrage :

```bash
docker compose logs -f model-init
docker compose logs --tail=100 api retrain-worker
docker compose ps -a
```

Les accès locaux sont les suivants :

| Service | Adresse | Utilisation |
|---|---|---|
| Interface Streamlit | http://localhost:8501 | Prédiction, feedback et historique |
| API FastAPI | http://localhost:8000/docs | Contrat des endpoints et essais manuels |
| MLflow | http://localhost:5000 | Runs, artefacts et registre des modèles |
| Prometheus | http://localhost:9090 | Métriques collectées depuis l'API |
| Grafana | http://localhost:3000 | Tableau de bord ; identifiants de démonstration `admin` / `admin` |

Le jeton `ADMIN_TOKEN` doit être conservé dans le même terminal pour les requêtes d'administration. La configuration actuelle est destinée à un environnement local de démonstration : elle publie les services sur les ports de l'hôte.

## Essayer le parcours

La manière la plus simple est d'ouvrir Streamlit. L'interface permet de renseigner les quatre variables du scénario retenu, de consulter la prédiction et son identifiant, puis de saisir ultérieurement une classe observée et de retrouver les enregistrements dans l'historique.

Le même parcours est disponible par l'API :

```bash
curl -sS http://localhost:8000/health

curl -sS -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "niveau_diplome": "Bac+2",
    "anciennete_poste_ans": 4.5,
    "code_rome_vise": "M1805",
    "synthese_entretien": "Profil autonome, recherche active."
  }'
```

La réponse de `/predict` contient notamment `prediction_id`, la classe prédite, les probabilités et la version du modèle. Pour enregistrer une vérité terrain, remplacer l'identifiant ci-dessous par celui de la réponse :

```bash
curl -sS -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"prediction_id":"IDENTIFIANT_RETOURNE","actual_class":1}'

curl -sS "http://localhost:8000/history?limit=10"
```

Une demande de réentraînement nécessite le jeton administrateur :

```bash
curl -sS -X POST http://localhost:8000/retrain \
  -H "X-Admin-Token: ${ADMIN_TOKEN}"

curl -sS "http://localhost:8000/retrain/IDENTIFIANT_DU_JOB" \
  -H "X-Admin-Token: ${ADMIN_TOKEN}"
```

Le worker rejette normalement la demande si moins de **20 prédictions distinctes disposent d'un feedback**. Au-delà de ce seuil, il entraîne un candidat et applique les critères codés de performance et de latence. Une promotion approuvée remplace l'artefact partagé ; l'API le recharge sans redémarrage.

## Composants et données

| Service Docker | Rôle |
|---|---|
| `model-init` | Entraîne le modèle initial si les artefacts manquent ; enregistre le run et la version dans MLflow ; s'arrête. |
| `mlflow` | Conserve les expériences, métriques, artefacts et versions enregistrées. |
| `api` | Sert la prédiction, le feedback, l'historique, la santé, les métriques et les demandes de réentraînement. |
| `ui` | Fournit l'interface Streamlit. |
| `retrain-worker` | Traite les demandes de réentraînement enregistrées dans SQLite. |
| `prometheus` | Collecte uniquement `api:8000/metrics`. |
| `grafana` | Affiche les métriques provenant de Prometheus. |

SQLite est embarqué dans les services qui en ont besoin ; il n'existe pas de conteneur SQLite séparé. Les volumes Docker conservent l'historique, les modèles, les expériences MLflow et les données de supervision entre deux démarrages. L'architecture détaillée et les choix techniques sont décrits au chapitre 6 du [notebook](notebooks/cas-usage-evrard.ipynb).

Le scénario retenu est **le scénario 2 « Éthique »** : une régression logistique associée à un prétraitement des variables tabulaires et de `synthese_entretien`. Le texte passe par des reformulations prévues pour quatre phrases connues avant sa vectorisation TF-IDF.

## Tests et intégration continue

Le smoke test démarre la pile Docker et vérifie l'initialisation du modèle, une prédiction, un feedback, l'historique, les métriques, Prometheus, Grafana, Streamlit et la présence du worker :

```bash
export ADMIN_TOKEN="choisir-un-jeton-local"
bash tests/smoke_test.sh
```

Pour lancer les vérifications Python localement, avec Python 3.13 :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -v
ruff check src tests --select E9,F63,F7,F82
```

Le workflow GitHub Actions lance les tests et Ruff, construit les images et exécute le smoke test lors des pushes et des pull requests sur `main`. Il ne déploie pas l'application. Le smoke test contrôle le démarrage du worker, mais ne couvre pas encore un cycle complet de réentraînement et de promotion avec 20 feedbacks.

## Limites du prototype

- Sur le jeu de test réservé, le pipeline retenu obtient un **F1 macro de 0,6453** et un **rappel de la classe 2 de 0,6778**. Il manque 29 des 90 dossiers réellement en classe 2. Ces résultats restent sous les objectifs présentés dans le notebook ; la prédiction ne doit pas automatiser l'orientation.
- Le nettoyage éthique reconnaît quatre formulations exactes ; il n'analyse pas de façon générale tous les textes libres qui pourraient être saisis.
- L'accès des conseillers, la politique de conservation des données et l'intégration au système d'information métier ne sont pas implémentés. En particulier, `/history` et `/feedback` ne sont pas protégés par une authentification utilisateur.
- La décision de promotion du candidat est automatique selon des seuils codés ; aucune approbation humaine du nouveau modèle n'est imposée.
- Les identifiants de démonstration Grafana, le jeton d'administration local et l'exposition des ports doivent être revus avant tout accès extérieur.

Pour arrêter les services tout en gardant les volumes :

```bash
docker compose down
```

Pour repartir d'un environnement vierge, `docker compose down -v` efface également les volumes du projet, notamment l'historique SQLite, les artefacts de modèles et les données MLflow.
