## MarketPulse AI

**Real-time Market Segmentation Engine using Unsupervised Learning.**

MarketPulse AI est un moteur de segmentation de marchés en temps (quasi) réel basé sur l’apprentissage non supervisé.  
Il automatise la collecte de données de marché (API & scraping web), leur prétraitement avancé, puis applique une chaîne **Standardisation → PCA → K-Means** pour détecter des **segments et tendances émergentes**.

Le backend est exposé via **FastAPI** afin de servir les résultats (centres de clusters, statistiques par segment, projections PCA, etc.) à un frontend type **React / Next.js** ou à tout autre consommateur d’API.

---

### 1. Objectifs du projet

- **Automatisation**
  - Mettre en place un flux capable de récupérer des données de marché sans intervention manuelle (tâches planifiées, scripts d’ingestion).
- **Analyse avancée**
  - Transformer des données brutes (prix, volumes, ratios, indicateurs techniques, fondamentaux…) en **segments de marché actionnables** via clustering.
- **Interopérabilité**
  - Exposer une **API REST** claire (FastAPI) pour alimenter un dashboard ou un portfolio **React / Next.js** (ou tout autre client).

---

### 2. Stack technique

- **Langage** : Python 3.x
- **Backend** : FastAPI
- **ML / Data** : scikit-learn (StandardScaler, PCA, KMeans, Silhouette Score), pandas, numpy
- **Ingestion** : `requests` (APIs), `BeautifulSoup4` (scraping HTML), éventuellement `schedule` / `APScheduler` pour l’ordonnancement
- **Base de données** : MongoDB Atlas (via `pymongo`) pour le stockage temporaire ou historique
- **Logging & utilitaires** : `logging`, utilitaires custom (`logger.py`, `db_client.py`, etc.)

---

### 3. Architecture globale (Pipeline)

Le système est découpé en plusieurs couches, chacune avec une responsabilité claire.

#### 3.1 Ingestion Layer (Data Engineering)

- **Objectif** : récupérer les données brutes depuis différentes sources.
- **Mécanismes** :
  - **API publiques ou privées** (ex. marchés financiers / crypto) via `requests`.
  - **Scraping HTML** de sites de marché via `BeautifulSoup4` lorsque l’API n’existe pas ou est limitée.
- **Stockage** :
  - Sauvegarde intermédiaire dans des fichiers (`data/` – CSV, JSON, parquet, etc.).
  - Option de **stockage historique** dans MongoDB Atlas (`pymongo`) pour rejouer des analyses ou entraîner des modèles plus robustes.

#### 3.2 Processing Layer (Logic)

- **Nettoyage des données** :
  - Gestion des valeurs manquantes (imputation, suppression, remplissage par médiane / moyenne / valeur constante).
  - Gestion des outliers (winsorisation, cap, filtrage par quantiles ou z-score).
  - Harmonisation des types, parsing de dates, tri chronologique, etc.
- **Standardisation** :
  - Application impérative du **Z-score** sur les variables numériques :
    - \( x' = \frac{x - \mu}{\sigma} \)
  - Utilisation de `StandardScaler` de scikit-learn pour garantir que les distances (Euclidienne / Manhattan) soient cohérentes pour le clustering.

#### 3.3 ML Layer (Intelligence)

- **PCA (Analyse en Composantes Principales)** :
  - Objectif : réduire la dimension tout en conservant le maximum de variance expliquée.
  - Sélection du nombre de composantes selon :
    - **Critère de Kaiser** (valeurs propres \> 1),
    - ou **règle du coude** sur la variance expliquée cumulée.
  - Production de matrices :
    - composantes principales,
    - variance expliquée par composante,
    - projection des observations dans l’espace PCA.

- **K-Means (Clustering)** :
  - Partitionnement en \(k\) clusters à partir des données (éventuellement projetées dans l’espace PCA).
  - **Recherche du nombre optimal de clusters** :
    - calcul du **Silhouette Score** pour plusieurs valeurs de \(k\),
    - choix du \(k\) qui maximise (ou se situe dans une zone stable de) la silhouette.
  - Sorties principales :
    - centres de clusters,
    - affectation cluster de chaque observation,
    - statistiques par segment (moyennes, médianes, dispersion, etc.).

#### 3.4 Serving Layer (Backend FastAPI)

- Exposition d’endpoints REST permettant à un frontend ou à des scripts de consommer l’intelligence produite :
  - récupération des **centres de clusters** ;
  - récupération des **statistiques agrégées par segment** ;
  - accès aux **paramètres du modèle** (nombre de composantes PCA, \(k\), scores de silhouette, etc.) ;
  - éventuellement déclenchement manuel d’une **nouvelle exécution du pipeline**.

---

### 4. Arborescence du projet

Arborescence indicative (peut évoluer avec le développement) :

```bash
marketpulse-ai/
├── data/               # Stockage local temporaire (CSV/JSON)
├── logs/               # Fichiers .log générés par le système
├── src/
│   ├── ingestion/      # Collecte : scraper.py, api_collector.py
│   ├── processing/     # Logique : cleaner.py, features.py
│   ├── models/         # IA : pca_model.py, clustering.py
│   ├── api/            # Backend : main.py, schemas.py
│   └── utils/          # Helpers : logger.py, db_client.py
├── .env                # Secrets (API Keys, MongoDB URI)
├── .gitignore          # Exclure .env, logs/ et data/
├── requirements.txt    # Dépendances Python
└── README.md           # Documentation du projet
```

---

### 5. Installation & démarrage

#### 5.1 Prérequis

- Python 3.11+ (recommandé)
- Accès à une base **MongoDB Atlas** (URI dans `.env`)
- Accès internet pour interroger les APIs / sites de marché

#### 5.2 Cloner le dépôt

```bash
git clone <URL_DU_DEPOT>
cd MarketPulse_AI
```

#### 5.3 Environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.\.venv\Scripts\activate       # Windows (PowerShell)
```

#### 5.4 Installation des dépendances

```bash
pip install -r requirements.txt
```

#### 5.5 Configuration des variables d’environnement

Créer un fichier `.env` à la racine du projet avec par exemple :

```bash
MONGODB_URI="mongodb+srv://..."
API_KEY_MARKET_DATA="..."
ENV="dev"
```

#### 5.6 Lancer l’API FastAPI

Depuis la racine du projet :

```bash
uvicorn src.api.main:app --reload
```

L’API sera disponible sur `http://127.0.0.1:8000` et la documentation interactive sur `http://127.0.0.1:8000/docs`.

---

### 6. Fonctionnement du pipeline

#### 6.1 Étapes principales

1. **Ingestion**
   - Appel d’APIs / scraping pour récupérer des données brutes (tickers, prix, volume, indicateurs, etc.).
   - Sauvegarde dans `data/` ou directement dans MongoDB.
2. **Prétraitement**
   - Nettoyage, gestion des valeurs manquantes, outliers, filtrage des variables inutiles.
   - Standardisation **Z-score** via `StandardScaler`.
3. **PCA**
   - Calcul des composantes principales sur les variables standardisées.
   - Sélection du nombre de composantes selon Kaiser / coude.
4. **Clustering K-Means**
   - Test de plusieurs \(k\) (par ex. 2 à 10).
   - Calcul du **Silhouette Score** pour chaque \(k\), choix du meilleur.
   - Entrainement du modèle K-Means final.
5. **Serving**
   - Sauvegarde des artefacts importants (modèles, paramètres, stats) dans MongoDB / fichiers.
   - Exposition des résultats via FastAPI.

#### 6.2 Exécution manuelle

Selon l’implémentation, le pipeline pourra être :

- déclenché par une **commande CLI** (ex. `python -m src.pipeline.run`), ou
- orchestré par des **jobs planifiés** (cron, GitHub Actions, scheduler externe), ou
- déclenché via un **endpoint FastAPI** (ex. `/pipeline/run` en `POST`).  

Les détails exacts seront documentés dans les modules correspondants (`ingestion`, `processing`, `models`).  

---

### 7. API FastAPI (exemples d’endpoints)

> Les chemins précis peuvent évoluer, mais l’idée est d’exposer les briques suivantes.

- **`GET /health`**
  - Vérifie l’état du service (ping, connexion DB, version).

- **`GET /clusters/centers`**
  - Retourne les centres des clusters K-Means dans l’espace des features (ou PCA).

- **`GET /clusters/stats`**
  - Retourne des statistiques agrégées par cluster (moyenne des features, taille du cluster, etc.).

- **`GET /pca/explained-variance`**
  - Donne la variance expliquée par composante + variance cumulée.

- **`POST /pipeline/run`** (optionnel)
  - Force l’exécution d’un pipeline complet d’ingestion → preprocessing → PCA → K-Means.

Chaque endpoint sera documenté dans `src/api/schemas.py` (Pydantic) et visible via la doc Swagger (`/docs`).  

---

### 8. Intégration avec un frontend React / Next.js

Le projet est pensé pour être consommé par un frontend moderne :

- **Consommation REST** depuis un dashboard Next.js / React :
  - visualisation des clusters sur des scatter plots (PCA 2D / 3D),
  - affichage de fiches segments (profils moyens, liste des actifs par segment, etc.),
  - filtres dynamiques (par période, classe d’actifs, liquidité…).
- L’API FastAPI renvoie des **JSON structurés** pour faciliter l’intégration avec des librairies comme Recharts, Victory, D3, etc.

---

### 9. Roadmap (indicative)

- **v0.1 – Prototype local**
  - Ingestion simple (une ou deux sources de données).
  - Preprocessing de base + PCA + K-Means.
  - Exposition minimale via FastAPI (`/health`, `/clusters/centers`). 
- **v0.2 – Stabilisation & métriques**
  - Amélioration du nettoyage, gestion robuste des valeurs manquantes / outliers.
  - Mise en place systématique du Silhouette Score + logs détaillés.
  - Stockage des modèles et résultats dans MongoDB.
- **v0.3 – Intégration front**
  - Ajout d’endpoints dédiés pour dashboard React/Next.js.
  - Documentation API renforcée (OpenAPI).
  - Premiers graphiques de visualisation côté frontend.

---

### 10. Contribution & bonnes pratiques

- **Structure** : respecter l’architecture des dossiers (`ingestion`, `processing`, `models`, `api`, `utils`).
- **Qualité de code** :
  - préférer des fonctions pures testables,
  - typer les fonctions (type hints),
  - ajouter des tests unitaires pour les briques critiques (standardisation, PCA, choix de \(k\), etc.).
- **Sécurité** :
  - ne jamais committer le fichier `.env`,
  - vérifier que les logs ne contiennent pas de secrets (API keys, URI complètes…).

---