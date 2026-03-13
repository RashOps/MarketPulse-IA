## Architecture du projet

marketpulse-ai/
├── data/               # Stockage local temporaire (CSV/JSON)
├── logs/               # Fichiers .log générés par ton système
├── src/
│   ├── ingestion/      # Collecte : scraper.py, api_collector.py
│   ├── processing/     # Logique : cleaner.py, features.py
│   ├── models/         # IA : pca_model.py, clustering.py
│   ├── api/            # Backend : main.py, schemas.js
│   └── utils/          # Helpers : logger.py, db_client.py
├── .env                # Secrets (API Keys, MongoDB URI)
├── .gitignore          # Exclure .env, logs/ et data/
├── requirements.txt    # Tes dépendances Python
└── README.md           # Ta doc projet