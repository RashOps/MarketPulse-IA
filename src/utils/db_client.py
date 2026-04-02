import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# .env à la racine du projet (indépendant du répertoire courant)
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

# Validate environment variables
# Variable globale privée pour stocker l'instance unique (Singleton)
_client: MongoClient | None = None

def get_db() -> Database:
    """Retourne l'objet Database MongoDB via Lazy Initialization."""
    global _client
    
    # On ne crée la connexion que si elle n'existe pas encore
    if _client is None:
        try:
            logger.debug("Initialisation de la connexion MongoDB...")
            _client = MongoClient(MONGO_URI)
            _client.admin.command('ping')
            logger.info("Connecté à MongoDB avec succès !")
        except Exception as e:
            logger.critical("Échec critique de la connexion à MongoDB : %s", e)
            raise e
            
    return _client[DB_NAME]