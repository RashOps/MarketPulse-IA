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
if MONGO_URI is None or DB_NAME is None:
    logger.error("MONGO_URI and DB_NAME must be set in the .env file.")
    raise ValueError("MONGO_URI and DB_NAME must be set in the .env file.")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Test the connection
try:
    client.admin.command('ping')
    logger.info("Connected to MongoDB successfully!")
    print("Connected to MongoDB successfully!")
except Exception as e:
    logger.error("Failed to connect to MongoDB.")
    print(e)

def get_db() -> Database:
    """Retourne l'objet Database MongoDB."""
    logger.debug("Returning database handle.")
    return db