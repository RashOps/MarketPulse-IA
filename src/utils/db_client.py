import os
from pymongo import MongoClient
from dotenv import load_dotenv
from logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Load environment variables
load_dotenv(dotenv_path="../../.env") # Or "../../.env.exemple" for testing
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

# Function to get the database connection
def get_db() -> MongoClient:
    """"Returns the MongoDB database connection."""

    if db is None:
        logger.error("Database connection is not established.")
        raise ConnectionError("Database connection is not established.")

    logger.info("Returning database connection.")
    return db