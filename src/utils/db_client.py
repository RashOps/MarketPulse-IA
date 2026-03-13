import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "marketpulse")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def get_db():
    return db
