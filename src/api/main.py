from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import Dict, Any
import pymongo

from src.utils.logger import get_logger
from src.processing.cleaner import load_market_data
from src.processing.features import engineer_features, scale_features
from src.ingestion.scraper import fetch_latest_financial_news, save_news_to_db
from src.models.pca_model import apply_pca
from src.models.clustering import apply_clustering
from src.models.profiling import generate_cluster_profiles, assign_business_labels
from src.utils.db_client import get_db
from src.api.schemas import MarketSegmentsResponse, MarketNewsResponse

logger = get_logger(__name__)

app = FastAPI(
    title="MarketPulse AI Engine",
    description="API de segmentation topologique des marchés financiers.",
    version="1.0.0"
)

# Configuration stricte des CORS pour le Front-end Next.js
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # Ajoute ici l'URL de production de ton portfolio plus tard
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

def run_full_pipeline() -> None:
    """
    Exécute le pipeline ML complet en arrière-plan.
    En production, les résultats devraient être sauvegardés dans MongoDB.
    """
    logger.info("Déclenchement manuel du pipeline ML End-to-End.")
    try:
        # 1. Pipeline ML
        df_raw = load_market_data()
        df_engineered = engineer_features(df_raw)
        df_scaled = scale_features(df_engineered, is_training=True)
        df_pca = apply_pca(df_scaled, n_components=2, is_training=True)
        df_final = apply_clustering(df_pca, max_k=5, is_training=True)
        logger.info("Pipeline ML exécuté avec succès.")
    except Exception as e:
        logger.error(f"Erreur critique lors de l'exécution du pipeline : {e}")


@app.get("/status", tags=["Health"])
async def get_status() -> Dict[str, str]:
    """Vérifie la santé de la pipeline."""
    return {"status": "online", "message": "MarketPulse AI is ready."}


# Ajout du response_model pour la validation stricte
@app.get("/market-segments", response_model=MarketSegmentsResponse, tags=["Intelligence"])
async def get_market_segments() -> Any:
    """
    Récupère les derniers segments identifiés avec leurs caractéristiques.
    """
    try:
        df_raw = load_market_data()
        if df_raw is None or df_raw.empty:
            raise HTTPException(status_code=404, detail="Aucune donnée de marché disponible.")

        df_engineered = engineer_features(df_raw)
        df_scaled = scale_features(df_engineered, is_training=False)
        df_pca = apply_pca(df_scaled, n_components=2, is_training=False)
        df_final = apply_clustering(df_pca, is_training=False)
        
        # Mapping Business
        cluster_profiles = generate_cluster_profiles(df_final)
        business_labels = assign_business_labels(cluster_profiles)
        df_final['business_segment'] = df_final['cluster'].map(business_labels).fillna("Segment Inconnu")

        # Nettoyage des NaNs pour le format JSON
        df_final = df_final.fillna(0)

        # Transformation en dictionnaire orienté 'records' pour l'API
        payload = {
            "metadata": {
                "total_assets": len(df_final),
                "clusters_identified": len(business_labels)
            },
            "data": df_final[['ticker', 'PCA_1', 'PCA_2', 'cluster', 'business_segment', 'metrics_volume']].to_dict(orient="records"),
            "profiles": cluster_profiles.to_dict(orient="records")
        }
        
        # FastAPI se charge de caster ce dictionnaire vers le MarketSegmentsResponse
        return payload

    except Exception as e:
        logger.exception("Échec de la génération des segments.")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trigger-update", tags=["Operations"])
async def trigger_update(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """
    Force une nouvelle collecte et un ré-entraînement du modèle en asynchrone.
    """
    background_tasks.add_task(run_full_pipeline)
    return {"status": "processing", "message": "Pipeline d'entraînement lancé en arrière-plan."}


# Ajout du response_model
@app.get("/market-news", response_model=MarketNewsResponse, tags=["News Feed"])
async def get_market_news(limit: int = 15, source: str = None) -> Any:
    """
    Récupère le flux d'actualités financières depuis MongoDB.
    Permet de filtrer par source et de limiter le nombre de résultats.
    """
    try:
        db = get_db()
        collection = db["market-news"]
        
        query = {}
        if source:
            query["source"] = source
            
        cursor = collection.find(query).sort("published", pymongo.DESCENDING).limit(limit)
        news_list = list(cursor)
        
        for news in news_list:
            # Conversion du ObjectId indéchiffrable par FastAPI en string classique
            news["_id"] = str(news["_id"])
            
            if "published" in news and news["published"]:
                news["published"] = news["published"].isoformat()
                
            # Gestion de la date d'ingestion pour le schéma Pydantic
            if "ingested_at" in news and news["ingested_at"]:
                news["ingested_at"] = news["ingested_at"].isoformat()
                
        logger.info(f"Serveur API : {len(news_list)} articles renvoyés.")
        
        payload = {
            "metadata": {
                "count": len(news_list),
                "filter_applied": source if source else "None"
            },
            "data": news_list
        }
        
        return payload

    except Exception as e:
        logger.error(f"Échec de la récupération des news : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne de la base de données.")

def background_news_scraper(limit: int = 5) -> None:
    """
    Fonction worker qui exécute le scraping réel hors du flux principal de l'API.
    """
    logger.info("Démarrage du job de scraping en arrière-plan...")
    try:
        news_dict = fetch_latest_financial_news(limit_per_source=limit)
        save_news_to_db(news_dict)
        logger.info("Job de scraping terminé avec succès.")
    except Exception as e:
        logger.error(f"Le job de scraping a échoué : {e}")


@app.post("/scrape-news", tags=["Operations", "Ingestion"])
async def trigger_news_scraping(background_tasks: BackgroundTasks, limit_per_source: int = 5) -> Dict[str, str]:
    """
    Déclenche le scraper de news financières à la demande.
    L'opération s'exécute en arrière-plan pour ne pas bloquer le client.
    """
    background_tasks.add_task(background_news_scraper, limit_per_source)
    
    return {
        "status": "processing", 
        "message": f"Le scraping de {limit_per_source} articles par source a été lancé en arrière-plan."
    }