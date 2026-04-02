from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import Dict, Any, Optional
import pymongo

from src.config import settings
from src.utils.logger import get_logger
from src.processing.cleaner import load_market_data
from src.processing.features import engineer_features
from src.ingestion.scraper import fetch_latest_financial_news, save_news_to_db
from src.models.pipeline import model_engine
from src.models.profiling import generate_cluster_profiles, assign_business_labels
from src.utils.db_client import get_db
from src.api.schemas import MarketSegmentsResponse, MarketNewsResponse
from src.utils.exceptions import handle_marketpulse_exception, MarketPulseError

logger = get_logger(__name__)

app = FastAPI(
    title="MarketPulse AI Engine",
    description="Topological segmentation API for financial markets.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

def run_full_pipeline() -> None:
    """
    Executes the End-to-End ML pipeline in the background.
    """
    logger.info("Manual trigger of the End-to-End ML pipeline.")
    try:
        df_raw = load_market_data()
        if df_raw is None or df_raw.empty:
            logger.error("No data available for training.")
            return
            
        df_engineered = engineer_features(df_raw)
        model_engine.train(df_engineered)
        logger.info("ML pipeline executed successfully.")
    except Exception as e:
        logger.error(f"Critical error during pipeline execution: {e}")

@app.get("/status", tags=["Health"])
async def get_status() -> Dict[str, str]:
    """Check pipeline health status."""
    return {"status": "online", "message": "MarketPulse AI is ready."}

@app.get("/market-segments", response_model=MarketSegmentsResponse, tags=["Intelligence"])
async def get_market_segments() -> Any:
    """
    Retrieves the latest identified segments with their characteristics.
    """
    try:
        df_raw = load_market_data()
        if df_raw is None or df_raw.empty:
            raise HTTPException(status_code=404, detail="No market data available.")

        df_engineered = engineer_features(df_raw)
        df_final, _ = model_engine.predict(df_engineered)
        
        cluster_profiles = generate_cluster_profiles(df_final)
        business_labels = assign_business_labels(cluster_profiles)
        df_final['business_segment'] = df_final['cluster'].map(business_labels).fillna("Unknown Segment")

        df_final = df_final.fillna(0)

        payload = {
            "metadata": {
                "total_assets": len(df_final),
                "clusters_identified": len(business_labels)
            },
            "data": df_final[['ticker', 'PCA_1', 'PCA_2', 'cluster', 'business_segment', 'metrics_volume']].to_dict(orient="records"),
            "profiles": cluster_profiles.to_dict(orient="records")
        }
        
        return payload

    except MarketPulseError as e:
        raise handle_marketpulse_exception(e)
    except Exception as e:
        logger.exception("Failed to generate segments.")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/latest-metrics", tags=["Operations"])
async def get_latest_metrics() -> Dict[str, Any]:
    """
    Retrieves the last training metrics for monitoring purposes.
    """
    try:
        db = get_db()
        latest_metrics = db["model-metrics"].find_one(sort=[("timestamp", pymongo.DESCENDING)])
        if not latest_metrics:
            raise HTTPException(status_code=404, detail="No metrics found. Model might not have been trained yet.")
        
        latest_metrics["_id"] = str(latest_metrics["_id"])
        latest_metrics["timestamp"] = latest_metrics["timestamp"].isoformat()
        return latest_metrics
    except Exception as e:
        logger.error("Failed to retrieve metrics: %s", e)
        raise HTTPException(status_code=500, detail="Database error.")

@app.post("/trigger-update", tags=["Operations"])
async def trigger_update(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """
    Forces an asynchronous data collection and model retraining.
    """
    background_tasks.add_task(run_full_pipeline)
    return {"status": "processing", "message": "Training pipeline launched in background."}

@app.get("/market-news", response_model=MarketNewsResponse, tags=["News Feed"])
async def get_market_news(limit: int = 15, source: Optional[str] = None) -> Any:
    """
    Retrieves the financial news feed from MongoDB.
    Allows filtering by source and limiting the number of results.
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
            news["_id"] = str(news["_id"])
            if "published" in news and news["published"]:
                news["published"] = news["published"].isoformat()
            if "ingested_at" in news and news["ingested_at"]:
                news["ingested_at"] = news["ingested_at"].isoformat()
                
        logger.info(f"API Server: Returned {len(news_list)} articles.")
        
        payload = {
            "metadata": {
                "count": len(news_list),
                "filter_applied": source if source else "None"
            },
            "data": news_list
        }
        
        return payload

    except Exception as e:
        logger.error(f"Failed to retrieve news: {e}")
        raise HTTPException(status_code=500, detail="Internal database error.")

def background_news_scraper(limit: int = 5) -> None:
    """
    Worker function that executes actual scraping outside the main API thread.
    """
    logger.info("Starting background scraping job...")
    try:
        news_dict = fetch_latest_financial_news(limit_per_source=limit)
        save_news_to_db(news_dict)
        logger.info("Scraping job completed successfully.")
    except Exception as e:
        logger.error(f"Scraping job failed: {e}")

@app.post("/scrape-news", tags=["Operations", "Ingestion"])
async def trigger_news_scraping(background_tasks: BackgroundTasks, limit_per_source: int = 5) -> Dict[str, str]:
    """
    Triggers the financial news scraper on demand.
    Runs in the background.
    """
    background_tasks.add_task(background_news_scraper, limit_per_source)
    return {
        "status": "processing", 
        "message": f"Scraping of {limit_per_source} articles per source launched in background."
    }

@app.post("/trigger-update", tags=["Operations"])
async def trigger_update(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """
    Forces an asynchronous data collection and model retraining.
    """
    background_tasks.add_task(run_full_pipeline)
    return {"status": "processing", "message": "Training pipeline launched in background."}

@app.get("/market-news", response_model=MarketNewsResponse, tags=["News Feed"])
async def get_market_news(limit: int = 15, source: Optional[str] = None) -> Any:
    """
    Retrieves the financial news feed from MongoDB.
    Allows filtering by source and limiting the number of results.
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
            news["_id"] = str(news["_id"])
            if "published" in news and news["published"]:
                news["published"] = news["published"].isoformat()
            if "ingested_at" in news and news["ingested_at"]:
                news["ingested_at"] = news["ingested_at"].isoformat()
                
        logger.info(f"API Server: Returned {len(news_list)} articles.")
        
        payload = {
            "metadata": {
                "count": len(news_list),
                "filter_applied": source if source else "None"
            },
            "data": news_list
        }
        
        return payload

    except Exception as e:
        logger.error(f"Failed to retrieve news: {e}")
        raise HTTPException(status_code=500, detail="Internal database error.")

def background_news_scraper(limit: int = 5) -> None:
    """
    Worker function that executes actual scraping outside the main API thread.
    """
    logger.info("Starting background scraping job...")
    try:
        news_dict = fetch_latest_financial_news(limit_per_source=limit)
        save_news_to_db(news_dict)
        logger.info("Scraping job completed successfully.")
    except Exception as e:
        logger.error(f"Scraping job failed: {e}")

@app.post("/scrape-news", tags=["Operations", "Ingestion"])
async def trigger_news_scraping(background_tasks: BackgroundTasks, limit_per_source: int = 5) -> Dict[str, str]:
    """
    Triggers the financial news scraper on demand.
    Runs in the background.
    """
    background_tasks.add_task(background_news_scraper, limit_per_source)
    return {
        "status": "processing", 
        "message": f"Scraping of {limit_per_source} articles per source launched in background."
    }