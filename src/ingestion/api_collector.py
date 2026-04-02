import yfinance as yf
import time
import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from io import StringIO
import requests
import random

# Resolving Path issue
# _project_root = Path(__file__).resolve().parent.parent.parent
# if str(_project_root) not in sys.path:
#     sys.path.insert(0, str(_project_root))

from src.utils.logger import get_logger
from src.utils.db_client import get_db

# Initialize logger
logger = get_logger(__name__)

def get_dynamic_tickers(limit: int = 50) -> list[str]:
    """
    Récupère dynamiquement les Tickers du S&P 500 via scraping de tableau HTML.
    Intègre StringIO pour la compatibilité Pandas 2.2+ et le ciblage CSS.
    """
    logger.info("Récupération dynamique de la liste des tickers S&P 500...")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        
        # 1. Conversion du texte brut en flux I/O (Requirement Pandas)
        html_stream = StringIO(response.text)
        
        # 2. Ciblage chirurgical : on ne veut QUE le tableau principal
        tables = pd.read_html(html_stream, attrs={'id': 'constituents'})
        
        if not tables:
            raise ValueError("Le tableau 'constituents' n'a pas été trouvé sur la page.")
            
        df_sp500 = tables[0]
        tickers = df_sp500['Symbol'].tolist()
        
        # 3. Nettoyage vital (ex: BRK.B devient BRK-B pour yfinance)
        tickers = [str(t).replace('.', '-') for t in tickers]
        
        logger.info("%s tickers identifiés avec succès.", len(tickers))
        return tickers[:limit]
        
    except Exception as e:
        logger.error("Échec de la récupération dynamique des tickers : %s", e)
        return ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

def parse_info(ticker: str | yf.Ticker) -> dict:
    """Build a payload from yfinance ticker info and analyst price targets."""
    ticker_obj = ticker if isinstance(ticker, yf.Ticker) else yf.Ticker(ticker.strip())

    try:
        ticker_info = ticker_obj.info
        if not ticker_info:
            logger.error("Ticker info is empty")
            raise ValueError("Empty ticker info (invalid symbol or no data)")

        ta = ticker_obj.analyst_price_targets
        if ta is None:
            ticker_analysis = {}
        elif hasattr(ta, "to_dict"):
            ticker_analysis = ta.to_dict()
        else:
            ticker_analysis = dict(ta) if isinstance(ta, dict) else {}

        payload = {
            "ticker": ticker_info.get("symbol"),
            "company_name": ticker_info.get("longName"),
            "timestamp": datetime.now(timezone.utc),
            "source_type": "yfinance (api)",
            "metrics": {
                "open": ticker_info.get("open"),
                "current_price": ticker_info.get("currentPrice"),
                "dayHigh": ticker_info.get("dayHigh"),
                "dayLow": ticker_info.get("dayLow"),
                "previousClose": ticker_info.get("previousClose"),
                "volume": ticker_info.get("volume"),
                "Global Metrics": {
                    "High Price": ticker_analysis.get("high"),
                    "Low Price": ticker_analysis.get("low"),
                    "Average Price": ticker_analysis.get("mean"),
                    "Median Price": ticker_analysis.get("median"),
                },
            },
            "metadata": {
                "status": "success",
                "message": "Ticker info fetched successfully",
            },
        }

        logger.info("Payload successfully created for %s", ticker_obj.ticker)
        return payload

    except Exception as e:
        logger.exception("Payload creation failed: %s", e)
        return {
            "ticker": getattr(ticker_obj, "ticker", None),
            "source_type": "yfinance (api)",
            "metadata": {
                "status": "error",
                "message": str(e),
            },
        }

# Get ticket list
def process_tickers(ticker_list: list) -> None:
    """Fetches data for a list of tickers and inserts them into MongoDB."""
    db = get_db()
    collection = db["raw-market-data"] # Lazy creation de la collection
    
    for ticker in ticker_list:
        payload = parse_info(ticker)
        
        # Insertion uniquement si la récupération a réussi
        if payload.get("metadata", {}).get("status") == "success":
            logger.info("Inserted %s data into MongoDB.", ticker)
            collection.insert_one(payload)
        else:
            logger.warning("Skipping DB insertion for %s due to fetch error.", ticker)
        
        # Rate Limiting : Pause aléatoire entre 1.5 et 3.5 secondes entre chaque requête
        time.sleep(random.uniform(1.5, 3.5))

# Load tickets test
if __name__ == "__main__":
    try:
        # On récupère automatiquement les 30 plus grosses capitalisations du moment
        dynamic_list = get_dynamic_tickers(limit=30)
        process_tickers(dynamic_list)
    except Exception as e:
        logger.critical("Le pipeline de collecte API a échoué : %s", e)