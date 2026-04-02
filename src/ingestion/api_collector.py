import yfinance as yf
import time
import sys
from datetime import datetime, timezone
import pandas as pd
from io import StringIO
import requests
import random

from src.utils.logger import get_logger
from src.utils.db_client import get_db

logger = get_logger(__name__)

def get_dynamic_tickers(limit: int = 50) -> list[str]:
    """
    Dynamically retrieves S&P 500 Tickers by scraping Wikipedia.

    Args:
        limit (int): Maximum number of tickers to retrieve.

    Returns:
        list[str]: List of ticker symbols.
    """
    logger.info("Dynamically retrieving S&P 500 tickers...")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status() 

        html_stream = StringIO(response.text)

        tables = pd.read_html(html_stream, attrs={'id': 'constituents'})

        if not tables:
            raise ValueError("The 'constituents' table was not found on the page.")

        df_sp500 = tables[0]
        tickers = df_sp500['Symbol'].tolist()

        # Clean formatting (e.g., BRK.B becomes BRK-B for yfinance)
        tickers = [str(t).replace('.', '-') for t in tickers]

        logger.info("Successfully identified %s tickers.", len(tickers))
        return tickers[:limit]

    except Exception as e:
        logger.error("Failed to dynamically retrieve tickers: %s", e)
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

def process_tickers(ticker_list: list) -> None:
    """Fetches data for a list of tickers and inserts them into MongoDB."""
    db = get_db()
    collection = db["raw-market-data"] 

    for ticker in ticker_list:
        payload = parse_info(ticker)

        if payload.get("metadata", {}).get("status") == "success":
            logger.info("Inserted %s data into MongoDB.", ticker)
            collection.insert_one(payload)
        else:
            logger.warning("Skipping DB insertion for %s due to fetch error.", ticker)

        # Rate Limiting: random pause between 1.5 and 3.5 seconds
        time.sleep(random.uniform(1.5, 3.5))

if __name__ == "__main__":
    try:
        dynamic_list = get_dynamic_tickers(limit=30)
        process_tickers(dynamic_list)
    except Exception as e:
        logger.critical("API collection pipeline failed: %s", e)