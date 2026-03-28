# Lab: dernières news financières via les flux RSS Yahoo Finance et Investing.com
from __future__ import annotations

import sys
from pathlib import Path

# Resolving Path issue
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import httpx
import pymongo

from src.utils.db_client import get_db
from src.utils.formatters import standardize_date
from src.utils.logger import get_logger

logger = get_logger(__name__)

YAHOO_FINANCE_NEWS_RSS = "https://finance.yahoo.com/news/rssindex"
INVESTING_STOCK_NEWS_RSS = "https://www.investing.com/rss/news_25.rss"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}


def _parse_rss_items(xml_text: str, limit: int) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.exception("Échec du parse XML RSS (flux invalide ou tronqué)")
        raise

    rows: list[dict[str, Any]] = []

    for item in root.findall(".//item"):
        if len(rows) >= limit:
            break

        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_raw = (item.findtext("pubDate") or "").strip()

        if not title and not link:
            continue

        normalized_date = standardize_date(pub_raw)
        is_inferred = False

        if normalized_date is None:
            normalized_date = datetime.now(timezone.utc)
            is_inferred = True

        rows.append({
            "title": title,
            "url": link,
            "published": normalized_date,
            "is_inferred_date": is_inferred,
        })

    return rows


def fetch_rss(url: str, limit: int = 5, *, timeout: float = 25.0) -> list[dict[str, Any]]:
    logger.debug("Requête RSS url=%s limit=%s", url, limit)
    try:
        with httpx.Client(headers=DEFAULT_HEADERS, timeout=timeout, follow_redirects=True) as client:
            r = client.get(url)
            r.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.error(
            "RSS HTTP %s pour url=%s",
            e.response.status_code,
            url,
        )
        raise
    except httpx.RequestError as e:
        logger.error("Erreur réseau RSS url=%s: %s", url, e)
        raise
    items = _parse_rss_items(r.text, limit)
    logger.info("RSS OK url=%s — %s article(s) parsé(s)", url, len(items))
    return items


def fetch_yahoo_finance_news(limit: int = 5) -> list[dict[str, Any]]:
    logger.debug("fetch_yahoo_finance_news limit=%s", limit)
    items = fetch_rss(YAHOO_FINANCE_NEWS_RSS, limit)
    for row in items:
        row["source"] = "Yahoo Finance"
    return items


def fetch_investing_news(limit: int = 5) -> list[dict[str, Any]]:
    logger.debug("fetch_investing_news limit=%s", limit)
    items = fetch_rss(INVESTING_STOCK_NEWS_RSS, limit)
    for row in items:
        row["source"] = "Investing.com"
    return items


def fetch_latest_financial_news(limit_per_source: int = 5) -> dict[str, list[dict[str, Any]]]:
    """Récupère les `limit_per_source` dernières entrées sur chaque source."""
    logger.info("Début fetch_latest_financial_news (limit_par_source=%s)", limit_per_source)
    out = {
        "yahoo_finance": fetch_yahoo_finance_news(limit_per_source),
        "investing": fetch_investing_news(limit_per_source),
    }
    logger.info(
        "Fin fetch_latest_financial_news — yahoo=%s investing=%s",
        len(out["yahoo_finance"]),
        len(out["investing"]),
    )
    return out


def save_news_to_db(news_dict: dict[str, list[dict[str, Any]]]) -> None:
    """Enregistre les news en base. TTL sur `ingested_at` (30 j), pas sur `published`."""
    db = get_db()
    collection = db["market-news"]
    collection.create_index(
        [("ingested_at", pymongo.ASCENDING)],
        expireAfterSeconds=2592000,
        name="ingested_at_ttl",
    )
    collection.create_index([("published", pymongo.DESCENDING)], name="published_desc")

    ingested_at = datetime.now(timezone.utc)
    for source, articles in news_dict.items():
        if articles:
            docs = [{**article, "ingested_at": ingested_at} for article in articles]
            logger.info("Inserted %s articles from %s into MongoDB.", len(docs), source)
            collection.insert_many(docs)


if __name__ == "__main__":
    try:
        news_by_source = fetch_latest_financial_news(limit_per_source=5)
        save_news_to_db(news_by_source)
    except Exception as e:
        logger.critical("Scraping pipeline failed: %s", e)
