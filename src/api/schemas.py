from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# --- SCHEMAS FOR NEWS FEED ---

class NewsItem(BaseModel):
    id: str = Field(..., alias="_id")
    title: str
    url: str
    published: str
    is_inferred_date: bool
    source: str
    ingested_at: Optional[str] = None

class NewsMetadata(BaseModel):
    count: int
    filter_applied: str

class MarketNewsResponse(BaseModel):
    metadata: NewsMetadata
    data: List[NewsItem]

# --- SCHEMAS FOR MARKET INTELLIGENCE ---

class MarketSegmentData(BaseModel):
    ticker: str
    PCA_1: float
    PCA_2: float
    cluster: int
    business_segment: str
    metrics_volume: int

class MarketSegmentsMetadata(BaseModel):
    total_assets: int
    clusters_identified: int

class MarketSegmentsResponse(BaseModel):
    metadata: MarketSegmentsMetadata
    data: List[MarketSegmentData]
    profiles: List[Dict]

# --- NEW SCHEMAS FOR EXTENDED OPS ---

class DataInventory(BaseModel):
    total_raw_records: int
    latest_ingestion_date: Optional[str] = None
    collection_name: str

class ModelHistoryItem(BaseModel):
    version: str
    timestamp: str
    silhouette_score: float
    optimal_k: int

class ModelHistoryResponse(BaseModel):
    total_runs: int
    history: List[ModelHistoryItem]

class SystemConfig(BaseModel):
    pca_components: int
    max_clusters: int
    db_name: str
    artifacts_dir: str