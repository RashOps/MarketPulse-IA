from pydantic import BaseModel

from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# --- SCHEMAS POUR LE NEWS FEED ---

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

# --- SCHEMAS POUR L'INTELLIGENCE DE MARCHÉ ---

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
    profiles: List[Dict] # Les profils générés par le groupby