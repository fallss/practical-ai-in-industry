from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class HotelSearchRequest(BaseModel):
    city: Optional[str] = Field(default="All", description="Target city destination or 'All'")
    user_budget: int = Field(default=1500000, ge=100000, le=50000000, description="User intended budget per night in IDR")
    pref_stars: int = Field(default=4, ge=1, le=5, description="Preferred star class rating (1-5)")
    max_price: Optional[int] = Field(default=None, ge=100000, description="Optional upper price ceiling filter")
    min_rating: Optional[float] = Field(default=None, ge=1.0, le=5.0, description="Optional minimum review score filter")
    sort_by: str = Field(default="ml_ranking", description="Sorting strategy: 'ml_ranking', 'price_asc', 'rating_desc'")
    limit: int = Field(default=20, ge=1, le=100, description="Number of results to return")

class HotelItem(BaseModel):
    hotel_id: str
    name: str
    description: str
    city: str
    star_rating: int
    price_per_night: int
    review_score: float
    review_count: int
    distance_to_center_km: float
    discount_percent: int
    amenities: List[str]
    image_url: Optional[str] = None
    ml_score: Optional[float] = None
    badges: List[str] = []

class HotelSearchResponse(BaseModel):
    total: int
    latency_ms: float
    sort_applied: str
    model_metadata: Optional[Dict[str, Any]] = None
    hotels: List[HotelItem]

class CrossSellItem(BaseModel):
    item_id: str
    name: str
    category: str
    price: int
    relevance_score: float
    confidence_percent: int

class CrossSellResponse(BaseModel):
    hotel_id: str
    hotel_name: str
    city: str
    latency_ms: float
    recommendations: List[CrossSellItem]

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    ml_models_ready: bool
    total_hotels_indexed: int
    model_details: Dict[str, Any]
    server_time: str
