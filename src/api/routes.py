import datetime
from fastapi import APIRouter, HTTPException, Query, status
from src.api.schemas import (
    HotelSearchRequest,
    HotelSearchResponse,
    CrossSellResponse,
    HealthResponse,
    HotelItem
)
from src.models.pipeline import MLServingPipeline
from src.config import API_TITLE, API_VERSION

router = APIRouter(prefix="/api/v1", tags=["Production ML Endpoints"])

@router.get("/health", response_model=HealthResponse, summary="System Health & Model Readiness")
async def health_check():
    pipeline = MLServingPipeline.get_instance()
    is_ready = pipeline.is_loaded
    ranking_meta = {}
    if pipeline.ranking_artifact:
        ranking_meta = {
            "model_type": pipeline.ranking_artifact.get("model_type"),
            "trained_at": pipeline.ranking_artifact.get("trained_at"),
            "metrics": pipeline.ranking_artifact.get("metrics")
        }
        
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        service=API_TITLE,
        version=API_VERSION,
        ml_models_ready=is_ready,
        total_hotels_indexed=len(pipeline.hotels_catalog),
        model_details=ranking_meta,
        server_time=datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

@router.post("/hotels/search", response_model=HotelSearchResponse, summary="Synchronous ML Hotel Search Ranking")
async def search_and_rank_hotels(request: HotelSearchRequest):
    pipeline = MLServingPipeline.get_instance()
    if not pipeline.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine Learning models are not initialized yet."
        )
        
    result = pipeline.rank_hotels(
        city=request.city,
        user_budget=request.user_budget,
        pref_stars=request.pref_stars,
        max_price=request.max_price,
        min_rating=request.min_rating,
        sort_by=request.sort_by,
        limit=request.limit
    )
    return result

@router.get("/hotels/{hotel_id}", response_model=HotelItem, summary="Get Hotel Details by ID")
async def get_hotel(hotel_id: str):
    pipeline = MLServingPipeline.get_instance()
    hotel = pipeline.hotels_map.get(hotel_id)
    if not hotel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hotel with ID '{hotel_id}' not found."
        )
    return hotel

@router.get("/recommend/cross-sell", response_model=CrossSellResponse, summary="Cross-Sell Recommendations for Hotel")
async def get_cross_sell_recommendations(
    hotel_id: str = Query(..., description="Target hotel ID to find co-booked cross-sell activities"),
    top_k: int = Query(default=3, ge=1, le=10, description="Top N recommendations")
):
    pipeline = MLServingPipeline.get_instance()
    if not pipeline.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommender models are not loaded."
        )
        
    res = pipeline.get_cross_sell(hotel_id=hotel_id, top_k=top_k)
    if not res.get("recommendations") and hotel_id not in pipeline.hotels_map:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hotel '{hotel_id}' not found in catalog."
        )
        
    return res
