import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from src.config import (
    HOTELS_DATA_PATH,
    RANKING_MODEL_PATH,
    RECOMMENDER_PATH
)
from src.models.train_ranking import FEATURE_COLUMNS, engineer_features

class MLServingPipeline:
    """Production ML Inference & Serving Pipeline for Hotel Ranking and Recommendations."""
    
    _instance = None
    
    def __init__(self):
        self.ranking_artifact = None
        self.recommender_artifact = None
        self.hotels_catalog: List[Dict[str, Any]] = []
        self.hotels_map: Dict[str, Dict[str, Any]] = {}
        self.is_loaded = False
        self.load_timestamp = None
        
    @classmethod
    def get_instance(cls) -> "MLServingPipeline":
        if cls._instance is None:
            cls._instance = MLServingPipeline()
        return cls._instance

    def load(self):
        """Load models and catalog into memory."""
        start_time = time.time()
        print("Loading ML models and catalog into memory...")
        
        # Load Hotels
        with open(HOTELS_DATA_PATH, "r", encoding="utf-8") as f:
            self.hotels_catalog = json.load(f)
        self.hotels_map = {h["hotel_id"]: h for h in self.hotels_catalog}
        
        # Load Ranking Model
        self.ranking_artifact = joblib.load(RANKING_MODEL_PATH)
        
        # Load Recommender Artifacts
        self.recommender_artifact = joblib.load(RECOMMENDER_PATH)
        
        self.is_loaded = True
        self.load_timestamp = time.time()
        duration = round((time.time() - start_time) * 1000, 2)
        print(f"MLServingPipeline successfully loaded in {duration}ms. Hotels: {len(self.hotels_catalog)}")

    def rank_hotels(
        self,
        city: Optional[str] = None,
        user_budget: int = 1500000,
        pref_stars: int = 4,
        max_price: Optional[int] = None,
        min_rating: Optional[float] = None,
        sort_by: str = "ml_ranking",
        limit: int = 20
    ) -> Dict[str, Any]:
        """Rank hotels synchronously with low-latency machine learning scoring."""
        t0 = time.time()
        
        # 1. Filter candidates
        candidates = self.hotels_catalog
        if city and city.lower() != "all":
            candidates = [h for h in candidates if h["city"].lower() == city.lower()]
        if max_price:
            candidates = [h for h in candidates if h["price_per_night"] <= max_price]
        if min_rating:
            candidates = [h for h in candidates if h["review_score"] >= min_rating]
            
        if not candidates:
            return {
                "total": 0,
                "latency_ms": round((time.time() - t0) * 1000, 2),
                "sort_applied": sort_by,
                "hotels": []
            }
            
        # 2. Score with ML if requested
        ranked_hotels = []
        if sort_by == "ml_ranking" and self.ranking_artifact:
            # Build feature frame
            df_cand = pd.DataFrame(candidates)
            df_cand["user_budget"] = user_budget
            df_cand["pref_stars"] = pref_stars
            
            X = engineer_features(df_cand)
            model = self.ranking_artifact["model"]
            probas = model.predict_proba(X)[:, 1]
            
            for idx, hotel in enumerate(candidates):
                h_copy = dict(hotel)
                # Compute conversion match score (0-100%)
                conversion_prob = float(probas[idx])
                h_copy["ml_score"] = round(conversion_prob * 100, 1)
                
                # Dynamic badges based on model features
                badges = []
                if hotel["discount_percent"] >= 20:
                    badges.append(f"Super Deal -{hotel['discount_percent']}%")
                if hotel["review_score"] >= 4.7:
                    badges.append("Guest Favorite")
                if hotel["distance_to_center_km"] <= 2.5:
                    badges.append("City Center")
                if conversion_prob >= 0.70:
                    badges.append("Top Recommended Match")
                h_copy["badges"] = badges
                ranked_hotels.append(h_copy)
                
            # Sort by ML probability descending
            ranked_hotels.sort(key=lambda x: x["ml_score"], reverse=True)
            
        elif sort_by == "price_asc":
            ranked_hotels = sorted(candidates, key=lambda x: x["price_per_night"])
        elif sort_by == "rating_desc":
            ranked_hotels = sorted(candidates, key=lambda x: x["review_score"], reverse=True)
        else:
            ranked_hotels = list(candidates)
            
        latency = round((time.time() - t0) * 1000, 2)
        
        return {
            "total": len(ranked_hotels),
            "latency_ms": latency,
            "sort_applied": sort_by,
            "model_metadata": {
                "model_type": self.ranking_artifact.get("model_type") if self.ranking_artifact else "Rule-based",
                "auc_score": self.ranking_artifact.get("metrics", {}).get("roc_auc") if self.ranking_artifact else None
            },
            "hotels": ranked_hotels[:limit]
        }

    def get_cross_sell(self, hotel_id: str, top_k: int = 3) -> Dict[str, Any]:
        """Fetch real-time cross-sell recommendations for a specific hotel."""
        t0 = time.time()
        hotel = self.hotels_map.get(hotel_id)
        if not hotel:
            return {"hotel_id": hotel_id, "recommendations": [], "found": False}
            
        recs = []
        if self.recommender_artifact:
            hotel_recs = self.recommender_artifact.get("hotel_recommendations", {})
            recs = hotel_recs.get(hotel_id, [])[:top_k]
            
        latency = round((time.time() - t0) * 1000, 2)
        return {
            "hotel_id": hotel_id,
            "hotel_name": hotel["name"],
            "city": hotel["city"],
            "latency_ms": latency,
            "recommendations": recs
        }
