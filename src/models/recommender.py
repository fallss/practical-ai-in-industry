import json
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any
from src.config import (
    INTERACTIONS_DATA_PATH,
    HOTELS_DATA_PATH,
    RECOMMENDER_PATH
)
from src.data.generator import CROSS_SELL_CATALOG

def build_recommender():
    """Build item-based cross-sell recommendation association matrix."""
    print(f"Building cross-sell recommender model...")
    df = pd.read_csv(INTERACTIONS_DATA_PATH)
    with open(HOTELS_DATA_PATH, "r", encoding="utf-8") as f:
        hotels = json.load(f)
    
    hotel_info = {h["hotel_id"]: h for h in hotels}
    catalog_map = {item["item_id"]: item for item in CROSS_SELL_CATALOG}
    
    # Co-occurrence counts between hotel_id and cross_sell_items
    hotel_item_counts = defaultdict(lambda: defaultdict(int))
    city_item_counts = defaultdict(lambda: defaultdict(int))
    item_overall_counts = defaultdict(int)
    total_hotel_bookings = defaultdict(int)
    
    booked_df = df[df["is_booked"] == 1]
    
    for _, row in booked_df.iterrows():
        hid = row["hotel_id"]
        city = row["city"]
        total_hotel_bookings[hid] += 1
        
        cs_str = str(row.get("cross_sell_items", ""))
        if cs_str and cs_str != "nan":
            items = cs_str.split(",")
            for item in items:
                item = item.strip()
                if item in catalog_map:
                    hotel_item_counts[hid][item] += 1
                    city_item_counts[city][item] += 1
                    item_overall_counts[item] += 1
                    
    # Compute association scores: P(item | hotel) with Bayesian smoothing
    hotel_recommendations = {}
    all_catalog_ids = list(catalog_map.keys())
    
    for hid, h_data in hotel_info.items():
        city = h_data["city"]
        stars = h_data["star_rating"]
        scores = []
        
        n_bookings = total_hotel_bookings[hid]
        for item_id in all_catalog_ids:
            item_data = catalog_map[item_id]
            co_count = hotel_item_counts[hid][item_id]
            city_count = city_item_counts[city][item_id]
            overall_count = item_overall_counts[item_id]
            
            # Priors based on hotel category and stars
            domain_bonus = 0.0
            if stars >= 4 and item_data["category"] in ["Wellness", "Dining"]:
                domain_bonus += 0.25
            if city in ["Bali", "Lombok", "Labuan Bajo"] and item_data["category"] in ["Adventure", "Tour"]:
                domain_bonus += 0.30
            if item_data["category"] == "Transport":
                domain_bonus += 0.20
                
            # Bayesian smoothed probability
            p_empirical = (co_count + 1.0) / (n_bookings + 10.0)
            score = (0.5 * p_empirical) + (0.3 * (city_count / (len(booked_df) + 1.0) * 10)) + domain_bonus
            
            scores.append({
                "item_id": item_id,
                "name": item_data["name"],
                "category": item_data["category"],
                "price": item_data["price"],
                "relevance_score": round(float(score), 3),
                "confidence_percent": min(98, int(score * 100))
            })
            
        # Sort by relevance_score descending
        scores.sort(key=lambda x: x["relevance_score"], reverse=True)
        hotel_recommendations[hid] = scores[:4]
        
    artifacts = {
        "hotel_recommendations": hotel_recommendations,
        "catalog": CROSS_SELL_CATALOG,
        "model_type": "Item-to-Category Association & Collaborative Filtering"
    }
    
    joblib.dump(artifacts, RECOMMENDER_PATH)
    print(f"Saved recommender artifacts to {RECOMMENDER_PATH}")

if __name__ == "__main__":
    build_recommender()
