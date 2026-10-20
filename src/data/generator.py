import json
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from src.config import HOTELS_DATA_PATH, INTERACTIONS_DATA_PATH

CITIES = ["Bali", "Jakarta", "Bandung", "Yogyakarta", "Surabaya", "Lombok", "Labuan Bajo"]

AMENITIES_LIST = [
    "Free Wi-Fi", "Swimming Pool", "Spa & Wellness", "Fitness Center",
    "Airport Shuttle", "Breakfast Included", "Ocean View", "Rooftop Bar",
    "Restaurant", "24-Hour Room Service"
]

CROSS_SELL_CATALOG = [
    {"item_id": "act_101", "name": "Airport Fast Track & Private Transfer", "category": "Transport", "price": 180000},
    {"item_id": "act_102", "name": "Sunset Catamaran & Dinner Cruise", "category": "Activity", "price": 450000},
    {"item_id": "act_103", "name": "Floating Breakfast Experience", "category": "Dining", "price": 220000},
    {"item_id": "act_104", "name": "Full-Day Cultural Tour & Temple Visit", "category": "Tour", "price": 380000},
    {"item_id": "act_105", "name": "Traditional Balinese Aromatherapy Spa (90 min)", "category": "Wellness", "price": 275000},
    {"item_id": "act_106", "name": "Snorkeling Excursion to Coral Reefs", "category": "Adventure", "price": 350000},
    {"item_id": "act_107", "name": "Premium Scooter Rental (3 Days)", "category": "Transport", "price": 210000},
    {"item_id": "act_108", "name": "Romantic 4-Course Candlelight Dinner", "category": "Dining", "price": 600000},
]

HOTEL_TEMPLATES = [
    ("The Haven Resort & Sanctuary", "Luxury resort offering tranquil scenery, lush gardens, and signature spa."),
    ("Grand Merdeka Skyline", "Centrally located high-rise with skyline panoramic views and business amenities."),
    ("Padma Heritage Boutique Villa", "Authentic artisanal design blended with world-class hospitality and private pools."),
    ("Alila Eco Cliff Suites", "Modern minimalist cliffside suites with direct private infinity pool access."),
    ("Aston Urban Express Hotel", "Vibrant, budget-friendly tech hotel equipped with high-speed internet and coworking cafe."),
    ("Plataran Rainforest Haven", "Exclusive nature retreat nestled inside tropical greenery with eco-conscious dining."),
    ("Aryaduta Grand Palace", "Classic 5-star elegance with majestic ballrooms and fine-dining restaurants."),
    ("Katamama Coastal Retreat", "Contemporary beachfront escape featuring craft mixology, beach club access, and yoga."),
    ("Santika Premiere Suites", "Comfortable, warm hospitality ideal for family vacations and corporate travelers."),
    ("Swiss-Belresort Hilltop", "Boutique hillside hotel with valley views, heated pool, and barbecue terrace.")
]

def generate_hotels(num_hotels: int = 60) -> List[Dict[str, Any]]:
    """Generate realistic hotel catalog data."""
    random.seed(42)
    np.random.seed(42)
    hotels = []
    
    for i in range(1, num_hotels + 1):
        city = random.choice(CITIES)
        tmpl_name, tmpl_desc = random.choice(HOTEL_TEMPLATES)
        star = random.choices([3, 4, 5], weights=[0.3, 0.45, 0.25])[0]
        
        # Price correlates with stars and city
        base_price = {3: 450000, 4: 950000, 5: 2200000}[star]
        city_factor = 1.3 if city in ["Bali", "Labuan Bajo"] else 1.0
        price = int(base_price * city_factor * np.random.uniform(0.8, 1.4))
        # Round to nearest 10,000
        price = round(price, -4)
        
        rating = round(float(np.random.uniform(3.9, 4.9)), 1)
        review_count = int(np.random.exponential(scale=350) + 20)
        distance = round(float(np.random.exponential(scale=4.5) + 0.5), 1)
        discount = random.choices([0, 10, 15, 20, 25, 30], weights=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05])[0]
        
        # Amenities
        k = random.randint(3, 7)
        amenities = random.sample(AMENITIES_LIST, k)
        
        hotel = {
            "hotel_id": f"htl_{i:03d}",
            "name": f"{tmpl_name} {city} #{i}",
            "description": tmpl_desc,
            "city": city,
            "star_rating": star,
            "price_per_night": price,
            "review_score": rating,
            "review_count": review_count,
            "distance_to_center_km": distance,
            "discount_percent": discount,
            "amenities": amenities,
            "historical_ctr": round(float(np.random.beta(2, 20)), 4),
            "historical_conversion": round(float(np.random.beta(2, 50)), 4)
        }
        hotels.append(hotel)
        
    return hotels

def generate_interaction_logs(hotels: List[Dict[str, Any]], num_sessions: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic search impressions, clicks, bookings, and cross-sell purchases.
    Used for:
    1. Training the Search Ranking Model (pairwise/pointwise conversion prediction).
    2. Building the Co-occurrence Matrix for Cross-Sell Recommendation.
    """
    random.seed(42)
    np.random.seed(42)
    
    rows = []
    hotel_map = {h["hotel_id"]: h for h in hotels}
    hotel_ids = list(hotel_map.keys())
    
    for session_id in range(1, num_sessions + 1):
        # User search intent
        city = random.choice(CITIES)
        user_budget = random.choice([500000, 1000000, 1800000, 3000000, 5000000])
        pref_stars = random.choice([3, 4, 5])
        
        # Filter candidate hotels in that city
        candidates = [h for h in hotels if h["city"] == city]
        if not candidates:
            continue
            
        # Sample impression list (e.g., 5-8 hotels presented to user)
        k = min(len(candidates), random.randint(4, 8))
        impressed = random.sample(candidates, k)
        
        for rank_pos, hotel in enumerate(impressed, start=1):
            # Compute conversion probability based on realistic business factors:
            # 1. Price alignment with user budget
            price_ratio = hotel["price_per_night"] / user_budget
            price_score = np.exp(-0.5 * ((price_ratio - 1.0) ** 2))
            
            # 2. Review score and review count effect
            rating_score = (hotel["review_score"] - 3.5) / 1.5
            
            # 3. Distance penalty
            distance_score = np.exp(-hotel["distance_to_center_km"] / 10.0)
            
            # 4. Star preference match
            star_match = 1.0 if hotel["star_rating"] == pref_stars else 0.5
            
            # 5. Discount incentive
            discount_boost = 1.0 + (hotel["discount_percent"] / 100.0)
            
            # Overall utility
            utility = (
                0.30 * price_score +
                0.25 * rating_score +
                0.15 * distance_score +
                0.15 * star_match +
                0.15 * (discount_boost - 1.0)
            )
            
            # Probability of click and booking with realistic business conversion signal
            p_booked = 1 / (1 + np.exp(-10 * (utility - 0.52)))
            booked = int(random.random() < p_booked)
            clicked = 1 if booked else int(random.random() < (p_booked + 0.15))
            cross_sells = []
            
            if booked:
                num_cs = random.choices([0, 1, 2], weights=[0.3, 0.5, 0.2])[0]
                if num_cs > 0:
                    sampled_cs = random.sample(CROSS_SELL_CATALOG, num_cs)
                    cross_sells = [item["item_id"] for item in sampled_cs]

            rows.append({
                "session_id": f"sess_{session_id:06d}",
                "city": city,
                "user_budget": user_budget,
                "pref_stars": pref_stars,
                "hotel_id": hotel["hotel_id"],
                "price_per_night": hotel["price_per_night"],
                "star_rating": hotel["star_rating"],
                "review_score": hotel["review_score"],
                "review_count": hotel["review_count"],
                "distance_to_center_km": hotel["distance_to_center_km"],
                "discount_percent": hotel["discount_percent"],
                "display_position": rank_pos,
                "is_clicked": clicked,
                "is_booked": booked,
                "cross_sell_items": ",".join(cross_sells)
            })
            
    return pd.DataFrame(rows)

def prepare_and_save_data():
    """Generate and save both hotels catalog and user interaction logs."""
    print("Generating synthetic hotel catalog...")
    hotels = generate_hotels(num_hotels=70)
    with open(HOTELS_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(hotels, f, indent=2)
    print(f"Saved {len(hotels)} hotels to {HOTELS_DATA_PATH}")

    print("Generating synthetic user interaction logs...")
    df_interactions = generate_interaction_logs(hotels, num_sessions=6000)
    df_interactions.to_csv(INTERACTIONS_DATA_PATH, index=False)
    print(f"Saved {len(df_interactions)} interaction logs to {INTERACTIONS_DATA_PATH}")

if __name__ == "__main__":
    prepare_and_save_data()
