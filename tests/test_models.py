import pytest
import numpy as np
import pandas as pd
from src.data.generator import generate_hotels, generate_interaction_logs
from src.models.train_ranking import engineer_features, FEATURE_COLUMNS
from src.models.pipeline import MLServingPipeline

def test_data_generation():
    hotels = generate_hotels(num_hotels=10)
    assert len(hotels) == 10
    h = hotels[0]
    assert "hotel_id" in h
    assert "price_per_night" in h
    assert "review_score" in h
    assert h["star_rating"] in [3, 4, 5]

def test_feature_engineering():
    df = pd.DataFrame([{
        "price_per_night": 1500000,
        "user_budget": 1500000,
        "pref_stars": 4,
        "star_rating": 4,
        "review_score": 4.8,
        "review_count": 250,
        "distance_to_center_km": 2.0,
        "discount_percent": 15
    }])
    X = engineer_features(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert np.isclose(X["price_ratio"].iloc[0], 1.0, atol=1e-3)
    assert X["star_diff"].iloc[0] == 0

def test_serving_pipeline_instance():
    pipeline = MLServingPipeline.get_instance()
    assert pipeline is not None
