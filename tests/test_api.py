import pytest
from fastapi.testclient import TestClient
from src.api.app import app
from src.models.pipeline import MLServingPipeline

@pytest.fixture(scope="module")
def client():
    # Trigger lifespan to load models
    pipeline = MLServingPipeline.get_instance()
    if not pipeline.is_loaded:
        pipeline.load()
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["ml_models_ready"] is True
    assert data["total_hotels_indexed"] > 0
    assert "X-Process-Time-Ms" in response.headers

def test_search_endpoint_ml_ranking(client):
    payload = {
        "city": "Bali",
        "user_budget": 1500000,
        "pref_stars": 4,
        "sort_by": "ml_ranking",
        "limit": 10
    }
    response = client.post("/api/v1/hotels/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sort_applied"] == "ml_ranking"
    assert len(data["hotels"]) > 0
    first_hotel = data["hotels"][0]
    assert "ml_score" in first_hotel
    assert 0.0 <= first_hotel["ml_score"] <= 100.0

def test_hotel_detail_endpoint(client):
    pipeline = MLServingPipeline.get_instance()
    sample_id = pipeline.hotels_catalog[0]["hotel_id"]
    response = client.get(f"/api/v1/hotels/{sample_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["hotel_id"] == sample_id

def test_cross_sell_endpoint(client):
    pipeline = MLServingPipeline.get_instance()
    sample_id = pipeline.hotels_catalog[0]["hotel_id"]
    response = client.get(f"/api/v1/recommend/cross-sell?hotel_id={sample_id}&top_k=3")
    assert response.status_code == 200
    data = response.json()
    assert data["hotel_id"] == sample_id
    assert len(data["recommendations"]) <= 3
    if len(data["recommendations"]) > 0:
        rec = data["recommendations"][0]
        assert "item_id" in rec
        assert "price" in rec
