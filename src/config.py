import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
PRESENTATION_DIR = BASE_DIR / "presentation"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
HOTELS_DATA_PATH = DATA_DIR / "hotels.json"
INTERACTIONS_DATA_PATH = DATA_DIR / "interactions.csv"
RANKING_MODEL_PATH = MODELS_DIR / "ranking_model.joblib"
RECOMMENDER_PATH = MODELS_DIR / "recommender_artifacts.joblib"

# Server configuration
API_TITLE = "OTA Production AI & Search Ranking Engine"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Production-grade Machine Learning service for Hotel Search Ranking and "
    "Cross-Sell Recommendations, inspired by real-world Online Travel Agent architecture."
)
