# Practical AI in Industry: Hotel Search Ranking & Cross-Sell Service

[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/API-FastAPI%200.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Deploy-Docker%20Ready-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)
[![Tests](https://img.shields.io/badge/Tests-PyTest%20Passing-brightgreen.svg)](tests/)
[![Slide Deck](https://img.shields.io/badge/Slides-Marp%20Deck-0288d1.svg)](presentation.html)

> **"In coursework, the model is the finish line. In industry, a trained model is only the starting line."**

A production-grade Machine Learning service and presentation guide inspired by real-world Online Travel Agent (OTA) architecture (e.g. **tiket.com**). This project demonstrates how modern AI systems bridge the gap between model training and real-world deployment with **sub-15ms synchronous inference**, robust Pydantic schemas, MLOps lifecycle, Docker containerization, and a modern interactive UI.

---

## 📑 Table of Contents

- [Live Demo & Visuals](#-live-demo--visuals)
- [System Architecture](#-system-architecture)
- [Features & Capabilities](#-features--capabilities)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start Guide](#-quick-start-guide)
- [API Reference](#-api-reference)
- [Running Automated Tests](#-running-automated-tests)
- [Docker Deployment](#-docker-deployment)
- [Presentation Slide Deck](#-presentation-slide-deck)
- [Attribution & License](#-attribution--license)

---

## 📸 Live Demo & Visuals

The service includes an interactive dark-glassmorphism dashboard with live telemetry, dynamic budget sliders, city filtering, and real-time cross-sell add-on modals.

- **Interactive Web UI**: `http://localhost:8000`
- **Interactive OpenAPI (Swagger) Docs**: `http://localhost:8000/docs`
- **Health & Telemetry Endpoint**: `http://localhost:8000/api/v1/health`

---

## 🏛 System Architecture

```text
[ User / Web Client ]
         │
         │ HTTP / JSON (Low-Latency Synchronous Request)
         ▼
[ FastAPI Application (Uvicorn ASGI) ]
    ├── Latency Middleware (X-Process-Time-Ms)
    ├── Pydantic V2 Request & Response Validation
    │
    ├── [ MLServingPipeline (In-Memory Singleton) ]
    │        ├── Pointwise Ranking Model (HistGradientBoostingClassifier)
    │        │       └── Features: Price ratio, Gaussian utility, star match, review decay
    │        │
    │        └── Collaborative-Filtering Recommender (Bayesian Association Matrix)
    │                └── P(Activity | Hotel, Destination)
    │
    └── [ Endpoints ]
         ├── POST /api/v1/hotels/search        (Synchronous Ranking)
         ├── GET  /api/v1/recommend/cross-sell (Cross-Sell Engine)
         ├── GET  /api/v1/hotels/{id}          (Catalog Lookup)
         └── GET  /api/v1/health               (Observability & Diagnostics)
```

---

## ✨ Features & Capabilities

1. **Synchronous Search Ranking**:
   - Converts hotel catalog items into feature vectors on-the-fly.
   - Evaluates conversion probability using `HistGradientBoostingClassifier`.
   - Returns ranked candidates in **< 15ms**.
2. **Cross-Sell Recommender Engine**:
   - Implements item-to-activity collaborative filtering with Bayesian smoothing.
   - Recommends airport transfers, wellness spas, floating breakfasts, and tours tailored to hotel star class and destination.
3. **Enterprise Software Practices**:
   - Pydantic v2 strict typing and schema validation.
   - Startup model lifecycle caching (zero cold-start latency per request).
   - Observability: custom `X-Process-Time-Ms` response header and `/health` telemetry.
4. **PyTest Automated Test Suite**:
   - Comprehensive unit and integration test coverage for models, data generation, and API endpoints.

---

## 🛠 Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **API Framework** | FastAPI & Starlette | High-performance async RESTful API serving |
| **ASGI Server** | Uvicorn (standard) | Production-ready HTTP/1.1 & WebSocket server |
| **Data Validation** | Pydantic v2 | Strict request/response payload typing |
| **Machine Learning** | Scikit-Learn | `HistGradientBoostingClassifier` for conversion ranking |
| **Data Processing** | Pandas & NumPy | Feature engineering & vectorized calculations |
| **Artifact Storage** | Joblib | Serialized model pipelines and association matrices |
| **Automated Testing** | PyTest & HTTPX | Integration testing and schema validation |
| **Containerization** | Docker & Docker Compose | Portable deployment and multi-stage builds |
| **Frontend UI** | HTML5, CSS3, Vanilla JS | Dark glassmorphism dashboard with zero external dependencies |

---

## 📂 Project Structure

```text
ai_ml_in_production/
├── Dockerfile                   # Multi-stage production container
├── docker-compose.yml           # Compose orchestration
├── requirements.txt             # Locked Python dependencies
├── pyproject.toml               # Python project configuration
├── pytest.ini                   # PyTest settings
├── README.md                    # Project documentation
├── presentation.html            # Rendered Marp slides (Interactive)
├── presentation.md              # Presentation source in Marp markdown
├── data/
│   ├── hotels.json              # Catalog dataset (70 hotels across Indonesia)
│   └── interactions.csv         # 34,000+ synthetic user interaction logs
├── models/
│   ├── ranking_model.joblib     # Serialized conversion ranking model
│   └── recommender_artifacts.joblib # Serialized cross-sell association matrix
├── src/
│   ├── config.py                # Configuration constants and paths
│   ├── data/
│   │   └── generator.py         # Realistic OTA dataset and interaction generator
│   ├── models/
│   │   ├── train_ranking.py     # Feature engineering & ranking model trainer
│   │   ├── recommender.py       # Cross-sell association engine
│   │   └── pipeline.py          # Low-latency serving pipeline singleton
│   └── api/
│       ├── app.py               # FastAPI app, middleware, and lifecycle
│       ├── routes.py            # API routing handlers
│       └── schemas.py           # Pydantic request/response schemas
├── static/
│   ├── index.html               # Responsive web UI
│   ├── style.css                # Glassmorphism aesthetic styles
│   └── app.js                   # Client logic, live telemetry & modals
└── tests/
    ├── test_api.py              # API endpoint integration tests
    └── test_models.py           # Feature engineering & model tests
```

---

## 🚀 Quick Start Guide

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/fallss/practical-ai-in-industry.git
cd practical-ai-in-industry

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data & Train Models

```bash
# Generate synthetic OTA catalog and interaction logs
python -m src.data.generator

# Train Gradient Boosted Search Ranking model
python -m src.models.train_ranking

# Build Cross-Sell Recommender matrix
python -m src.models.recommender
```

### 3. Run the Development Server

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to:
- Dashboard: [http://localhost:8000](http://localhost:8000)
- OpenAPI Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 API Reference

### 1. Health & Telemetry
- **Endpoint**: `GET /api/v1/health`
- **Response**:
```json
{
  "status": "healthy",
  "service": "OTA Production AI & Search Ranking Engine",
  "version": "1.0.0",
  "ml_models_ready": true,
  "total_hotels_indexed": 70,
  "model_details": {
    "model_type": "HistGradientBoostingClassifier (Pointwise Conversion Ranker)",
    "metrics": {
      "roc_auc": 0.7459,
      "pr_auc": 0.8088
    }
  }
}
```

### 2. Hotel Search Ranking
- **Endpoint**: `POST /api/v1/hotels/search`
- **Request Body**:
```json
{
  "city": "Bali",
  "user_budget": 1500000,
  "pref_stars": 4,
  "sort_by": "ml_ranking",
  "limit": 10
}
```
- **Response**: List of scored hotels with `ml_score` (0–100%), personalized badges, and latency metrics.

### 3. Cross-Sell Recommendations
- **Endpoint**: `GET /api/v1/recommend/cross-sell?hotel_id=htl_032&top_k=3`
- **Response**:
```json
{
  "hotel_id": "htl_032",
  "hotel_name": "Grand Merdeka Skyline Bali #32",
  "city": "Bali",
  "latency_ms": 0.01,
  "recommendations": [
    {
      "item_id": "act_106",
      "name": "Snorkeling Excursion to Coral Reefs",
      "category": "Adventure",
      "price": 350000,
      "confidence_percent": 39
    }
  ]
}
```

---

## 🧪 Running Automated Tests

Run the full PyTest suite:

```bash
pytest tests/ -v
```

Output:
```text
tests/test_api.py::test_health_endpoint PASSED
tests/test_api.py::test_search_endpoint_ml_ranking PASSED
tests/test_api.py::test_hotel_detail_endpoint PASSED
tests/test_api.py::test_cross_sell_endpoint PASSED
tests/test_models.py::test_data_generation PASSED
tests/test_models.py::test_feature_engineering PASSED
tests/test_models.py::test_serving_pipeline_instance PASSED
========================= 7 passed in 3.55s =========================
```

---

## 🐳 Docker Deployment

Build and run using Docker Compose:

```bash
# Build and launch container
docker compose up --build -d

# Check container logs
docker compose logs -f

# Verify container health
docker ps
```

---

## 🎓 Presentation Slide Deck

The repository includes the complete keynote presentation deck:
- **Interactive Slides**: Open [presentation.html](presentation.html) in any browser (Press `P` for presenter mode, `F` for fullscreen).
- **Markdown Deck**: Edit [presentation.md](presentation.md) with Marp CLI:
  ```bash
  npx @marp-team/marp-cli -s .
  ```

---

## 👤 Attribution & License

- **Original Presentation by**: **Setia Budi** (Machine Learning Engineer at **tiket.com**)
- **Architecture & Implementation**: Production ML engineering reference implementation.
- **License**: Distributed under the [GNU General Public License v3.0](LICENSE).
