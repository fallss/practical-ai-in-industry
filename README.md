# Practical AI in Industry: Keynote Presentation & Production ML Engine

<div align="center">

[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI%200.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Container-Docker%20Ready-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)
[![Tests: 7 Passed](https://img.shields.io/badge/PyTest-7%20Passed%20(100%25)-brightgreen.svg?logo=pytest&logoColor=white)](tests/)
[![Serving Latency](https://img.shields.io/badge/Serving%20Latency-%3C15ms-success.svg)](#ml-model-pipeline--feature-engineering)
[![Presentation: Marp](https://img.shields.io/badge/Slides-Marp%20Interactive-0288d1.svg)](presentation/presentation.html)

<p align="center">
  <b>A dual-purpose industry resource:</b><br>
  An executive keynote presentation deck detailing real-world AI implementation and career preparation, paired with an end-to-end, production-grade <b>Online Travel Agent (OTA) Search Ranking & Cross-Sell Recommendation Service</b>.
</p>

[Explore Slide Deck](presentation/presentation.html) • [Live Dashboard](http://localhost:8000) • [Swagger API Docs](http://localhost:8000/docs) • [Architecture](#system-architecture) • [Deployment Guide](#cloud-deployment-guide)

---

</div>

> **"In coursework, the model is the finish line. In industry, a trained model is merely the starting line."**  
> *— Setia Budi, Machine Learning Engineer at tiket.com*

---

## 📑 Table of Contents

1. [Folder Structure & Project Organization](#folder-structure--project-organization)
2. [Executive Summary & Vision](#executive-summary--vision)
3. [Part I: The Presentation Deck (Theory & Industry Insights)](#part-i-the-presentation-deck-theory--industry-insights)
   - [Speaker & Background](#speaker--background)
   - [Curriculum & Presentation Breakdown](#curriculum--presentation-breakdown)
   - [Key Industry Mental Models](#key-industry-mental-models)
   - [How to View and Deliver the Slides](#how-to-view-and-deliver-the-slides)
4. [Part II: The Production ML Engine (Implementation)](#part-ii-the-production-ml-engine-implementation)
   - [Problem Statement & OTA Context](#problem-statement--ota-context)
   - [System Architecture](#system-architecture)
   - [ML Model Pipeline & Feature Engineering](#ml-model-pipeline--feature-engineering)
   - [Cross-Sell Collaborative Filtering Engine](#cross-sell-collaborative-filtering-engine)
   - [Interactive Dashboard & Telemetry](#interactive-dashboard--telemetry)
5. [Part III: Developer & Operations Guide](#part-iii-developer--operations-guide)
   - [Quick Start Installation](#quick-start-installation)
   - [API Reference & Endpoints](#api-reference--endpoints)
   - [Automated Testing with PyTest](#automated-testing-with-pytest)
   - [Production Docker Deployment](#production-docker-deployment)
6. [Part IV: Cloud Deployment Guide (Free & Production)](#part-iv-cloud-deployment-guide-free--production)
   - [Option A: 1-Click Deploy on Render.com (Recommended Free Hosting)](#option-a-1-click-deploy-on-rendercom-recommended-free-hosting)
   - [Option B: Railway.app (Automatic Docker Deployment)](#option-b-railwayapp-automatic-docker-deployment)
   - [Option C: Hugging Face Spaces (Free ML Portfolio Hosting)](#option-c-hugging-face-spaces-free-ml-portfolio-hosting)
   - [Option D: Self-Hosted Cloud VPS (Docker Compose)](#option-d-self-hosted-cloud-vps-docker-compose)
7. [Attribution & License](#attribution--license)

---

## 🗂 Folder Structure & Project Organization

The repository is organized into distinct, clean modules separating the **Keynote Presentation** from the **Production ML Engineering Application**:

```text
practical-ai-in-industry/
│
├── 📊 presentation/                    # 1. KEYNOTE PRESENTATION MODULE
│   ├── presentation.md                # Source presentation in Marp Markdown
│   └── presentation.html              # Standalone interactive slides with speaker notes
│
├── 🧠 src/                            # 2. CORE MACHINE LEARNING SERVICE
│   ├── config.py                      # Global configuration and path management
│   ├── data/
│   │   └── generator.py               # Synthetic OTA catalog and interaction generator
│   ├── models/
│   │   ├── train_ranking.py           # Feature engineering & HistGradientBoosting trainer
│   │   ├── recommender.py             # Bayesian collaborative filtering co-booking matrix
│   │   └── pipeline.py                # In-memory ML inference pipeline singleton (<15ms)
│   └── api/
│       ├── app.py                     # FastAPI application, latency tracing & static mounts
│       ├── routes.py                  # API routes (/search, /cross-sell, /health)
│       └── schemas.py                 # Pydantic V2 strict request/response schemas
│
├── 🎨 static/                         # 3. INTERACTIVE WEB DASHBOARD
│   ├── index.html                     # Dark glassmorphism dashboard layout
│   ├── style.css                      # Premium UI design system with glowing accents
│   └── app.js                         # Dynamic telemetry, sliders, and cross-sell modal
│
├── 🧪 tests/                          # 4. AUTOMATED TEST SUITE
│   ├── test_api.py                    # Integration tests for FastAPI endpoints
│   └── test_models.py                 # Unit tests for data generator & feature pipeline
│
├── 📦 data/                           # 5. DATA ASSETS
│   ├── hotels.json                    # Catalog dataset (70 hotels in Indonesia)
│   └── interactions.csv               # 34,000+ user session & conversion logs
│
├── 💾 models/                         # 6. SERIALIZED ML ARTIFACTS
│   ├── ranking_model.joblib           # Trained conversion ranking model
│   └── recommender_artifacts.joblib   # Serialized cross-sell co-booking matrix
│
├── 🐳 Deployment & Config
│   ├── Dockerfile                     # Multi-stage production container build
│   ├── docker-compose.yml             # Container orchestration
│   ├── render.yaml                    # 1-Click Blueprint for Render.com deployment
│   ├── pyproject.toml / pytest.ini    # Python build and test configurations
│   ├── requirements.txt               # Locked Python dependencies
│   ├── LICENSE                        # GNU GPL-3.0 License
│   └── README.md                      # Comprehensive documentation
```

---

## 💡 Executive Summary & Vision

Most academic curricula and tutorials treat machine learning as an isolated exercise in Kaggle competitions or Jupyter Notebooks. In reality, **over 80% of a Machine Learning Engineer's day-to-day work is software and systems engineering**: data pipelines, low-latency API serving, containerization, monitoring, and business alignment.

This repository serves as a **complete bridge from theory to production**:
- **Part I**: Slides and lessons learned from a practicing Machine Learning Engineer at **tiket.com**.
- **Part II**: A runnable, production-grade microservice demonstrating real-world ranking and recommendation algorithms.

---

## 🎤 Part I: The Presentation Deck (Theory & Industry Insights)

### Speaker & Background
- **Speaker:** **Setia Budi**
- **Role:** Machine Learning Engineer at **tiket.com** (Online Travel Agent).
- **Format:** [Marp](https://marp.app/) Markdown (`presentation/presentation.md`) & Standalone Interactive HTML (`presentation/presentation.html`).

### Curriculum & Presentation Breakdown

The presentation is organized into 8 sequential modules:

```text
[1. Business Economics] ──▶ [2. Real-World Use Cases] ──▶ [3. The ML Engineer Role] ──▶ [4. Deployment Patterns]
                                                                                                 │
[8. Q&A & Action Plan]  ◀── [7. Career Preparation]   ◀── [6. Fundamental Skills]   ◀── [5. Modern Tech Stacks]
```

1. **Why Data (and AI) Matters in Business:**
   - *Data is the new oil:* Raw data has no standalone value until it is refined through ingestion, validation, and analytics pipelines.
   - The two non-negotiable business mandates: **Reduce Operational Cost** (automation, support chatbots) and **Increase Revenue** (conversion-driven search ranking, cross-sell recommendations).
2. **What AI in Industry Actually Looks Like:**
   - De-hyping AI: The best production AI is invisible, processing millions of micro-decisions per second rather than flashing robots.
   - Practical OTA examples: Search autocomplete, hotel ranking, cross-sell recommendation, review sentiment classification, and image quality moderation.
3. **The Role of an ML / AI Engineer:**
   - Moving beyond `model.fit()`: Writing reliable, tested, and containerized software services that run 24/7 without intervention.
4. **How AI Services Are Deployed:**
   - **Synchronous (User-facing):** Real-time, ultra-low latency (< 50ms), e.g. search ranking.
   - **Asynchronous (Batch):** High throughput offline scoring (Airflow, Spark, dbt).
   - **Scheduled / Event-Driven:** Periodic retraining and feature store materialization.
5. **Modern Industry Tech Stack:**
   - *Cloud & Compute:* AWS, GCP, Azure, Kubernetes, ArgoCD.
   - *Databases & Vector Stores:* PostgreSQL, Redis, Milvus, FalkorDB.
   - *MLOps & LLMOps:* MLflow, AI Gateways, Prompt Registries.
   - *Python Toolbox:* `uv`, `FastAPI`, `Pydantic`, `PyTest`, `Polars`.
6. **Fundamental Skills that Matter:**
   - Software engineering hygiene (Clean Code, Git, Docker, System Design).
   - Growth mindset: The stack evolves every 18 months; learning agility is paramount.
   - Clear communication: Translating technical trade-offs into business KPIs.
7. **How to Prepare Before Graduation (Actionable Roadmap):**
   - Focus on building **2–3 end-to-end deployed projects** rather than ten half-finished tutorials.
   - Master SQL, the Linux command line, and Docker.
   - Put clean code on GitHub with comprehensive documentation—*your public code is your true resume*.
   - Pursue real industry internships to understand production constraints and code reviews.

### Key Industry Mental Models

| Concept | Coursework / Academic ML | Production / Industry ML |
| :--- | :--- | :--- |
| **Primary Metric** | Benchmark Accuracy / F1-Score | Business Impact (Conversion Rate, GMV, Latency, Cost) |
| **Code Deliverable** | Jupyter Notebook (`.ipynb`) | Tested, Versioned, Containerized REST/gRPC Service |
| **Data Nature** | Static, clean CSVs (Kaggle/UCI) | Noisy, drifting, streaming, real-time data pipelines |
| **Model Lifecycle** | Ends once training converges | Begins *after* the model is deployed to production |
| **Work Allocation** | ~80% Modeling & Tuning | ~80% Systems, Data Engineering, Testing & Observability |

### How to View and Deliver the Slides

The slide deck is built with [Marp](https://marp.app/) and includes custom GitHub Dark styling, typography, and presenter notes for every slide.

#### 1. Open the Pre-compiled Interactive Slides
Open `presentation/presentation.html` directly in any web browser, or access `/slides` when running the application:
```bash
# On Linux
xdg-open presentation/presentation.html

# On macOS
open presentation/presentation.html
```
- **Press `P`**: Enters **Presenter Mode** with dual-screen view, speaker notes, and real-time elapsed timer.
- **Press `F`**: Enters **Fullscreen Presentation Mode**.
- **Press `O`**: Opens **Slide Grid Overview** for rapid navigation.

#### 2. Edit or Recompile with Marp CLI
```bash
# Run local live-reload preview server
npx @marp-team/marp-cli -s presentation/

# Export to HTML, PDF, or PowerPoint
npx @marp-team/marp-cli presentation/presentation.md -o presentation/presentation.html
npx @marp-team/marp-cli presentation/presentation.md --pdf -o presentation/presentation.pdf
npx @marp-team/marp-cli presentation/presentation.md --pptx -o presentation/presentation.pptx
```

---

## ⚙️ Part II: The Production ML Engine (Implementation)

### Problem Statement & OTA Context
When a user searches for hotels in Bali or Jakarta with a specific budget, the platform must:
1. Filter candidate accommodations instantly.
2. Rank candidates to maximize conversion probability while matching user constraints.
3. Dynamically suggest relevant add-on activities (e.g., airport pickup, catamaran cruise, or spa) based on previous co-booking affinities.

### System Architecture

```text
[ Client / Web Browser / Mobile App ]
                    │
                    │ HTTP POST / GET (JSON)
                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      FASTAPI ASGI APPLICATION                          │
│                                                                        │
│   ├── Latency Tracing Middleware (Injects 'X-Process-Time-Ms' header)  │
│   ├── Pydantic V2 Request & Response Validation                        │
│   │                                                                    │
│   └── Lifespan Singleton Pipeline (In-Memory Preloaded Cache)          │
│            │                                                           │
│            ├── [ Pointwise Ranking Engine ]                            │
│            │   • Model: HistGradientBoostingClassifier                 │
│            │   • Input: Query Context + Hotel Feature Vectors          │
│            │   • Output: P(Booking | Query, Hotel) score (0–100%)       │
│            │                                                           │
│            └── [ Collaborative Filtering Engine ]                      │
│                • Association Matrix: P(Addon | Hotel, City, Stars)     │
│                • Output: Ranked add-on packages with confidence scores │
└────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
      [ High-Performance Response: < 15ms Round-Trip ]
```

### ML Model Pipeline & Feature Engineering

The search ranking model is formulated as a **pointwise conversion probability estimator** using `HistGradientBoostingClassifier`.

#### Feature Engineering Vector (14 Dimensions):

| Feature Name | Type | Mathematical Logic | Business Rationale |
| :--- | :--- | :--- | :--- |
| `price_ratio` | Float | `price_per_night / user_budget` | Relative price alignment to user's intended spend |
| `price_diff_norm` | Float | `abs(price - budget) / budget` | Normalized price distance penalty |
| `price_gaussian` | Float | `exp(-0.5 * (price_ratio - 1.0)^2)` | Non-linear consumer willingness curve |
| `star_exact_match` | Binary | `1.0 if star_rating == pref_stars else 0.0` | Exact star preference match |
| `star_diff` | Integer | `abs(star_rating - pref_stars)` | Delta penalty between actual and desired star tier |
| `review_score` | Float | `review_score (3.0 - 5.0)` | Direct guest satisfaction score |
| `rating_scaled` | Float | `(review_score - 3.5) / 1.5` | Standardized quality baseline |
| `log_review_count` | Float | `log(1 + review_count)` | Logarithmic transformation to model diminishing returns |
| `distance_to_center_km` | Float | Physical distance in km | Proximity to downtown and attractions |
| `distance_decay` | Float | `exp(-distance / 10.0)` | Exponential decay penalty for distant properties |
| `discount_percent` | Integer | Discount percentage (0% - 30%) | Deal promotion incentive |
| `discount_factor` | Float | `discount_percent / 100.0` | Scaled promotional boost |
| `value_score` | Float | `(review_score * 1,000,000) / (price + 1)` | Price-to-quality value metric |

#### Evaluation Metrics:
- **ROC-AUC:** `0.7459`
- **PR-AUC (Average Precision):** `0.8088`
- **Inference Latency:** `~2.4ms` per query batch.

### Cross-Sell Collaborative Filtering Engine

The cross-sell recommender evaluates co-booking transaction patterns from over 34,000 historical session logs to calculate Bayesian-smoothed item association:

```text
Association Score = [ (Co_Occurrences(Hotel, Item) + 1.0) / (Total_Bookings(Hotel) + 10.0) ]
                    + [ 0.3 * (City_Occurrences / Total_City_Bookings) ]
                    + Domain_Priors(City, Star_Class)
```

- Suggests high-intent add-ons (Airport Shuttle, Private Dinner, Guided Diving, Balinese Spa) tailored specifically to the booked property's class and geography.
- **Inference Latency:** `< 1.0ms`.

### Interactive Dashboard & Telemetry

The application comes with a pre-configured dashboard (`static/index.html`) demonstrating production observability:
- **Live Telemetry Banner:** Real-time server latency ticker, model type, ROC-AUC score, and candidate count.
- **Dynamic Query Controls:** City selector, interactive budget slider with real-time IDR currency formatting, star class selector, and algorithmic sorting toggles (`⚡ ML Conversion AI`, `Lowest Price`, `Top Rated`).
- **Real-Time Cross-Sell Modal:** Instant add-on recommendation preview with co-booking affinity percentages upon clicking any hotel card.
- **Direct Link to Keynote Slides:** Navbar includes quick navigation to `/slides` for presentation delivery.

---

## 🛠 Part III: Developer & Operations Guide

### Quick Start Installation

```bash
# 1. Clone the repository
git clone https://github.com/fallss/practical-ai-in-industry.git
cd practical-ai-in-industry

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate datasets and train ML models
python -m src.data.generator
python -m src.models.train_ranking
python -m src.models.recommender

# 5. Launch the FastAPI application
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Access the services in your browser:
- **Interactive UI:** [http://localhost:8000](http://localhost:8000)
- **Keynote Slide Deck:** [http://localhost:8000/slides](http://localhost:8000/slides)
- **Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

### API Reference & Endpoints

#### 1. System Health & Diagnostics
```http
GET /api/v1/health
```
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
      "pr_auc": 0.8088,
      "test_samples": 6843
    }
  },
  "server_time": "2026-10-20T14:17:39.531719+00:00"
}
```

#### 2. Synchronous Search Ranking
```http
POST /api/v1/hotels/search
Content-Type: application/json
```
```json
{
  "city": "Bali",
  "user_budget": 1500000,
  "pref_stars": 4,
  "sort_by": "ml_ranking",
  "limit": 5
}
```
**Response Headers:** `X-Process-Time-Ms: 6.87`  
**Response Body:**
```json
{
  "total": 10,
  "latency_ms": 5.12,
  "sort_applied": "ml_ranking",
  "hotels": [
    {
      "hotel_id": "htl_032",
      "name": "Grand Merdeka Skyline Bali #32",
      "city": "Bali",
      "star_rating": 4,
      "price_per_night": 1240000,
      "review_score": 4.8,
      "ml_score": 92.1,
      "badges": ["Guest Favorite", "Top Recommended Match"]
    }
  ]
}
```

#### 3. Cross-Sell Recommendations
```http
GET /api/v1/recommend/cross-sell?hotel_id=htl_032&top_k=3
```
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
    },
    {
      "item_id": "act_104",
      "name": "Full-Day Cultural Tour & Temple Visit",
      "category": "Tour",
      "price": 380000,
      "confidence_percent": 39
    }
  ]
}
```

---

### Automated Testing with PyTest

A solid test suite ensures production stability across feature transformations and API contracts:

```bash
pytest tests/ -v
```

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.7.0
rootdir: /home/ifhalfaizi/Downloads/ai_ml_in_production
configfile: pytest.ini

tests/test_api.py::test_health_endpoint PASSED                           [ 14%]
tests/test_api.py::test_search_endpoint_ml_ranking PASSED                [ 28%]
tests/test_api.py::test_hotel_detail_endpoint PASSED                     [ 42%]
tests/test_api.py::test_cross_sell_endpoint PASSED                       [ 57%]
tests/test_models.py::test_data_generation PASSED                        [ 71%]
tests/test_models.py::test_feature_engineering PASSED                    [ 85%]
tests/test_models.py::test_serving_pipeline_instance PASSED              [100%]

========================= 7 passed in 3.41s =========================
```

---

### Production Docker Deployment

Deploy the entire service locally or on a server using Docker:

```bash
# 1. Build and run in detached mode
docker compose up --build -d

# 2. Inspect real-time logs
docker compose logs -f

# 3. Check health status
curl -i http://localhost:8000/api/v1/health

# 4. Stop service
docker compose down
```

---

## 🚀 Part IV: Cloud Deployment Guide (Free & Production)

You can easily deploy this repository online so that recruiters, colleagues, and users can access your live API, Web Dashboard, and Presentation Deck anywhere in the world.

### Option A: 1-Click Deploy on Render.com (Recommended Free Hosting)

[Render](https://render.com) offers free cloud hosting with native Docker support and automatic GitHub synchronization.

1. Create a free account at [render.com](https://render.com) using your GitHub account (`fallss`).
2. Click **New +** $\rightarrow$ **Web Service**.
3. Select and connect your repository: `fallss/practical-ai-in-industry`.
4. Configure the service:
   - **Name:** `practical-ai-service`
   - **Environment:** `Docker`
   - **Region:** `Singapore` (or closest to your users)
   - **Plan:** `Free`
5. *(Optional)* Render will automatically detect the included [render.yaml](render.yaml) blueprint!
6. Click **Deploy Web Service**.
7. In ~2–3 minutes, Render will build the container, train the models, and provide a live public HTTPS URL:
   ```text
   https://practical-ai-service.onrender.com
   ```
   - **Live Web UI:** `https://practical-ai-service.onrender.com`
   - **Slide Deck:** `https://practical-ai-service.onrender.com/slides`
   - **Swagger Docs:** `https://practical-ai-service.onrender.com/docs`

---

### Option B: Railway.app (Automatic Docker Deployment)

[Railway](https://railway.app) is an ultra-fast platform with instant Docker deployment:

1. Log in to [Railway](https://railway.app) via GitHub.
2. Click **New Project** $\rightarrow$ **Deploy from GitHub repo**.
3. Select `fallss/practical-ai-in-industry`.
4. Railway will automatically detect the [Dockerfile](Dockerfile), build the application, and assign a live public domain under **Settings $\rightarrow$ Networking $\rightarrow$ Generate Domain**.

---

### Option C: Hugging Face Spaces (Free ML Portfolio Hosting)

[Hugging Face Spaces](https://huggingface.co/spaces) is the premier portfolio platform for Machine Learning applications:

1. Create an account at [huggingface.co](https://huggingface.co).
2. Go to **New Space**.
3. Set **Space Name** to `practical-ai-in-industry` and select **Docker** as the Space SDK (Blank template).
4. Clone the Space repo or push this codebase to the Hugging Face Git remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/practical-ai-in-industry
   git push hf main
   ```
5. Hugging Face will automatically launch the container and host your interactive dashboard for free.

---

### Option D: Self-Hosted Cloud VPS (Docker Compose)

To host on your own Linux server (AWS EC2, DigitalOcean, Google Cloud Compute, Hetzner, or IDCloudHost):

```bash
# 1. SSH into your VPS
ssh user@your-server-ip

# 2. Clone the repository
git clone https://github.com/fallss/practical-ai-in-industry.git
cd practical-ai-in-industry

# 3. Launch with Docker Compose
docker compose up --build -d

# 4. (Optional) Set up Nginx Reverse Proxy with Certbot SSL
# Point your custom domain (e.g., ai.yourdomain.com) to port 8000
```

---

## 👤 Attribution & License

- **Keynote Presentation**: Authored by **Setia Budi** (Machine Learning Engineer at **tiket.com**).
- **Engineering Implementation**: Production AI architecture & codebase built for practical industry reference.
- **License**: Released under the **[GNU General Public License v3.0](LICENSE)**.
