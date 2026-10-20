import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.config import API_TITLE, API_VERSION, API_DESCRIPTION, STATIC_DIR, PRESENTATION_DIR
from src.models.pipeline import MLServingPipeline
from src.api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML models & catalog
    pipeline = MLServingPipeline.get_instance()
    pipeline.load()
    yield
    # Shutdown: Clean up resources if needed
    print("Shutting down ML Serving Pipeline...")

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Latency Tracking Middleware (Production Observability)
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response

# API Routes
app.include_router(router)

# Mount static web directory
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Mount presentation directory for online viewing
if PRESENTATION_DIR.exists():
    app.mount("/presentation", StaticFiles(directory=PRESENTATION_DIR, html=True), name="presentation")

@app.get("/", include_in_schema=False)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Welcome to OTA Production AI Engine API. Visit /docs for OpenAPI specs."}

@app.get("/slides", include_in_schema=False)
async def serve_slides():
    slide_file = PRESENTATION_DIR / "presentation.html"
    if slide_file.exists():
        return FileResponse(slide_file)
    return {"message": "Presentation slides not found."}
