"""Serve the built MoleculeLens frontend and API from one staging origin."""

from pathlib import Path

from fastapi.staticfiles import StaticFiles

from app.main import app


frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if not (frontend_dist / "index.html").is_file():
    raise RuntimeError("Build the frontend with npm run build before starting staging.")

# Register static files after the API routes so relative frontend requests reach FastAPI.
app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
