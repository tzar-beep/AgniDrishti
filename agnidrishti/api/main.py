"""FastAPI app. Run with ``uvicorn agnidrishti.api.main:app``."""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from agnidrishti import config
from agnidrishti.api.routes import Store, router
from agnidrishti.utils import setup_logging

log = logging.getLogger(__name__)


def create_app(
    processed_dir: Path = config.DATA_PROCESSED,
    model_json: Path = config.MODELS_DIR / f"{config.MODEL_NAME}.json",
) -> FastAPI:
    """Build the app, loading precomputed tables from ``processed_dir`` and ``model_json``."""
    setup_logging()
    app = FastAPI(title="AgniDrishti", version="0.1.0")
    app.state.store = Store.load(processed_dir, model_json)
    log.info("Loaded %d cells, %d dates", len(app.state.store.geometry), len(app.state.store.dates))
    app.include_router(router)
    app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")
    return app


_app: FastAPI | None = None


def __getattr__(name: str) -> FastAPI:
    """Create the default ``app`` lazily so importing this module needs no data on disk."""
    global _app
    if name != "app":
        raise AttributeError(name)
    if _app is None:
        _app = create_app()
    return _app
