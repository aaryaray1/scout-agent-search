"""Logging setup shared by the API server and the CLI.

Scout's own loggers run at the configured level; everything else stays at
WARNING. Without that split, turning Scout up to INFO also turns on
sentence-transformers' model-loading chatter and httpx's per-request
logging, which buries Scout's output in third-party noise -- roughly two
dozen extra lines on every server startup, and enough to hide what
scout-ingest is actually doing.
"""
import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
CLI_FORMAT = "%(message)s"


def configure_logging(level, fmt=LOG_FORMAT):
    """Send Scout's logs to stderr at `level`, leaving dependencies at
    WARNING.

    A host that has already configured logging (uvicorn, gunicorn, pytest)
    keeps its own handlers and formatting; only Scout's level is adjusted,
    so this never fights the process that owns the root logger.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.WARNING, format=fmt)
    logging.getLogger("scout").setLevel(level)
