"""Logging setup shared by the API server and the CLI."""
import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
CLI_FORMAT = "%(message)s"


def configure_logging(level, fmt=LOG_FORMAT):
    """Set Scout's loggers to `level`, leaving dependencies at WARNING.

    Only Scout's level is touched, so a host that owns the root logger
    (uvicorn, pytest) keeps its handlers. See docs/design/api.md.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.WARNING, format=fmt)
    logging.getLogger("scout").setLevel(level)
