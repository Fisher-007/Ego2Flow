"""Write timestamped progress events."""
import logging
import os
from pathlib import Path


def progress(event, message, level="info", **fields):
    logger = logging.getLogger("Ego2Flow.progress")
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s", "%Y-%m-%d %H:%M:%S")
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)
        for destination in dict.fromkeys(os.environ.get(key) for key in ("E2F_PROGRESS_LOG", "E2F_WORKER_LOG") if os.environ.get(key)):
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(path, mode="a")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    context = {"city": os.environ.get("E2F_CITY", ""), **fields}
    if os.environ.get("E2F_WORKER_ID"):
        context = {"city": os.environ.get("E2F_CITY", ""), "worker": os.environ["E2F_WORKER_ID"], **fields}
    details = " ".join(f"{key}={value}" for key, value in context.items() if value is not None)
    getattr(logger, level)(f"{event:<10} {message} {details}".rstrip())
