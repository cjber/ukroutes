import logging

from rich.logging import RichHandler


def setup_logging():
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        handler = RichHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    return logger
