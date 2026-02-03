import logging
import sys


def setup_logging(level=logging.INFO):
    """Set up universal logging configuration for use across modules."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("project.log", mode="a")
        ]
    )
