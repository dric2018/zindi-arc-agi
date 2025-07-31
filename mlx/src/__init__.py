import logging
import sys

logger = logging.getLogger("ARCLogger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console)

__all__ = ["logger"]

