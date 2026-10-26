"""Logging configuration for the BANFF pipeline."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Configure logging for the BANFF pipeline.

    Args:
        level: Logging level (default: INFO).
        log_file: Optional path to log file.
        format_string: Custom format string for log messages.

    Returns:
        Root logger instance.
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    return logging.getLogger(__name__)
