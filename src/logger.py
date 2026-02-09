import logging
import sys
from typing import Any

import structlog


def setup_logging(name: str = "diffusionGPT", log_level: str = "INFO") -> Any:
    """Configures professional logging compatible with cloud & local dev."""

    # Configure standard logging to capture library logs (Transformers, etc.)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # Use JSON in production (files), ConsoleRenderer for dev
            structlog.dev.ConsoleRenderer()
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger(name)
