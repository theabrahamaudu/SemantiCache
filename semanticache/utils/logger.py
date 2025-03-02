"""
Configuration module for logging to console
"""

import logging
from logging import Logger


def logger(level: str = "DEBUG") -> Logger:
    """Format logger, configure file handler and add handler
    for data fetching logger.

    Returns:
        Logger: Logger for data fetching logs
    """

    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s:%(filename)s:%(funcName)s:%(levelname)s:%(message)s:'
    )

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
