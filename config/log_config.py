import logging
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

STREAM_FORMATTER = logging.Formatter(
    fmt='[%(name)s] %(levelname)s: %(message)s'
)

FILE_FORMATTER = logging.Formatter(
    fmt='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

LEVEL = getattr(logging, LOG_LEVEL, logging.DEBUG)
