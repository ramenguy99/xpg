from pyxpg import *

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.NOTSET,
    format='[%(asctime)s.%(msecs)03d] %(levelname)-6s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

log_levels = {
    LogLevel.TRACE: logging.DEBUG,
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARN: logging.WARN,
    LogLevel.ERROR: logging.ERROR,
}

def log_callback(level, ctx, s):
    logger.log(log_levels[level], f"[{ctx}] {s}")

log_ovverride = LogCapture(log_callback)

ctx = Context(
    enable_validation_layer=True,
    enable_synchronization_validation=True,
)
