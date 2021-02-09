# -*- coding: utf-8 -*-

__all__ = ["setup_logger"]

import sys
import logging


def exception_hook(exc_type, exc_value, exc_traceback):
    logging.error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )


def setup_logger(**kwargs):
    logging.basicConfig(**kwargs)
    sys.excepthook = exception_hook
