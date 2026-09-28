"""
Unit tests for utils/logger.py.
"""

import logging

from project.utils.logger import LoggerUtils


def test_setup_logs_to_file_and_counts_levels(tmp_path):
    logger = logging.getLogger("test_logger")
    log_path = tmp_path / "test.log"
    counter = LoggerUtils.setup(logger, str(log_path))
    logger.info("info")
    logger.warning("warning")
    logger.error("error 1")
    logger.critical("error 2")
    assert (counter.warnings, counter.errors) == (1, 2)
    text = log_path.read_text(encoding="utf-8")
    assert "INFO    | info" in text and "WARNING | warning" in text
    # a second setup replaces the handlers instead of adding more
    counter = LoggerUtils.setup(logger, str(log_path))
    assert len(logger.handlers) == 3
    assert counter.warnings == counter.errors == 0
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
