"""
Logging to the console and to a file, with a count of warnings and errors for end-of-run summaries.
"""

import logging


class LevelCounter(logging.Handler):
    """
    Logging handler that counts warnings and errors.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.warnings = 0
        self.errors = 0

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.ERROR:
            self.errors += 1
        else:
            self.warnings += 1


class LoggerUtils:
    @staticmethod
    def setup(logger: logging.Logger, log_path: str) -> LevelCounter:
        """
        Method to make a logger write to the console and to a file; previous handlers of the logger are removed.
        :param logger: logger to configure.
        :param log_path: path to the log file (overwritten).
        :return: handler counting warnings and errors.
        """
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.propagate = False
        # e.g. "[2026-09-29 16:35:16] [INFO] message"
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        for handler in (logging.StreamHandler(), logging.FileHandler(log_path, mode="w", encoding="utf-8")):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        counter = LevelCounter()
        logger.addHandler(counter)
        return counter
