"""
Logging to the console and to a file, with a count of warnings and errors for end-of-run summaries.

The pipeline logs to one file, {data root}/{version}/pipeline.log, set up once by main.py (setup_pipeline): the
orchestrator to "pipeline", every stage to "stage.<key>" and every plot to "plot.<key>". The stages of the
classification also write their own logs into their results (LoggerUtils.setup).
"""

import logging

PIPELINE_LOGGER = "pipeline"
# parent loggers of the orchestrator, the stages and the plots
PIPELINE_LOGGERS = (PIPELINE_LOGGER, "stage", "plot")
PIPELINE_LOG = "pipeline.log"


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
    def setup(logger: logging.Logger, log_path: str, mode: str = "w") -> LevelCounter:
        """
        Method to make a logger write to the console and to a file; previous handlers of the logger are removed.
        :param logger: logger to configure.
        :param log_path: path to the log file.
        :param mode: "w" to overwrite the log file, "a" to append to it.
        :return: handler counting warnings and errors.
        """
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.propagate = False
        # e.g. "[2026-09-29 16:35:16] [INFO] message"
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        for handler in (logging.StreamHandler(), logging.FileHandler(log_path, mode=mode, encoding="utf-8")):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        counter = LevelCounter()
        logger.addHandler(counter)
        return counter

    @staticmethod
    def setup_pipeline(names: tuple[str, ...], log_path: str | None, level: str = "INFO") -> LevelCounter:
        """
        Method to make the loggers of the pipeline (and their children, e.g. stage.costs) share one console handler and
        one file handler (appended); every line names its logger. Previous handlers of the loggers are removed.
        :param names: names of the parent loggers, e.g. ("pipeline", "stage", "plot").
        :param log_path: path to the log file (None: console only).
        :param level: level of the loggers, e.g. "INFO".
        :return: handler counting warnings and errors.
        """
        # e.g. "[2026-10-15 16:35:16] [INFO] [stage.costs] message"
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        handlers: list[logging.Handler] = [logging.StreamHandler()]
        if log_path is not None:
            handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))
        for handler in handlers:
            handler.setFormatter(formatter)
        counter = LevelCounter()
        for name in names:
            logger = logging.getLogger(name)
            for handler in logger.handlers:
                handler.close()
            logger.handlers.clear()
            logger.setLevel(level)
            logger.propagate = False
            for handler in [*handlers, counter]:
                logger.addHandler(handler)
        return counter
