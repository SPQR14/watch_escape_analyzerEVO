from __future__ import annotations

import logging
import os
from pathlib import Path


class LoggerSetup:
    """
    A utility class for setting up the application's root logger.

    This class provides a static method to configure the root logger with
    two handlers: a FileHandler (INFO by default) that writes log messages to
    a specified log file and a StreamHandler (WARNING by default) that outputs
    log messages to the console. Levels can be overridden with the
    ``WEA_LOG_LEVEL`` and ``WEA_CONSOLE_LOG_LEVEL`` environment variables.

    The log file is created in a dedicated ``logs`` directory located in
    the same directory as this module. If the directory does not exist,
    it will be created automatically.

    Attributes:
        None

    Methods:
        setup_logger(name): Configures the root logger with the specified name.
    """

    BASE_DIR: Path = Path(__file__).resolve().parents[3]
    LOG_DIR: Path = BASE_DIR / "logs"
    LOG_FILE: Path = LOG_DIR / "wea.log"
    # DEBUG logging on every DSP window and drift update is real I/O on the hot path (console
    # writes are especially slow on Windows). Override with WEA_LOG_LEVEL / WEA_CONSOLE_LOG_LEVEL.
    DEFAULT_FILE_LEVEL: str = "INFO"
    DEFAULT_CONSOLE_LEVEL: str = "WARNING"
    _configured: bool = False

    @staticmethod
    def _resolve_level(env_var: str, default: str) -> int:
        """Return the logging level named by ``env_var`` (falls back to ``default``)."""
        name = os.environ.get(env_var, default).strip().upper()
        level = logging.getLevelName(name)
        return level if isinstance(level, int) else logging.getLevelName(default)

    @staticmethod
    def setup_logger(name: str = "wea") -> logging.Logger:
        """
        Configure the application's root logger.

        This function creates a dedicated ``logs`` directory in the same
        directory as this module if it does not already exist. It then
        configures the root logger with two handlers:

        - A FileHandler that writes log messages to ``logs/<name>.log``
          (level ``WEA_LOG_LEVEL``, INFO by default).
        - A StreamHandler that outputs log messages to the console
          (level ``WEA_CONSOLE_LOG_LEVEL``, WARNING by default).

        Any existing logging configuration is replaced to ensure that the
        specified handlers and logging level are applied.

        Args:
            name (str): Name used to generate the log file. The resulting log
                file will be created as ``logs/<name>.log``.

        Returns:
            None
        """

        # Defines log directory and creates it if it doesn't exist
        LoggerSetup.LOG_DIR.mkdir(parents=True, exist_ok=True)

        root_logger = logging.getLogger()
        has_unified_handler = any(
            isinstance(h, logging.FileHandler)
            and Path(getattr(h, "baseFilename", "")).resolve() == LoggerSetup.LOG_FILE.resolve()
            for h in root_logger.handlers
        )

        if not LoggerSetup._configured or not has_unified_handler:
            file_level = LoggerSetup._resolve_level("WEA_LOG_LEVEL", LoggerSetup.DEFAULT_FILE_LEVEL)
            console_level = LoggerSetup._resolve_level(
                "WEA_CONSOLE_LOG_LEVEL", LoggerSetup.DEFAULT_CONSOLE_LEVEL
            )
            file_handler = logging.FileHandler(LoggerSetup.LOG_FILE, encoding="utf-8")
            file_handler.setLevel(file_level)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(console_level)
            logging.basicConfig(
                level=min(file_level, console_level),
                format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
                datefmt="%d-%m-%Y %H:%M:%S",
                handlers=[file_handler, console_handler],
                force=True,
            )
            LoggerSetup._configured = True

        logger = logging.getLogger(name)

        logger.debug(
            "Logger '%s' initialized and attached to unified log '%s'",
            name,
            LoggerSetup.LOG_FILE.name,
        )

        return logger
