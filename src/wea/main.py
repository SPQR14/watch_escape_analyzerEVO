"""Executable entry point for Watch Escape Analyzer EVO."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication, QMessageBox

from wea.i18n.translator import Translator
from wea.ui.main_window import MainWin
from wea.utils.logger_setup import LoggerSetup

logger = LoggerSetup.setup_logger(__name__)


def global_exception_handler(exc_type, exc_value, exc_traceback):
    """
    Catch any unhandled exception in the main thread or Qt threads.
    Prevents the application from crashing silently.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Allow terminal Ctrl+C to work normally
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical(
        "Unhandled global exception",
        exc_info=(exc_type, exc_value, exc_traceback),
    )


def main(argv: list[str] | None = None) -> int:
    """Launch the Watch Escape Analyzer EVO PyQt6 application."""
    logger.info("Starting Watch Escape Analyzer EVO application")

    # 1. Inject the global hook before starting the Qt machinery
    sys.excepthook = global_exception_handler

    args = argv if argv is not None else sys.argv
    app = QApplication.instance() or QApplication(args)
    app.setApplicationName("Watch Escape Analyzer EVO")
    app.setOrganizationName("WatchEscapeAnalyzerEVO")

    # 2. Initialization isolation: Catch I/O, memory, or hardware failures here
    try:
        # Will fail here if the translation file is missing or corrupted
        translator = Translator("en")

        # Will fail here if the UI tries to load a missing asset or invalid
        # layout.
        window = MainWin(translator=translator)

        # NOTE: Instantiate the DSPEngine and audio interface here in the
        # future.

    except Exception as e:
        logger.critical(f"Critical failure during initialization: {e}")
        # Visual alert to the user before aborting
        QMessageBox.critical(
            None,
            "Fatal Startup Error",
            f"Could not initialize the application.\n\nDetails: {str(e)}",
        )
        return 1  # Exit with system error code

    # 3. Window system calls (free of try/except blocks)
    window.show()
    logger.info("Watch Escape Analyzer EVO window displayed successfully")

    exit_code = int(app.exec())
    logger.info("Watch Escape Analyzer EVO exiting with code %d", exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
