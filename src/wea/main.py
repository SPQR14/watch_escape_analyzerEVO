"""Executable entry point for Watch Escape Analyzer EVO."""

from __future__ import annotations

import sys
from PyQt6.QtWidgets import QApplication

from wea.i18n.translator import Translator
from wea.ui.main_window import MainWin


def main(argv: list[str] | None = None) -> int:
    """Launch the Watch Escape Analyzer EVO PyQt6 application."""
    args = argv if argv is not None else sys.argv
    app = QApplication.instance() or QApplication(args)
    app.setApplicationName("Watch Escape Analyzer EVO")
    app.setOrganizationName("WatchEscapeAnalyzerEVO")

    translator = Translator("en")
    window = MainWin(translator=translator)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
