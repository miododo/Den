from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from src.core.standards import StandardsLibrary
from src.ui.main_window import MainWindow
from src.utils.logging_utils import setup_logging


APP_DIR = Path(__file__).resolve().parent
EXPORTS_DIR = APP_DIR / "exports"
CONFIG_PATH = APP_DIR / "config" / "standards.json"


def main() -> int:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(APP_DIR)
    app = QApplication(sys.argv)
    standards = StandardsLibrary(CONFIG_PATH)
    window = MainWindow(standards, EXPORTS_DIR)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
