from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class StandardsLibrary:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._data: Dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        self._data = json.loads(self.config_path.read_text(encoding="utf-8"))

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    def names(self) -> Dict[str, str]:
        return {key: value.get("name", key) for key, value in self._data.items()}

    def get(self, key: str) -> Dict[str, Any]:
        return self._data.get(key, {})
