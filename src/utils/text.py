from __future__ import annotations

import re
from datetime import datetime
from typing import Optional


_DATE_PATTERNS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%Y年%m月%d日",
    "%Y年%m月%d",
]


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def normalize_date_str(value: str) -> str:
    value = value.strip().replace("-", "/").replace(".", "/")
    value = value.replace("年", "/").replace("月", "/").replace("日", "")
    value = re.sub(r"/+", "/", value).strip("/")
    return value


def parse_date(value: str) -> Optional[datetime]:
    value = value.strip()
    for fmt in _DATE_PATTERNS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    normalized = normalize_date_str(value)
    for fmt in ("%Y/%m/%d", "%Y/%m"):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return None


def safe_preview(text: str, limit: int = 1200) -> str:
    text = normalize_whitespace(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
