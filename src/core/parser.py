from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import fitz

from src.utils.text import normalize_whitespace

logger = logging.getLogger(__name__)

INDICATOR_KEYWORDS = [
    "pH",
    "溶解氧",
    "高锰酸盐指数",
    "氨氮",
    "总磷",
    "总氮",
    "化学需氧量",
    "五日生化需氧量",
    "浊度",
    "色度",
    "挥发酚",
    "石油类",
    "氰化物",
    "六价铬",
    "总氯",
    "总氮",
    "总磷",
]

METHOD_PATTERNS = [
    re.compile(r"([\u4e00-\u9fffA-Za-z0-9、（）()]+?)\s+(HJ\s?\d{2,4}[-—]\d{4})"),
    re.compile(r"([\u4e00-\u9fffA-Za-z0-9、（）()]+?)\s+(GB/?T?\s?\d{2,5}(?:[.-]\d{2,4})?)"),
]


class PDFTextExtractor:
    def extract(self, file_path: str) -> Tuple[str, List[str]]:
        pages: List[str] = []
        doc = fitz.open(file_path)
        try:
            for page in doc:
                text = page.get_text("text")
                pages.append(normalize_whitespace(text))
        finally:
            doc.close()
        return "\n\n".join(pages), pages


class ReportParser:
    def __init__(self) -> None:
        self.extractor = PDFTextExtractor()

    def parse(self, file_path: str) -> Dict[str, object]:
        text, pages = self.extractor.extract(file_path)
        joined = normalize_whitespace(text)
        first_pages = "\n".join(pages[:5])

        report_no = self._search(joined, r"([\u4e00-\u9fff()（）\[\]A-Za-z0-9-]*字\[[0-9]{4}\][^\n]{0,40}?号)")
        report_date = self._search(joined, r"报告日期[:：]?\s*([0-9]{4}[年./-][0-9]{1,2}[月./-][0-9]{1,2}日?)")
        commissioning_unit = self._search(joined, r"委托单位[:：]?\s*([^\n]+)")
        inspected_unit = self._search(joined, r"受检单位(?:\(项目名称\))?[:：]?\s*([^\n]+)")
        project_name = self._search(joined, r"项目名称[:：]?\s*([^\n]+)")
        report_type = self._infer_report_type(joined)
        cma_code, cma_valid_from, cma_valid_to = self._extract_cma_info(first_pages)
        indicators = [name for name in INDICATOR_KEYWORDS if name in joined]
        methods = self._extract_methods(joined)

        warnings: List[str] = []
        if len(joined) < 200:
            warnings.append("文本提取较少，当前 PDF 可能是扫描件，建议使用 OCR 版 PDF 或接入本地 OCR。")

        return {
            "file_path": str(Path(file_path).resolve()),
            "raw_text": joined,
            "pages": pages,
            "report_no": report_no,
            "report_date": report_date,
            "commissioning_unit": commissioning_unit,
            "inspected_unit": inspected_unit,
            "project_name": project_name,
            "report_type": report_type,
            "cma_code": cma_code,
            "cma_valid_from": cma_valid_from,
            "cma_valid_to": cma_valid_to,
            "supported_indicators": indicators,
            "methods_detected": methods,
            "warnings": warnings,
        }

    def _search(self, text: str, pattern: str) -> str:
        match = re.search(pattern, text)
        return match.group(1).strip() if match else ""

    def _infer_report_type(self, text: str) -> str:
        if "地表水" in text:
            return "surface_water"
        if "地下水" in text:
            return "groundwater"
        if "废水" in text:
            return "wastewater"
        if "空气" in text:
            return "air"
        return "unknown"

    def _extract_cma_info(self, text: str) -> Tuple[str, str, str]:
        code = self._search(text, r"CMA\s*([0-9]{8,})")
        if not code:
            code = self._search(text, r"([0-9]{9,})\s*[\r\n ]*(?:20\d{2}[./-])")

        validity_match = re.search(
            r"(20\d{2}[./-]\d{1,2}[./-]\d{1,2})\s*[-—~至]\s*(20\d{2}[./-]\d{1,2}[./-]\d{1,2})",
            text,
        )
        if validity_match:
            return code, validity_match.group(1), validity_match.group(2)
        return code, "", ""

    def _extract_methods(self, text: str) -> Dict[str, str]:
        methods: Dict[str, str] = {}
        for line in text.splitlines():
            for pattern in METHOD_PATTERNS:
                for match in pattern.finditer(line):
                    name = match.group(1).strip()
                    method = match.group(2).replace(" ", "")
                    if len(name) > 40:
                        continue
                    methods[name] = method
        return methods
