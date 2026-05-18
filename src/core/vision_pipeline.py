from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import importlib.util
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import fitz
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from src.core.ai_client import AIClient
from src.core.formula_engine import FormulaDatabase, FormulaVerifier
from src.core.models import DetectionRecord, ProcessedPage, SamplingInfo, StructuredReportResult
from src.core.standards import StandardsLibrary
from src.utils.text import normalize_whitespace, safe_preview

logger = logging.getLogger(__name__)

try:  # OpenCV gives the best scanner-like perspective correction when installed.
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional acceleration
    cv2 = None  # type: ignore


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PDF_SUFFIXES = {".pdf"}
CONFIDENCE_REVIEW_THRESHOLD = 0.95

INDICATOR_ALIASES: Dict[str, List[str]] = {
    "PM2.5": ["PM2.5", "PM₂.₅", "PM 2.5", "细颗粒物", "粒径小于等于2.5"],
    "PM10": ["PM10", "PM₁₀", "PM 10", "可吸入颗粒物", "粒径小于等于10"],
    "SO2": ["SO2", "SO₂", "二氧化硫"],
    "NO2": ["NO2", "NO₂", "二氧化氮"],
    "NOx": ["NOx", "NOX", "氮氧化物"],
    "CO": ["CO", "一氧化碳"],
    "O3": ["O3", "O₃", "臭氧"],
    "COD": ["COD", "化学需氧量"],
    "BOD5": ["BOD5", "BOD₅", "五日生化需氧量", "生化需氧量"],
    "氨氮": ["氨氮", "NH3-N", "NH₃-N", "NH4-N", "铵氮"],
    "总磷": ["总磷", "TP"],
    "总磷酸盐": ["总磷酸盐", "磷酸盐"],
    "总氮": ["总氮", "TN"],
    "pH": ["pH", "PH", "氢离子浓度指数"],
    "浊度": ["浊度", "浊度计"],
    "电导率": ["电导率", "也导率", "屯导率", "电导"],
    "溶解氧": ["溶解氧", "DO"],
    "高锰酸盐指数": ["高锰酸盐指数", "CODMn", "CODMn"],
    "石油类": ["石油类"],
    "悬浮物": ["悬浮物", "SS"],
}

UNIT_PATTERN = re.compile(
    r"(μg\s*/\s*m[³3]|µg\s*/\s*m[³3]|ug\s*/\s*m[³3]|mg\s*/\s*m[³3]|"
    r"μg\s*/\s*L|µg\s*/\s*L|ug\s*/\s*L|mg\s*/\s*L|"
    r"mg/L|μg/L|µg/L|ug/L|mg/m3|ug/m3|μg/m3|µg/m3|"
    r"μS\s*/\s*cm|µS\s*/\s*cm|uS\s*/\s*cm|mS\s*/\s*cm|"
    r"无量纲|%|NTU|dB)",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"(?P<prefix><|≤|<=|≥|>=)?\s*(?P<number>\d+(?:\.\d+)?)")
STANDARD_CODE_PATTERN = re.compile(
    r"(?:HJ|GB|GB/T|GBZ|NY/T|SL|CJ/T|ISO)\s*[-/]?\s*\d{2,5}(?:[-—./]\d{2,4})?",
    re.IGNORECASE,
)
NON_RESULT_LINE_PATTERN = re.compile(
    r"("
    r"分析方法|检测方法|监测分析方法|方法及依据|分析方法名称|检测人员|采样人员|分析人员|"
    r"仪器名称|仪器型号|仪器编号|有效期|溯源|检定|校准|精度|报告说明|检测报告说明|"
    r"合同|协议|任务书|通知单|目录|登记人|地址|电话|网址|页共|第\d+版|修订|"
    r"检测点位|检测项目|检测频次|频次一览表|企业基本情况|备注|计算公式|标准溶液|"
    r"缓冲溶液|质控|自控|加标|回收|空白|审核|校对|签名|日期|样品前处理"
    r")",
    re.IGNORECASE,
)
RESULT_LINE_HINT_PATTERN = re.compile(
    r"(检测值|监测值|测定值|测量值|报出值|报出结果|计算结果|检测结果|监测结果|分析结果|样品编号|原始记录表|mg/L|μg/L|µg/L|ug/L|mg/m|μg/m|ug/m)",
    re.IGNORECASE,
)
OCR_PRIORITY_KEYWORDS = [
    "检测结果",
    "监测结果",
    "结果一览表",
    "pH值分析原始记录表",
    "pH值分析",
    "电导率仪分析原始记录表",
    "溶解氧仪分析原始记录表",
    "便携式浊度计分析原始记录表",
    "分光光度法原始记录表",
    "分析原始记录表",
    "样品编号",
    "报出值",
    "报出结果",
    "计算结果",
    "检测日期",
]


@dataclass
class PageBundle:
    processed: ProcessedPage
    text: str = ""


class ScanPreprocessor:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.images_dir = output_dir / "enhanced_pages"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def process_files(self, file_paths: Sequence[str]) -> Tuple[List[PageBundle], List[str], str]:
        warnings: List[str] = []
        bundles: List[PageBundle] = []
        max_workers = max(1, min(4, len(file_paths)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(self._process_one_file, Path(path)) for path in file_paths]
            for future in concurrent.futures.as_completed(futures):
                try:
                    file_bundles, file_warnings = future.result()
                    bundles.extend(file_bundles)
                    warnings.extend(file_warnings)
                except Exception as exc:
                    logger.exception("Image preprocessing failed")
                    warnings.append(f"图像预处理失败：{exc}")

        bundles.sort(key=lambda item: (item.processed.source_path, item.processed.page_index))
        enhanced_pdf = self._write_enhanced_pdf([item.processed for item in bundles])
        return bundles, warnings, enhanced_pdf

    def _process_one_file(self, path: Path) -> Tuple[List[PageBundle], List[str]]:
        suffix = path.suffix.lower()
        if suffix in PDF_SUFFIXES:
            return self._process_pdf(path)
        if suffix in IMAGE_SUFFIXES:
            bundle = self._process_image(path, 1, "")
            return [bundle], []
        return [], [f"已跳过不支持的文件类型：{path}"]

    def _process_pdf(self, path: Path) -> Tuple[List[PageBundle], List[str]]:
        bundles: List[PageBundle] = []
        warnings: List[str] = []
        doc = fitz.open(str(path))
        try:
            for index, page in enumerate(doc, start=1):
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = normalize_whitespace(page.get_text("text"))
                bundle = self._process_pil_image(image, path, index, text)
                bundles.append(bundle)
        finally:
            doc.close()
        if not any(item.text for item in bundles):
            warnings.append("PDF 未提取到嵌入文本，已按扫描页处理；系统将尝试本地 OCR 或 AI 视觉读取。")
        return bundles, warnings

    def _process_image(self, path: Path, page_index: int, text: str) -> PageBundle:
        image = Image.open(path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        return self._process_pil_image(image, path, page_index, text)

    def _process_pil_image(self, image: Image.Image, source_path: Path, page_index: int, text: str) -> PageBundle:
        original_width, original_height = image.size
        working = self._limit_size(image)
        steps = ["EXIF 方向校正", "长边限制为 3200px 以内"]
        warnings: List[str] = []

        corrected, perspective_confidence, perspective_steps, perspective_warnings = self._perspective_correct(working)
        steps.extend(perspective_steps)
        warnings.extend(perspective_warnings)

        enhanced, enhance_steps = self._scanner_enhance(corrected)
        steps.extend(enhance_steps)
        out_path = self._enhanced_path(source_path, page_index)
        enhanced.save(out_path, format="PNG", optimize=True)

        processed = ProcessedPage(
            source_path=str(source_path.resolve()),
            page_index=page_index,
            original_width=original_width,
            original_height=original_height,
            enhanced_image_path=str(out_path.resolve()),
            transform_confidence=round(float(perspective_confidence), 3),
            preprocessing_steps=steps,
            warnings=warnings,
        )
        return PageBundle(processed=processed, text=text)

    def _limit_size(self, image: Image.Image, max_side: int = 3200) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        if side <= max_side:
            return image.copy()
        ratio = max_side / side
        return image.resize((int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS)

    def _perspective_correct(self, image: Image.Image) -> Tuple[Image.Image, float, List[str], List[str]]:
        if cv2 is not None:
            result = self._opencv_perspective_correct(image)
            if result is not None:
                corrected, confidence = result
                return corrected, confidence, ["OpenCV 透视变换校正"], []
        cropped, confidence = self._fallback_document_crop(image)
        warning = "未检测到 OpenCV 或未稳定识别四边形轮廓，已使用文档边界裁切增强。"
        return cropped, confidence, ["文档边界裁切"], [warning]

    def _opencv_perspective_correct(self, image: Image.Image) -> Optional[Tuple[Image.Image, float]]:
        arr_rgb = np.array(image.convert("RGB"))
        arr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
        ratio = arr.shape[0] / 700.0
        resized = cv2.resize(arr, (int(arr.shape[1] / ratio), 700))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 60, 180)
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) != 4:
                continue
            area_ratio = cv2.contourArea(approx) / float(resized.shape[0] * resized.shape[1])
            if area_ratio < 0.18:
                continue
            points = approx.reshape(4, 2).astype("float32") * ratio
            warped = self._four_point_transform(arr, points)
            confidence = max(0.55, min(0.99, area_ratio * 1.4))
            pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            return pil, confidence
        return None

    def _four_point_transform(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        rect = self._order_points(points)
        tl, tr, br, bl = rect
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_width = max(1, int(max(width_a, width_b)))
        max_height = max(1, int(max(height_a, height_b)))
        dst = np.array(
            [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, matrix, (max_width, max_height), flags=cv2.INTER_CUBIC)

    def _order_points(self, points: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        summed = points.sum(axis=1)
        rect[0] = points[np.argmin(summed)]
        rect[2] = points[np.argmax(summed)]
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]
        return rect

    def _fallback_document_crop(self, image: Image.Image) -> Tuple[Image.Image, float]:
        gray = ImageOps.grayscale(image)
        arr = np.asarray(gray)
        cutoff = min(245, max(170, int(np.percentile(arr, 70)) - 10))
        mask = arr < cutoff
        coords = np.argwhere(mask)
        if coords.size < 100:
            return image.copy(), 0.25
        y0, x0 = np.percentile(coords, 1, axis=0)
        y1, x1 = np.percentile(coords, 99, axis=0)
        margin = int(min(image.size) * 0.04)
        left = max(0, int(x0) - margin)
        top = max(0, int(y0) - margin)
        right = min(image.width, int(x1) + margin)
        bottom = min(image.height, int(y1) + margin)
        crop_area = max(1, (right - left) * (bottom - top))
        confidence = min(0.65, max(0.3, crop_area / float(image.width * image.height)))
        return image.crop((left, top, right, bottom)), confidence

    def _scanner_enhance(self, image: Image.Image) -> Tuple[Image.Image, List[str]]:
        image = self._remove_colored_noise(image)
        gray = ImageOps.grayscale(image)
        radius = max(18, min(gray.size) // 28)
        background = gray.filter(ImageFilter.GaussianBlur(radius=radius))
        gray_arr = np.asarray(gray).astype(np.float32)
        bg_arr = np.asarray(background).astype(np.float32)
        corrected = gray_arr / np.maximum(bg_arr, 1.0) * 238.0
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        corrected_image = Image.fromarray(corrected)
        corrected_image = ImageOps.autocontrast(corrected_image, cutoff=1)
        corrected_image = corrected_image.filter(ImageFilter.UnsharpMask(radius=1.6, percent=180, threshold=3))
        arr = np.asarray(corrected_image).copy()
        white_cutoff = max(218, int(np.percentile(arr, 82)))
        arr[arr >= white_cutoff] = 255
        arr[arr <= 55] = np.maximum(0, arr[arr <= 55] * 0.72).astype(np.uint8)
        enhanced = Image.fromarray(arr).convert("RGB")
        return enhanced, ["红章/彩色笔迹弱化", "阴影背景估计与去除", "自动对比度", "文字边缘锐化", "背景白化"]

    def _remove_colored_noise(self, image: Image.Image) -> Image.Image:
        arr = np.array(image.convert("RGB"))
        r = arr[:, :, 0].astype(np.int16)
        g = arr[:, :, 1].astype(np.int16)
        b = arr[:, :, 2].astype(np.int16)
        red_stamp = (r > g + 34) & (r > b + 34) & (r > 115)
        blue_pen = (b > r + 28) & (b > g + 12) & (b > 90)
        green_pen = (g > r + 30) & (g > b + 10) & (g > 90)
        mask = red_stamp | blue_pen | green_pen
        arr[mask] = [255, 255, 255]
        return Image.fromarray(arr)

    def _enhanced_path(self, source_path: Path, page_index: int) -> Path:
        digest = hashlib.sha1(f"{source_path.resolve()}::{page_index}".encode("utf-8")).hexdigest()[:12]
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", source_path.stem).strip("_") or "page"
        return self.images_dir / f"{stem}_{page_index:03d}_{digest}.png"

    def _write_enhanced_pdf(self, pages: Sequence[ProcessedPage]) -> str:
        if not pages:
            return ""
        output = self.output_dir / "enhanced_scan.pdf"
        images: List[Image.Image] = []
        for page in pages:
            try:
                images.append(Image.open(page.enhanced_image_path).convert("RGB"))
            except Exception as exc:
                logger.warning("Cannot add enhanced image to PDF: %s", exc)
        if not images:
            return ""
        first, rest = images[0], images[1:]
        first.save(output, "PDF", resolution=200.0, save_all=True, append_images=rest)
        for image in images:
            image.close()
        return str(output.resolve())


class OptionalLocalOCR:
    def __init__(self) -> None:
        self.paddleocr_available = importlib.util.find_spec("paddleocr") is not None
        self.rapidocr_available = importlib.util.find_spec("rapidocr") is not None
        self.tesseract_available = importlib.util.find_spec("pytesseract") is not None
        self.available = self.paddleocr_available or self.rapidocr_available or self.tesseract_available
        self.warning = "" if self.available else "未安装可用本地 OCR，本地图片 OCR 已跳过。"
        self._paddleocr = None
        self._rapidocr = None
        self.last_engine = ""

    def extract(self, image_path: str) -> Tuple[str, str]:
        if not self.available:
            return "", self.warning
        if self.paddleocr_available:
            text, warning = self._extract_with_paddleocr(image_path)
            if text:
                self.last_engine = "PaddleOCR"
                return text, warning
            if warning and not (self.rapidocr_available or self.tesseract_available):
                return "", warning
        if self.rapidocr_available:
            text, warning = self._extract_with_rapidocr(image_path)
            if text:
                self.last_engine = "RapidOCR"
                return text, warning
            if warning and not self.tesseract_available:
                return "", warning
        try:
            import pytesseract  # type: ignore

            text = pytesseract.image_to_string(Image.open(image_path), lang="chi_sim+eng")
            self.last_engine = "Tesseract"
            return normalize_whitespace(text), ""
        except Exception as exc:
            return "", f"本地 OCR 调用失败：{exc}"

    def _extract_with_paddleocr(self, image_path: str) -> Tuple[str, str]:
        try:
            if self._paddleocr is None:
                from paddleocr import PaddleOCR  # type: ignore

                attempts = [
                    {
                        "use_angle_cls": False,
                        "lang": "ch",
                        "show_log": False,
                        "enable_mkldnn": False,
                        "use_gpu": False,
                    },
                    {
                        "lang": "ch",
                        "text_detection_model_name": "PP-OCRv5_mobile_det",
                        "text_recognition_model_name": "PP-OCRv5_mobile_rec",
                        "use_doc_orientation_classify": False,
                        "use_doc_unwarping": False,
                        "use_textline_orientation": False,
                    },
                    {
                        "lang": "ch",
                        "use_doc_orientation_classify": False,
                        "use_doc_unwarping": False,
                        "use_textline_orientation": False,
                    },
                    {"lang": "ch"},
                ]
                last_error: Optional[Exception] = None
                for kwargs in attempts:
                    try:
                        self._paddleocr = PaddleOCR(**kwargs)
                        break
                    except Exception as exc:
                        last_error = exc
                if self._paddleocr is None:
                    raise last_error or RuntimeError("PaddleOCR 初始化失败")

            try:
                if hasattr(self._paddleocr, "ocr"):
                    result = self._paddleocr.ocr(image_path, cls=False)
                else:
                    result = self._paddleocr.predict(image_path)
            except TypeError as exc:
                if "cls" in str(exc) and hasattr(self._paddleocr, "predict"):
                    result = self._paddleocr.predict(image_path)
                else:
                    raise
            lines = self._parse_paddleocr_result(result)
            return normalize_whitespace("\n".join(lines)), ""
        except Exception as exc:
            return "", f"PaddleOCR 调用失败：{exc}"

    def _parse_paddleocr_result(self, result: object) -> List[str]:
        lines: List[str] = []
        if not result:
            return lines
        if isinstance(result, list):
            for block in result:
                if hasattr(block, "json"):
                    lines.extend(self._parse_paddleocr_json(getattr(block, "json")))
                    continue
                if isinstance(block, dict):
                    lines.extend(self._parse_paddleocr_json(block))
                    continue
                if isinstance(block, list):
                    for row in block:
                        if (
                            isinstance(row, (list, tuple))
                            and len(row) >= 2
                            and isinstance(row[1], (list, tuple))
                            and row[1]
                        ):
                            text = str(row[1][0]).strip()
                            score = float(row[1][1]) if len(row[1]) > 1 else 1.0
                            if text and score >= 0.45:
                                lines.append(text)
        elif hasattr(result, "json"):
            lines.extend(self._parse_paddleocr_json(getattr(result, "json")))
        return lines

    def _parse_paddleocr_json(self, data: object) -> List[str]:
        if callable(data):
            data = data()
        if not isinstance(data, dict):
            return []
        payload = data.get("res", data)
        if not isinstance(payload, dict):
            return []
        texts = payload.get("rec_texts") or payload.get("txts") or []
        scores = payload.get("rec_scores") or payload.get("scores") or [1.0] * len(texts)
        lines = []
        for text, score in zip(texts, scores):
            if str(text).strip() and float(score) >= 0.45:
                lines.append(str(text).strip())
        return lines

    def _extract_with_rapidocr(self, image_path: str) -> Tuple[str, str]:
        try:
            if self._rapidocr is None:
                from rapidocr import RapidOCR  # type: ignore

                self._rapidocr = RapidOCR()
            result = self._rapidocr(image_path)
            txts = list(getattr(result, "txts", None) or [])
            scores = list(getattr(result, "scores", None) or [])
            lines = [str(text).strip() for text, score in zip(txts, scores or [1.0] * len(txts)) if str(text).strip() and float(score) >= 0.45]
            return normalize_whitespace("\n".join(lines)), ""
        except Exception as exc:
            return "", f"RapidOCR 调用失败：{exc}"


class TableAwareExtractor:
    def extract(self, text: str) -> Tuple[List[SamplingInfo], List[DetectionRecord], List[str]]:
        warnings: List[str] = []
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sampling = self._extract_sampling(lines)
        default_sampling = sampling[0] if sampling else SamplingInfo()
        records: List[DetectionRecord] = self._extract_sample_mapped_result_tables(text, default_sampling)
        for page_no, line in self._iter_page_lines(text):
            record = self._extract_line_record(line, page_no, default_sampling)
            if record:
                records.append(record)

        if not records and text.strip():
            records = self.fallback_extract_surface_water_records(text, default_sampling)
            if records:
                warnings.append(f"规则抽取为空，已启用 OCR 兜底抽取保留 {len(records)} 条低置信度记录。")
            else:
                warnings.append("未从文本中稳定识别到“污染物-数值-单位”表格行，建议启用 AI 视觉识别或人工复核。")
        return sampling, self._dedupe_records(records), warnings

    def fallback_extract_surface_water_records(self, text: str, sampling: Optional[SamplingInfo] = None) -> List[DetectionRecord]:
        default_sampling = sampling or SamplingInfo()
        records: List[DetectionRecord] = []
        current_item = ""
        current_page = 1
        recent_context: List[str] = []
        item_patterns: Dict[str, str] = {
            "pH": r"(?:pH|PH|pH值)",
            "浊度": r"浊度",
            "电导率": r"电导率|也导率|屯导率|电导",
            "溶解氧": r"溶解氧|浴解氧",
            "总磷": r"总磷",
            "总氮": r"总氮",
            "氨氮": r"氨氮|氨氯|氮氨",
            "高锰酸盐指数": r"高锰酸盐指数|高猛酸盐指数|高轻酸盐指数",
            "COD": r"化学需氧量|COD|C0D|coo|c00",
            "BOD5": r"五日生化需氧量|BOD5|B005",
        }
        sample_re = re.compile(r"\b(?:HS|WS|S)\s*\d{1,3}(?:\s*[-—–~]\s*\d{1,3}){0,3}\b", re.IGNORECASE)
        raw_lines = text.splitlines()
        for line_index, raw_line in enumerate(raw_lines):
            line = raw_line.strip()
            if not line:
                continue
            marker = re.match(r"^\[page:(\d+)\]$", line)
            if marker:
                current_page = int(marker.group(1))
                recent_context = []
                continue
            compact = line.replace(" ", "")
            for item, pattern in item_patterns.items():
                if re.search(pattern, compact, re.IGNORECASE):
                    current_item = item
                    recent_context.append(line)
                    recent_context = recent_context[-4:]
                    break
            if any(keyword in compact for keyword in ["原始记录表", "样品编号", "报出值", "报出结果", "计算结果", "测量值"]):
                recent_context.append(line)
                recent_context = recent_context[-4:]
            line_for_sample = self._normalize_sample_text(line)
            samples = [sample.replace(" ", "").replace("~", "-").upper() for sample in sample_re.findall(line_for_sample)]
            if not current_item or not samples:
                continue
            if any(keyword in compact for keyword in ["空白", "空向", "质控", "保证值", "实测值"]):
                continue
            window = [line]
            next_index = line_index + 1
            while next_index < len(raw_lines) and len(window) < 14:
                next_line = raw_lines[next_index].strip()
                if not next_line:
                    next_index += 1
                    continue
                if re.match(r"^\[page:(\d+)\]$", next_line):
                    break
                normalized_next_sample = self._normalize_sample_text(next_line)
                if sample_re.search(normalized_next_sample):
                    break
                compact_next = next_line.replace(" ", "")
                if any(
                    keyword in compact_next
                    for keyword in ["质控样品编号", "保证值", "实测值", "是否合格", "准确度检查", "精密度检查", "准确度", "精密度"]
                ):
                    break
                if any(
                    re.search(pattern, compact_next, re.IGNORECASE)
                    for item, pattern in item_patterns.items()
                    if item != current_item
                ):
                    break
                window.append(next_line)
                next_index += 1
            selected_value = self._fallback_value_from_window(current_item, window, sample_re)
            if not selected_value:
                continue
            raw_value, value = selected_value
            context = " / ".join((recent_context + [line])[-5:])
            if len(window) > 1:
                context = " / ".join((recent_context + window[:6])[-8:])
            for sample in samples:
                records.append(
                    DetectionRecord(
                        indicator=current_item,
                        normalized_indicator=canonical_indicator(current_item),
                        value=value,
                        unit=self._guess_unit(current_item),
                        sample_time=default_sampling.sample_time,
                        sample_point=sample,
                        frequency=default_sampling.frequency,
                        status="review",
                        confidence=0.45,
                        needs_review=True,
                        source_page=current_page,
                        source_line=context,
                        raw_value=raw_value,
                        notes=["OCR兜底抽取：检测项目 + 样品编号 + 检测值，需人工复核"],
                    )
                )
        return records

    def _normalize_sample_text(self, line: str) -> str:
        normalized = line.replace("）", "").replace(")", "").replace("~", "-")
        normalized = re.sub(r"\bH[S5]\s*[Iil|!]\b", "HS1", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bHSS\b", "HS5", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bHs\b", "HS", normalized, flags=re.IGNORECASE)
        return normalized

    def _fallback_value_from_window(
        self, item: str, window: Sequence[str], sample_re: re.Pattern[str]
    ) -> Optional[Tuple[str, float]]:
        values = self._fallback_numeric_values(window, sample_re)
        if not values:
            return None
        if item == "pH":
            candidates = [(raw, value) for raw, value in values if 4 <= value <= 10 and self._fallback_has_decimal(raw)]
            if candidates:
                return candidates[-1]
        if item == "溶解氧":
            candidates = [(raw, value) for raw, value in values if 0.5 <= value <= 20 and self._fallback_has_decimal(raw)]
            if candidates:
                return candidates[-1]
        if item == "电导率":
            candidates = [(raw, value) for raw, value in values if value >= 20]
            if candidates:
                return max(candidates, key=lambda item_value: item_value[1])
        if item in {"总磷", "总氮", "氨氮", "高锰酸盐指数", "COD", "BOD5"}:
            candidates = [
                (raw, value)
                for raw, value in values
                if not (len(values) > 3 and abs(value - 10.0) < 0.0001 and raw.replace(" ", "") in {"10.0", "10-0"})
            ]
            decimal_candidates = [(raw, value) for raw, value in candidates if self._fallback_has_decimal(raw)]
            if decimal_candidates:
                return decimal_candidates[-1]
            if candidates:
                return candidates[-1]
        return values[-1]

    def _fallback_numeric_values(
        self, window: Sequence[str], sample_re: re.Pattern[str]
    ) -> List[Tuple[str, float]]:
        token_re = re.compile(
            r"[<>≤≥]?\s*[0-9OBIlSo]+(?:\s*[.\-·:]\s*[0-9OBIlSo]+)?(?:\s*[xX×]\s*10(?:\s*\^?\s*\d+)?)?",
            re.IGNORECASE,
        )
        values: List[Tuple[str, float]] = []
        for line in window:
            line_for_value = self._normalize_sample_text(line)
            line_for_value = sample_re.sub(" ", line_for_value)
            line_for_value = re.sub(r"\b20\d{2}[./-]?\d{0,2}[./-]?\d{0,2}\b", " ", line_for_value)
            for match in token_re.finditer(line_for_value):
                raw = match.group(0).strip()
                parsed = self._clean_ocr_number_token(raw)
                if parsed is None:
                    continue
                if self._fallback_value_plausible("", parsed):
                    values.append((raw, parsed))
        return values

    def _clean_ocr_number_token(self, token: str) -> Optional[float]:
        cleaned = token.strip().replace(" ", "")
        cleaned = cleaned.replace("，", ".").replace("·", ".").replace(":", ".")
        cleaned = cleaned.replace("O", "0").replace("o", "0")
        cleaned = cleaned.replace("B", "8").replace("I", "1").replace("l", "1").replace("|", "1").replace("!", "1")
        cleaned = re.sub(r"(?<=\d)[\-–—](?=\d)", ".", cleaned)
        cleaned = re.sub(r"^[<>≤≥]+", "", cleaned)
        cleaned = re.sub(r"[^0-9.+\-xX×^]", "", cleaned)
        if not cleaned or cleaned in {".", "-", "+", "+.", "-."}:
            return None
        sci_match = re.match(r"^([+-]?\d+(?:\.\d+)?)[xX×]10(?:\^?(\d+))?$", cleaned)
        if sci_match:
            exponent = int(sci_match.group(2) or "1")
            return float(sci_match.group(1)) * (10 ** exponent)
        number_match = re.search(r"[+-]?\d+(?:\.\d+)?", cleaned)
        if not number_match:
            return None
        try:
            return float(number_match.group(0))
        except ValueError:
            return None

    def _fallback_has_decimal(self, raw: str) -> bool:
        return bool(re.search(r"[.·:]", raw) or re.search(r"\d\s*[-–—]\s*\d", raw))

    def _fallback_value_plausible(self, item: str, value: float) -> bool:
        if item == "pH":
            return 0 <= value <= 14
        if item == "溶解氧":
            return 0 <= value <= 30
        if item in {"总磷", "总氮", "氨氮", "高锰酸盐指数", "COD", "BOD5"}:
            return 0 <= value <= 10000
        if item == "浊度":
            return 0 <= value <= 100000
        if item == "电导率":
            return 0 <= value <= 1000000
        return 0 <= value <= 100000

    def _guess_unit(self, item: str) -> str:
        if item == "pH":
            return "无量纲"
        if item == "浊度":
            return "NTU"
        if item == "电导率":
            return "μS/cm"
        if item in {"溶解氧", "总磷", "总氮", "氨氮", "高锰酸盐指数", "COD", "BOD5"}:
            return "mg/L"
        return ""

    def _extract_sample_mapped_result_tables(self, text: str, sampling: SamplingInfo) -> List[DetectionRecord]:
        records: List[DetectionRecord] = []
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sample_indicator = self._sample_indicator_map(lines)
        if not sample_indicator:
            return records
        page_by_line: Dict[int, int] = {}
        page_no = 1
        for idx, line in enumerate(text.splitlines()):
            marker = re.match(r"^\[page:(\d+)\]$", line.strip())
            if marker:
                page_no = int(marker.group(1))
            page_by_line[idx] = page_no

        in_result_section = False
        for idx, line in enumerate(lines):
            if re.search(r"(检测结果|监测结果|结果一览表)", line):
                in_result_section = True
                continue
            if in_result_section and re.search(r"(方法检出限|备注|编制|审核|签发|以下空白)", line):
                in_result_section = False
            if not in_result_section:
                continue
            sample_match = re.search(r"\b(WS\d+|HS\d+|S\d+)\b", line, flags=re.IGNORECASE)
            if not sample_match:
                continue
            sample_id = sample_match.group(1).upper()
            indicator = sample_indicator.get(sample_id)
            if not indicator:
                continue
            values = self._candidate_values_near_sample(line)
            source_parts = [line]
            lookahead = 1
            while idx + lookahead < len(lines) and lookahead <= 8:
                nxt = lines[idx + lookahead]
                if re.search(r"\b(WS\d+|HS\d+|S\d+)\b", nxt, flags=re.IGNORECASE):
                    break
                next_values = self._candidate_values_near_sample(nxt)
                if next_values:
                    values.extend(next_values)
                    source_parts.append(nxt)
                lookahead += 1
            value = self._choose_value_for_indicator(indicator, values)
            if value is None:
                continue
            source = " / ".join(source_parts[:5])
            confidence = 0.91 if "/" in source or lookahead > 1 else 0.94
            records.append(
                DetectionRecord(
                    indicator=indicator,
                    normalized_indicator=canonical_indicator(indicator),
                    value=value,
                    unit="mg/L",
                    sample_time=sampling.sample_time,
                    sample_point=sample_id,
                    frequency=sampling.frequency,
                    confidence=confidence,
                    needs_review=confidence < CONFIDENCE_REVIEW_THRESHOLD,
                    source_page=self._estimate_page_for_line(text, source),
                    source_line=source,
                    raw_value=f"{value:g}",
                    notes=["按样品编号-检测项目映射恢复宽表结果"],
                )
            )
        return records

    def _sample_indicator_map(self, lines: Sequence[str]) -> Dict[str, str]:
        sample_indicator: Dict[str, str] = {}
        for idx, line in enumerate(lines):
            sample_match = re.search(r"\b(WS\d+|HS\d+|S\d+)\b", line, flags=re.IGNORECASE)
            if not sample_match:
                continue
            sample_id = sample_match.group(1).upper()
            indicator, _, _ = self._find_indicator(line)
            if indicator and not self._is_non_result_line(line):
                sample_indicator[sample_id] = indicator
                continue
            # Detection-content tables often list the indicator one line after the sample id.
            for nxt in lines[idx + 1 : min(len(lines), idx + 4)]:
                if re.search(r"\b(WS\d+|HS\d+|S\d+)\b", nxt, flags=re.IGNORECASE):
                    break
                indicator, _, _ = self._find_indicator(nxt)
                if indicator:
                    sample_indicator[sample_id] = indicator
                    break
        return sample_indicator

    def _candidate_values_near_sample(self, line: str) -> List[float]:
        values: List[float] = []
        cleaned = re.sub(r"\b(?:LS|YS|WT|SY)\d{6,}\b", " ", line, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(WS|HS|S)\d+\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b20\d{2}[./-]?\d{0,2}[./-]?\d{0,2}\b", " ", cleaned)
        date_fragment = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})", cleaned.strip())
        if date_fragment:
            month = int(date_fragment.group(1))
            day = int(date_fragment.group(2))
            if len(date_fragment.group(1)) == 2 and 1 <= month <= 12 and 1 <= day <= 31:
                return values
        for match in NUMBER_PATTERN.finditer(cleaned):
            number_text = match.group("number")
            start, end = match.span()
            after = cleaned[end : end + 6]
            before = cleaned[max(0, start - 3) : start]
            if "." in number_text and (re.search(r"\d[./-]$", before) or re.match(r"\s*[./-]\d", after)):
                continue
            if re.match(r"\s*(页|版|次|天|年|月|日|号)", after):
                continue
            try:
                values.append(float(number_text))
            except ValueError:
                continue
        return values

    def _choose_value_for_indicator(self, indicator: str, values: Sequence[float]) -> Optional[float]:
        if not values:
            return None
        if indicator == "总氮":
            larger = [value for value in values if value >= 1 and value not in {1.0, 2.0, 3.0, 4.0}]
            return larger[-1] if larger else values[-1]
        if indicator in {"总磷", "氨氮"}:
            decimal = [value for value in values if value < 20]
            return decimal[-1] if decimal else values[-1]
        if indicator == "COD":
            larger = [value for value in values if value >= 4]
            return larger[0] if larger else values[-1]
        return values[-1]

    def _estimate_page_for_line(self, text: str, source: str) -> int:
        target = source.split(" / ", 1)[0]
        page_no = 1
        for line in text.splitlines():
            marker = re.match(r"^\[page:(\d+)\]$", line.strip())
            if marker:
                page_no = int(marker.group(1))
                continue
            if target and target in line:
                return page_no
        return page_no

    def _iter_page_lines(self, text: str) -> Iterable[Tuple[int, str]]:
        page_no = 1
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            marker = re.match(r"^\[page:(\d+)\]$", line)
            if marker:
                page_no = int(marker.group(1))
                continue
            yield page_no, line

    def _extract_sampling(self, lines: Sequence[str]) -> List[SamplingInfo]:
        info = SamplingInfo()
        confidence = 0.0
        source_lines: List[str] = []
        for line in lines:
            if not info.sample_time:
                value = self._search_value(
                    line,
                    [
                        r"采样时间[:：]?\s*(.+?)(?=\s*(样点|采样点|监测点|采样频|频次|$))",
                        r"采样日期[:：]?\s*(.+?)(?=\s*(样点|采样点|监测点|采样频|频次|$))",
                        r"(\d{4}[年./-]\d{1,2}[月./-]\d{1,2}日?(?:\s*\d{1,2}[:：]\d{2})?)",
                    ],
                )
                if value:
                    info.sample_time = value
                    source_lines.append(line)
                    confidence += 0.32
            if not info.sample_point:
                value = self._search_value(
                    line,
                    [
                        r"样点(?:名称|编号|点位)?[:：]?\s*(.+?)(?=\s*(采样时间|采样日期|采样频|频次|检测项目|$))",
                        r"采样点(?:位|名称)?[:：]?\s*(.+?)(?=\s*(采样时间|采样日期|采样频|频次|检测项目|$))",
                        r"监测点(?:位|名称)?[:：]?\s*(.+?)(?=\s*(采样时间|采样日期|采样频|频次|检测项目|$))",
                    ],
                )
                if value:
                    info.sample_point = value
                    source_lines.append(line)
                    confidence += 0.32
            if not info.frequency:
                value = self._search_value(
                    line,
                    [
                        r"采样(?:频次|频率)[:：]?\s*(.+?)(?=\s*(采样时间|采样日期|样点|采样点|监测点|$))",
                        r"频次[:：]?\s*(.+?)(?=\s*(采样时间|采样日期|样点|采样点|监测点|$))",
                    ],
                )
                if value:
                    info.frequency = value
                    source_lines.append(line)
                    confidence += 0.24
        if source_lines:
            info.confidence = min(0.98, confidence)
            info.source_line = " / ".join(list(dict.fromkeys(source_lines))[:3])
            return [info]
        return []

    def _search_value(self, line: str, patterns: Sequence[str]) -> str:
        for pattern in patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                return self._clean_cell(match.group(1))
        return ""

    def _extract_line_record(
        self, line: str, page_no: int, sampling: SamplingInfo
    ) -> Optional[DetectionRecord]:
        clean = self._clean_cell(line)
        if self._is_non_result_line(clean):
            return None
        indicator, alias, alias_pos = self._find_indicator(clean)
        if not indicator:
            return None
        tail = clean[alias_pos + len(alias) :]
        numbers = self._numbers_after_indicator(tail)
        if not numbers:
            return None
        unit = self._extract_unit(tail) or self._extract_unit(clean)
        value_prefix, raw_value, value = numbers[0]
        average_time = self._extract_average_time(clean)
        confidence = 0.98
        notes: List[str] = []
        if not unit and indicator != "pH":
            confidence -= 0.12
            notes.append("未在同一行识别到单位")
        if len(numbers) > 3:
            confidence -= 0.04
            notes.append("同一行存在多个数字，已按表格感知优先选取指标后的首个检测值")
        if value_prefix in {"<", "≤", "<="}:
            notes.append("检测值带小于号，按检出限数值参与保守校验")
        if self._looks_like_header_or_footer(clean):
            confidence -= 0.25
            notes.append("疑似页眉页脚或标准说明行")
        if re.search(r"[OoIl]\d|\d[OoIl]", clean):
            confidence -= 0.04
            notes.append("数字附近存在易混淆字符")

        return DetectionRecord(
            indicator=alias,
            normalized_indicator=indicator,
            value=value,
            unit=normalize_unit(unit) if unit else ("无量纲" if indicator == "pH" else ""),
            sample_time=sampling.sample_time,
            sample_point=sampling.sample_point,
            frequency=sampling.frequency,
            confidence=round(max(0.0, min(0.99, confidence)), 3),
            needs_review=confidence < CONFIDENCE_REVIEW_THRESHOLD,
            source_page=page_no,
            source_line=clean,
            average_time=average_time,
            raw_value=f"{value_prefix or ''}{raw_value}".strip(),
            notes=notes,
        )

    def _find_indicator(self, line: str) -> Tuple[str, str, int]:
        candidates: List[Tuple[str, str, int]] = []
        compact = line.replace(" ", "")
        for canonical, aliases in INDICATOR_ALIASES.items():
            for alias in aliases:
                compact_alias = alias.replace(" ", "")
                if compact_alias.isascii() and len(compact_alias) <= 3:
                    match = re.search(
                        rf"(?<![A-Za-z0-9]){re.escape(compact_alias)}(?![A-Za-z0-9])",
                        compact,
                        flags=re.IGNORECASE,
                    )
                    pos = match.start() if match else -1
                else:
                    pos = compact.lower().find(compact_alias.lower())
                if pos >= 0:
                    candidates.append((canonical, alias, pos))
        if not candidates:
            return "", "", -1
        candidates.sort(key=lambda item: (len(item[1]), -item[2]), reverse=True)
        return candidates[0]

    def _numbers_after_indicator(self, text: str) -> List[Tuple[str, str, float]]:
        text = STANDARD_CODE_PATTERN.sub(" ", text)
        values: List[Tuple[str, str, float]] = []
        for match in NUMBER_PATTERN.finditer(text):
            start, end = match.span()
            before = text[max(0, start - 5) : start]
            after = text[end : end + 8]
            number_text = match.group("number")
            if re.search(r"(年|月|日|:|：|-|—)$", before) or re.match(
                r"\s*(小时|h|H|次|天|个|页|版|级|类|年|月|日|工作日)",
                after,
            ):
                continue
            if re.search(r"(GB|HJ|T)\s*$", before, flags=re.IGNORECASE):
                continue
            try:
                number = float(number_text)
            except ValueError:
                continue
            if number > 100000:
                continue
            values.append((match.group("prefix") or "", number_text, number))
        return values

    def _extract_unit(self, text: str) -> str:
        match = UNIT_PATTERN.search(text)
        return match.group(0) if match else ""

    def _extract_average_time(self, text: str) -> str:
        if re.search(r"24\s*(?:h|H|小时|时)|日均|日平均|24小时平均", text):
            return "24h"
        if re.search(r"1\s*(?:h|H|小时|时)|小时平均", text):
            return "1h"
        if re.search(r"8\s*(?:h|H|小时|时)|8小时", text):
            return "8h"
        if "年平均" in text or "年均" in text:
            return "annual"
        return ""

    def _looks_like_header_or_footer(self, line: str) -> bool:
        return bool(re.search(r"第\s*\d+\s*页|页码|报告编号|声明|注意事项|地址|电话|网址", line))

    def _is_non_result_line(self, line: str) -> bool:
        if not line:
            return True
        if NON_RESULT_LINE_PATTERN.search(line):
            return True
        if STANDARD_CODE_PATTERN.search(line) and not RESULT_LINE_HINT_PATTERN.search(line):
            return True
        if re.search(r"^\d+[、.．]\s*", line) and not RESULT_LINE_HINT_PATTERN.search(line):
            return True
        if re.search(r"(pH|溶解氧|氨氮|化学需氧量|总磷|总氮|高锰酸盐).*?(的测定|测定|方法|法)", line, flags=re.IGNORECASE):
            return True
        return False

    def _clean_cell(self, value: str) -> str:
        value = value.replace("\t", " ")
        value = re.sub(r"\s{2,}", " ", value)
        return value.strip(" ：:;；,，")

    def _dedupe_records(self, records: Sequence[DetectionRecord]) -> List[DetectionRecord]:
        seen = set()
        result: List[DetectionRecord] = []
        for item in records:
            key = (
                item.normalized_indicator,
                item.value,
                item.unit,
                item.sample_time,
                item.sample_point,
                item.source_page,
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result


class StandardEvaluator:
    def __init__(self, standards: StandardsLibrary) -> None:
        self.standards = standards

    def evaluate_records(self, records: Sequence[DetectionRecord], standard_key: str) -> None:
        standard = self.standards.get(standard_key)
        standard_name = standard.get("name", standard_key)
        for record in records:
            self._evaluate_one(record, standard, standard_name)
        self._run_domain_consistency(records)

    def _evaluate_one(self, record: DetectionRecord, standard: Dict[str, Any], standard_name: str) -> None:
        indicator_name = record.normalized_indicator or canonical_indicator(record.indicator)
        rule = self._find_rule(standard, indicator_name)
        matched_indicator = self._find_rule_name(standard, indicator_name)
        record.standard_name = standard_name
        record.database_match = {
            "standard_database": str(self.standards.config_path),
            "standard_key_matched": bool(standard),
            "standard_name": standard_name,
            "indicator_input": record.indicator,
            "indicator_normalized": indicator_name,
            "indicator_matched": bool(rule),
            "matched_indicator": matched_indicator,
            "history_database_matched": False,
            "history_message": "历史记录/样本数据库未配置，点位、日期、报告编号仅做格式与一致性提示。",
        }
        if not rule:
            record.status = "unknown"
            record.needs_review = True
            record.database_match["reason"] = "数据库未匹配到对应依据：标准库未覆盖该检测项目或别名。"
            record.notes.append("数据库未匹配到对应依据：标准库未覆盖该指标")
            return
        if record.value is None:
            record.status = "unknown"
            record.needs_review = True
            record.database_match["reason"] = "检测值缺失或无法解析，无法进行限值比对。"
            return
        limit_value, limit_unit, mode, label = self._resolve_limit(rule, standard, record.average_time)
        record.database_match.update(
            {
                "indicator_matched": True,
                "aliases": rule.get("aliases", []),
                "expected_unit": normalize_unit(rule.get("unit", "")),
                "limit_mode": mode,
                "limit_label": label,
                "limit_value": limit_value,
                "limit_unit": limit_unit,
                "limit_source": "standards.json",
            }
        )
        if limit_value is None:
            record.status = "unknown"
            record.needs_review = True
            record.database_match["reason"] = "数据库未匹配到对应依据：标准库存在指标但缺少适用级别/时段限值。"
            record.notes.append("未找到匹配的标准限值")
            return
        converted = convert_unit(record.value, record.unit, limit_unit)
        if converted is None:
            record.status = "unknown"
            record.needs_review = True
            record.standard_limit = str(limit_value)
            record.limit_unit = limit_unit
            record.database_match["reason"] = f"单位不一致且无法自动换算：{record.unit or '未知单位'} -> {limit_unit}"
            record.notes.append(f"无法将 {record.unit or '未知单位'} 换算为 {limit_unit}")
            return
        record.database_match["converted_value"] = converted
        record.limit_unit = limit_unit
        record.standard_limit = self._format_limit(limit_value, mode, label)
        if mode == "max":
            ok = converted <= float(limit_value)
            record.comparison = f"{converted:g} <= {float(limit_value):g}"
        elif mode == "min":
            ok = converted >= float(limit_value)
            record.comparison = f"{converted:g} >= {float(limit_value):g}"
        elif mode == "range" and isinstance(limit_value, (list, tuple)) and len(limit_value) == 2:
            low, high = float(limit_value[0]), float(limit_value[1])
            ok = low <= converted <= high
            record.comparison = f"{low:g} <= {converted:g} <= {high:g}"
        else:
            record.status = "unknown"
            record.needs_review = True
            record.notes.append(f"暂不支持的限值模式：{mode}")
            return
        record.status = "pass" if ok else "fail"
        record.needs_review = record.needs_review or record.confidence < CONFIDENCE_REVIEW_THRESHOLD
        record.detection_conclusion = "通过" if ok else "不通过"
        record.database_match["final_status"] = record.status
        record.database_match["reason"] = "数据库限值匹配成功，已完成合规判断。"

    def _find_rule(self, standard: Dict[str, Any], indicator: str) -> Dict[str, Any]:
        indicators = standard.get("indicators", {})
        if indicator in indicators:
            return indicators[indicator]
        target = indicator.lower()
        for name, rule in indicators.items():
            aliases = [name] + rule.get("aliases", [])
            if any(target == alias.lower() for alias in aliases):
                return rule
        return {}

    def _find_rule_name(self, standard: Dict[str, Any], indicator: str) -> str:
        indicators = standard.get("indicators", {})
        if indicator in indicators:
            return indicator
        target = indicator.lower()
        for name, rule in indicators.items():
            aliases = [name] + rule.get("aliases", [])
            if any(target == alias.lower() for alias in aliases):
                return name
        return ""

    def _resolve_limit(
        self, rule: Dict[str, Any], standard: Dict[str, Any], average_time: str
    ) -> Tuple[Optional[Any], str, str, str]:
        mode = rule.get("mode", "max")
        unit = normalize_unit(rule.get("unit", ""))
        grade = standard.get("default_grade", "III")
        avg = average_time or standard.get("default_average_time", "")
        limits = rule.get("limits")
        if isinstance(limits, dict):
            bucket: Any = limits.get(avg) if avg else None
            if bucket is None and avg:
                bucket = limits.get(normalize_average_time(avg))
            if bucket is None and standard.get("default_average_time"):
                bucket = limits.get(standard.get("default_average_time"))
            if bucket is None and len(limits) == 1:
                bucket = next(iter(limits.values()))
            if isinstance(bucket, dict):
                for key in [grade, "II", "III", "一级", "二级", "default"]:
                    if key in bucket:
                        return bucket[key], unit, mode, f"{avg or standard.get('default_average_time', '')} {key}".strip()
            if isinstance(bucket, (int, float, list, tuple)):
                return bucket, unit, mode, avg

        if grade in rule:
            return rule[grade], unit or normalize_unit(rule.get("unit", "")), mode, grade
        for key in ["II", "III", "一级", "二级", "default"]:
            if key in rule:
                return rule[key], unit or normalize_unit(rule.get("unit", "")), mode, key
        return None, unit, mode, ""

    def _format_limit(self, value: Any, mode: str, label: str) -> str:
        prefix = {"max": "≤", "min": "≥", "range": ""}.get(mode, "")
        if isinstance(value, (list, tuple)) and len(value) == 2:
            formatted = f"{value[0]:g}~{value[1]:g}"
        elif isinstance(value, (int, float)):
            formatted = f"{prefix}{float(value):g}"
        else:
            formatted = f"{prefix}{value}"
        return f"{formatted} ({label})" if label else formatted

    def _run_domain_consistency(self, records: Sequence[DetectionRecord]) -> None:
        by_sample: Dict[Tuple[str, str], Dict[str, DetectionRecord]] = {}
        for record in records:
            key = (record.sample_point, record.sample_time)
            by_sample.setdefault(key, {})[record.normalized_indicator] = record
        for group in by_sample.values():
            self._flag_if_greater(group, "BOD5", "COD", "BOD5 通常不应高于 COD，建议复核是否跨行错位或单位错误。")
            self._flag_if_greater(group, "总磷", "总磷酸盐", "总磷高于总磷酸盐，建议复核计量口径、单位或跨行识别。")

    def _flag_if_greater(self, group: Dict[str, DetectionRecord], left: str, right: str, message: str) -> None:
        a = group.get(left)
        b = group.get(right)
        if not a or not b or a.value is None or b.value is None:
            return
        b_value = convert_unit(b.value, b.unit, a.unit)
        if b_value is None:
            return
        if a.value > b_value:
            for record in (a, b):
                record.needs_review = True
                if message not in record.notes:
                    record.notes.append(message)


class DetectionRecordQualityFilter:
    def __init__(self) -> None:
        self.table_extractor = TableAwareExtractor()

    def filter(self, records: Sequence[DetectionRecord]) -> Tuple[List[DetectionRecord], List[str]]:
        kept: List[DetectionRecord] = []
        dropped = 0
        for record in records:
            reason = self._reject_reason(record)
            if reason:
                dropped += 1
                logger.debug("Dropped detection record: %s | %s", reason, record.to_dict())
                continue
            kept.append(record)
        warnings: List[str] = []
        if dropped:
            warnings.append(f"已过滤 {dropped} 条疑似非检测结果记录（方法标准号、合同说明、频次或不合理数值）。")
        return kept, warnings

    def _reject_reason(self, record: DetectionRecord) -> str:
        indicator = record.normalized_indicator or canonical_indicator(record.indicator)
        line = record.source_line or ""
        value = record.value
        if line and self.table_extractor._is_non_result_line(line):
            return "non-result-line"
        if value is None:
            return "missing-value"
        if indicator == "pH" and not (0 <= value <= 14):
            return "ph-out-of-range"
        if indicator == "溶解氧" and not (0 <= value <= 30):
            return "do-out-of-range"
        if indicator in {"COD", "BOD5", "氨氮", "总磷", "总氮", "高锰酸盐指数"} and value > 10000:
            return "water-value-out-of-range"
        if 1900 <= value <= 2100 and (STANDARD_CODE_PATTERN.search(line) or indicator == "pH"):
            return "standard-year"
        if indicator != "pH" and not normalize_unit(record.unit):
            if not RESULT_LINE_HINT_PATTERN.search(line):
                return "missing-unit-without-result-context"
        if len(line) <= 4 and not normalize_unit(record.unit):
            return "too-short"
        return ""


class EnvironmentReportVisionPipeline:
    def __init__(self, standards: StandardsLibrary, output_root: Path, ai_client: Optional[AIClient] = None) -> None:
        self.standards = standards
        self.output_root = output_root
        self.ai_client = ai_client or AIClient()
        self.extractor = TableAwareExtractor()
        self.evaluator = StandardEvaluator(standards)
        formula_path = standards.config_path.parent / "calculation_formulas.json"
        self.formula_verifier = FormulaVerifier(FormulaDatabase(formula_path)) if formula_path.exists() else None

    def analyze_files(self, file_paths: Sequence[str], standard_key: str, use_ai: bool = True) -> StructuredReportResult:
        batch_dir = self._create_batch_dir(file_paths)
        preprocessor = ScanPreprocessor(batch_dir)
        bundles, warnings, enhanced_pdf = preprocessor.process_files(file_paths)
        force_local_ocr = self._env_flag("OCR_FORCE_LOCAL", False)
        warnings.extend(self._run_optional_ocr(bundles, force=force_local_ocr))

        text = self._combine_text(bundles)
        sampling, records, extraction_warnings = self.extractor.extract(text)
        warnings.extend(extraction_warnings)
        processing_trace: Dict[str, Any] = {
            "route": "local_rules_only",
            "api": {
                "enabled_requested": use_ai,
                "configured": self.ai_client.is_enabled,
                "provider": self.ai_client.provider,
                "base_url": self.ai_client.base_url,
                "model": self.ai_client.model,
                "vision_expected": self.ai_client.likely_supports_vision(),
            },
            "ocr": {
                "total_pages": len(bundles),
                "pages_with_text": sum(1 for item in bundles if item.text),
                "max_pages": self._ocr_max_pages(total_pages=len(bundles)) if bundles else 0,
                "force_local_ocr": force_local_ocr,
                "text_length": len(text),
            },
            "ai": {"attempts": []},
            "database": {
                "standard_key": standard_key,
                "standard_name": self.standards.names().get(standard_key, standard_key),
                "standard_database": str(self.standards.config_path),
                "formula_database": str(self.standards.config_path.parent / "calculation_formulas.json"),
                "history_database": "未配置",
            },
        }

        if use_ai and self.ai_client.is_enabled and bundles:
            if not self.ai_client.likely_supports_vision() and not text.strip():
                warnings.append("当前模型未标记为视觉模型，已强制使用本地 OCR 生成文本后再交给 AI 抽取。")
                warnings.extend(self._run_optional_ocr(bundles, force=True))
                text = self._combine_text(bundles)
            ai_sampling, ai_records, ai_warnings, ai_attempts = self._extract_with_ai(
                [item.processed for item in bundles],
                text,
                standard_key,
            )
            warnings.extend(ai_warnings)
            processing_trace["ai"]["attempts"].extend(ai_attempts)
            if ai_attempts:
                processing_trace["route"] = ai_attempts[-1].get("mode", "ai")
            sampling = self._merge_sampling(sampling, ai_sampling)
            records = self._merge_records(records, ai_records)
        elif use_ai:
            warnings.append("AI 视觉抽取未启用：请配置 AI_API_KEY / AI_BASE_URL / AI_MODEL 后重试。")
            processing_trace["api"]["error"] = "AI API 未启用"

        quality_filter = DetectionRecordQualityFilter()
        records, quality_warnings = quality_filter.filter(records)
        warnings.extend(quality_warnings)
        if not records and text.strip():
            fallback_sampling = sampling[0] if sampling else None
            fallback_records = self.extractor.fallback_extract_surface_water_records(text, fallback_sampling)
            if fallback_records:
                fallback_records, fallback_quality_warnings = quality_filter.filter(fallback_records)
                warnings.extend(fallback_quality_warnings)
                if fallback_records:
                    records = fallback_records
                    warnings.append(f"结构化结果经质量过滤后为空，已启用 OCR 兜底抽取保留 {len(records)} 条低置信度记录。")
        needs_ocr_retry = not records or (len(records) < 3 and len(bundles) > 5)
        if needs_ocr_retry and not force_local_ocr and bundles and self._env_flag("OCR_FALLBACK_ON_ZERO_RECORDS", True):
            reason = "未形成检测记录" if not records else f"仅形成 {len(records)} 条检测记录"
            warnings.append(f"嵌入文本{reason}，已强制使用本地 OCR 重新识别增强页。")
            warnings.extend(self._run_optional_ocr(bundles, force=True))
            text = self._combine_text(bundles)
            fallback_sampling, fallback_records, fallback_warnings = self.extractor.extract(text)
            warnings.extend(fallback_warnings)
            if fallback_sampling:
                sampling = fallback_sampling
            records, quality_warnings = quality_filter.filter(fallback_records)
            warnings.extend(quality_warnings)
        self.evaluator.evaluate_records(records, standard_key)
        if self.formula_verifier:
            warnings.extend(self.formula_verifier.annotate_records(records))
        document_metadata = self._extract_report_metadata(text)
        for record in records:
            record.sample_id = record.sample_id or record.sample_point
            record.report_no = record.report_no or document_metadata.get("report_no", "")
            record.detection_date = record.detection_date or record.sample_time
            record.database_match.setdefault("formula_matched", bool(record.formula_verification))
            record.database_match.setdefault("history_database_matched", False)
        standard_name = self.standards.names().get(standard_key, standard_key)
        processing_trace["ocr"].update(
            {
                "pages_with_text": sum(1 for item in bundles if item.text),
                "text_length": len(text),
                "text_preview": safe_preview(text, limit=4000),
            }
        )
        processing_trace["database"].update(
            {
                "records_checked": len(records),
                "standard_matches": sum(1 for item in records if item.database_match.get("indicator_matched")),
                "formula_matches": sum(1 for item in records if item.database_match.get("formula_matched")),
                "history_message": "历史记录/样本数据库未配置；如需点位、日期、报告编号历史合理性校验，请接入历史样本表。",
            }
        )
        processing_trace["document_metadata"] = document_metadata
        if records:
            processing_trace["database"]["record_match_preview"] = [
                {
                    "indicator": item.normalized_indicator or item.indicator,
                    "sample_id": item.sample_id or item.sample_point,
                    "status": item.status,
                    "database_match": item.database_match,
                }
                for item in records[:20]
            ]
        result = StructuredReportResult(
            input_files=[str(Path(path).resolve()) for path in file_paths],
            standard_key=standard_key,
            standard_name=standard_name,
            logical_report_id=batch_dir.name,
            enhanced_pdf_path=enhanced_pdf,
            pages=[item.processed for item in bundles],
            sampling=sampling,
            records=records,
            warnings=list(dict.fromkeys(warnings + [warning for item in bundles for warning in item.processed.warnings])),
            raw_text_preview=safe_preview(text, limit=20000),
            visualization_suggestions=self._visualization_suggestions(records),
            processing_trace=processing_trace,
        )
        result.brief_summary = self._build_summary(result)
        return result

    def _run_optional_ocr(self, bundles: Sequence[PageBundle], force: bool = False) -> List[str]:
        warnings: List[str] = []
        if not force and all(bundle.text for bundle in bundles):
            return warnings
        ocr = OptionalLocalOCR()
        if not ocr.available:
            return [ocr.warning]
        max_pages = self._ocr_max_pages(total_pages=len(bundles))
        candidates = self._select_ocr_bundles(bundles, force=force, max_pages=max_pages)
        if force and len(bundles) > len(candidates):
            selected_pages = ", ".join(str(bundle.processed.page_index) for bundle in candidates[:12])
            suffix = "..." if len(candidates) > 12 else ""
            warnings.append(f"OCR 已按检测结果关键词优先选择 {len(candidates)} 页扫描图（页码：{selected_pages}{suffix}）。")
        ocr_count = 0
        for bundle in candidates:
            text, warning = ocr.extract(bundle.processed.enhanced_image_path)
            if text:
                bundle.text = text
                ocr_count += 1
            if warning:
                warnings.append(warning)
        skipped = sum(1 for bundle in bundles if not bundle.text)
        if ocr_count:
            engine = ocr.last_engine or "本地 OCR"
            mode = "强制" if force else ""
            warnings.append(f"已{mode}使用 {engine} 对 {ocr_count} 页扫描图进行文字读取。")
        if skipped:
            warnings.append(f"本地 OCR 已处理 {ocr_count} 页，仍有 {skipped} 页未提取文本；可设置 OCR_MAX_PAGES 调大页数上限。")
        return warnings

    def _select_ocr_bundles(
        self, bundles: Sequence[PageBundle], force: bool, max_pages: int
    ) -> List[PageBundle]:
        candidates = [bundle for bundle in bundles if force or not bundle.text]
        if not candidates:
            return []
        if not force:
            return candidates[:max_pages]
        scored = [(self._ocr_priority_score(bundle), index, bundle) for index, bundle in enumerate(candidates)]
        if not any(score for score, _, _ in scored):
            return candidates[:max_pages]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [bundle for _, _, bundle in scored[:max_pages]]

    def _ocr_priority_score(self, bundle: PageBundle) -> int:
        text = bundle.text or ""
        compact = text.replace(" ", "")
        score = 0
        for keyword in OCR_PRIORITY_KEYWORDS:
            if keyword in compact:
                score += 50
        if re.search(r"\b(?:HS|WS|S)\s*\d{1,3}(?:[-—–]\d{1,3}){0,3}\b", text, re.IGNORECASE):
            score += 20
        for aliases in INDICATOR_ALIASES.values():
            if any(re.search(re.escape(alias), compact, re.IGNORECASE) for alias in aliases):
                score += 10
        if UNIT_PATTERN.search(text) or re.search(r"(无量纲|NTU|μS/cm|µS/cm|uS/cm|mS/cm)", text, re.IGNORECASE):
            score += 5
        return score

    def _env_flag(self, name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() not in {"0", "false", "no", "off", ""}

    def _ocr_max_pages(self, total_pages: int = 60) -> int:
        raw_value = os.getenv("OCR_MAX_PAGES", "all").strip().lower()
        if raw_value in {"", "0", "all", "max", "full", "全部", "全量"}:
            return max(1, total_pages)
        try:
            return max(1, int(raw_value))
        except ValueError:
            return max(1, total_pages)

    def _combine_text(self, bundles: Sequence[PageBundle]) -> str:
        chunks = []
        for bundle in bundles:
            if bundle.text:
                chunks.append(f"[page:{bundle.processed.page_index}]\n{bundle.text}")
        return "\n".join(chunks)

    def _extract_report_metadata(self, text: str) -> Dict[str, str]:
        metadata: Dict[str, str] = {}
        report_match = re.search(
            r"(?:报告编号|报告号|报告代码|编号)\s*[:：]?\s*([A-Za-z0-9\u4e00-\u9fff][A-Za-z0-9_\-—/（）()第号字\u4e00-\u9fff]{3,60})",
            text,
        )
        if report_match:
            metadata["report_no"] = self.extractor._clean_cell(report_match.group(1))
        date_match = re.search(r"(20\d{2})\s*[年/\-.]\s*(\d{1,2})\s*[月/\-.]\s*(\d{1,2})\s*日?", text)
        if date_match:
            metadata["detected_date"] = f"{int(date_match.group(1)):04d}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"
        if not metadata:
            metadata["message"] = "未从文本中稳定提取报告编号或报告日期。"
        return metadata

    def _extract_with_ai(
        self, pages: Sequence[ProcessedPage], text: str, standard_key: str
    ) -> Tuple[List[SamplingInfo], List[DetectionRecord], List[str], List[Dict[str, Any]]]:
        warnings: List[str] = []
        attempts: List[Dict[str, Any]] = []
        standard_name = self.standards.names().get(standard_key, standard_key)
        if self.ai_client.likely_supports_vision():
            try:
                payload = self._call_vision_model(pages, standard_name)
                sampling, records = self._parse_ai_payload(payload, source_note="AI 视觉抽取")
                attempts.append({"mode": "ai_vision", "status": "success", "records": len(records)})
                return sampling, records, warnings, attempts
            except Exception as exc:
                logger.exception("AI vision extraction failed")
                attempts.append({"mode": "ai_vision", "status": "failed", "error": str(exc)})
                warnings.append(f"AI 视觉抽取失败，已自动降级 OCR 文本抽取：{exc}")
        else:
            attempts.append({"mode": "ai_vision", "status": "skipped", "reason": "模型未标记为视觉模型"})
            warnings.append("当前模型未标记为视觉模型，已自动降级为 OCR 文本 + AI 结构化抽取。")

        if not text.strip():
            warnings.append("OCR 文本为空，AI 文本抽取无法执行；已保留本地规则结果。")
            attempts.append({"mode": "ocr_text_ai", "status": "skipped", "reason": "OCR 文本为空"})
            return [], [], warnings, attempts
        try:
            payload = self._call_text_model(text, standard_name)
            sampling, records = self._parse_ai_payload(payload, source_note="AI OCR文本抽取")
            attempts.append({"mode": "ocr_text_ai", "status": "success", "records": len(records)})
            return sampling, records, warnings, attempts
        except Exception as exc:
            logger.exception("AI text extraction failed")
            attempts.append({"mode": "ocr_text_ai", "status": "failed", "error": str(exc)})
            return [], [], warnings + [f"AI OCR 文本抽取失败，已保留本地规则结果：{exc}"], attempts

    def _parse_ai_payload(self, payload: Dict[str, Any], source_note: str) -> Tuple[List[SamplingInfo], List[DetectionRecord]]:
        sampling = [
            SamplingInfo(
                sample_time=str(item.get("sample_time", "")),
                sample_point=str(item.get("sample_point", "")),
                frequency=str(item.get("frequency", "")),
                confidence=float(item.get("confidence", 0.9) or 0.9),
                source_line=str(item.get("source_line", "")),
            )
            for item in payload.get("sampling", [])
            if isinstance(item, dict)
        ]
        records: List[DetectionRecord] = []
        default_sampling = sampling[0] if sampling else SamplingInfo()
        for item in payload.get("records", []):
            if not isinstance(item, dict):
                continue
            indicator = str(item.get("indicator", "")).strip()
            value = parse_float(item.get("value"))
            normalized = canonical_indicator(indicator)
            confidence = float(item.get("confidence", 0.9) or 0.9)
            records.append(
                DetectionRecord(
                    indicator=indicator,
                    normalized_indicator=normalized or indicator,
                    value=value,
                    unit=normalize_unit(str(item.get("unit", ""))),
                    sample_id=str(item.get("sample_id", "")),
                    report_no=str(item.get("report_no", "")),
                    detection_date=str(item.get("detection_date", "")),
                    sample_time=str(item.get("sample_time", default_sampling.sample_time)),
                    sample_point=str(item.get("sample_point", default_sampling.sample_point)),
                    frequency=str(item.get("frequency", default_sampling.frequency)),
                    standard_limit=str(item.get("standard_limit", "")),
                    confidence=round(max(0.0, min(0.99, confidence)), 3),
                    needs_review=confidence < CONFIDENCE_REVIEW_THRESHOLD,
                    source_page=int(item.get("source_page", 0) or 0),
                    source_line=str(item.get("source_line", "")),
                    average_time=normalize_average_time(str(item.get("average_time", ""))),
                    raw_value=str(item.get("value", "")),
                    detection_conclusion=str(item.get("detection_conclusion", "")),
                    notes=[source_note] if confidence >= CONFIDENCE_REVIEW_THRESHOLD else [f"{source_note}，低置信度需复核"],
                )
            )
        return sampling, records

    def _call_vision_model(self, pages: Sequence[ProcessedPage], standard_name: str) -> Dict[str, Any]:
        prompt = {
            "role": "环境检测报告智能识别与结构化系统",
            "task": "从增强后的报告图片中提取结构化数据。只返回 JSON，不要解释。",
            "requirements": [
                "只提取检测表格里的采样信息、检测项目、检测值、单位、标准限值或评价。",
                "保持数值与单位严格同一行/同一表格单元关联，无法确认时 confidence 低于 0.95。",
                "过滤页眉、页脚、声明、地址、电话、签章文字。",
                "必须尽量提取报告编号、样品编号、检测日期、检测点位、标准名称和检测结论；无法确认则留空并降低 confidence。",
                "字段 records[].confidence 范围 0~1；低于 0.95 的记录会进入人工复核。",
                f"优先按标准库 {standard_name} 的指标名称归一化。",
            ],
            "schema": {
                "sampling": [
                    {
                        "sample_time": "string",
                        "sample_point": "string",
                        "frequency": "string",
                        "confidence": 0.0,
                        "source_line": "string",
                    }
                ],
                "records": [
                    {
                        "indicator": "PM2.5/COD/氨氮等",
                        "value": 0.0,
                        "unit": "mg/L 或 μg/m³",
                        "sample_id": "string",
                        "report_no": "string",
                        "detection_date": "string",
                        "average_time": "24h/1h/8h/annual/空",
                        "sample_time": "string",
                        "sample_point": "string",
                        "frequency": "string",
                        "standard_name": "string",
                        "standard_limit": "string",
                        "detection_conclusion": "string",
                        "source_page": 1,
                        "source_line": "string",
                        "confidence": 0.0,
                    }
                ],
            },
        }
        content: List[Dict[str, Any]] = [{"type": "text", "text": json.dumps(prompt, ensure_ascii=False)}]
        for page in list(pages)[:12]:
            with open(page.enhanced_image_path, "rb") as fh:
                encoded = base64.b64encode(fh.read()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )
        response_text = self.ai_client.chat_with_content(content, temperature=0.0, response_format={"type": "json_object"})
        return self._parse_json_object(response_text)

    def _call_text_model(self, text: str, standard_name: str) -> Dict[str, Any]:
        prompt = {
            "role": "环境检测报告 OCR 文本结构化抽取与核验助手",
            "task": "根据 OCR/嵌入文本提取环境检测报告核心数据。只返回 JSON，不要解释。",
            "requirements": [
                "输入文本来自 PDF OCR，可能存在错字、断行和跨列表格；必须保持检测项目、样品编号、点位、检测值、单位同一行或同一表格区域关联。",
                "抽取字段包括：检测项目、检测点位、检测值、单位、检测日期、样品编号、报告编号、标准名称、标准限值、检测结论。",
                "不要把分析方法标准号、合同说明、频次说明、质控样、空白样当作检测结果。",
                "模糊数字、单位不确定、跨行表格无法确认时 confidence 必须低于 0.95。",
                f"执行标准优先参考：{standard_name}；标准限值如文本中缺失可留空，后端会用标准库二次匹配。",
            ],
            "schema": {
                "sampling": [
                    {
                        "sample_time": "string",
                        "sample_point": "string",
                        "frequency": "string",
                        "confidence": 0.0,
                        "source_line": "string",
                    }
                ],
                "records": [
                    {
                        "indicator": "string",
                        "value": 0.0,
                        "unit": "string",
                        "sample_id": "string",
                        "report_no": "string",
                        "detection_date": "string",
                        "sample_time": "string",
                        "sample_point": "string",
                        "frequency": "string",
                        "standard_name": "string",
                        "standard_limit": "string",
                        "detection_conclusion": "string",
                        "average_time": "string",
                        "source_page": 1,
                        "source_line": "string",
                        "confidence": 0.0,
                    }
                ],
            },
            "ocr_text": safe_preview(text, limit=120000),
        }
        content: List[Dict[str, Any]] = [{"type": "text", "text": json.dumps(prompt, ensure_ascii=False)}]
        response_text = self.ai_client.chat_with_content(content, temperature=0.0, response_format={"type": "json_object"})
        return self._parse_json_object(response_text)

    def _parse_json_object(self, text: str) -> Dict[str, Any]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise
            data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("AI 返回内容不是 JSON 对象")
        return data

    def _merge_sampling(self, local: Sequence[SamplingInfo], ai: Sequence[SamplingInfo]) -> List[SamplingInfo]:
        merged = list(ai) + [item for item in local if item.to_dict() not in [x.to_dict() for x in ai]]
        return merged[:8]

    def _merge_records(self, local: Sequence[DetectionRecord], ai: Sequence[DetectionRecord]) -> List[DetectionRecord]:
        result: List[DetectionRecord] = []
        seen = set()
        for item in list(ai) + list(local):
            key = (
                item.normalized_indicator or item.indicator,
                item.value,
                item.unit,
                item.sample_time,
                item.sample_point,
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def _create_batch_dir(self, file_paths: Sequence[str]) -> Path:
        digest = hashlib.sha1("::".join(sorted(file_paths)).encode("utf-8")).hexdigest()[:12]
        batch_dir = self.output_root / f"vision_batch_{digest}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        return batch_dir

    def _build_summary(self, result: StructuredReportResult) -> str:
        judged = result.pass_count + result.fail_count
        rate = f"{result.compliance_rate * 100:.1f}%" if judged else "暂无可判定数据"
        pollutants = "、".join(result.exceeded_pollutants) if result.exceeded_pollutants else "未发现"
        review = result.review_count
        suggestions = []
        if result.fail_count:
            suggestions.append("优先复测超标项目，并核对样点、采样时段、单位换算和执行标准。")
        if review:
            suggestions.append("低置信度记录需人工复核原图，重点查看模糊数字、跨行表格和红章覆盖区域。")
        if not result.records:
            suggestions.append("当前未形成检测记录，建议使用更清晰照片、启用 AI 视觉或安装本地 OCR。")
        if not suggestions:
            suggestions.append("本批次未发现超标记录，可继续按日期或样点做趋势对比。")
        return (
            f"合格率：{rate}；主要超标污染物：{pollutants}；"
            f"共识别 {len(result.records)} 条检测记录，其中 {review} 条需人工复核。"
            f"检测建议：{' '.join(suggestions)}"
        )

    def _visualization_suggestions(self, records: Sequence[DetectionRecord]) -> List[str]:
        return [
            "按样点 + 指标展示达标仪表盘，绿色为合格、红色为超标、黄色为低置信度待复核。",
            "同一指标跨日期展示趋势折线图，并叠加标准限值水平线。",
            "对超标项目展示污染物贡献排行，支持点击回看原始增强页和 source_line。",
            "批量报告页面使用复核队列，只展示 confidence < 0.95、单位缺失或逻辑校验异常的数据行。",
        ]


def canonical_indicator(name: str) -> str:
    target = name.replace(" ", "").lower()
    for canonical, aliases in INDICATOR_ALIASES.items():
        if target == canonical.replace(" ", "").lower():
            return canonical
        if any(target == alias.replace(" ", "").lower() for alias in aliases):
            return canonical
    return name


def normalize_unit(unit: str) -> str:
    unit = (unit or "").strip()
    unit = unit.replace("µ", "μ").replace("ug", "μg").replace("UG", "μg")
    unit = unit.replace("µS", "μS").replace("uS", "μS").replace("US", "μS")
    unit = unit.replace("m3", "m³").replace("M3", "m³")
    unit = re.sub(r"\s+", "", unit)
    replacements = {
        "mg/m³": "mg/m³",
        "μg/m³": "μg/m³",
        "mg/L": "mg/L",
        "μg/L": "μg/L",
        "mg/l": "mg/L",
        "μg/l": "μg/L",
        "μS/cm": "μS/cm",
        "mS/cm": "mS/cm",
    }
    return replacements.get(unit, unit)


def normalize_average_time(value: str) -> str:
    text = value.strip().lower()
    if not text:
        return ""
    if "24" in text or "日" in text:
        return "24h"
    if "8" in text:
        return "8h"
    if "1" in text or "小时" in text:
        return "1h"
    if "annual" in text or "年" in text:
        return "annual"
    return value


def convert_unit(value: float, from_unit: str, to_unit: str) -> Optional[float]:
    from_unit = normalize_unit(from_unit)
    to_unit = normalize_unit(to_unit)
    if not to_unit or from_unit == to_unit:
        return value
    if not from_unit:
        return value if to_unit == "无量纲" else None
    factors = {
        ("mg/m³", "μg/m³"): 1000.0,
        ("μg/m³", "mg/m³"): 0.001,
        ("mg/L", "μg/L"): 1000.0,
        ("μg/L", "mg/L"): 0.001,
    }
    factor = factors.get((from_unit, to_unit))
    if factor is None:
        return None
    return value * factor


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(float(value)):
            return None
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0))
