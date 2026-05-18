from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.ai_client import AIClient
from src.core.exporter import export_json, export_structured_excel, export_structured_html
from src.core.standards import StandardsLibrary
from src.core.vision_pipeline import EnvironmentReportVisionPipeline


APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config" / "standards.json"
EXPORT_DIR = APP_DIR / "simple_test_exports"


class AnalyzeWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, files: List[str], standard_key: str, use_ai: bool, max_pages: int) -> None:
        super().__init__()
        self.files = files
        self.standard_key = standard_key
        self.use_ai = use_ai
        self.max_pages = max_pages

    def run(self) -> None:
        try:
            os.environ["OCR_MAX_PAGES"] = str(self.max_pages)
            standards = StandardsLibrary(CONFIG_PATH)
            pipeline = EnvironmentReportVisionPipeline(standards, EXPORT_DIR / "vision", AIClient())
            result = pipeline.analyze_files(self.files, self.standard_key, self.use_ai)
            data = result.to_dict()
            stem = data.get("logical_report_id") or "simple_test"
            export_json(data, EXPORT_DIR / f"{stem}_structured.json")
            export_structured_html(data, EXPORT_DIR / f"{stem}_structured.html")
            export_structured_excel(data, EXPORT_DIR / f"{stem}_structured.xlsx")
            self.finished.emit(data)
        except Exception:
            self.failed.emit(traceback.format_exc())


class SimpleTester(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("环境检测报告 OCR 结构化测试工具")
        self.resize(1320, 860)
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.standards = StandardsLibrary(CONFIG_PATH)
        self.files: List[str] = []
        self.current_data: Dict[str, Any] = {}
        self.worker_thread: QThread | None = None
        self.worker: AnalyzeWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

        form_box = QGroupBox("测试输入")
        form = QFormLayout(form_box)

        file_row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        choose_btn = QPushButton("选择 PDF/图片")
        choose_btn.clicked.connect(self.choose_files)
        file_row.addWidget(self.file_edit)
        file_row.addWidget(choose_btn)
        file_widget = QWidget()
        file_widget.setLayout(file_row)
        form.addRow("报告文件", file_widget)

        self.standard_combo = QComboBox()
        for key, name in self.standards.names().items():
            self.standard_combo.addItem(name, key)
        form.addRow("执行标准", self.standard_combo)

        self.max_pages = QSpinBox()
        self.max_pages.setRange(1, 300)
        self.max_pages.setValue(int(os.getenv("OCR_MAX_PAGES", "60")))
        form.addRow("OCR 页数上限", self.max_pages)

        self.ai_checkbox = QCheckBox("启用 AI 视觉抽取")
        self.ai_checkbox.setChecked(False)
        form.addRow("AI", self.ai_checkbox)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("开始识别")
        self.run_btn.clicked.connect(self.run_analysis)
        self.export_dir_btn = QPushButton("导出目录")
        self.export_dir_btn.clicked.connect(self.show_export_dir)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.export_dir_btn)
        run_row.addStretch()
        run_widget = QWidget()
        run_widget.setLayout(run_row)
        form.addRow("", run_widget)
        layout.addWidget(form_box)

        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)

        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(110)
        layout.addWidget(self.summary_text)

        tabs = QTabWidget()
        self.record_table = QTableWidget(0, 10)
        self.record_table.setHorizontalHeaderLabels(["样点", "指标", "检测值", "单位", "限值", "状态", "公式核验", "置信度", "来源页", "来源行"])
        self.record_table.horizontalHeader().setStretchLastSection(True)
        tabs.addTab(self.record_table, "结构化结果")

        self.candidate_text = QPlainTextEdit()
        self.candidate_text.setReadOnly(True)
        tabs.addTab(self.candidate_text, "候选行")

        self.raw_text = QPlainTextEdit()
        self.raw_text.setReadOnly(True)
        tabs.addTab(self.raw_text, "OCR 原文")

        self.json_text = QPlainTextEdit()
        self.json_text.setReadOnly(True)
        tabs.addTab(self.json_text, "JSON")

        self.warning_text = QPlainTextEdit()
        self.warning_text.setReadOnly(True)
        tabs.addTab(self.warning_text, "提示/警告")

        splitter = QSplitter(Qt.Vertical)
        tabs_holder = QWidget()
        tabs_layout = QVBoxLayout(tabs_holder)
        tabs_layout.setContentsMargins(0, 0, 0, 0)
        tabs_layout.addWidget(tabs)
        splitter.addWidget(tabs_holder)
        splitter.setSizes([640])
        layout.addWidget(splitter)

        self.setCentralWidget(root)

    def choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择报告 PDF 或图片",
            "",
            "Report Files (*.pdf *.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)",
        )
        if not files:
            return
        self.files = files
        self.file_edit.setText("; ".join(files))
        lower = " ".join(files).lower()
        if "废水" in lower or "污水" in lower:
            self._select_standard("wastewater_custom")
        elif "地下水" in lower:
            self._select_standard("groundwater_custom")
        elif "地表水" in lower:
            self._select_standard("surface_water_gb3838_subset")
        elif "空气" in lower or "废气" in lower:
            self._select_standard("air_gb3095_2012_grade2")

    def _select_standard(self, key: str) -> None:
        idx = self.standard_combo.findData(key)
        if idx >= 0:
            self.standard_combo.setCurrentIndex(idx)

    def run_analysis(self) -> None:
        if not self.files:
            QMessageBox.information(self, "提示", "请先选择 PDF 或图片文件。")
            return
        self.run_btn.setEnabled(False)
        self.status_label.setText("识别中：正在预处理、PaddleOCR 读取和结构化抽取...")
        self.summary_text.clear()
        self.record_table.setRowCount(0)
        self.candidate_text.clear()
        self.raw_text.clear()
        self.json_text.clear()
        self.warning_text.clear()

        self.worker_thread = QThread()
        self.worker = AnalyzeWorker(
            self.files,
            self.standard_combo.currentData(),
            self.ai_checkbox.isChecked(),
            self.max_pages.value(),
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._analysis_finished)
        self.worker.failed.connect(self._analysis_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def _analysis_finished(self, data: Dict[str, Any]) -> None:
        self.current_data = data
        self.run_btn.setEnabled(True)
        stats = data.get("statistics", {})
        judged = (stats.get("pass_count", 0) or 0) + (stats.get("fail_count", 0) or 0)
        rate = f"{stats.get('compliance_rate', 0) * 100:.1f}%" if judged else "暂无可判定数据"
        self.status_label.setText("完成")
        self.summary_text.setPlainText(
            f"合格率：{rate}\n"
            f"记录数：{stats.get('total_records', 0)}，超标：{stats.get('fail_count', 0)}，复核：{stats.get('review_count', 0)}\n"
            f"{data.get('brief_summary', '')}\n"
            f"增强 PDF：{data.get('enhanced_pdf_path', '')}\n"
            f"导出目录：{EXPORT_DIR}"
        )
        self._populate_records(data.get("records", []))
        raw = data.get("raw_text_preview", "")
        self.raw_text.setPlainText(raw)
        self.candidate_text.setPlainText(self._build_candidates(raw))
        self.warning_text.setPlainText("\n".join(data.get("warnings", [])))
        self.json_text.setPlainText(json.dumps(data, ensure_ascii=False, indent=2))

    def _analysis_failed(self, error: str) -> None:
        self.run_btn.setEnabled(True)
        self.status_label.setText("失败")
        self.warning_text.setPlainText(error)
        QMessageBox.critical(self, "识别失败", error[:2000])

    def _populate_records(self, records: List[Dict[str, Any]]) -> None:
        self.record_table.setRowCount(len(records))
        for row, item in enumerate(records):
            values = [
                item.get("sample_point", ""),
                item.get("normalized_indicator") or item.get("indicator", ""),
                item.get("raw_value") or item.get("value", ""),
                item.get("unit", ""),
                f"{item.get('standard_limit', '')} {item.get('limit_unit', '')}".strip(),
                ("复核/" if item.get("needs_review") else "") + item.get("status", ""),
                (item.get("formula_verification") or {}).get("status", ""),
                item.get("confidence", ""),
                item.get("source_page", ""),
                item.get("source_line", ""),
            ]
            for col, value in enumerate(values):
                self.record_table.setItem(row, col, QTableWidgetItem(str(value)))
        self.record_table.resizeColumnsToContents()

    def _build_candidates(self, raw: str) -> str:
        keywords = ["检测结果", "监测结果", "结果一览表", "样品编号", "WS", "HS", "COD", "化学需氧量", "氨氮", "总磷", "总氮", "mg/L"]
        lines = [line for line in raw.splitlines() if any(key in line for key in keywords)]
        return "\n".join(lines[:500])

    def show_export_dir(self) -> None:
        QMessageBox.information(self, "导出目录", str(EXPORT_DIR))


def main() -> int:
    app = QApplication([])
    win = SimpleTester()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
