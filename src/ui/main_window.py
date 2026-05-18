from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
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
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.ai_client import AIClient
from src.core.exporter import (
    export_json,
    export_report_html,
    export_sensor_html,
    export_structured_excel,
    export_structured_html,
)
from src.core.report_analyzer import ReportAnalyzer
from src.core.sensor_analyzer import SensorAnalyzer
from src.core.standards import StandardsLibrary
from src.core.vision_pipeline import EnvironmentReportVisionPipeline


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(5, 3), tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)

    def plot_series(self, xs, ys, title: str) -> None:
        self.ax.clear()
        self.ax.plot(xs, ys, marker="o")
        self.ax.set_title(title)
        self.ax.tick_params(axis="x", rotation=20)
        self.draw_idle()


class ReportTab(QWidget):
    def __init__(self, standards: StandardsLibrary, exports_dir: Path):
        super().__init__()
        self.standards = standards
        self.exports_dir = exports_dir
        self.analyzer = ReportAnalyzer(standards, AIClient())
        self.current_result = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form_box = QGroupBox("报告核验")
        form = QFormLayout(form_box)

        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        select_btn = QPushButton("选择 PDF")
        select_btn.clicked.connect(self.choose_file)
        row = QHBoxLayout()
        row.addWidget(self.file_edit)
        row.addWidget(select_btn)
        row_w = QWidget()
        row_w.setLayout(row)
        form.addRow("报告文件", row_w)

        self.standard_combo = QComboBox()
        for key, name in self.standards.names().items():
            self.standard_combo.addItem(name, key)
        form.addRow("核验标准", self.standard_combo)

        self.ai_checkbox = QCheckBox("启用 AI 总结（需配置 AI_API_KEY / AI_BASE_URL / AI_MODEL）")
        self.ai_checkbox.setChecked(True)
        form.addRow("AI", self.ai_checkbox)

        run_btn = QPushButton("开始核验")
        run_btn.clicked.connect(self.run_analysis)
        form.addRow("", run_btn)
        layout.addWidget(form_box)

        splitter = QSplitter(Qt.Vertical)
        self.result_text = QPlainTextEdit()
        self.result_text.setPlaceholderText("核验结果会显示在这里。")
        self.result_text.setReadOnly(True)
        splitter.addWidget(self.result_text)

        self.check_table = QTableWidget(0, 3)
        self.check_table.setHorizontalHeaderLabels(["项目", "状态", "说明"])
        self.check_table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self.check_table)
        splitter.setSizes([300, 240])
        layout.addWidget(splitter)

        export_row = QHBoxLayout()
        export_json_btn = QPushButton("导出 JSON")
        export_json_btn.clicked.connect(self.export_json_file)
        export_html_btn = QPushButton("导出 HTML")
        export_html_btn.clicked.connect(self.export_html_file)
        export_row.addWidget(export_json_btn)
        export_row.addWidget(export_html_btn)
        export_row.addStretch()
        layout.addLayout(export_row)

    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 PDF 报告", "", "PDF Files (*.pdf)")
        if path:
            self.file_edit.setText(path)

    def run_analysis(self) -> None:
        file_path = self.file_edit.text().strip()
        if not file_path:
            QMessageBox.warning(self, "提示", "请先选择 PDF 文件。")
            return
        standard_key = self.standard_combo.currentData()
        try:
            self.analyzer.ai_client = AIClient()
            result = self.analyzer.analyze(file_path, standard_key, self.ai_checkbox.isChecked())
        except Exception as exc:
            QMessageBox.critical(self, "核验失败", str(exc))
            return
        self.current_result = result
        self.result_text.setPlainText(result.ai_summary + "\n\n" + result.raw_text_preview)
        self.check_table.setRowCount(len(result.checks))
        for row, item in enumerate(result.checks):
            self.check_table.setItem(row, 0, QTableWidgetItem(item.name))
            self.check_table.setItem(row, 1, QTableWidgetItem(item.status))
            self.check_table.setItem(row, 2, QTableWidgetItem(item.detail))
        self.check_table.resizeColumnsToContents()

    def export_json_file(self) -> None:
        if not self.current_result:
            QMessageBox.information(self, "提示", "请先运行核验。")
            return
        output = self.exports_dir / f"{Path(self.current_result.file_path).stem}_report.json"
        export_json(self.current_result.to_dict(), output)
        QMessageBox.information(self, "导出完成", f"已保存：{output}")

    def export_html_file(self) -> None:
        if not self.current_result:
            QMessageBox.information(self, "提示", "请先运行核验。")
            return
        output = self.exports_dir / f"{Path(self.current_result.file_path).stem}_report.html"
        export_report_html(self.current_result.to_dict(), output)
        QMessageBox.information(self, "导出完成", f"已保存：{output}")


class VisionReportTab(QWidget):
    def __init__(self, standards: StandardsLibrary, exports_dir: Path):
        super().__init__()
        self.standards = standards
        self.exports_dir = exports_dir
        self.pipeline = EnvironmentReportVisionPipeline(standards, exports_dir / "vision")
        self.file_paths: list[str] = []
        self.current_result = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form_box = QGroupBox("报告照片扫描增强与结构化识别")
        form = QFormLayout(form_box)

        self.files_text = QPlainTextEdit()
        self.files_text.setReadOnly(True)
        self.files_text.setPlaceholderText("选择拍摄照片、扫描图片或 PDF；多张图片会自动拼成一个逻辑报告。")
        self.files_text.setMaximumHeight(82)
        select_btn = QPushButton("选择照片/PDF")
        select_btn.clicked.connect(self.choose_files)
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(self.clear_files)
        row = QHBoxLayout()
        row.addWidget(self.files_text)
        row.addWidget(select_btn)
        row.addWidget(clear_btn)
        row_w = QWidget()
        row_w.setLayout(row)
        form.addRow("输入文件", row_w)

        self.standard_combo = QComboBox()
        for key, name in self.standards.names().items():
            self.standard_combo.addItem(name, key)
        form.addRow("执行标准", self.standard_combo)

        self.ai_checkbox = QCheckBox("启用 AI 视觉抽取（需配置 AI_API_KEY / AI_BASE_URL / AI_MODEL）")
        self.ai_checkbox.setChecked(True)
        form.addRow("高精度模式", self.ai_checkbox)

        run_btn = QPushButton("开始扫描识别")
        run_btn.clicked.connect(self.run_analysis)
        form.addRow("", run_btn)
        layout.addWidget(form_box)

        splitter = QSplitter(Qt.Vertical)
        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlaceholderText("简报总结、告警和增强 PDF 路径会显示在这里。")
        splitter.addWidget(self.summary_text)

        self.record_table = QTableWidget(0, 8)
        self.record_table.setHorizontalHeaderLabels(["指标", "检测值", "单位", "限值", "状态", "置信度", "样点", "来源行"])
        self.record_table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self.record_table)

        self.json_preview = QPlainTextEdit()
        self.json_preview.setReadOnly(True)
        self.json_preview.setPlaceholderText("结构化 JSON 预览。")
        splitter.addWidget(self.json_preview)
        splitter.setSizes([160, 260, 220])
        layout.addWidget(splitter)

        export_row = QHBoxLayout()
        export_json_btn = QPushButton("导出 JSON")
        export_json_btn.clicked.connect(self.export_json_file)
        export_html_btn = QPushButton("导出 HTML")
        export_html_btn.clicked.connect(self.export_html_file)
        export_excel_btn = QPushButton("导出 Excel")
        export_excel_btn.clicked.connect(self.export_excel_file)
        export_row.addWidget(export_json_btn)
        export_row.addWidget(export_html_btn)
        export_row.addWidget(export_excel_btn)
        export_row.addStretch()
        layout.addLayout(export_row)

    def choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择报告照片或 PDF",
            "",
            "Report Images/PDF (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.pdf)",
        )
        if not paths:
            return
        self.file_paths = paths
        self.files_text.setPlainText("\n".join(paths))

    def clear_files(self) -> None:
        self.file_paths = []
        self.files_text.clear()
        self.summary_text.clear()
        self.json_preview.clear()
        self.record_table.setRowCount(0)
        self.current_result = None

    def run_analysis(self) -> None:
        if not self.file_paths:
            QMessageBox.warning(self, "提示", "请先选择至少一张报告照片或 PDF。")
            return
        standard_key = self.standard_combo.currentData()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.pipeline.ai_client = AIClient()
            result = self.pipeline.analyze_files(
                self.file_paths,
                standard_key,
                use_ai=self.ai_checkbox.isChecked(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "识别失败", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.current_result = result
        data = result.to_dict()
        warnings = "\n".join(data.get("warnings", []))
        self.summary_text.setPlainText(
            f"{result.brief_summary}\n\n增强 PDF：{result.enhanced_pdf_path}\n\n警告：\n{warnings}"
        )
        self.json_preview.setPlainText(json.dumps(data, ensure_ascii=False, indent=2))
        self._populate_record_table(data)

    def _populate_record_table(self, data: dict) -> None:
        records = data.get("records", [])
        self.record_table.setRowCount(len(records))
        for row, item in enumerate(records):
            values = [
                item.get("normalized_indicator") or item.get("indicator", ""),
                item.get("raw_value") or item.get("value", ""),
                item.get("unit", ""),
                f"{item.get('standard_limit', '')} {item.get('limit_unit', '')}".strip(),
                ("复核/" if item.get("needs_review") else "") + item.get("status", ""),
                item.get("confidence", ""),
                item.get("sample_point", ""),
                item.get("source_line", ""),
            ]
            for col, value in enumerate(values):
                self.record_table.setItem(row, col, QTableWidgetItem(str(value)))
        self.record_table.resizeColumnsToContents()

    def _current_data(self) -> Optional[dict]:
        if not self.current_result:
            QMessageBox.information(self, "提示", "请先运行照片识别。")
            return None
        return self.current_result.to_dict()

    def _output_stem(self) -> str:
        if self.current_result:
            return self.current_result.logical_report_id
        return "vision_report"

    def export_json_file(self) -> None:
        data = self._current_data()
        if not data:
            return
        output = self.exports_dir / f"{self._output_stem()}_structured.json"
        export_json(data, output)
        QMessageBox.information(self, "导出完成", f"已保存：{output}")

    def export_html_file(self) -> None:
        data = self._current_data()
        if not data:
            return
        output = self.exports_dir / f"{self._output_stem()}_structured.html"
        export_structured_html(data, output)
        QMessageBox.information(self, "导出完成", f"已保存：{output}")

    def export_excel_file(self) -> None:
        data = self._current_data()
        if not data:
            return
        output = self.exports_dir / f"{self._output_stem()}_structured.xlsx"
        export_structured_excel(data, output)
        QMessageBox.information(self, "导出完成", f"已保存：{output}")


class SensorTab(QWidget):
    def __init__(self, exports_dir: Path):
        super().__init__()
        self.exports_dir = exports_dir
        self.analyzer = SensorAnalyzer()
        self.current_result = None
        self.df_cache = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form_box = QGroupBox("监测数据异常分析")
        form = QFormLayout(form_box)

        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        select_btn = QPushButton("选择 CSV/XLSX")
        select_btn.clicked.connect(self.choose_file)
        row = QHBoxLayout()
        row.addWidget(self.file_edit)
        row.addWidget(select_btn)
        row_w = QWidget()
        row_w.setLayout(row)
        form.addRow("数据文件", row_w)

        self.timestamp_combo = QComboBox()
        self.station_combo = QComboBox()
        self.indicator_combo = QComboBox()
        self.value_combo = QComboBox()
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItem("Z-Score", "zscore")
        self.algorithm_combo.addItem("Isolation Forest", "iforest")
        form.addRow("时间列", self.timestamp_combo)
        form.addRow("站点列", self.station_combo)
        form.addRow("指标列", self.indicator_combo)
        form.addRow("数值列", self.value_combo)
        form.addRow("算法", self.algorithm_combo)

        run_btn = QPushButton("开始分析")
        run_btn.clicked.connect(self.run_analysis)
        form.addRow("", run_btn)
        layout.addWidget(form_box)

        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlaceholderText("分析摘要会显示在这里。")
        layout.addWidget(self.summary_text)

        self.plot_canvas = MplCanvas()
        layout.addWidget(self.plot_canvas)

        self.anomaly_table = QTableWidget(0, 5)
        self.anomaly_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.anomaly_table)

        export_row = QHBoxLayout()
        export_json_btn = QPushButton("导出 JSON")
        export_json_btn.clicked.connect(self.export_json_file)
        export_html_btn = QPushButton("导出 HTML")
        export_html_btn.clicked.connect(self.export_html_file)
        export_row.addWidget(export_json_btn)
        export_row.addWidget(export_html_btn)
        export_row.addStretch()
        layout.addLayout(export_row)

    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "Data Files (*.csv *.xlsx *.xls)")
        if not path:
            return
        self.file_edit.setText(path)
        try:
            df = self.analyzer.load_dataframe(path)
            guessed = self.analyzer.guess_columns(df)
        except Exception as exc:
            QMessageBox.critical(self, "读取失败", str(exc))
            return
        self.df_cache = df
        for combo in [self.timestamp_combo, self.station_combo, self.indicator_combo, self.value_combo]:
            combo.clear()
            combo.addItems(guessed["all"])
        self._set_current(self.timestamp_combo, guessed["timestamp"])
        self._set_current(self.station_combo, guessed["station"])
        self._set_current(self.indicator_combo, guessed["indicator"])
        self._set_current(self.value_combo, guessed["value"])

    def _set_current(self, combo: QComboBox, value: str) -> None:
        idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def run_analysis(self) -> None:
        file_path = self.file_edit.text().strip()
        if not file_path:
            QMessageBox.warning(self, "提示", "请先选择数据文件。")
            return
        try:
            result = self.analyzer.analyze(
                file_path=file_path,
                timestamp_col=self.timestamp_combo.currentText(),
                station_col=self.station_combo.currentText(),
                indicator_col=self.indicator_combo.currentText(),
                value_col=self.value_combo.currentText(),
                algorithm=self.algorithm_combo.currentData(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "分析失败", str(exc))
            return
        self.current_result = result
        self.summary_text.setPlainText(result.summary + "\n\n" + json.dumps(result.stats, ensure_ascii=False, indent=2))
        self._populate_anomaly_table(result)
        self._plot_result(file_path)

    def _populate_anomaly_table(self, result) -> None:
        keys = []
        for row in result.anomalies[:1]:
            keys = list(row.keys())
        self.anomaly_table.clear()
        self.anomaly_table.setColumnCount(len(keys))
        self.anomaly_table.setHorizontalHeaderLabels(keys)
        self.anomaly_table.setRowCount(len(result.anomalies))
        for r, row in enumerate(result.anomalies):
            for c, key in enumerate(keys):
                self.anomaly_table.setItem(r, c, QTableWidgetItem(str(row.get(key, ""))))
        self.anomaly_table.resizeColumnsToContents()

    def _plot_result(self, file_path: str) -> None:
        if self.df_cache is None:
            return
        ts_col = self.timestamp_combo.currentText()
        value_col = self.value_combo.currentText()
        df = self.df_cache.copy()
        try:
            df[ts_col] = df[ts_col].astype("datetime64[ns]")
        except Exception:
            pass
        df[value_col] = df[value_col].astype(float)
        df = df.sort_values(ts_col).head(120)
        self.plot_canvas.plot_series(df[ts_col], df[value_col], title=Path(file_path).name)

    def export_json_file(self) -> None:
        if not self.current_result:
            QMessageBox.information(self, "提示", "请先运行分析。")
            return
        output = self.exports_dir / f"{Path(self.current_result.file_path).stem}_sensor.json"
        export_json(self.current_result.to_dict(), output)
        QMessageBox.information(self, "导出完成", f"已保存：{output}")

    def export_html_file(self) -> None:
        if not self.current_result:
            QMessageBox.information(self, "提示", "请先运行分析。")
            return
        output = self.exports_dir / f"{Path(self.current_result.file_path).stem}_sensor.html"
        export_sensor_html(self.current_result.to_dict(), output)
        QMessageBox.information(self, "导出完成", f"已保存：{output}")


class MainWindow(QMainWindow):
    def __init__(self, standards: StandardsLibrary, exports_dir: Path):
        super().__init__()
        self.setWindowTitle("环境检测 AI 智能核验系统 MVP")
        self.resize(1280, 860)

        tabs = QTabWidget()
        tabs.addTab(ReportTab(standards, exports_dir), "报告核验")
        tabs.addTab(VisionReportTab(standards, exports_dir), "照片识别结构化")
        tabs.addTab(SensorTab(exports_dir), "数据异常分析")
        self.setCentralWidget(tabs)
