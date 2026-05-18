from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CheckItem:
    name: str
    status: str
    detail: str
    severity: str = "info"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReportAnalysisResult:
    file_path: str
    report_type: str = "unknown"
    report_no: str = ""
    commissioning_unit: str = ""
    inspected_unit: str = ""
    project_name: str = ""
    report_date: str = ""
    cma_code: str = ""
    cma_valid_from: str = ""
    cma_valid_to: str = ""
    extracted_text_length: int = 0
    supported_indicators: List[str] = field(default_factory=list)
    methods_detected: Dict[str, str] = field(default_factory=dict)
    checks: List[CheckItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    ai_summary: str = ""
    raw_text_preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["checks"] = [item.to_dict() for item in self.checks]
        return data


@dataclass
class SensorAnalysisResult:
    file_path: str
    rows: int
    timestamp_column: str
    indicator_column: str
    value_column: str
    station_column: Optional[str] = None
    algorithm: str = "zscore"
    anomaly_count: int = 0
    summary: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessedPage:
    source_path: str
    page_index: int
    original_width: int
    original_height: int
    enhanced_image_path: str
    transform_confidence: float = 0.0
    preprocessing_steps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SamplingInfo:
    sample_time: str = ""
    sample_point: str = ""
    frequency: str = ""
    confidence: float = 0.0
    source_line: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionRecord:
    indicator: str
    value: Optional[float]
    unit: str = ""
    sample_id: str = ""
    report_no: str = ""
    detection_date: str = ""
    sample_time: str = ""
    sample_point: str = ""
    frequency: str = ""
    standard_name: str = ""
    standard_limit: str = ""
    limit_unit: str = ""
    comparison: str = ""
    status: str = "unknown"
    confidence: float = 0.0
    needs_review: bool = True
    source_page: int = 0
    source_line: str = ""
    average_time: str = ""
    raw_value: str = ""
    normalized_indicator: str = ""
    detection_conclusion: str = ""
    notes: List[str] = field(default_factory=list)
    database_match: Dict[str, Any] = field(default_factory=dict)
    formula_verification: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredReportResult:
    input_files: List[str]
    standard_key: str
    standard_name: str
    logical_report_id: str = ""
    enhanced_pdf_path: str = ""
    pages: List[ProcessedPage] = field(default_factory=list)
    sampling: List[SamplingInfo] = field(default_factory=list)
    records: List[DetectionRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    raw_text_preview: str = ""
    brief_summary: str = ""
    visualization_suggestions: List[str] = field(default_factory=list)
    processing_trace: Dict[str, Any] = field(default_factory=dict)

    @property
    def pass_count(self) -> int:
        return sum(1 for item in self.records if item.status == "pass")

    @property
    def fail_count(self) -> int:
        return sum(1 for item in self.records if item.status == "fail")

    @property
    def review_count(self) -> int:
        return sum(1 for item in self.records if item.needs_review or item.status in {"review", "unknown"})

    @property
    def compliance_rate(self) -> float:
        judged = [item for item in self.records if item.status in {"pass", "fail"}]
        if not judged:
            return 0.0
        return round(self.pass_count / len(judged), 4)

    @property
    def exceeded_pollutants(self) -> List[str]:
        names = []
        for item in self.records:
            name = item.normalized_indicator or item.indicator
            if item.status == "fail" and name not in names:
                names.append(name)
        return names

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_files": self.input_files,
            "standard_key": self.standard_key,
            "standard_name": self.standard_name,
            "logical_report_id": self.logical_report_id,
            "enhanced_pdf_path": self.enhanced_pdf_path,
            "pages": [item.to_dict() for item in self.pages],
            "sampling": [item.to_dict() for item in self.sampling],
            "records": [item.to_dict() for item in self.records],
            "statistics": {
                "total_records": len(self.records),
                "judged_count": self.pass_count + self.fail_count,
                "pass_count": self.pass_count,
                "fail_count": self.fail_count,
                "review_count": self.review_count,
                "compliance_rate": self.compliance_rate,
                "exceeded_pollutants": self.exceeded_pollutants,
            },
            "warnings": self.warnings,
            "raw_text_preview": self.raw_text_preview,
            "brief_summary": self.brief_summary,
            "visualization_suggestions": self.visualization_suggestions,
            "processing_trace": self.processing_trace,
        }
