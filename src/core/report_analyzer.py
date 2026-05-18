from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from src.core.ai_client import AIClient
from src.core.models import CheckItem, ReportAnalysisResult
from src.core.parser import ReportParser
from src.core.standards import StandardsLibrary
from src.utils.text import parse_date, safe_preview

logger = logging.getLogger(__name__)


class ReportAnalyzer:
    def __init__(self, standards: StandardsLibrary, ai_client: Optional[AIClient] = None):
        self.standards = standards
        self.ai_client = ai_client
        self.parser = ReportParser()

    def analyze(self, file_path: str, standard_key: str, use_ai: bool = True) -> ReportAnalysisResult:
        parsed = self.parser.parse(file_path)
        result = ReportAnalysisResult(
            file_path=str(Path(file_path).resolve()),
            report_type=parsed["report_type"],
            report_no=parsed["report_no"],
            commissioning_unit=parsed["commissioning_unit"],
            inspected_unit=parsed["inspected_unit"],
            project_name=parsed["project_name"],
            report_date=parsed["report_date"],
            cma_code=parsed["cma_code"],
            cma_valid_from=parsed["cma_valid_from"],
            cma_valid_to=parsed["cma_valid_to"],
            extracted_text_length=len(parsed["raw_text"]),
            supported_indicators=parsed["supported_indicators"],
            methods_detected=parsed["methods_detected"],
            warnings=list(parsed["warnings"]),
            raw_text_preview=safe_preview(parsed["raw_text"]),
        )
        self._run_compliance_checks(result, standard_key)
        if use_ai and self.ai_client and self.ai_client.is_enabled:
            try:
                result.ai_summary = self.ai_client.summarize_report(result)
            except Exception as exc:  # pragma: no cover
                logger.exception("AI summary failed")
                result.warnings.append(f"AI 分析失败，已退回本地规则模式：{exc}")
        if not result.ai_summary:
            result.ai_summary = self._build_local_summary(result, standard_key)
        return result

    def _run_compliance_checks(self, result: ReportAnalysisResult, standard_key: str) -> None:
        required_fields = {
            "报告编号": result.report_no,
            "委托单位": result.commissioning_unit,
            "受检单位": result.inspected_unit,
            "项目名称": result.project_name,
            "报告日期": result.report_date,
        }
        for label, value in required_fields.items():
            result.checks.append(
                CheckItem(
                    name=f"字段完整性 - {label}",
                    status="pass" if value else "warn",
                    severity="warning" if not value else "info",
                    detail=value if value else "未识别到，建议人工复核。",
                )
            )

        if result.cma_valid_to:
            report_dt = parse_date(result.report_date) if result.report_date else None
            valid_to_dt = parse_date(result.cma_valid_to)
            if report_dt and valid_to_dt:
                ok = report_dt <= valid_to_dt
                result.checks.append(
                    CheckItem(
                        name="CMA 有效期核验",
                        status="pass" if ok else "fail",
                        severity="error" if not ok else "info",
                        detail=(
                            f"报告日期 {result.report_date} {'早于或等于' if ok else '晚于'} CMA 到期日 {result.cma_valid_to}。"
                        ),
                    )
                )
            else:
                result.checks.append(
                    CheckItem(
                        name="CMA 有效期核验",
                        status="warn",
                        severity="warning",
                        detail="已识别到 CMA 信息，但日期格式不完整，建议人工复核。",
                    )
                )
        else:
            result.checks.append(
                CheckItem(
                    name="CMA 有效期核验",
                    status="warn",
                    severity="warning",
                    detail="未识别到完整 CMA 有效期。扫描件建议先做 OCR。",
                )
            )

        if result.methods_detected:
            result.checks.append(
                CheckItem(
                    name="分析方法识别",
                    status="pass",
                    detail=f"已识别 {len(result.methods_detected)} 条方法/标准号。",
                )
            )
        else:
            result.checks.append(
                CheckItem(
                    name="分析方法识别",
                    status="warn",
                    severity="warning",
                    detail="未识别出足够的方法标准号，建议检查表格页 OCR 质量。",
                )
            )

        standard = self.standards.get(standard_key)
        indicator_rules = standard.get("indicators", {})
        if result.report_type == "surface_water" and indicator_rules:
            coverage = len(set(indicator_rules.keys()) & set(result.supported_indicators))
            result.checks.append(
                CheckItem(
                    name="标准库覆盖度",
                    status="pass" if coverage else "warn",
                    severity="warning" if not coverage else "info",
                    detail=f"当前标准库与报告文本重合指标 {coverage} 项。",
                )
            )
        else:
            result.checks.append(
                CheckItem(
                    name="标准库覆盖度",
                    status="warn",
                    severity="warning",
                    detail="当前报告类型默认只做完整性与资质核验；如需超标判定，请补充自定义标准 JSON。",
                )
            )

    def _build_local_summary(self, result: ReportAnalysisResult, standard_key: str) -> str:
        standard_name = self.standards.names().get(standard_key, standard_key)
        ok_count = sum(1 for c in result.checks if c.status == "pass")
        issue_count = sum(1 for c in result.checks if c.status in {"warn", "fail"})
        indicators = "、".join(result.supported_indicators[:12]) if result.supported_indicators else "未识别"
        return (
            f"已完成《{Path(result.file_path).name}》的本地规则分析。"
            f"报告类型判定为 {result.report_type}，使用标准库：{standard_name}。"
            f"当前识别到的关键指标包括：{indicators}。"
            f"共生成 {len(result.checks)} 项核验，其中通过 {ok_count} 项，需人工关注 {issue_count} 项。"
            f"建议优先复核字段缺失、CMA 有效期以及方法标准号覆盖情况。"
        )
