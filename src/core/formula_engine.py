from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.core.models import DetectionRecord


SAFE_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sqrt": math.sqrt,
    "log10": math.log10,
    "pow": pow,
}


class FormulaError(ValueError):
    pass


class SafeExpressionEvaluator(ast.NodeVisitor):
    def __init__(self, variables: Dict[str, float]) -> None:
        self.variables = variables

    def evaluate(self, expression: str) -> float:
        tree = ast.parse(expression, mode="eval")
        return float(self.visit(tree.body))

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise FormulaError("公式中只允许数字常量")

    def visit_Name(self, node: ast.Name) -> float:
        if node.id not in self.variables:
            raise FormulaError(f"缺少变量：{node.id}")
        return float(self.variables[node.id])

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise FormulaError("除数不能为 0")
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        if isinstance(node.op, ast.Mod):
            return left % right
        raise FormulaError("公式中包含不支持的运算符")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise FormulaError("公式中包含不支持的一元运算符")

    def visit_Call(self, node: ast.Call) -> float:
        if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCTIONS:
            raise FormulaError("公式中包含不允许的函数")
        args = [self.visit(arg) for arg in node.args]
        return float(SAFE_FUNCTIONS[node.func.id](*args))

    def generic_visit(self, node: ast.AST) -> float:
        raise FormulaError(f"公式中包含不支持的语法：{type(node).__name__}")


def _norm(value: str) -> str:
    return (value or "").replace(" ", "").lower()


@dataclass
class FormulaDatabase:
    config_path: Path

    def __post_init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        self.data = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.formulas: List[Dict[str, Any]] = list(self.data.get("formulas", []))

    def list_formulas(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": item.get("id"),
                "indicator": item.get("indicator"),
                "method_name": item.get("method_name"),
                "method_standard": item.get("method_standard"),
                "result_unit": item.get("result_unit"),
                "required_inputs": item.get("required_inputs", []),
                "source_files": item.get("source_files", []),
            }
            for item in self.formulas
        ]

    def find(self, indicator: str, method_id: str = "") -> Optional[Dict[str, Any]]:
        if method_id:
            for formula in self.formulas:
                if formula.get("id") == method_id:
                    return formula
        target = _norm(indicator)
        for formula in self.formulas:
            names = [formula.get("indicator", "")] + list(formula.get("aliases", []))
            if any(_norm(name) == target for name in names):
                return formula
        return None

    def find_all(self, indicator: str) -> List[Dict[str, Any]]:
        target = _norm(indicator)
        matches = []
        for formula in self.formulas:
            names = [formula.get("indicator", "")] + list(formula.get("aliases", []))
            if any(_norm(name) == target for name in names):
                matches.append(formula)
        return matches


class FormulaVerifier:
    def __init__(self, database: FormulaDatabase) -> None:
        self.database = database

    def annotate_records(self, records: Sequence[DetectionRecord]) -> List[str]:
        matched = 0
        for record in records:
            formula = self.database.find(record.normalized_indicator or record.indicator)
            if not formula:
                continue
            matched += 1
            record.formula_verification = {
                "status": "formula_available_missing_raw_inputs",
                "formula_id": formula.get("id"),
                "method_standard": formula.get("method_standard"),
                "method_name": formula.get("method_name"),
                "required_inputs": formula.get("required_inputs", []),
                "source_files": formula.get("source_files", []),
                "message": "已匹配计算公式库；报告未提供吸光度、滴定体积等原始实验参数，需通过 /api/formula/verify 或人工录入后复算。",
            }
            record.database_match.setdefault("formula_database", str(self.database.config_path))
            record.database_match["formula_matched"] = True
            record.database_match["formula_id"] = formula.get("id")
            record.database_match["formula_method_standard"] = formula.get("method_standard")
            note = f"公式库已匹配：{formula.get('method_standard', formula.get('id'))}，待原始实验参数复算"
            if note not in record.notes:
                record.notes.append(note)
        if matched:
            return [f"公式库已匹配 {matched} 条记录；如需复算，请录入原始实验参数或调用 /api/formula/verify。"]
        return []

    def verify(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        indicator = str(payload.get("indicator", "")).strip()
        method_id = str(payload.get("method_id", "")).strip()
        reported_value = self._optional_float(payload.get("reported_value"))
        inputs = payload.get("inputs", {}) or {}
        if not isinstance(inputs, dict):
            raise FormulaError("inputs 必须是对象")
        formula = self.database.find(indicator, method_id)
        if not formula:
            return {"status": "unsupported", "message": f"公式库未覆盖指标：{indicator}", "indicator": indicator}

        context: Dict[str, float] = {}
        for key, value in formula.get("defaults", {}).items():
            context[key] = float(value)
        for key, value in inputs.items():
            context[key] = float(value)

        required = list(formula.get("required_inputs", []))
        missing = [key for key in required if key not in context]
        base = self._base_result(formula, indicator, method_id, reported_value, inputs)
        if missing:
            base.update(
                {
                    "status": "missing_inputs",
                    "missing_inputs": missing,
                    "message": "缺少机器复算所需原始实验参数。",
                }
            )
            return base

        evaluator = SafeExpressionEvaluator(context)
        try:
            for step in formula.get("derived", []):
                name = step["name"]
                context[name] = evaluator.evaluate(step["expression"])
                evaluator = SafeExpressionEvaluator(context)
            calculated = evaluator.evaluate(str(formula.get("result_expression", "")))
        except Exception as exc:
            base.update({"status": "formula_error", "message": str(exc)})
            return base

        tolerance = formula.get("tolerance", {}) or {}
        abs_tol = float(tolerance.get("absolute", 0) or 0)
        rel_tol = float(tolerance.get("relative", 0) or 0)
        comparison: Dict[str, Any] = {"reported_value": reported_value, "calculated_value": round(calculated, 6)}
        if reported_value is None:
            status = "calculated"
            message = "已完成机器复算，未提供报告值用于比对。"
        else:
            allowed = max(abs_tol, abs(calculated) * rel_tol)
            diff = abs(reported_value - calculated)
            comparison.update({"difference": round(diff, 6), "allowed_tolerance": round(allowed, 6)})
            status = "pass" if diff <= allowed else "fail"
            message = "机器复算与报告值一致。" if status == "pass" else "机器复算与报告值不一致，建议 AI/人工复核。"

        base.update(
            {
                "status": status,
                "message": message,
                "calculated_value": round(calculated, 6),
                "result_unit": formula.get("result_unit", ""),
                "derived_values": {key: round(value, 6) for key, value in context.items() if key not in inputs},
                "comparison": comparison,
            }
        )
        return base

    def _base_result(
        self,
        formula: Dict[str, Any],
        indicator: str,
        method_id: str,
        reported_value: Optional[float],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "indicator": indicator or formula.get("indicator", ""),
            "method_id": method_id or formula.get("id", ""),
            "method_name": formula.get("method_name", ""),
            "method_standard": formula.get("method_standard", ""),
            "source_files": formula.get("source_files", []),
            "required_inputs": formula.get("required_inputs", []),
            "provided_inputs": inputs,
            "reported_value": reported_value,
            "formula_expression": formula.get("result_expression", ""),
            "notes": formula.get("notes", ""),
        }

    def _optional_float(self, value: Any) -> Optional[float]:
        if value in {None, ""}:
            return None
        return float(value)
