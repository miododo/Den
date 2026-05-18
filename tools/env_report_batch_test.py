from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.core.ai_client import AIClient
from src.core.exporter import export_json, export_structured_excel, export_structured_html
from src.core.models import StructuredReportResult
from src.core.standards import StandardsLibrary
from src.core.vision_pipeline import EnvironmentReportVisionPipeline


DEFAULT_SAMPLE_FILES = [
    Path(r"D:\个人资料\比赛内容\环境检测pdf扫描版\2023年生态环境监测委托监测服务（地表水）(OCR).pdf"),
    Path(r"D:\个人资料\比赛内容\环境检测pdf扫描版\江北区2024年水资源质量检测项目（地表水）.pdf"),
    Path(r"D:\个人资料\比赛内容\环境检测pdf扫描版\桐梓县渝能水电开发有限公司（废水）.pdf"),
    Path(r"D:\个人资料\比赛内容\环境检测pdf扫描版\重庆市映天氯碱化工有限公司（地下水）(OCR).pdf"),
    Path(r"D:\个人资料\比赛内容\环境检测pdf扫描版\重庆市映天氯碱化工有限公司（地下水）.pdf"),
    Path(r"D:\个人资料\比赛内容\环境检测pdf扫描版\重庆中法唐家沱污水处理有限公司（废水）.pdf"),
]

IMAGE_OR_PDF_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def safe_name(value: str, limit: int = 90) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", value, flags=re.UNICODE).strip("_.")
    return (cleaned or "sample")[:limit]


def suggest_standard(path: Path) -> str:
    name = path.name.lower()
    if "废水" in name or "污水" in name:
        return "wastewater_custom"
    if "地下水" in name:
        return "groundwater_custom"
    if "地表水" in name:
        return "surface_water_gb3838_subset"
    if "空气" in name or "废气" in name:
        return "air_gb3095_2012_grade2"
    return "surface_water_gb3838_subset"


def collect_files(args: argparse.Namespace) -> List[Path]:
    files: List[Path] = []
    if args.files:
        files.extend(Path(item) for item in args.files)
    if args.input_dir:
        root = Path(args.input_dir)
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_OR_PDF_SUFFIXES)
    if not files:
        files.extend(DEFAULT_SAMPLE_FILES)
    unique: List[Path] = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            unique.append(resolved)
        else:
            print(f"[skip] 文件不存在：{path}")
    return unique


def split_records(items: Sequence[Any], eval_ratio: float, seed: int) -> Tuple[List[Any], List[Any]]:
    rng = random.Random(seed)
    copied = list(items)
    rng.shuffle(copied)
    eval_count = max(1, int(len(copied) * eval_ratio)) if len(copied) > 1 else 0
    return copied[eval_count:], copied[:eval_count]


class PaddleWeakDatasetBuilder:
    """Build PaddleOCR-compatible weak labels from the current OCR output.

    These labels are intended for review and bootstrapping. They should be
    manually corrected before being treated as real training ground truth.
    """

    def __init__(self, root: Path, min_confidence: float = 0.86, eval_ratio: float = 0.2, seed: int = 2026) -> None:
        self.root = root
        self.min_confidence = min_confidence
        self.eval_ratio = eval_ratio
        self.seed = seed
        self.det_image_dir = root / "det" / "images"
        self.rec_image_dir = root / "rec" / "images"
        self.det_image_dir.mkdir(parents=True, exist_ok=True)
        self.rec_image_dir.mkdir(parents=True, exist_ok=True)
        self.det_records: List[Tuple[str, List[Dict[str, Any]]]] = []
        self.rec_records: List[Tuple[str, str]] = []
        self.meta_records: List[Dict[str, Any]] = []
        self._ocr: Any = None

    def add_report(self, result: StructuredReportResult, source_file: Path, max_pages: Optional[int] = None) -> None:
        pages = sorted(result.pages, key=lambda item: item.page_index)
        if max_pages:
            pages = pages[:max_pages]
        for page in pages:
            image_path = Path(page.enhanced_image_path)
            if not image_path.exists():
                continue
            base = f"{safe_name(source_file.stem, 72)}_p{page.page_index:03d}.png"
            target = self.det_image_dir / base
            shutil.copy2(image_path, target)
            ocr_lines = self._ocr_image(target)
            labels: List[Dict[str, Any]] = []
            with Image.open(target) as image:
                for line_index, (points, text, score) in enumerate(ocr_lines, start=1):
                    if not text.strip() or score < self.min_confidence:
                        continue
                    labels.append({"transcription": text.strip(), "points": points})
                    crop_rel = self._crop_rec_image(image, base, line_index, points)
                    if crop_rel:
                        self.rec_records.append((crop_rel, text.strip()))
                    self.meta_records.append(
                        {
                            "source_file": str(source_file),
                            "page_index": page.page_index,
                            "image": f"det/images/{base}",
                            "text": text.strip(),
                            "score": round(score, 4),
                            "points": points,
                            "label_source": "paddleocr_weak",
                        }
                    )
            if labels:
                self.det_records.append((f"det/images/{base}", labels))

    def write(self) -> Dict[str, Any]:
        det_train, det_eval = split_records(self.det_records, self.eval_ratio, self.seed)
        rec_train, rec_eval = split_records(self.rec_records, self.eval_ratio, self.seed)
        self._write_det(self.root / "det_gt_train.txt", det_train)
        self._write_det(self.root / "det_gt_eval.txt", det_eval)
        self._write_rec(self.root / "rec_gt_train.txt", rec_train)
        self._write_rec(self.root / "rec_gt_eval.txt", rec_eval)
        with (self.root / "weak_labels.jsonl").open("w", encoding="utf-8") as fh:
            for item in self.meta_records:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        readme = self.root / "README.md"
        readme.write_text(
            "\n".join(
                [
                    "# 环境检测报告 PaddleOCR 弱标注样本集",
                    "",
                    "这些标签由当前 PaddleOCR 推理结果自动生成，只适合做候选标注和回归测试。",
                    "正式训练前，请先人工复核 `det_gt_*.txt`、`rec_gt_*.txt` 与 `weak_labels.jsonl`。",
                    "",
                    "检测任务格式：`图片路径\\tjson.dumps([{transcription, points}, ...])`。",
                    "识别任务格式：`裁剪图路径\\t文字标签`。",
                ]
            ),
            encoding="utf-8",
        )
        return {
            "dataset_dir": str(self.root),
            "det_train": len(det_train),
            "det_eval": len(det_eval),
            "rec_train": len(rec_train),
            "rec_eval": len(rec_eval),
            "weak_labels": len(self.meta_records),
        }

    def _ocr_image(self, image_path: Path) -> List[Tuple[List[List[int]], str, float]]:
        if self._ocr is None:
            from paddleocr import PaddleOCR  # type: ignore

            self._ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        raw = self._ocr.ocr(str(image_path), cls=True)
        return list(self._iter_paddle_lines(raw))

    def _iter_paddle_lines(self, raw: Any) -> Iterable[Tuple[List[List[int]], str, float]]:
        if not raw:
            return
        lines = raw
        if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
            lines = raw[0]
        for item in lines or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            points_raw, text_score = item[0], item[1]
            if not isinstance(text_score, (list, tuple)) or len(text_score) < 2:
                continue
            text = str(text_score[0]).strip()
            try:
                score = float(text_score[1])
            except (TypeError, ValueError):
                score = 0.0
            points: List[List[int]] = []
            try:
                for point in points_raw:
                    points.append([int(round(float(point[0]))), int(round(float(point[1])))])
            except Exception:
                continue
            if len(points) >= 4:
                yield points[:4], text, score

    def _crop_rec_image(self, image: Image.Image, base_name: str, line_index: int, points: Sequence[Sequence[int]]) -> str:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        pad = 3
        left = max(0, min(xs) - pad)
        top = max(0, min(ys) - pad)
        right = min(image.width, max(xs) + pad)
        bottom = min(image.height, max(ys) + pad)
        if right <= left or bottom <= top:
            return ""
        crop_name = f"{Path(base_name).stem}_l{line_index:04d}.png"
        crop_path = self.rec_image_dir / crop_name
        image.crop((left, top, right, bottom)).save(crop_path)
        return f"rec/images/{crop_name}"

    def _write_det(self, path: Path, rows: Sequence[Tuple[str, List[Dict[str, Any]]]]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for image_rel, labels in rows:
                fh.write(f"{image_rel}\t{json.dumps(labels, ensure_ascii=False)}\n")

    def _write_rec(self, path: Path, rows: Sequence[Tuple[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for image_rel, text in rows:
                fh.write(f"{image_rel}\t{text}\n")


def write_batch_outputs(output_dir: Path, rows: Sequence[Dict[str, Any]], dataset_summary: Optional[Dict[str, Any]]) -> None:
    json_path = output_dir / "batch_summary.json"
    json_path.write_text(json.dumps({"reports": rows, "paddleocr_dataset": dataset_summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output_dir / "batch_summary.csv"
    fieldnames = [
        "file",
        "standard_key",
        "standard_name",
        "total_records",
        "judged_count",
        "pass_count",
        "fail_count",
        "review_count",
        "compliance_rate",
        "warnings",
        "json_path",
        "html_path",
        "excel_path",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    html_rows = "\n".join(
        "<tr>"
        f"<td>{row.get('file','')}</td>"
        f"<td>{row.get('standard_name','')}</td>"
        f"<td>{row.get('total_records',0)}</td>"
        f"<td>{row.get('pass_count',0)}</td>"
        f"<td>{row.get('fail_count',0)}</td>"
        f"<td>{row.get('review_count',0)}</td>"
        f"<td>{row.get('warnings','')}</td>"
        "</tr>"
        for row in rows
    )
    (output_dir / "batch_summary.html").write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>环境检测报告批量测试结果</title>
  <style>
    body {{ font-family: Arial, 'Microsoft YaHei', sans-serif; margin: 24px; color: #1f2933; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border: 1px solid #d7dee8; padding: 7px 8px; vertical-align: top; }}
    th {{ background: #eef4fb; }}
    .box {{ background: #f8fbff; border: 1px solid #dce8f5; padding: 12px; margin: 12px 0; }}
  </style>
</head>
<body>
  <h1>环境检测报告批量测试结果</h1>
  <div class="box"><pre>{json.dumps(dataset_summary or {}, ensure_ascii=False, indent=2)}</pre></div>
  <table>
    <thead><tr><th>文件</th><th>标准</th><th>记录</th><th>合格</th><th>超标</th><th>复核</th><th>提示</th></tr></thead>
    <tbody>{html_rows}</tbody>
  </table>
</body>
</html>""",
        encoding="utf-8",
    )


def run_batch(args: argparse.Namespace) -> Path:
    files = collect_files(args)
    if not files:
        raise SystemExit("没有找到可测试的 PDF/图片文件。")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else APP_DIR / "training_runtime" / f"env_report_batch_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OCR_MAX_PAGES"] = str(args.max_pages)

    standards = StandardsLibrary(APP_DIR / "config" / "standards.json")
    dataset_builder = (
        PaddleWeakDatasetBuilder(output_dir / "paddleocr_weak_dataset", args.weak_label_min_confidence, args.eval_ratio, args.seed)
        if args.make_paddle_dataset
        else None
    )

    rows: List[Dict[str, Any]] = []
    for index, file_path in enumerate(files, start=1):
        standard_key = args.standard_key or suggest_standard(file_path)
        standard_name = standards.names().get(standard_key, standard_key)
        print(f"[{index}/{len(files)}] {file_path.name} -> {standard_name}")
        report_dir = output_dir / "reports" / safe_name(file_path.stem)
        pipeline = EnvironmentReportVisionPipeline(standards, report_dir / "vision", AIClient())
        result = pipeline.analyze_files([str(file_path)], standard_key, use_ai=args.use_ai)
        data = result.to_dict()
        data["dataset_role"] = "training_test_sample"
        data["label_status"] = "unreviewed_weak_or_rule_extracted"
        stem = safe_name(result.logical_report_id or file_path.stem)
        json_path = export_json(data, report_dir / f"{stem}_structured.json")
        html_path = export_structured_html(data, report_dir / f"{stem}_structured.html")
        excel_path = export_structured_excel(data, report_dir / f"{stem}_structured.xlsx")
        if dataset_builder:
            dataset_builder.add_report(result, file_path, max_pages=args.dataset_max_pages)
        stats = data.get("statistics", {})
        row = {
            "file": str(file_path),
            "standard_key": standard_key,
            "standard_name": standard_name,
            "total_records": stats.get("total_records", 0),
            "judged_count": stats.get("judged_count", 0),
            "pass_count": stats.get("pass_count", 0),
            "fail_count": stats.get("fail_count", 0),
            "review_count": stats.get("review_count", 0),
            "compliance_rate": stats.get("compliance_rate", 0),
            "warnings": "；".join(data.get("warnings", [])),
            "json_path": str(json_path),
            "html_path": str(html_path),
            "excel_path": str(excel_path),
        }
        rows.append(row)
        print(
            "    "
            f"records={row['total_records']} pass={row['pass_count']} "
            f"fail={row['fail_count']} review={row['review_count']}"
        )

    dataset_summary = dataset_builder.write() if dataset_builder else None
    write_batch_outputs(output_dir, rows, dataset_summary)
    print(f"[done] 输出目录：{output_dir}")
    if dataset_summary:
        print(f"[done] PaddleOCR 弱标注数据集：{dataset_summary['dataset_dir']}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="环境检测报告 PaddleOCR 批量测试与弱标注数据集生成")
    parser.add_argument("--input-dir", help="扫描 PDF/图片目录；为空时使用比赛样本默认路径")
    parser.add_argument("--files", nargs="*", help="指定一个或多个 PDF/图片文件")
    parser.add_argument("--output-dir", help="输出目录；默认写入 training_runtime/env_report_batch_时间戳")
    parser.add_argument("--standard-key", help="强制使用某个标准；为空时按文件名自动匹配")
    parser.add_argument("--max-pages", type=int, default=12, help="每份报告参与 OCR/结构化的最大页数")
    parser.add_argument("--use-ai", action="store_true", help="启用 AI 视觉抽取")
    parser.add_argument("--make-paddle-dataset", action="store_true", help="生成 PaddleOCR 检测/识别弱标注数据")
    parser.add_argument("--dataset-max-pages", type=int, default=8, help="每份报告用于弱标注数据集的最大页数")
    parser.add_argument("--weak-label-min-confidence", type=float, default=0.86, help="生成弱标注的 OCR 最低置信度")
    parser.add_argument("--eval-ratio", type=float, default=0.2, help="评估集比例")
    parser.add_argument("--seed", type=int, default=2026, help="训练/评估拆分随机种子")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_batch(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
