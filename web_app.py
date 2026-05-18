from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from src.core.ai_client import AIClient
from src.core.exporter import export_json, export_structured_excel, export_structured_html
from src.core.formula_engine import FormulaDatabase, FormulaVerifier
from src.core.standards import StandardsLibrary
from src.core.vision_pipeline import EnvironmentReportVisionPipeline


APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config" / "standards.json"
FORMULA_CONFIG_PATH = APP_DIR / "config" / "calculation_formulas.json"
WEB_DIR = APP_DIR / "web_runtime"
UPLOADS_DIR = WEB_DIR / "uploads"
EXPORTS_DIR = WEB_DIR / "exports"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

standards = StandardsLibrary(CONFIG_PATH)
formula_database = FormulaDatabase(FORMULA_CONFIG_PATH)
formula_verifier = FormulaVerifier(formula_database)
app = FastAPI(title="Env AI Validator Web", version="0.2.0")
app.mount("/exports", StaticFiles(directory=EXPORTS_DIR), name="exports")


INDEX_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>环境检测报告智能识别</title>
  <style>
    :root {
      --ink: #17212b;
      --muted: #66717f;
      --line: #d8e1ea;
      --panel: #f7fafc;
      --blue: #1565c0;
      --green: #18794e;
      --red: #c62828;
      --amber: #9a5b00;
      --bg: #ffffff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      border-bottom: 1px solid var(--line);
      padding: 18px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }
    h1 { font-size: 22px; margin: 0; font-weight: 700; letter-spacing: 0; }
    main { padding: 18px 24px 28px; display: grid; grid-template-columns: 360px 1fr; gap: 18px; }
    section { min-width: 0; }
    .panel {
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 14px;
      border-radius: 8px;
    }
    .controls { display: grid; gap: 12px; }
    label { display: grid; gap: 6px; font-size: 13px; color: var(--muted); }
    input[type="file"], select {
      width: 100%;
      border: 1px solid var(--line);
      background: white;
      border-radius: 6px;
      padding: 9px;
      color: var(--ink);
      font-size: 14px;
    }
    .toggle { display: flex; align-items: center; gap: 8px; color: var(--ink); }
    button {
      border: 0;
      border-radius: 6px;
      background: var(--blue);
      color: white;
      height: 40px;
      padding: 0 14px;
      font-size: 14px;
      cursor: pointer;
    }
    button:disabled { opacity: .55; cursor: default; }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(130px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: white;
      min-height: 76px;
    }
    .metric span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 8px; }
    .metric strong { font-size: 20px; }
    .summary {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      padding: 12px;
      margin-bottom: 14px;
      line-height: 1.55;
      min-height: 72px;
    }
    .warnings {
      display: none;
      border-color: #f0c36d;
      background: #fff8e6;
      color: #5f3b00;
      min-height: 0;
    }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
    .actions a {
      color: var(--blue);
      border: 1px solid var(--line);
      background: white;
      border-radius: 6px;
      padding: 8px 10px;
      text-decoration: none;
      font-size: 13px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      background: white;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 8px;
      text-align: left;
      vertical-align: top;
      font-size: 13px;
      word-break: break-word;
    }
    th { color: var(--muted); background: #eef3f8; font-weight: 700; }
    td:nth-child(1) { width: 12%; }
    .pass { color: var(--green); font-weight: 700; }
    .fail { color: var(--red); font-weight: 700; }
    .unknown, .review { color: var(--amber); font-weight: 700; }
    .json-box {
      margin-top: 14px;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 360px;
      overflow: auto;
      border: 1px solid var(--line);
      background: #101820;
      color: #e9f2ff;
      border-radius: 8px;
      padding: 12px;
      font-size: 12px;
    }
    .status { color: var(--muted); font-size: 13px; min-height: 18px; }
    @media (max-width: 920px) {
      main { grid-template-columns: 1fr; padding: 14px; }
      header { padding: 14px; align-items: flex-start; flex-direction: column; }
      .summary-grid { grid-template-columns: repeat(2, minmax(130px, 1fr)); }
    }
  </style>
</head>
<body>
  <header>
    <h1>环境检测报告智能识别</h1>
    <div class="status" id="status">就绪</div>
  </header>
  <main>
    <section class="panel">
      <form class="controls" id="uploadForm">
        <label>报告文件
          <input id="files" name="files" type="file" multiple accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp,.pdf" />
        </label>
        <label>执行标准
          <select id="standard" name="standard_key"></select>
        </label>
        <label class="toggle">
          <input id="useAi" name="use_ai" type="checkbox" />
          <span>AI 视觉抽取</span>
        </label>
        <button id="runBtn" type="submit">开始识别</button>
      </form>
    </section>
    <section>
      <div class="summary-grid">
        <div class="metric"><span>合格率</span><strong id="rate">-</strong></div>
        <div class="metric"><span>记录数</span><strong id="total">-</strong></div>
        <div class="metric"><span>超标</span><strong id="fail">-</strong></div>
        <div class="metric"><span>复核</span><strong id="review">-</strong></div>
      </div>
      <div class="summary" id="summary">暂无结果</div>
      <div class="summary warnings" id="warnings"></div>
      <div class="actions" id="downloads"></div>
      <table>
        <thead>
          <tr>
            <th>指标</th><th>检测值</th><th>单位</th><th>限值</th><th>状态</th><th>公式核验</th><th>置信度</th><th>样点</th><th>来源行</th>
          </tr>
        </thead>
        <tbody id="records"></tbody>
      </table>
      <pre class="json-box" id="jsonPreview">{}</pre>
    </section>
  </main>
  <script>
    const statusEl = document.getElementById("status");
    const form = document.getElementById("uploadForm");
    const standard = document.getElementById("standard");
    const runBtn = document.getElementById("runBtn");
    const filesInput = document.getElementById("files");

    async function loadStandards() {
      const res = await fetch("/api/standards");
      const data = await res.json();
      standard.innerHTML = "";
      Object.entries(data).forEach(([key, name]) => {
        const option = document.createElement("option");
        option.value = key;
        option.textContent = name;
        standard.appendChild(option);
      });
    }

    function suggestStandard(files) {
      const names = [...files].map(file => file.name).join(" ").toLowerCase();
      let key = "";
      if (names.includes("废水") || names.includes("污水")) key = "wastewater_custom";
      else if (names.includes("地下水")) key = "groundwater_custom";
      else if (names.includes("地表水")) key = "surface_water_gb3838_subset";
      else if (names.includes("空气") || names.includes("废气")) key = "air_gb3095_2012_grade2";
      if (key && [...standard.options].some(option => option.value === key)) {
        standard.value = key;
        statusEl.textContent = "已根据文件名自动匹配执行标准";
      }
    }

    function setText(id, text) {
      document.getElementById(id).textContent = text;
    }

    function render(result) {
      const stats = result.statistics || {};
      const judged = (stats.pass_count || 0) + (stats.fail_count || 0);
      setText("rate", judged ? `${((stats.compliance_rate || 0) * 100).toFixed(1)}%` : "-");
      setText("total", stats.total_records ?? 0);
      setText("fail", stats.fail_count ?? 0);
      setText("review", stats.review_count ?? 0);
      setText("summary", result.brief_summary || "暂无结果");
      const warningBox = document.getElementById("warnings");
      const warnings = result.warnings || [];
      warningBox.style.display = warnings.length ? "block" : "none";
      warningBox.textContent = warnings.length ? `提示：${warnings.slice(0, 5).join("；")}` : "";

      const downloads = document.getElementById("downloads");
      downloads.innerHTML = "";
      Object.entries(result.downloads || {}).forEach(([label, url]) => {
        if (!url) return;
        const a = document.createElement("a");
        a.href = url;
        a.target = "_blank";
        a.textContent = label;
        downloads.appendChild(a);
      });

      const body = document.getElementById("records");
      body.innerHTML = "";
      (result.records || []).forEach((item) => {
        const tr = document.createElement("tr");
        const status = item.needs_review ? `复核/${item.status}` : item.status;
        const cells = [
          item.normalized_indicator || item.indicator || "",
          item.raw_value || item.value || "",
          item.unit || "",
          `${item.standard_limit || ""} ${item.limit_unit || ""}`.trim(),
          status,
          (item.formula_verification && item.formula_verification.status) || "",
          item.confidence ?? "",
          item.sample_point || "",
          item.source_line || ""
        ];
        cells.forEach((value, index) => {
          const td = document.createElement("td");
          td.textContent = value;
          if (index === 4) td.className = item.needs_review ? "review" : (item.status || "");
          tr.appendChild(td);
        });
        body.appendChild(tr);
      });
      document.getElementById("jsonPreview").textContent = JSON.stringify(result, null, 2);
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const files = filesInput.files;
      if (!files.length) {
        statusEl.textContent = "请选择文件";
        return;
      }
      const data = new FormData();
      [...files].forEach(file => data.append("files", file));
      data.append("standard_key", standard.value);
      data.append("use_ai", document.getElementById("useAi").checked ? "true" : "false");
      runBtn.disabled = true;
      statusEl.textContent = "识别中";
      try {
        const res = await fetch("/api/analyze", { method: "POST", body: data });
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.detail || "识别失败");
        render(payload);
        statusEl.textContent = "完成";
      } catch (error) {
        statusEl.textContent = error.message;
      } finally {
        runBtn.disabled = false;
      }
    });

    filesInput.addEventListener("change", () => suggestStandard(filesInput.files));
    loadStandards().catch((error) => { statusEl.textContent = error.message; });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "standards": len(standards.names())}


@app.get("/api/standards")
def list_standards() -> dict:
    return standards.names()


@app.get("/api/formulas")
def list_formulas() -> dict:
    return {"version": formula_database.data.get("version"), "formulas": formula_database.list_formulas()}


@app.post("/api/formula/verify")
async def verify_formula(payload: Dict[str, Any] = Body(...)) -> dict:
    use_ai = bool(payload.get("use_ai"))
    try:
        machine_result = await run_in_threadpool(formula_verifier.verify, payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ai_review = ""
    if use_ai:
        ai_client = AIClient()
        if ai_client.is_enabled:
            try:
                ai_review = await run_in_threadpool(ai_client.review_formula_verification, payload, machine_result)
            except Exception as exc:
                ai_review = f"AI 复检失败，已保留机器复算结果：{exc}"
        else:
            ai_review = "AI 复检未启用：请配置 AI_API_KEY / AI_BASE_URL / AI_MODEL。"
    return {"machine": machine_result, "ai_review": ai_review}


@app.post("/api/analyze")
async def analyze(
    files: List[UploadFile] = File(...),
    standard_key: str = Form(...),
    use_ai: str = Form("false"),
) -> dict:
    if standard_key not in standards.names():
        raise HTTPException(status_code=400, detail="未知执行标准")
    if not files:
        raise HTTPException(status_code=400, detail="请上传文件")

    upload_dir = UPLOADS_DIR / uuid.uuid4().hex
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []
    try:
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            name = _safe_filename(Path(upload.filename or f"upload{suffix}").name)
            target = upload_dir / name
            with target.open("wb") as fh:
                shutil.copyfileobj(upload.file, fh)
            saved_paths.append(str(target))

        result = await run_in_threadpool(
            _run_pipeline,
            saved_paths,
            standard_key,
            use_ai.lower() in {"1", "true", "yes", "on"},
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _run_pipeline(file_paths: List[str], standard_key: str, use_ai: bool) -> dict:
    pipeline = EnvironmentReportVisionPipeline(standards, EXPORTS_DIR / "vision", AIClient())
    result = pipeline.analyze_files(file_paths, standard_key, use_ai=use_ai)
    data = result.to_dict()
    suggested_standard = _suggest_standard_for_paths(file_paths)
    if suggested_standard and suggested_standard != standard_key:
        data.setdefault("warnings", []).insert(
            0,
            f"文件名更像“{standards.names().get(suggested_standard, suggested_standard)}”，当前选择为“{standards.names().get(standard_key, standard_key)}”；"
            "若合格率异常，请先确认执行标准是否匹配报告类型。",
        )
    stem = result.logical_report_id or uuid.uuid4().hex
    json_path = export_json(data, EXPORTS_DIR / f"{stem}_structured.json")
    html_path = export_structured_html(data, EXPORTS_DIR / f"{stem}_structured.html")
    excel_path = export_structured_excel(data, EXPORTS_DIR / f"{stem}_structured.xlsx")
    data["downloads"] = {
        "JSON": _export_url(json_path),
        "HTML": _export_url(html_path),
        "Excel": _export_url(excel_path),
        "增强 PDF": _export_url(Path(result.enhanced_pdf_path)) if result.enhanced_pdf_path else "",
    }
    return data


def _suggest_standard_for_paths(file_paths: List[str]) -> str:
    joined = " ".join(Path(path).name for path in file_paths).lower()
    if "废水" in joined or "污水" in joined:
        return "wastewater_custom"
    if "地下水" in joined:
        return "groundwater_custom"
    if "地表水" in joined:
        return "surface_water_gb3838_subset"
    if "空气" in joined or "废气" in joined:
        return "air_gb3095_2012_grade2"
    return ""


def _export_url(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(EXPORTS_DIR.resolve())
    except ValueError:
        return ""
    return "/exports/" + relative.as_posix()


def _safe_filename(value: str) -> str:
    stem = Path(value).stem or "upload"
    suffix = Path(value).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "_", stem).strip("_") or "upload"
    return f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
