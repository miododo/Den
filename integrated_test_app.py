from __future__ import annotations

import json
import os
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
RUNTIME_DIR = APP_DIR / "integrated_runtime"
UPLOADS_DIR = RUNTIME_DIR / "uploads"
EXPORTS_DIR = RUNTIME_DIR / "exports"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

standards = StandardsLibrary(CONFIG_PATH)
formula_database = FormulaDatabase(FORMULA_CONFIG_PATH)
formula_verifier = FormulaVerifier(formula_database)

AI_PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "name": "OpenAI / ChatGPT",
        "api_style": "openai",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-5.5",
        "requires_key": True,
        "note": "官方 API 模型会按账号权限变化；以读取模型列表为最终准。",
        "models": [
            {"id": "gpt-5.5", "name": "GPT-5.5", "vision": True, "status": "官方当前"},
            {"id": "gpt-5.4", "name": "GPT-5.4", "vision": True, "status": "官方当前"},
            {"id": "gpt-5.4-mini", "name": "GPT-5.4 mini", "vision": True, "status": "官方当前"},
            {"id": "gpt-5.4-nano", "name": "GPT-5.4 nano", "vision": True, "status": "官方当前"},
            {"id": "gpt-5.2", "name": "GPT-5.2", "vision": True, "status": "官方稳定"},
            {"id": "gpt-5.2-pro", "name": "GPT-5.2 Pro", "vision": True, "status": "官方稳定"},
            {"id": "gpt-5.1", "name": "GPT-5.1", "vision": True, "status": "官方稳定"},
            {"id": "gpt-4o", "name": "GPT-4o", "vision": True, "status": "视觉稳定"},
            {"id": "gpt-4o-mini", "name": "GPT-4o mini", "vision": True, "status": "视觉稳定"},
            {"id": "gpt-4.1", "name": "GPT-4.1", "vision": True, "status": "官方稳定"},
            {"id": "gpt-4.1-mini", "name": "GPT-4.1 mini", "vision": True, "status": "官方稳定"},
            {"id": "o3", "name": "o3", "vision": True, "status": "推理"},
            {"id": "o4-mini", "name": "o4-mini", "vision": True, "status": "推理"},
        ],
    },
    "deepseek": {
        "name": "DeepSeek",
        "api_style": "openai",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-v4-flash",
        "requires_key": True,
        "note": "官方当前列表以 V4 Flash / V4 Pro 为主，旧兼容名不再放入推荐。",
        "models": [
            {"id": "deepseek-v4-flash", "name": "DeepSeek V4 Flash", "vision": False, "status": "官方当前"},
            {"id": "deepseek-v4-pro", "name": "DeepSeek V4 Pro", "vision": False, "status": "官方当前"},
        ],
    },
    "gemini": {
        "name": "Google Gemini（OpenAI 兼容）",
        "api_style": "openai",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-3.1-flash-lite",
        "requires_key": True,
        "note": "使用 Gemini OpenAI-compatible 端点；预览模型可能需要账号权限。",
        "models": [
            {"id": "gemini-3.1-flash-lite", "name": "Gemini 3.1 Flash-Lite", "vision": True, "status": "官方当前"},
            {"id": "gemini-3.1-pro-preview", "name": "Gemini 3.1 Pro", "vision": True, "status": "预览"},
            {"id": "gemini-3-flash-preview", "name": "Gemini 3 Flash", "vision": True, "status": "预览"},
            {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "vision": True, "status": "官方稳定"},
            {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "vision": True, "status": "官方稳定"},
            {"id": "gemini-2.5-flash-lite", "name": "Gemini 2.5 Flash-Lite", "vision": True, "status": "官方稳定"},
        ],
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "api_style": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "default_model": "claude-sonnet-4-0",
        "requires_key": True,
        "note": "Claude 走 Anthropic Messages API，不是 OpenAI 兼容协议。",
        "models": [
            {"id": "claude-opus-4-1", "name": "Claude Opus 4.1", "vision": True, "status": "官方当前"},
            {"id": "claude-opus-4-0", "name": "Claude Opus 4", "vision": True, "status": "官方稳定"},
            {"id": "claude-sonnet-4-0", "name": "Claude Sonnet 4", "vision": True, "status": "官方稳定"},
            {"id": "claude-3-7-sonnet-latest", "name": "Claude Sonnet 3.7", "vision": True, "status": "官方稳定"},
            {"id": "claude-3-5-haiku-latest", "name": "Claude Haiku 3.5", "vision": True, "status": "官方稳定"},
        ],
    },
    "dashscope": {
        "name": "阿里云百炼 / 通义千问",
        "api_style": "openai",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen3.6-plus",
        "requires_key": True,
        "note": "百炼 OpenAI 兼容模式；OCR/图片优先选 Qwen-VL 或 Qwen-OCR。",
        "models": [
            {"id": "qwen3.6-plus", "name": "Qwen3.6 Plus", "vision": True, "status": "官方当前"},
            {"id": "qwen3.6-flash", "name": "Qwen3.6 Flash", "vision": True, "status": "官方当前"},
            {"id": "qwen3.6-max-preview", "name": "Qwen3.6 Max", "vision": True, "status": "预览"},
            {"id": "qwen3-vl-plus", "name": "Qwen3-VL Plus", "vision": True, "status": "视觉"},
            {"id": "qwen3-vl-flash", "name": "Qwen3-VL Flash", "vision": True, "status": "视觉"},
            {"id": "qwen-vl-ocr-latest", "name": "Qwen OCR", "vision": True, "status": "OCR"},
            {"id": "qwen-plus", "name": "Qwen Plus", "vision": False, "status": "兼容稳定"},
        ],
    },
    "siliconflow": {
        "name": "硅基流动 SiliconFlow",
        "api_style": "openai",
        "base_url": "https://api.siliconflow.cn/v1",
        "default_model": "deepseek-ai/DeepSeek-V3",
        "requires_key": True,
        "note": "聚合平台模型随平台上下架变化，建议用“读取模型列表”确认。",
        "models": [
            {"id": "deepseek-ai/DeepSeek-V3", "name": "DeepSeek-V3", "vision": False, "status": "聚合示例"},
            {"id": "deepseek-ai/DeepSeek-R1", "name": "DeepSeek-R1", "vision": False, "status": "聚合示例"},
            {"id": "Qwen/Qwen3-235B-A22B", "name": "Qwen3-235B-A22B", "vision": False, "status": "聚合示例"},
            {"id": "Qwen/Qwen2.5-VL-72B-Instruct", "name": "Qwen2.5-VL 72B", "vision": True, "status": "视觉示例"},
        ],
    },
    "moonshot": {
        "name": "Moonshot / Kimi",
        "api_style": "openai",
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-32k",
        "requires_key": True,
        "note": "Kimi 新模型名称更新频繁；旧 moonshot-v1 作为稳定兼容项保留。",
        "models": [
            {"id": "moonshot-v1-8k", "name": "Moonshot v1 8K", "vision": False, "status": "兼容稳定"},
            {"id": "moonshot-v1-32k", "name": "Moonshot v1 32K", "vision": False, "status": "兼容稳定"},
            {"id": "moonshot-v1-128k", "name": "Moonshot v1 128K", "vision": False, "status": "兼容稳定"},
            {"id": "kimi-k2-0711-preview", "name": "Kimi K2", "vision": False, "status": "预览/需权限"},
        ],
    },
    "zhipu": {
        "name": "智谱 GLM",
        "api_style": "openai",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-5.1",
        "requires_key": True,
        "note": "GLM 新版本会分批开放，视觉复核优先选 GLM-5V。",
        "models": [
            {"id": "glm-5.1", "name": "GLM-5.1", "vision": False, "status": "官方当前"},
            {"id": "glm-5", "name": "GLM-5", "vision": False, "status": "官方当前"},
            {"id": "glm-4.7", "name": "GLM-4.7", "vision": False, "status": "官方稳定"},
            {"id": "glm-4.7-flashx", "name": "GLM-4.7 FlashX", "vision": False, "status": "官方稳定"},
            {"id": "glm-5v-turbo", "name": "GLM-5V Turbo", "vision": True, "status": "视觉"},
        ],
    },
    "baidu": {
        "name": "百度千帆 / 文心",
        "api_style": "openai",
        "base_url": "https://qianfan.baidubce.com/v2",
        "default_model": "ernie-4.5-turbo-128k",
        "requires_key": True,
        "note": "千帆 v2 兼容 OpenAI；模型可用性受地域和账号开通状态影响。",
        "models": [
            {"id": "ernie-4.5-turbo-128k", "name": "ERNIE 4.5 Turbo 128K", "vision": False, "status": "官方稳定"},
            {"id": "ernie-x1-turbo-32k", "name": "ERNIE X1 Turbo 32K", "vision": False, "status": "官方稳定"},
            {"id": "ernie-4.5-turbo-vl", "name": "ERNIE 4.5 Turbo VL", "vision": True, "status": "视觉"},
            {"id": "ernie-5.0-thinking-preview", "name": "ERNIE 5.0 Thinking", "vision": False, "status": "预览/需权限"},
        ],
    },
    "tencent": {
        "name": "腾讯混元",
        "api_style": "openai",
        "base_url": "https://api.hunyuan.cloud.tencent.com/v1",
        "default_model": "hunyuan-turbos-latest",
        "requires_key": True,
        "note": "腾讯混元 OpenAI 兼容端点，latest 别名由官方动态指向。",
        "models": [
            {"id": "hunyuan-turbos-latest", "name": "Hunyuan TurboS", "vision": False, "status": "官方稳定"},
            {"id": "hunyuan-t1-latest", "name": "Hunyuan T1", "vision": False, "status": "推理"},
            {"id": "hunyuan-vision", "name": "Hunyuan Vision", "vision": True, "status": "视觉"},
        ],
    },
    "volcengine": {
        "name": "火山方舟 Ark",
        "api_style": "openai",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "default_model": "ep-xxxxxxxx",
        "requires_key": True,
        "note": "方舟生产调用通常填推理接入点 ID：ep-...；不要直接保留占位符。",
        "models": [
            {"id": "ep-xxxxxxxx", "name": "你的方舟接入点 ID", "vision": True, "status": "必须替换"},
            {"id": "doubao-seed-1-6-250615", "name": "Doubao Seed 1.6", "vision": False, "status": "需账号开通"},
            {"id": "doubao-seed-1-6-thinking-250715", "name": "Doubao Seed 1.6 Thinking", "vision": False, "status": "需账号开通"},
            {"id": "doubao-seed-1-6-vision-250815", "name": "Doubao Seed 1.6 Vision", "vision": True, "status": "视觉/需账号"},
        ],
    },
    "minimax": {
        "name": "MiniMax",
        "api_style": "openai",
        "base_url": "https://api.minimaxi.com/v1",
        "default_model": "MiniMax-M2.5",
        "requires_key": True,
        "note": "MiniMax 新平台模型更新较快；若官方账号返回专属模型，请用读取模型列表覆盖。",
        "models": [
            {"id": "MiniMax-M2.7", "name": "MiniMax M2.7", "vision": False, "status": "官方当前"},
            {"id": "MiniMax-M2.5", "name": "MiniMax M2.5", "vision": False, "status": "官方当前"},
            {"id": "MiniMax-Text-01", "name": "MiniMax Text", "vision": False, "status": "兼容稳定"},
            {"id": "MiniMax-VL-01", "name": "MiniMax VL", "vision": True, "status": "视觉"},
        ],
    },
    "baichuan": {
        "name": "百川智能",
        "api_style": "openai",
        "base_url": "https://api.baichuan-ai.com/v1",
        "default_model": "Baichuan4",
        "requires_key": True,
        "note": "百川商业 API 更新较慢，建议用自定义模型或读取模型列表确认账号可用项。",
        "models": [
            {"id": "Baichuan4", "name": "Baichuan4", "vision": False, "status": "需账号确认"},
            {"id": "Baichuan4-Turbo", "name": "Baichuan4 Turbo", "vision": False, "status": "需账号确认"},
        ],
    },
    "yi": {
        "name": "零一万物 01.AI",
        "api_style": "openai",
        "base_url": "https://api.lingyiwanwu.com/v1",
        "default_model": "yi-large",
        "requires_key": True,
        "note": "01.AI 商业 API 可用性以账号后台为准。",
        "models": [
            {"id": "yi-large", "name": "Yi Large", "vision": False, "status": "需账号确认"},
            {"id": "yi-lightning", "name": "Yi Lightning", "vision": False, "status": "需账号确认"},
            {"id": "yi-vision", "name": "Yi Vision", "vision": True, "status": "视觉/需确认"},
        ],
    },
    "mistral": {
        "name": "Mistral AI",
        "api_style": "openai",
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-large-latest",
        "requires_key": True,
        "note": "Mistral 使用 latest 别名降低版本号过期风险。",
        "models": [
            {"id": "mistral-large-latest", "name": "Mistral Large", "vision": False, "status": "官方稳定"},
            {"id": "mistral-small-latest", "name": "Mistral Small", "vision": False, "status": "官方稳定"},
            {"id": "pixtral-large-latest", "name": "Pixtral Large", "vision": True, "status": "视觉"},
            {"id": "codestral-latest", "name": "Codestral", "vision": False, "status": "代码"},
        ],
    },
    "xai": {
        "name": "xAI Grok",
        "api_style": "openai",
        "base_url": "https://api.x.ai/v1",
        "default_model": "grok-4",
        "requires_key": True,
        "note": "xAI 模型权限按控制台开通状态变化。",
        "models": [
            {"id": "grok-4", "name": "Grok 4", "vision": False, "status": "官方当前"},
            {"id": "grok-3", "name": "Grok 3", "vision": False, "status": "官方稳定"},
            {"id": "grok-2-vision-1212", "name": "Grok Vision", "vision": True, "status": "视觉"},
        ],
    },
    "groq": {
        "name": "Groq",
        "api_style": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "requires_key": True,
        "note": "Groq 模型上架节奏快，建议读取 /models。",
        "models": [
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "vision": False, "status": "官方稳定"},
            {"id": "meta-llama/llama-4-scout-17b-16e-instruct", "name": "Llama 4 Scout", "vision": True, "status": "视觉"},
            {"id": "openai/gpt-oss-120b", "name": "GPT-OSS 120B", "vision": False, "status": "官方稳定"},
            {"id": "moonshotai/kimi-k2-instruct", "name": "Kimi K2 Instruct", "vision": False, "status": "官方稳定"},
        ],
    },
    "openrouter": {
        "name": "OpenRouter",
        "api_style": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "openai/gpt-4o",
        "requires_key": True,
        "note": "OpenRouter 是聚合网关，具体 slug 和价格以读取模型列表为准。",
        "models": [
            {"id": "openai/gpt-4o", "name": "OpenAI GPT-4o", "vision": True, "status": "网关示例"},
            {"id": "anthropic/claude-sonnet-4", "name": "Claude Sonnet 4", "vision": True, "status": "网关示例"},
            {"id": "google/gemini-2.5-pro", "name": "Gemini 2.5 Pro", "vision": True, "status": "网关示例"},
            {"id": "deepseek/deepseek-r1", "name": "DeepSeek R1", "vision": False, "status": "网关示例"},
        ],
    },
    "perplexity": {
        "name": "Perplexity",
        "api_style": "openai",
        "base_url": "https://api.perplexity.ai",
        "default_model": "sonar-pro",
        "requires_key": True,
        "note": "Perplexity Sonar 适合联网问答，报告图片视觉抽取不建议作为首选。",
        "models": [
            {"id": "sonar-pro", "name": "Sonar Pro", "vision": False, "status": "官方稳定"},
            {"id": "sonar-reasoning-pro", "name": "Sonar Reasoning Pro", "vision": False, "status": "推理"},
            {"id": "sonar", "name": "Sonar", "vision": False, "status": "官方稳定"},
        ],
    },
    "together": {
        "name": "Together AI",
        "api_style": "openai",
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "requires_key": True,
        "note": "Together 是模型托管平台，具体可用项以 /models 返回为准。",
        "models": [
            {"id": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "name": "Llama 3.3 70B", "vision": False, "status": "托管示例"},
            {"id": "Qwen/Qwen3-235B-A22B-fp8-tput", "name": "Qwen3 235B", "vision": False, "status": "托管示例"},
            {"id": "deepseek-ai/DeepSeek-R1", "name": "DeepSeek R1", "vision": False, "status": "托管示例"},
        ],
    },
    "fireworks": {
        "name": "Fireworks AI",
        "api_style": "openai",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "default_model": "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "requires_key": True,
        "note": "Fireworks 模型路径较长，复制控制台模型 ID 最稳。",
        "models": [
            {"id": "accounts/fireworks/models/llama-v3p3-70b-instruct", "name": "Llama 3.3 70B", "vision": False, "status": "托管示例"},
            {"id": "accounts/fireworks/models/deepseek-r1", "name": "DeepSeek R1", "vision": False, "status": "托管示例"},
            {"id": "accounts/fireworks/models/qwen2p5-vl-72b-instruct", "name": "Qwen2.5 VL 72B", "vision": True, "status": "视觉示例"},
        ],
    },
    "azure_openai": {
        "name": "Azure OpenAI",
        "api_style": "azure",
        "base_url": "https://YOUR_RESOURCE.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT/chat/completions?api-version=2024-10-21",
        "default_model": "YOUR_DEPLOYMENT",
        "requires_key": True,
        "note": "Azure 填部署名，不直接填基础模型名。",
        "models": [
            {"id": "YOUR_DEPLOYMENT", "name": "Azure 部署名", "vision": True, "status": "必须替换"},
        ],
    },
    "ollama": {
        "name": "Ollama 本地",
        "api_style": "openai",
        "base_url": "http://127.0.0.1:11434/v1",
        "default_model": "qwen2.5vl",
        "requires_key": False,
        "note": "本地模型无需 Key，但必须已在 Ollama 拉取并启动。",
        "models": [
            {"id": "qwen2.5vl", "name": "Qwen2.5-VL", "vision": True, "status": "本地/需拉取"},
            {"id": "llama3.2-vision", "name": "Llama 3.2 Vision", "vision": True, "status": "本地/需拉取"},
            {"id": "qwen2.5", "name": "Qwen2.5", "vision": False, "status": "本地/需拉取"},
            {"id": "deepseek-r1", "name": "DeepSeek R1", "vision": False, "status": "本地/需拉取"},
        ],
    },
    "lmstudio": {
        "name": "LM Studio 本地",
        "api_style": "openai",
        "base_url": "http://127.0.0.1:1234/v1",
        "default_model": "local-model",
        "requires_key": False,
        "note": "LM Studio 选择当前加载模型，模型名可由 /models 自动读取。",
        "models": [
            {"id": "local-model", "name": "LM Studio 当前加载模型", "vision": True, "status": "本地"},
        ],
    },
    "custom": {
        "name": "自定义 OpenAI-compatible",
        "api_style": "openai",
        "base_url": "",
        "default_model": "",
        "requires_key": False,
        "note": "用于 One API、New API、LiteLLM、公司私有网关等兼容接口。",
        "models": [],
    },
}

app = FastAPI(title="Env AI Integrated Test Console", version="0.3.0")
app.mount("/exports", StaticFiles(directory=EXPORTS_DIR), name="exports")


def _load_index_html():
    """Load HTML template from file (cached on first read)."""
    cache_key = "_index_html_cache"
    if hasattr(_load_index_html, cache_key):
        return getattr(_load_index_html, cache_key)
    html_path = Path(__file__).resolve().parent / "templates" / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    setattr(_load_index_html, cache_key, html)
    return html


_INDEX_HTML_DEPRECATED = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>环境检测智能识别与公式核验测试台</title>
  <style>
    :root {
      --ink: #17212b;
      --muted: #627084;
      --line: #d8e1ea;
      --panel: #f7fafc;
      --blue: #1565c0;
      --green: #18794e;
      --red: #c62828;
      --amber: #9a5b00;
      --bg: #ffffff;
      --soft: #eef4fb;
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
      padding: 16px 22px;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
    }
    h1 { margin: 0; font-size: 22px; letter-spacing: 0; }
    h2 { margin: 0 0 10px; font-size: 16px; }
    main {
      padding: 18px 22px 26px;
      display: grid;
      gap: 16px;
    }
    .top-grid {
      display: grid;
      grid-template-columns: minmax(360px, 0.95fr) minmax(420px, 1.05fr);
      gap: 16px;
      align-items: start;
    }
    section {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      padding: 14px;
      min-width: 0;
    }
    .controls {
      display: grid;
      gap: 12px;
    }
    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      font: inherit;
      color: var(--ink);
      background: white;
    }
    textarea {
      min-height: 130px;
      resize: vertical;
      font-family: Consolas, "Microsoft YaHei", monospace;
      font-size: 12px;
    }
    button {
      height: 40px;
      border: 0;
      border-radius: 6px;
      background: var(--blue);
      color: white;
      padding: 0 14px;
      font: inherit;
      cursor: pointer;
    }
    button.secondary {
      background: white;
      color: var(--blue);
      border: 1px solid var(--line);
    }
    button:disabled { opacity: .55; cursor: default; }
    .inline {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .inline label {
      display: flex;
      align-items: center;
      gap: 6px;
      color: var(--ink);
    }
    input[type="checkbox"] { width: auto; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(130px, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: var(--panel);
      min-height: 72px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    .metric strong { font-size: 21px; }
    .summary, .warnings {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      line-height: 1.55;
      background: white;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .warnings {
      display: none;
      border-color: #f0c36d;
      background: #fff8e6;
      color: #5f3b00;
    }
    .tabs {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }
    .tab-btn {
      background: white;
      color: var(--blue);
      border: 1px solid var(--line);
    }
    .tab-btn.active {
      background: var(--blue);
      color: white;
      border-color: var(--blue);
    }
    .tab { display: none; }
    .tab.active { display: block; }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      border: 1px solid var(--line);
      background: white;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 8px;
      vertical-align: top;
      text-align: left;
      font-size: 13px;
      word-break: break-word;
    }
    th {
      color: var(--muted);
      background: var(--soft);
      font-weight: 700;
    }
    pre {
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      max-height: 520px;
      overflow: auto;
      background: #101820;
      color: #dce7f3;
      font-size: 12px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .pass { color: var(--green); font-weight: 700; }
    .fail { color: var(--red); font-weight: 700; }
    .review, .unknown { color: var(--amber); font-weight: 700; }
    .downloads {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .downloads a {
      color: var(--blue);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      text-decoration: none;
      font-size: 13px;
    }
    .formula-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(180px, 1fr));
      gap: 10px;
    }
    .input-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(160px, 1fr));
      gap: 10px;
    }
    .status { color: var(--muted); min-height: 20px; font-size: 13px; }
    @media (max-width: 980px) {
      header { align-items: flex-start; flex-direction: column; }
      .top-grid, .formula-grid, .input-grid { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, minmax(130px, 1fr)); }
    }
  </style>
</head>
<body>
  <header>
    <h1>环境检测智能识别与公式核验测试台</h1>
    <div class="status" id="status">就绪</div>
  </header>
  <main>
    <section>
      <h2>AI API 配置</h2>
      <div class="controls">
        <div class="formula-grid">
          <label>服务商
            <select id="aiProvider"></select>
          </label>
          <label>模型预设
            <select id="aiModelSelect"></select>
          </label>
        </div>
        <div class="formula-grid">
          <label>模型名 / 部署名
            <input id="aiModel" type="text" placeholder="可选预设，也可手动输入私有模型名" />
          </label>
          <label>Base URL
            <input id="aiBaseUrl" type="text" placeholder="自动带入，也可填写你的私有代理地址" />
          </label>
        </div>
        <div class="formula-grid">
          <label>API Key
            <input id="aiApiKey" type="password" autocomplete="off" placeholder="只保存在本机浏览器，不写入后台文件" />
          </label>
          <label>连接状态
            <input id="aiConfigStatus" type="text" readonly value="未测试" />
          </label>
        </div>
        <div class="inline">
          <button class="secondary" id="saveAiConfigBtn" type="button">保存到本机浏览器</button>
          <button class="secondary" id="clearAiConfigBtn" type="button">清空本机配置</button>
          <button class="secondary" id="refreshAiModelsBtn" type="button">读取模型列表</button>
          <button class="secondary" id="testAiConfigBtn" type="button">测试 API</button>
          <span class="status">带“视觉”的模型会优先读 PDF 图片；文本模型会自动降级为 OCR 文本抽取。</span>
        </div>
      </div>
    </section>

    <div class="top-grid">
      <section>
        <h2>报告识别</h2>
        <form class="controls" id="uploadForm">
          <label>报告 PDF / 图片
            <input id="files" name="files" type="file" multiple accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp,.pdf" />
          </label>
          <label>执行标准
            <select id="standard" name="standard_key"></select>
          </label>
          <div class="inline">
            <label><input id="useAi" name="use_ai" type="checkbox" /> AI 智能抽取</label>
            <button id="runBtn" type="submit">开始识别</button>
          </div>
        </form>
      </section>

      <section>
        <h2>公式复算与 AI 复检</h2>
        <div class="controls">
          <div class="formula-grid">
            <label>公式/方法
              <select id="formulaSelect"></select>
            </label>
            <label>报告值
              <input id="reportedValue" type="number" step="any" placeholder="可为空，仅机器计算" />
            </label>
          </div>
          <div class="input-grid" id="formulaInputs"></div>
          <div class="inline">
            <label><input id="formulaAi" type="checkbox" /> AI 复检</label>
            <button id="verifyBtn" type="button">执行公式核验</button>
            <button class="secondary" id="fillDemoBtn" type="button">填入示例</button>
          </div>
        </div>
      </section>
    </div>

    <section>
      <div class="metrics">
        <div class="metric"><span>合格率</span><strong id="rate">-</strong></div>
        <div class="metric"><span>记录数</span><strong id="total">0</strong></div>
        <div class="metric"><span>超标</span><strong id="fail">0</strong></div>
        <div class="metric"><span>复核</span><strong id="review">0</strong></div>
      </div>
    </section>

    <section>
      <h2>简报与下载</h2>
      <div class="summary" id="summary">暂无结果</div>
      <div class="warnings" id="warnings"></div>
      <div class="downloads" id="downloads"></div>
    </section>

    <section>
      <div class="tabs">
        <button class="tab-btn active" data-tab="recordsTab" type="button">结构化结果</button>
        <button class="tab-btn" data-tab="formulaTab" type="button">公式核验结果</button>
        <button class="tab-btn" data-tab="libraryTab" type="button">公式库</button>
        <button class="tab-btn" data-tab="jsonTab" type="button">完整 JSON</button>
      </div>

      <div class="tab active" id="recordsTab">
        <table>
          <thead>
            <tr>
              <th>指标</th><th>检测值</th><th>单位</th><th>限值</th><th>状态</th><th>数据库匹配</th><th>公式核验</th><th>置信度</th><th>样品/点位</th><th>检测日期</th><th>来源行</th>
            </tr>
          </thead>
          <tbody id="records"></tbody>
        </table>
      </div>

      <div class="tab" id="formulaTab">
        <pre id="formulaResult">{}</pre>
      </div>

      <div class="tab" id="libraryTab">
        <table>
          <thead><tr><th>指标</th><th>方法</th><th>标准号</th><th>所需参数</th><th>来源文件</th></tr></thead>
          <tbody id="formulaLibrary"></tbody>
        </table>
      </div>

      <div class="tab" id="jsonTab">
        <pre id="jsonPreview">{}</pre>
      </div>
    </section>
  </main>

  <script>
    const statusEl = document.getElementById("status");
    const filesInput = document.getElementById("files");
    const standard = document.getElementById("standard");
    const form = document.getElementById("uploadForm");
    const runBtn = document.getElementById("runBtn");
    const formulaSelect = document.getElementById("formulaSelect");
    const formulaInputs = document.getElementById("formulaInputs");
    const formulaResult = document.getElementById("formulaResult");
    const aiProvider = document.getElementById("aiProvider");
    const aiModelSelect = document.getElementById("aiModelSelect");
    const aiBaseUrl = document.getElementById("aiBaseUrl");
    const aiModel = document.getElementById("aiModel");
    const aiApiKey = document.getElementById("aiApiKey");
    const aiConfigStatus = document.getElementById("aiConfigStatus");
    let formulas = [];
    let aiProviders = {};

    function setText(id, value) {
      document.getElementById(id).textContent = value;
    }

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function collectAiConfig() {
      return {
        ai_provider: aiProvider.value,
        ai_base_url: aiBaseUrl.value.trim(),
        ai_model: aiModel.value.trim(),
        ai_api_key: aiApiKey.value.trim()
      };
    }

    function appendAiConfig(formData) {
      const config = collectAiConfig();
      Object.entries(config).forEach(([key, value]) => formData.append(key, value));
    }

    async function loadAiProviders() {
      const res = await fetch("/api/ai/providers");
      const data = await res.json();
      aiProviders = data.providers || {};
      aiProvider.innerHTML = "";
      Object.entries(aiProviders).forEach(([key, item]) => {
        const option = document.createElement("option");
        option.value = key;
        option.textContent = item.name || key;
        aiProvider.appendChild(option);
      });
    }

    function currentAiPreset() {
      return aiProviders[aiProvider.value] || aiProviders.custom || { models: [] };
    }

    function selectedModelPreset() {
      const preset = currentAiPreset();
      return (preset.models || []).find(item => item.id === aiModel.value.trim()) || {};
    }

    function updateAiCapabilityHint() {
      const item = selectedModelPreset();
      if (item.id && item.vision === false && document.getElementById("useAi").checked) {
        aiConfigStatus.value = "当前模型偏文本，视觉抽取建议换视觉模型";
      }
    }

    function renderAiModelOptions(currentModel = "") {
      const preset = currentAiPreset();
      const models = preset.models || [];
      aiModelSelect.innerHTML = "";
      models.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.id;
        const tags = [];
        if (item.vision) tags.push("视觉");
        if (item.status) tags.push(item.status);
        if (item.note) tags.push(item.note);
        option.textContent = `${item.name || item.id}${tags.length ? " / " + tags.join(" / ") : ""}`;
        aiModelSelect.appendChild(option);
      });
      const custom = document.createElement("option");
      custom.value = "__custom__";
      custom.textContent = "自定义模型名";
      aiModelSelect.appendChild(custom);
      const matched = models.some(item => item.id === currentModel);
      aiModelSelect.value = matched ? currentModel : "__custom__";
      updateAiCapabilityHint();
    }

    function applyAiPreset(force = false) {
      const preset = currentAiPreset();
      const defaultModel = preset.default_model || ((preset.models || [])[0] || {}).id || "";
      if (force || !aiBaseUrl.value.trim()) aiBaseUrl.value = preset.base_url || "";
      if (force || !aiModel.value.trim()) aiModel.value = defaultModel;
      renderAiModelOptions(aiModel.value.trim());
      aiConfigStatus.value = preset.note || (preset.requires_key === false ? "本地/可免 Key，建议先测试" : "未测试");
    }

    function saveAiConfig() {
      localStorage.setItem("envAiValidator.aiConfig", JSON.stringify(collectAiConfig()));
      setStatus("AI API 配置已保存到本机浏览器");
    }

    function loadAiConfig() {
      const raw = localStorage.getItem("envAiValidator.aiConfig");
      if (!raw) {
        applyAiPreset(true);
        return;
      }
      try {
        const config = JSON.parse(raw);
        aiProvider.value = config.ai_provider || "openai";
        if (!aiProvider.value) aiProvider.value = "custom";
        aiBaseUrl.value = config.ai_base_url || "";
        aiModel.value = config.ai_model || "";
        aiApiKey.value = config.ai_api_key || "";
        renderAiModelOptions(aiModel.value.trim());
      } catch {
        applyAiPreset(true);
      }
    }

    function clearAiConfig() {
      localStorage.removeItem("envAiValidator.aiConfig");
      aiApiKey.value = "";
      applyAiPreset(true);
      setStatus("已清空本机 AI API 配置");
    }

    async function refreshAiModels() {
      const config = collectAiConfig();
      aiConfigStatus.value = "读取中...";
      setStatus("正在读取模型列表");
      try {
        const res = await fetch("/api/ai/models", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(config)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "读取模型列表失败");
        const preset = currentAiPreset();
        preset.models = data.models || preset.models || [];
        renderAiModelOptions(aiModel.value.trim());
        aiConfigStatus.value = `已读取 ${preset.models.length} 个模型`;
        setStatus("模型列表已更新");
      } catch (error) {
        aiConfigStatus.value = "读取失败";
        setStatus(error.message);
      }
    }

    async function testAiConfig() {
      const config = collectAiConfig();
      aiConfigStatus.value = "测试中...";
      setStatus("正在测试 AI API");
      try {
        const res = await fetch("/api/ai/diagnose", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(config)
        });
        const data = await res.json();
        if (!res.ok || data.ok === false) throw new Error((data.errors || [data.detail || "API 测试失败"]).join("；"));
        const visionText = data.vision_ok ? "支持视觉" : "不支持视觉/将 OCR 降级";
        aiConfigStatus.value = `连接成功，${visionText}`;
        const modelText = data.model_available === false ? "；模型列表未直接命中，已通过实际调用验证" : "";
        const warningText = (data.warnings || []).length ? `；提示：${data.warnings.join("；")}` : "";
        setStatus(`AI API 诊断完成：文本调用正常，${visionText}${modelText}${warningText}`);
      } catch (error) {
        aiConfigStatus.value = "连接失败";
        setStatus(error.message);
      }
    }

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

    async function loadFormulas() {
      const res = await fetch("/api/formulas");
      const data = await res.json();
      formulas = data.formulas || [];
      formulaSelect.innerHTML = "";
      formulas.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.id;
        option.textContent = `${item.indicator} - ${item.method_standard} - ${item.method_name}`;
        formulaSelect.appendChild(option);
      });
      renderFormulaLibrary();
      renderFormulaInputs();
    }

    function selectedFormula() {
      return formulas.find(item => item.id === formulaSelect.value) || formulas[0] || {};
    }

    function renderFormulaInputs() {
      const item = selectedFormula();
      formulaInputs.innerHTML = "";
      (item.required_inputs || []).forEach((name) => {
        const label = document.createElement("label");
        label.textContent = name;
        const input = document.createElement("input");
        input.type = "number";
        input.step = "any";
        input.dataset.inputName = name;
        label.appendChild(input);
        formulaInputs.appendChild(label);
      });
    }

    function renderFormulaLibrary() {
      const body = document.getElementById("formulaLibrary");
      body.innerHTML = "";
      formulas.forEach((item) => {
        const tr = document.createElement("tr");
        [
          item.indicator || "",
          item.method_name || "",
          item.method_standard || "",
          (item.required_inputs || []).join(", "),
          (item.source_files || []).join("\\n")
        ].forEach((value) => {
          const td = document.createElement("td");
          td.textContent = value;
          tr.appendChild(td);
        });
        body.appendChild(tr);
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
        setStatus("已根据文件名自动匹配执行标准");
      }
    }

    function renderAnalyze(result) {
      const stats = result.statistics || {};
      const judged = (stats.pass_count || 0) + (stats.fail_count || 0);
      setText("rate", judged ? `${((stats.compliance_rate || 0) * 100).toFixed(1)}%` : "-");
      setText("total", stats.total_records ?? 0);
      setText("fail", stats.fail_count ?? 0);
      setText("review", stats.review_count ?? 0);
      setText("summary", result.brief_summary || "暂无简报");

      const warnings = document.getElementById("warnings");
      const warningText = (result.warnings || []).join("\\n");
      warnings.textContent = warningText;
      warnings.style.display = warningText ? "block" : "none";

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
        const formulaStatus = (item.formula_verification && item.formula_verification.status) || "";
        const db = item.database_match || {};
        const dbStatus = db.indicator_matched ? "标准库已匹配" : (db.reason || "数据库未匹配到对应依据");
        const cells = [
          item.normalized_indicator || item.indicator || "",
          item.raw_value || item.value || "",
          item.unit || "",
          `${item.standard_limit || ""} ${item.limit_unit || ""}`.trim(),
          status,
          dbStatus,
          formulaStatus,
          item.confidence ?? "",
          [item.sample_id || "", item.sample_point || ""].filter(Boolean).join(" / "),
          item.detection_date || item.sample_time || "",
          item.source_line || ""
        ];
        cells.forEach((value, index) => {
          const td = document.createElement("td");
          td.textContent = value;
          if (index === 4) td.className = item.needs_review ? "review" : (item.status || "");
          if (index === 5 && !db.indicator_matched) td.className = "review";
          if (index === 6 && formulaStatus.includes("missing")) td.className = "review";
          tr.appendChild(td);
        });
        body.appendChild(tr);
      });
      document.getElementById("jsonPreview").textContent = JSON.stringify(result, null, 2);
    }

    function collectFormulaInputs() {
      const inputs = {};
      formulaInputs.querySelectorAll("input[data-input-name]").forEach((input) => {
        if (input.value !== "") inputs[input.dataset.inputName] = Number(input.value);
      });
      return inputs;
    }

    function fillDemo() {
      const item = selectedFormula();
      const samples = {
        cod_dichromate_hj828_2017: {
          reportedValue: 200,
          inputs: {
            c_ferrous_ammonium_sulfate_mol_l: 0.25,
            blank_volume_ml: 10,
            sample_volume_titrant_ml: 8,
            sample_volume_ml: 20,
            dilution_factor: 1
          }
        },
        nh3n_salicylic_hj536_2009: {
          reportedValue: 0.5,
          inputs: { absorbance: 0.26, blank_absorbance: 0.01, slope: 0.5, intercept: 0, sample_volume_ml: 1 }
        },
        ph_electrode_hj1147_2020: {
          reportedValue: 7.2,
          inputs: { ph_value: 7.2 }
        }
      };
      const sample = samples[item.id] || { reportedValue: "", inputs: {} };
      document.getElementById("reportedValue").value = sample.reportedValue;
      formulaInputs.querySelectorAll("input[data-input-name]").forEach((input) => {
        input.value = sample.inputs[input.dataset.inputName] ?? "";
      });
    }

    async function verifyFormula() {
      const item = selectedFormula();
      const reportedValue = document.getElementById("reportedValue").value;
      const payload = {
        indicator: item.indicator,
        method_id: item.id,
        reported_value: reportedValue === "" ? null : Number(reportedValue),
        use_ai: document.getElementById("formulaAi").checked,
        inputs: collectFormulaInputs(),
        ...collectAiConfig()
      };
      setStatus("公式核验中");
      document.getElementById("verifyBtn").disabled = true;
      try {
        const res = await fetch("/api/formula/verify", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "公式核验失败");
        formulaResult.textContent = JSON.stringify(data, null, 2);
        activateTab("formulaTab");
        setStatus("公式核验完成");
      } catch (error) {
        setStatus(error.message);
      } finally {
        document.getElementById("verifyBtn").disabled = false;
      }
    }

    function activateTab(tabId) {
      document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.toggle("active", btn.dataset.tab === tabId));
      document.querySelectorAll(".tab").forEach(tab => tab.classList.toggle("active", tab.id === tabId));
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const files = filesInput.files;
      if (!files.length) {
        setStatus("请选择文件");
        return;
      }
      const data = new FormData();
      [...files].forEach(file => data.append("files", file));
      data.append("standard_key", standard.value);
      data.append("use_ai", document.getElementById("useAi").checked ? "true" : "false");
      appendAiConfig(data);
      runBtn.disabled = true;
      setStatus("识别中：正在预处理、OCR、结构化、标准判定和公式库匹配");
      try {
        const res = await fetch("/api/analyze", { method: "POST", body: data });
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.detail || "识别失败");
        renderAnalyze(payload);
        activateTab("recordsTab");
        setStatus("报告识别完成");
      } catch (error) {
        setStatus(error.message);
      } finally {
        runBtn.disabled = false;
      }
    });

    filesInput.addEventListener("change", () => suggestStandard(filesInput.files));
    formulaSelect.addEventListener("change", renderFormulaInputs);
    aiProvider.addEventListener("change", () => applyAiPreset(true));
    aiModelSelect.addEventListener("change", () => {
      if (aiModelSelect.value !== "__custom__") aiModel.value = aiModelSelect.value;
      updateAiCapabilityHint();
    });
    aiModel.addEventListener("input", () => renderAiModelOptions(aiModel.value.trim()));
    document.getElementById("useAi").addEventListener("change", updateAiCapabilityHint);
    document.getElementById("saveAiConfigBtn").addEventListener("click", saveAiConfig);
    document.getElementById("clearAiConfigBtn").addEventListener("click", clearAiConfig);
    document.getElementById("refreshAiModelsBtn").addEventListener("click", refreshAiModels);
    document.getElementById("testAiConfigBtn").addEventListener("click", testAiConfig);
    document.getElementById("verifyBtn").addEventListener("click", verifyFormula);
    document.getElementById("fillDemoBtn").addEventListener("click", fillDemo);
    document.querySelectorAll(".tab-btn").forEach(btn => btn.addEventListener("click", () => activateTab(btn.dataset.tab)));

    Promise.all([loadAiProviders(), loadStandards(), loadFormulas()])
      .then(() => {
        loadAiConfig();
        setStatus("就绪");
      })
      .catch((error) => setStatus(error.message));
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _load_index_html()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "standards": len(standards.names()),
        "formulas": len(formula_database.formulas),
        "runtime": str(RUNTIME_DIR),
        "ocr_max_pages": os.getenv("OCR_MAX_PAGES", "60"),
        "ocr_force_local": os.getenv("OCR_FORCE_LOCAL", "0"),
        "ocr_fallback_on_zero_records": os.getenv("OCR_FALLBACK_ON_ZERO_RECORDS", "1"),
    }


@app.get("/api/standards")
def list_standards() -> dict:
    return standards.names()


@app.get("/api/formulas")
def list_formulas() -> dict:
    return {"version": formula_database.data.get("version"), "formulas": formula_database.list_formulas()}


@app.get("/api/ai/providers")
def list_ai_providers() -> dict:
    return {"providers": AI_PROVIDER_PRESETS}


@app.post("/api/ai/models")
async def list_ai_models(payload: Dict[str, Any] = Body(...)) -> dict:
    provider = str(payload.get("ai_provider") or "openai")
    preset = AI_PROVIDER_PRESETS.get(provider, AI_PROVIDER_PRESETS["custom"])
    fallback_models = preset.get("models", [])
    ai_client = _ai_client_from_payload(payload)
    if not ai_client.base_url:
        return {"models": fallback_models, "source": "preset", "warning": "未配置 Base URL，已返回内置模型预设。"}
    if preset.get("requires_key", True) and not ai_client.api_key:
        return {"models": fallback_models, "source": "preset", "warning": "未填写 API Key，已返回内置模型预设。"}
    try:
        models = await run_in_threadpool(ai_client.list_models)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"读取模型列表失败：{exc}") from exc
    return {"models": models or fallback_models, "source": "api" if models else "preset"}


@app.post("/api/ai/test")
async def test_ai_connection(payload: Dict[str, Any] = Body(...)) -> dict:
    ai_client = _ai_client_from_payload(payload)
    if not ai_client.is_enabled:
        raise HTTPException(status_code=400, detail="AI API 未启用：请确认服务商、Base URL、模型名和 API Key（本地模型可免 Key）。")
    placeholder_message = _ai_placeholder_message(ai_client)
    if placeholder_message:
        raise HTTPException(status_code=400, detail=placeholder_message)
    try:
        response = await run_in_threadpool(ai_client.test_connection)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"AI API 调用失败：{exc}") from exc
    return {
        "ok": True,
        "provider": ai_client.provider,
        "base_url": ai_client.base_url,
        "model": ai_client.model,
        "response": response[:200],
    }


@app.post("/api/ai/diagnose")
async def diagnose_ai_connection(payload: Dict[str, Any] = Body(...)) -> dict:
    ai_client = _ai_client_from_payload(payload)
    placeholder_message = _ai_placeholder_message(ai_client)
    if placeholder_message:
        return {
            "ok": False,
            "provider": ai_client.provider,
            "base_url": ai_client.base_url,
            "model": ai_client.model,
            "api_key_present": bool(ai_client.api_key),
            "errors": [placeholder_message],
        }
    result = await run_in_threadpool(ai_client.diagnose, True)
    return {"ok": bool(result.get("text_ok")), **result}


@app.post("/api/formula/verify")
async def verify_formula(payload: Dict[str, Any] = Body(...)) -> dict:
    use_ai = bool(payload.get("use_ai"))
    try:
        machine_result = await run_in_threadpool(formula_verifier.verify, payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ai_review = ""
    if use_ai:
        ai_client = _ai_client_from_payload(payload)
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
    ai_provider: str = Form(""),
    ai_base_url: str = Form(""),
    ai_model: str = Form(""),
    ai_api_key: str = Form(""),
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
            {
                "ai_provider": ai_provider,
                "ai_base_url": ai_base_url,
                "ai_model": ai_model,
                "ai_api_key": ai_api_key,
            },
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _run_pipeline(file_paths: List[str], standard_key: str, use_ai: bool, ai_config: Dict[str, str] | None = None) -> dict:
    pipeline = EnvironmentReportVisionPipeline(standards, EXPORTS_DIR / "vision", _ai_client_from_payload(ai_config or {}))
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


def _ai_client_from_payload(payload: Dict[str, Any]) -> AIClient:
    provider = str(payload.get("ai_provider") or "openai")
    preset = AI_PROVIDER_PRESETS.get(provider, AI_PROVIDER_PRESETS["custom"])
    base_url = str(payload.get("ai_base_url") or preset.get("base_url") or "")
    model = str(payload.get("ai_model") or preset.get("default_model") or "")
    return AIClient(
        api_key=str(payload.get("ai_api_key") or ""),
        base_url=base_url,
        model=model,
        provider=provider,
    )


def _ai_placeholder_message(ai_client: AIClient) -> str:
    base_url = ai_client.base_url or ""
    model = ai_client.model or ""
    if any(token in base_url for token in ("YOUR_RESOURCE", "YOUR_DEPLOYMENT")):
        return "当前 Base URL 仍包含 Azure 占位符，请替换成你的 Azure OpenAI 资源名和部署名后再测试。"
    if model in {"YOUR_DEPLOYMENT", "ep-xxxxxxxx"} or model.lower().startswith("your_") or "xxxxxxxx" in model.lower():
        if ai_client.provider == "volcengine":
            return "火山方舟需要填写真实推理接入点 ID（ep-...），不能使用 ep-xxxxxxxx 占位符。"
        return "当前模型名仍是占位符，请替换为控制台中的真实模型名或部署名后再测试。"
    return ""


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
