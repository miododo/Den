from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from src.core.models import ReportAnalysisResult


class AIClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key or os.getenv("AI_API_KEY", "")
        self.base_url = (base_url or os.getenv("AI_BASE_URL", "")).rstrip("/")
        self.model = model or os.getenv("AI_MODEL", "")
        self.provider = (provider or os.getenv("AI_PROVIDER", "openai")).strip().lower()
        self.timeout = timeout

    @property
    def is_enabled(self) -> bool:
        return bool(self.base_url and self.model and (self.api_key or self._allows_blank_api_key()))

    def likely_supports_vision(self) -> bool:
        model = (self.model or "").lower()
        provider = (self.provider or "").lower()
        if provider in {"anthropic", "claude", "gemini", "lmstudio"}:
            return True
        vision_tokens = [
            "vision",
            "visual",
            "vl",
            "pixtral",
            "gpt-4o",
            "gpt-4.1",
            "gpt-5",
            "o3",
            "o4",
            "gemini",
            "claude",
            "qwen-vl",
            "qwen2.5-vl",
            "qwen3-vl",
            "qwen-vl-ocr",
            "glm-5v",
            "ernie-4.5-turbo-vl",
            "hunyuan-vision",
            "doubao-seed-1-6-vision",
            "minimax-vl",
            "llama3.2-vision",
        ]
        return any(token in model for token in vision_tokens)

    def diagnose(self, probe_vision: bool = True) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_present": bool(self.api_key),
            "enabled": self.is_enabled,
            "base_url_configured": bool(self.base_url),
            "model_configured": bool(self.model),
            "model_list_ok": False,
            "model_available": None,
            "text_ok": False,
            "vision_expected": self.likely_supports_vision(),
            "vision_ok": None,
            "errors": [],
            "warnings": [],
        }
        if not self.is_enabled:
            result["errors"].append("AI API 未启用：缺少 Base URL、模型名或 API Key。")
            return result

        try:
            models = self.list_models()
            result["model_list_ok"] = True
            result["models_seen"] = len(models)
            if models:
                model_ids = {str(item.get("id", "")) for item in models}
                result["model_available"] = self.model in model_ids
                if not result["model_available"]:
                    result["warnings"].append("模型列表中未直接找到当前模型名；部分平台不返回全部授权模型，可继续做调用测试。")
        except Exception as exc:
            result["model_list_error"] = str(exc)
            result["warnings"].append("模型列表读取失败；将继续通过 chat/completions 测试模型是否可调用。")

        try:
            result["text_response"] = self.test_connection()[:200]
            result["text_ok"] = True
        except Exception as exc:
            result["text_error"] = str(exc)
            result["errors"].append(f"文本调用失败：{exc}")
            return result

        if probe_vision:
            try:
                result["vision_response"] = self.probe_vision()[:200]
                result["vision_ok"] = True
            except Exception as exc:
                result["vision_error"] = str(exc)
                result["vision_ok"] = False
                result["warnings"].append("图片/视觉探测失败；生产流程会自动降级到 OCR 文本处理。")
        return result

    def test_connection(self) -> str:
        return self._chat_text(
            "你是 API 连通性测试助手。",
            "请只回复：AI API 连接成功",
            0.0,
        )

    def probe_vision(self) -> str:
        tiny_png = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )
        return self.chat_with_content(
            [
                {"type": "text", "text": "这是图片能力探测。若你能接收图片，请只回复：VISION_OK"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{tiny_png}"}},
            ],
            temperature=0.0,
        )

    def list_models(self) -> List[Dict[str, str]]:
        if not self.base_url:
            return []
        url = self._models_url()
        response = requests.get(url, headers=self._headers(), timeout=min(self.timeout, 30))
        if not response.ok:
            self._raise_for_response(response)
        data = response.json()
        items = data.get("data") if isinstance(data, dict) else data
        if items is None and isinstance(data, dict):
            items = data.get("models") or data.get("model_list")
        models: List[Dict[str, str]] = []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, str):
                    models.append({"id": item, "name": item})
                elif isinstance(item, dict):
                    model_id = str(item.get("id") or item.get("name") or item.get("model") or "").strip()
                    if model_id:
                        models.append({"id": model_id, "name": str(item.get("display_name") or item.get("name") or model_id)})
        return models

    def summarize_report(self, result: ReportAnalysisResult) -> str:
        if not self.is_enabled:
            return ""

        prompt = {
            "task": "你是环境检测报告核验助手。请根据输入字段输出精炼中文结论。",
            "requirements": [
                "先写总体结论，再写 3-5 条需要人工复核的点。",
                "如果 CMA 到期时间早于报告日期，明确标成高风险。",
                "不要编造未识别到的数据。",
            ],
            "report": result.to_dict(),
        }
        return self._chat_text("你是严谨的环境检测报告审核助手。", json.dumps(prompt, ensure_ascii=False), 0.1)

    def chat_with_content(
        self,
        content: List[Dict[str, Any]],
        temperature: float = 0.1,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        if not self.is_enabled:
            return ""

        if self.provider in {"anthropic", "claude"}:
            return self._anthropic_chat_with_content(content, temperature)
        return self._openai_chat_with_content(content, temperature, response_format)

    def _openai_chat_with_content(
        self,
        content: List[Dict[str, Any]],
        temperature: float = 0.1,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        url = self._openai_chat_url()
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是严谨的环境检测报告识别助手，只输出可审计的结果。"},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format
        response = self._post_json(url, payload)
        if not response.ok and response_format and self._should_retry_without_response_format(response):
            payload.pop("response_format", None)
            response = self._post_json(url, payload)
        if not response.ok and "temperature" in payload and self._should_retry_without_temperature(response):
            payload.pop("temperature", None)
            response = self._post_json(url, payload)
        if not response.ok:
            self._raise_for_response(response)
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def _anthropic_chat_with_content(self, content: List[Dict[str, Any]], temperature: float = 0.1) -> str:
        url = self._anthropic_messages_url()
        anthropic_content = self._to_anthropic_content(content)
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": temperature,
            "system": "你是严谨的环境检测报告识别助手，只输出可审计的结果。",
            "messages": [{"role": "user", "content": anthropic_content}],
        }
        response = requests.post(url, headers=self._anthropic_headers(), json=payload, timeout=self.timeout)
        if not response.ok:
            self._raise_for_response(response)
        data = response.json()
        return "".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text").strip()

    def _chat_text(self, system: str, prompt: str, temperature: float = 0.1) -> str:
        if not self.is_enabled:
            return ""
        if self.provider in {"anthropic", "claude"}:
            payload = {
                "model": self.model,
                "max_tokens": 2048,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            }
            response = requests.post(self._anthropic_messages_url(), headers=self._anthropic_headers(), json=payload, timeout=self.timeout)
            if not response.ok:
                self._raise_for_response(response)
            data = response.json()
            return "".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text").strip()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        response = self._post_json(self._openai_chat_url(), payload)
        if not response.ok and "temperature" in payload and self._should_retry_without_temperature(response):
            payload.pop("temperature", None)
            response = self._post_json(self._openai_chat_url(), payload)
        if not response.ok:
            self._raise_for_response(response)
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def review_formula_verification(self, payload: Dict[str, Any], machine_result: Dict[str, Any]) -> str:
        if not self.is_enabled:
            return ""
        prompt = {
            "task": "你是环境检测实验数据复核助手。请基于计算公式、输入参数、机器复算结果和报告值，输出可审计复核意见。",
            "requirements": [
                "先判断机器复算是否支持结论。",
                "指出缺失参数、单位口径、校准曲线口径或稀释倍数风险。",
                "不要编造未提供的数据。",
                "输出 3 条以内中文要点。",
            ],
            "payload": payload,
            "machine_result": machine_result,
        }
        return self._chat_text("你是严谨的环境检测公式复核助手。", json.dumps(prompt, ensure_ascii=False), 0.0)

    def _openai_chat_url(self) -> str:
        if "/chat/completions" in self.base_url:
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _models_url(self) -> str:
        if self.provider in {"anthropic", "claude"}:
            base = self.base_url or "https://api.anthropic.com/v1"
            if base.endswith("/models"):
                return base
            if base.endswith("/messages"):
                return f"{base.rsplit('/messages', 1)[0]}/models"
            return f"{base.rstrip('/')}/models"
        base = self.base_url.rstrip("/")
        if "/chat/completions" in base:
            return f"{base.rsplit('/chat/completions', 1)[0]}/models"
        return f"{base}/models"

    def _anthropic_messages_url(self) -> str:
        base = self.base_url or "https://api.anthropic.com/v1"
        if base.endswith("/messages"):
            return base
        return f"{base.rstrip('/')}/messages"

    def _anthropic_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
            "Content-Type": "application/json",
        }

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.provider in {"anthropic", "claude"}:
            return self._anthropic_headers()
        if self.provider in {"azure", "azure_openai"}:
            if self.api_key:
                headers["api-key"] = self.api_key
            return headers
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post_json(self, url: str, payload: Dict[str, Any]) -> requests.Response:
        return requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)

    def _raise_for_response(self, response: requests.Response) -> None:
        detail = ""
        try:
            payload = response.json()
            detail = json.dumps(payload, ensure_ascii=False)
        except Exception:
            detail = response.text[:1000]
        raise requests.HTTPError(f"{response.status_code} {response.reason}: {detail}", response=response)

    def _should_retry_without_response_format(self, response: requests.Response) -> bool:
        text = response.text.lower()
        return response.status_code in {400, 404, 422} and "response_format" in text

    def _should_retry_without_temperature(self, response: requests.Response) -> bool:
        text = response.text.lower()
        return response.status_code in {400, 422} and "temperature" in text

    def _allows_blank_api_key(self) -> bool:
        if self.provider in {"ollama", "lmstudio", "local_openai", "local"}:
            return True
        host = urlparse(self.base_url).hostname or ""
        return host in {"127.0.0.1", "localhost", "::1"}

    def _to_anthropic_content(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for item in content:
            if item.get("type") == "text":
                converted.append({"type": "text", "text": str(item.get("text", ""))})
                continue
            if item.get("type") == "image_url":
                url = str((item.get("image_url") or {}).get("url", ""))
                match = re.match(r"data:(image/[^;]+);base64,(.+)", url, flags=re.DOTALL)
                if match:
                    converted.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": match.group(1),
                                "data": match.group(2),
                            },
                        }
                    )
        return converted
