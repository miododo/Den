# 环境检测报告智能识别与核验系统

这是一个可本地部署的环境检测报告识别测试台，支持扫描版 PDF / 图片增强、OCR、AI 视觉或文本抽取、标准库限值匹配、公式库复核和结果导出。

## 主要功能

- 上传 PDF、扫描件或照片，生成扫描仪风格增强页和增强 PDF。
- 对报告文本进行 OCR，支持 PaddleOCR、RapidOCR、Tesseract 回退。
- 接入 DeepSeek、OpenAI、Gemini、Claude、通义千问、智谱、百度、腾讯、火山方舟、MiniMax、Ollama、LM Studio 等 OpenAI-compatible 或专用接口。
- 自动判断模型能力：支持视觉则优先读 PDF 页面图像；不支持视觉或视觉调用失败时自动降级为 OCR 文本 + AI 抽取。
- 抽取检测项目、检测点位、检测值、单位、检测日期、样品编号、报告编号、标准名称、标准限值和检测结论。
- 使用 `config/standards.json` 的标准库进行限值校验。
- 使用 `config/calculation_formulas.json` 的公式库标记可复算项目。
- 输出 JSON、HTML、Excel 和增强 PDF。

## 目录说明

```text
.
├─ integrated_test_app.py          # FastAPI Web 测试台
├─ app.py                          # 桌面入口
├─ web_app.py                      # 旧 Web 入口，保留用于兼容
├─ simple_test_app.py              # 简化测试入口
├─ src/
│  ├─ core/
│  │  ├─ ai_client.py              # 大模型 API、能力诊断、视觉探测
│  │  ├─ vision_pipeline.py        # PDF/图片增强、OCR、AI 抽取、标准校验
│  │  ├─ standards.py              # 标准库读取
│  │  ├─ formula_engine.py         # 公式库与机器复算
│  │  ├─ exporter.py               # JSON/HTML/Excel 导出
│  │  └─ models.py                 # 数据结构
│  ├─ ui/                          # PySide6 桌面 UI
│  └─ utils/
├─ config/
│  ├─ standards.json               # 标准限值库
│  └─ calculation_formulas.json    # 计算公式库
├─ docs/                           # 公式和 OCR 训练说明
├─ sample_data/                    # 合成示例数据，不含敏感报告
├─ requirements.txt
├─ .env.example
├─ install_windows.bat
├─ start_release_web.bat
└─ start_release_web.sh
```

## 环境要求

- Python 3.10 到 3.12，推荐 Python 3.11。
- Windows 10/11、macOS 或 Linux。
- CPU 可运行；PaddleOCR 在 CPU 上处理多页 PDF 会比较慢。
- 可选：如果使用 Tesseract 回退，需要额外安装系统级 Tesseract OCR。

## 安装依赖

### Windows

```bat
install_windows.bat
```

或手动执行：

```bat
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-minimal.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-minimal.txt
```

`requirements-minimal.txt` 用于先跑通 Web 测试台。扫描版 PDF 的高质量本地 OCR 属于可选增强；Windows 用户可在基础安装成功后运行：

```bat
install_ocr_windows.bat
```

或手动安装：

```bash
pip install -r requirements-ocr.txt
```

完整桌面端、异常分析和打包工具仍保留在 `requirements.txt` 中，二次开发时再按需安装。

## 配置环境变量

复制 `.env.example` 为 `.env`，或在系统环境变量中配置：

```text
AI_PROVIDER=openai
AI_API_KEY=your_api_key_here
AI_BASE_URL=https://api.openai.com/v1
AI_MODEL=gpt-4o

OCR_MAX_PAGES=all
OCR_FORCE_LOCAL=1
OCR_FALLBACK_ON_ZERO_RECORDS=1
APP_HOST=127.0.0.1
APP_PORT=8010
```

说明：

- `AI_PROVIDER`：服务商标识，例如 `openai`、`deepseek`、`gemini`、`anthropic`、`dashscope`、`zhipu`、`ollama`、`lmstudio`、`custom`。
- `AI_API_KEY`：你的私人大模型 Key。不要把真实 Key 提交给别人。
- `AI_BASE_URL`：API 地址，例如 OpenAI-compatible 的 `/v1` 地址。
- `AI_MODEL`：模型名或部署名。Azure OpenAI 填部署名，火山方舟通常填 `ep-...` 接入点。
- `OCR_MAX_PAGES=all`：OCR 最大页数，测试大 PDF 时可改为 `10` 或 `20`。
- `OCR_FORCE_LOCAL=1`：强制对增强页运行本地 OCR，更适合扫描版 PDF。

页面中也提供 API 配置区，可以直接填写服务商、Base URL、模型名和 API Key，并点击“测试 API”检查文本调用和视觉能力。

## 启动服务

### Windows

```bat
start_release_web.bat
```

### macOS / Linux

```bash
chmod +x start_release_web.sh
./start_release_web.sh
```

浏览器打开：

```text
http://127.0.0.1:8010/
```

## 最小测试流程

1. 解压 ZIP。
2. 安装依赖。
3. 启动 `start_release_web.bat` 或 `start_release_web.sh`。
4. 打开 `http://127.0.0.1:8010/`。
5. 可先不填 API Key，上传 `sample_data/sample_env_report.pdf`，执行标准选择 `GB 3838-2002 地表水环境质量标准（III 类，MVP 子集）`，点击“开始识别”。这个示例 PDF 有嵌入文本，即使未安装 OCR 也可以用于基础流程测试。
6. 查看结构化结果、完整 JSON、简报和导出文件。
7. 如需 AI 抽取，填写 API Key、Base URL 和模型名，点击“测试 API”，成功后勾选“AI 智能抽取”再上传 PDF。

## API 能力诊断

页面“测试 API”会调用：

```text
POST /api/ai/diagnose
```

它会检查：

- API Key 是否存在；
- Base URL 是否可访问；
- 模型名称是否能调用；
- 文本 chat/completions 是否正常；
- 图片输入是否可用；
- 如果图片能力失败，系统是否需要 OCR 降级。

## 主要接口

- `GET /`：Web 测试台。
- `GET /api/standards`：查看标准库列表。
- `GET /api/formulas`：查看公式库列表。
- `GET /api/ai/providers`：查看内置 AI 服务商和模型预设。
- `POST /api/ai/models`：读取模型列表。
- `POST /api/ai/diagnose`：API 与模型能力诊断。
- `POST /api/analyze`：上传 PDF/图片并返回结构化核验结果。
- `POST /api/formula/verify`：手动输入实验原始参数做公式复算。

## 输出结果

每次识别会在运行时目录中生成：

- JSON 结构化数据；
- HTML 报告；
- Excel 汇总表；
- 增强 PDF；
- `processing_trace`，包含 OCR、AI、数据库匹配和最终校验过程。

这些运行产物不会包含在交付 ZIP 中。

## 标准库与公式库

当前内置的是 MVP 子集：

- `GB 3095-2012` 环境空气二级常见指标；
- `GB 3838-2002` 地表水 III 类常见指标；
- 废水 / 地下水自定义规则入口；
- pH、总氮、氨氮、COD、高锰酸盐指数、溶解氧、BOD5、总磷等公式库。

商用前需要根据项目行业、排口、功能区、地方标准和排污许可继续扩展。

## 注意事项

- ZIP 不包含真实 API Key、历史上传文件、真实报告 PDF、导出结果、缓存文件或训练中间产物。
- PaddleOCR 首次运行可能下载模型，耗时较长。
- OCR 对低清晰度、强反光、红章覆盖、跨页复杂表格仍可能产生低置信度结果，系统会标记为“需人工复核”。
- 历史记录/样本数据库当前未内置，结果中会明确显示“历史记录/样本数据库未配置”。如需历史点位、日期、报告编号一致性校验，请接入业务数据库后扩展。
