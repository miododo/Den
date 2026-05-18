# PaddleOCR 训练/测试样本工作流

本项目已经接入 PaddleOCR 推理，并新增批量测试与弱标注数据集生成脚本：

```bat
tools\env_report_batch_test.py --make-paddle-dataset
```

默认样本来自：

- `D:\个人资料\比赛内容\环境检测pdf扫描版`

输出目录默认位于：

- `D:\project\env-ai-validator\env_ai_validator_app\training_runtime\env_report_batch_时间戳`

## 生成内容

- `batch_summary.json/csv/html`：每份报告抽取记录数、超标数、复核数、警告。
- `reports/*/*_structured.json/html/xlsx`：单份报告结构化结果。
- `paddleocr_weak_dataset/det_gt_train.txt`、`det_gt_eval.txt`：PaddleOCR 文本检测格式。
- `paddleocr_weak_dataset/rec_gt_train.txt`、`rec_gt_eval.txt`：PaddleOCR 文本识别格式。
- `paddleocr_weak_dataset/weak_labels.jsonl`：带来源文件、页码、置信度的候选标注。

## 重要说明

这些标签是当前 PaddleOCR 自动识别结果，属于弱标注。正式用于训练前，需要人工复核并修正。

官方 PaddleOCR 仓库：

- https://github.com/PaddlePaddle/PaddleOCR/tree/main

官方数据格式说明：

- 文本检测：图片路径 + `json.dumps([{transcription, points}, ...])`
- 文本识别：裁剪图路径 + 文本标签，中间用制表符分隔
- 官方说明页：https://www.paddleocr.ai/main/en/datasets/ocr_datasets.html

## 推荐命令

快速生成前 12 页批量测试和前 8 页弱标注：

```bat
C:\Users\Lenovo\Desktop\环境检测样本批量训练测试.bat
```

完整跑全部默认样本更多页：

```bat
"C:\Users\Lenovo\Desktop\env_ai_validator_app\.venv\Scripts\python.exe" tools\env_report_batch_test.py --max-pages 60 --dataset-max-pages 60 --make-paddle-dataset
```

只跑某一个文件：

```bat
"C:\Users\Lenovo\Desktop\env_ai_validator_app\.venv\Scripts\python.exe" tools\env_report_batch_test.py --files "D:\个人资料\比赛内容\环境检测pdf扫描版\重庆中法唐家沱污水处理有限公司（废水）.pdf" --make-paddle-dataset
```
