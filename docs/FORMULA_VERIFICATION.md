# 计算公式库与 AI 复检机制

公式库位置：

- `D:\project\env-ai-validator\env_ai_validator_app\config\calculation_formulas.json`

已整理纳入的指标：

- pH
- 总氮
- 氨氮（纳氏试剂法、水杨酸法）
- 化学需氧量 COD
- 高锰酸盐指数
- 溶解氧（温度校正、气压校正）
- 五日生化需氧量 BOD5
- 总磷

## 运行机制

1. OCR/AI 抽取报告中的检测项目、检测值、单位。
2. 标准库先做达标判定。
3. 公式库匹配对应指标和测定方法。
4. 如果报告中只有最终检测值，系统会标记为 `formula_available_missing_raw_inputs`，提示需要录入吸光度、滴定体积、稀释倍数等原始实验参数。
5. 如果通过 API 提供原始参数，系统会执行机器复算，并和报告值比较。
6. `use_ai=true` 时，同一接口会返回 AI 复检意见，用来解释缺项、单位口径或曲线口径风险。

## API

查看公式库：

```http
GET http://127.0.0.1:8000/api/formulas
```

机器复算 + AI 复检：

```http
POST http://127.0.0.1:8000/api/formula/verify
Content-Type: application/json

{
  "indicator": "COD",
  "method_id": "cod_dichromate_hj828_2017",
  "reported_value": 200,
  "use_ai": true,
  "inputs": {
    "c_ferrous_ammonium_sulfate_mol_l": 0.25,
    "blank_volume_ml": 10,
    "sample_volume_titrant_ml": 8,
    "sample_volume_ml": 20,
    "dilution_factor": 1
  }
}
```

返回结构包含：

- `machine.status`: `pass`、`fail`、`calculated`、`missing_inputs`、`unsupported`
- `machine.calculated_value`: 机器复算结果
- `machine.comparison`: 报告值与复算值差异
- `ai_review`: AI 复检意见

## 说明

部分 Word 文档中的公式是图片或嵌入对象，文本抽取无法完整读取。当前公式库已按指标和常用测定标准整理为可编辑 JSON，后续可把人工校正后的公式继续补充进同一文件。
