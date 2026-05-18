from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.core.models import SensorAnalysisResult


COMMON_TIMESTAMP_COLUMNS = ["timestamp", "time", "日期", "时间", "监测时间", "采样时间"]
COMMON_STATION_COLUMNS = ["station", "site", "点位", "站点", "监测点"]
COMMON_INDICATOR_COLUMNS = ["indicator", "factor", "项目", "指标", "因子"]
COMMON_VALUE_COLUMNS = ["value", "数值", "结果", "检测值", "浓度"]


class SensorAnalyzer:
    def load_dataframe(self, file_path: str) -> pd.DataFrame:
        suffix = Path(file_path).suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(file_path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(file_path)
        raise ValueError("仅支持 CSV / XLS / XLSX 文件。")

    def guess_columns(self, df: pd.DataFrame) -> dict:
        columns = {str(c): str(c) for c in df.columns}
        return {
            "timestamp": self._guess(df.columns, COMMON_TIMESTAMP_COLUMNS),
            "station": self._guess(df.columns, COMMON_STATION_COLUMNS),
            "indicator": self._guess(df.columns, COMMON_INDICATOR_COLUMNS),
            "value": self._guess(df.columns, COMMON_VALUE_COLUMNS),
            "all": list(columns.keys()),
        }

    def analyze(
        self,
        file_path: str,
        timestamp_col: str,
        indicator_col: str,
        value_col: str,
        station_col: Optional[str] = None,
        algorithm: str = "zscore",
    ) -> SensorAnalysisResult:
        df = self.load_dataframe(file_path).copy()
        if timestamp_col not in df.columns or indicator_col not in df.columns or value_col not in df.columns:
            raise ValueError("列名未找到，请检查字段映射。")

        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[timestamp_col, value_col, indicator_col]).sort_values(timestamp_col)
        if df.empty:
            raise ValueError("清洗后没有可用数据，请检查时间列和值列格式。")

        group_cols = [indicator_col]
        if station_col and station_col in df.columns:
            group_cols.insert(0, station_col)

        anomalies = []
        scored_frames = []
        for _, group in df.groupby(group_cols):
            g = group.copy()
            if len(g) < 5:
                g["is_anomaly"] = False
                g["score"] = 0.0
                scored_frames.append(g)
                continue

            if algorithm == "iforest":
                model = IsolationForest(contamination=0.08, random_state=42)
                pred = model.fit_predict(g[[value_col]])
                g["is_anomaly"] = pred == -1
                g["score"] = -model.score_samples(g[[value_col]])
            else:
                mean = g[value_col].mean()
                std = g[value_col].std(ddof=0)
                if std == 0 or np.isnan(std):
                    g["is_anomaly"] = False
                    g["score"] = 0.0
                else:
                    z = (g[value_col] - mean) / std
                    g["score"] = z.abs()
                    g["is_anomaly"] = g["score"] >= 2.5
            scored_frames.append(g)

        result_df = pd.concat(scored_frames, ignore_index=True)
        anomaly_df = result_df[result_df["is_anomaly"]].copy()
        anomaly_df = anomaly_df.sort_values([indicator_col, timestamp_col])

        columns = [c for c in [timestamp_col, station_col, indicator_col, value_col, "score"] if c and c in anomaly_df.columns]
        anomalies = anomaly_df[columns].to_dict(orient="records")
        stats = {
            "mean": float(result_df[value_col].mean()),
            "min": float(result_df[value_col].min()),
            "max": float(result_df[value_col].max()),
            "std": float(result_df[value_col].std(ddof=0)),
            "indicator_count": int(result_df[indicator_col].nunique()),
        }
        summary = (
            f"已分析 {len(result_df)} 行监测数据，识别出 {len(anomaly_df)} 行异常点。"
            f"算法：{'Isolation Forest' if algorithm == 'iforest' else 'Z-Score'}；"
            f"指标种类 {stats['indicator_count']} 个，数值范围 {stats['min']:.4f} ~ {stats['max']:.4f}。"
        )
        return SensorAnalysisResult(
            file_path=str(Path(file_path).resolve()),
            rows=int(len(result_df)),
            timestamp_column=timestamp_col,
            station_column=station_col,
            indicator_column=indicator_col,
            value_column=value_col,
            algorithm=algorithm,
            anomaly_count=int(len(anomaly_df)),
            summary=summary,
            stats=stats,
            anomalies=anomalies,
        )

    def _guess(self, columns, candidates):
        lower_map = {str(c).lower(): str(c) for c in columns}
        for candidate in candidates:
            if candidate.lower() in lower_map:
                return lower_map[candidate.lower()]
        for col in columns:
            text = str(col).lower()
            for candidate in candidates:
                if candidate.lower() in text:
                    return str(col)
        return str(columns[0]) if len(columns) else ""
