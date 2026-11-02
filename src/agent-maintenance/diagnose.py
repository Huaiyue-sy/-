"""诊断流水线：文件 → 特征 → 异常检测 → 故障分类/定位 → RUL 预测。

故障定位规则：比较包络谱特征幅值 env_BPFO / env_BPFI / env_BSF 的
相对大小，最大者对应的元件视为疑似故障源（工程近似，报告中已注明）。
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.bearing import FS_HZ, BearingGeometry, condition_rpm
from common.features import extract_channels

FAULT_LOCATION = {
    "BPFO": "外圈故障",
    "BPFI": "内圈故障",
    "BSF": "滚动体故障",
    "FTF": "保持架故障",
}
STAGE_NAMES = {0: "健康", 1: "退化", 2: "故障"}


@dataclass
class Diagnosis:
    file_path: str
    anomaly_score: float
    is_anomaly: bool
    stage: int | None = None
    stage_name: str | None = None
    fault_location: str | None = None
    fault_evidence: dict = field(default_factory=dict)
    rel_rul: float | None = None
    rul_note: str = ""


class MaintenanceAgent:
    def __init__(self, features_csv: str, classifier_path: str = "models/week1_classifier.joblib",
                 rul_path: str = "models/week1_rul.joblib", contamination: float = 0.05):
        self.df = pd.read_csv(features_csv)
        self.features = [c for c in self.df.columns if c not in
                         ("condition", "bearing", "file_idx", "bearing_key",
                          "stage", "rel_rul", "file_path")]
        self.classifier, self.rul_model, self.classifier_features, self.rul_features = None, None, None, None
        if Path(classifier_path).exists():
            blob = pickle.loads(Path(classifier_path).read_bytes())
            self.classifier = blob["model"]
            self.classifier_features = blob["features"]
        if Path(rul_path).exists():
            blob = pickle.loads(Path(rul_path).read_bytes())
            self.rul_model = blob["model"]
            self.rul_features = blob["features"]

        from anomaly import IsolationDetector
        healthy = self.df.loc[self.df["stage"] == 0, self.features].to_numpy()
        if len(healthy) < 8:
            healthy = self.df[self.features].to_numpy()
        self.detector = IsolationDetector(contamination=contamination).fit_healthy(healthy)

    def diagnose_file(self, csv_path: str, condition: str, geometry: BearingGeometry | None = None,
                      fs: float = FS_HZ) -> Diagnosis:
        """对单个振动文件执行完整诊断链路。"""
        fr = condition_rpm(condition) / 60.0
        geo = geometry or BearingGeometry()
        feats = extract_channels(csv_path, fs, geo, fr)
        x = np.array([[feats[c] for c in self.features]])
        score = self.detector.score(x)
        is_anom = self.detector.is_anomaly(x)

        diag = Diagnosis(file_path=str(csv_path), anomaly_score=score, is_anomaly=is_anom)

        if is_anom:
            if self.classifier is not None:
                xc = np.array([[feats[c] for c in self.classifier_features]])
                diag.stage = int(self.classifier.predict(xc)[0])
                diag.stage_name = STAGE_NAMES[diag.stage]
            if self.rul_model is not None:
                xr = np.array([[feats[c] for c in self.rul_features]])
                diag.rel_rul = float(self.rul_model.predict(xr)[0])
                diag.rul_note = (
                    f"预计已消耗约 {(1 - diag.rel_rul) * 100:.0f}% 寿命，"
                    f"剩余寿命比例约 {max(diag.rel_rul, 0) * 100:.0f}%")
            # 故障元件定位（包络谱证据）
            ev = {k: float(feats.get(f"env_{k}_v", 0.0) + feats.get(f"env_{k}_h", 0.0))
                  for k in ("BPFO", "BPFI", "BSF", "FTF")}
            strongest = max(ev, key=ev.get)
            diag.fault_location = FAULT_LOCATION[strongest]
            diag.fault_evidence = {k: round(v, 3) for k, v in ev.items()}
        return diag

    def scan_bearing(self, features_csv_rows: pd.DataFrame, condition: str,
                     geometry: BearingGeometry | None = None) -> list[Diagnosis]:
        """按行扫描某轴承的若干文件（DataFrame 需含 file_path 列）。"""
        out = []
        for _, row in features_csv_rows.iterrows():
            out.append(self.diagnose_file(row["file_path"], condition, geometry))
        return out
