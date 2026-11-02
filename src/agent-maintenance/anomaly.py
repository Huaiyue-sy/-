"""无监督异常检测模块：Isolation Forest。

用健康期样本（features.csv 中 stage=0 的行）训练孤立森林，
对在线读取的振动文件做异常打分。也支持"用其他轴承历史数据训练、
未训练轴承回测"的方案（见实训指南 3.2 节）。
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationDetector:
    def __init__(self, contamination: float = 0.05, seed: int = 42):
        self.model = IsolationForest(n_estimators=200, contamination=contamination,
                                     random_state=seed, n_jobs=-1)

    def fit_healthy(self, X: np.ndarray) -> "IsolationDetector":
        """用健康期特征矩阵训练（stage=0 的行）。"""
        self.model.fit(X)
        return self

    def score(self, x: np.ndarray) -> float:
        """返回异常分（decision_function，越小越异常）。"""
        return float(self.model.decision_function(np.atleast_2d(x))[0])

    def is_anomaly(self, x: np.ndarray) -> bool:
        return bool(self.model.predict(np.atleast_2d(x))[0] == -1)

    def score_series(self, X: np.ndarray) -> np.ndarray:
        """批量打分，用于整轴承扫描/趋势图。"""
        return self.model.decision_function(np.asarray(X))
