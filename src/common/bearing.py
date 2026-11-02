"""XJTU-SY 轴承数据集：目录发现、CSV 加载、健康阶段标注、轴承几何参数。

数据集布局：data/XJTU-SY_Bearing_Datasets/<工况>/<轴承名>/<序号>.csv
每个 CSV 为 32768 点 × 2 通道（水平/垂直振动），采样率 25.6 kHz。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FS_HZ = 25_600.0
CONDITIONS = ("35Hz12kN", "37.5Hz11kN", "40Hz10kN")


@dataclass
class BearingGeometry:
    """深沟球轴承几何参数，用于计算故障特征频率。

    XJTU-SY 试验轴承型号 LDK UER204，参数取自
    《XJTU-SY滚动轴承加速寿命试验数据集解读》（机械工程学报 2019）表 1：
    滚珠直径 7.92 mm、轴承中径 34.55 mm、滚珠个数 8、接触角 0°。
    """

    z: int = 8                  # 滚珠个数
    ball_d_mm: float = 7.92     # 滚珠直径
    pitch_d_mm: float = 34.55   # 轴承中径（节径）
    contact_angle_deg: float = 0.0

    def fault_frequencies(self, fr_hz: float) -> dict[str, float]:
        """给定转频 fr（Hz），返回各故障特征频率（Hz）。"""
        z, d, D = self.z, self.ball_d_mm, self.pitch_d_mm
        alpha = np.deg2rad(self.contact_angle_deg)
        c = d / D * np.cos(alpha)
        return {
            "BPFO": z / 2 * fr_hz * (1 - c),
            "BPFI": z / 2 * fr_hz * (1 + c),
            "BSF": D / (2 * d) * fr_hz * (1 - c**2),
            "FTF": fr_hz / 2 * (1 - c),
        }

    @classmethod
    def from_json(cls, path: str | Path) -> "BearingGeometry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            z=int(data["z"]),
            ball_d_mm=float(data["ball_d_mm"]),
            pitch_d_mm=float(data["pitch_d_mm"]),
            contact_angle_deg=float(data.get("contact_angle_deg", 0.0)),
        )


@dataclass
class BearingRun:
    condition: str      # 工况目录名，如 35Hz12kN
    name: str           # 轴承目录名，如 Bearing1_1
    files: list[Path]   # 按采集序号排序的 CSV

    @property
    def key(self) -> str:
        return f"{self.condition}/{self.name}"


def condition_rpm(condition: str) -> float:
    """从工况目录名解析转速（rpm），如 35Hz12kN -> 2100。"""
    m = re.search(r"(\d+(?:\.\d+)?)Hz", condition)
    if not m:
        raise ValueError(f"工况名 {condition!r} 无法解析转频")
    return float(m.group(1)) * 60.0


def discover(
    data_dir: str | Path,
    conditions: list[str] | None = None,
    bearings: list[str] | None = None,
    max_files: int | None = None,
) -> list[BearingRun]:
    """扫描数据目录，返回轴承运行列表。可按工况名/轴承名过滤，max_files 限制每轴承文件数。"""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"未找到数据目录 {root}。请先按 data/README.md 下载并放置 XJTU-SY 数据集。"
        )
    runs: list[BearingRun] = []
    conds = conditions or list(CONDITIONS)
    for cond in sorted(root.iterdir()):
        if not cond.is_dir() or cond.name not in conds:
            continue
        for bearing in sorted(cond.iterdir()):
            if not bearing.is_dir():
                continue
            if bearings and bearing.name not in bearings:
                continue
            files = sorted(
                (f for f in bearing.glob("*.csv")),
                key=lambda p: int(re.sub(r"\D", "", p.stem) or 0),
            )
            if not files:
                continue
            if max_files:
                files = files[:max_files]
            runs.append(BearingRun(condition=cond.name, name=bearing.name, files=files))
    if not runs:
        raise RuntimeError(f"{root} 下没有找到任何轴承数据，请检查目录布局。")
    return runs


def load_vibration(csv_path: str | Path) -> np.ndarray:
    """加载单个振动 CSV，返回 (n_samples, 2) 数组。兼容带/不带表头。"""
    path = Path(csv_path)
    try:
        arr = np.loadtxt(path, delimiter=",", ndmin=2)
    except ValueError:
        arr = np.loadtxt(path, delimiter=",", ndmin=2, skiprows=1)
    if arr.shape[0] < 2 or arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{csv_path} 不是两列振动数据")
    return arr[:, :2]


def rms_series(run: BearingRun, channel: int = 1) -> np.ndarray:
    """按文件序号计算 RMS 退化曲线（默认垂直通道，列索引 1）。"""
    return np.array([float(np.sqrt(np.mean(load_vibration(f)[:, channel] ** 2))) for f in run.files])


def health_stage_labels(
    rms: np.ndarray, healthy_ratio: float = 0.2,
    degrade_mult: float = 1.5, failure_mult: float = 3.0,
) -> np.ndarray:
    """根据 RMS 退化曲线给每个文件打健康阶段标签。

    以该轴承前 healthy_ratio 比例文件的 RMS 中位数作为"健康基线"：
    0=健康（<=基线×degrade_mult），1=退化，2=故障（>基线×failure_mult）。
    阈值是工程近似，可用 CLI 参数调整。
    """
    n = len(rms)
    base_n = max(1, int(round(n * healthy_ratio)))
    base = float(np.median(rms[:base_n]))
    labels = np.ones(n, dtype=int)
    labels[rms <= base * degrade_mult] = 0
    labels[rms > base * failure_mult] = 2
    return labels


def bearing_split_mask(keys) -> tuple["np.ndarray", "np.ndarray"]:
    """按轴承留出划分：每 4 个轴承留 1 个做测试集。

    keys 为每行样本所属轴承键的数组/列表，返回 (train_mask, test_mask)。
    """
    keys = np.asarray(keys)
    uniq = sorted(set(keys.tolist()))
    test_keys = set(uniq[i] for i in range(3, len(uniq), 4))
    if not test_keys:
        test_keys = {uniq[-1]}
    is_test = np.isin(keys, list(test_keys))
    return ~is_test, is_test
