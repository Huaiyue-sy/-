"""振动信号特征提取：时域、频域、包络谱特征。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew

from common.bearing import BearingGeometry, load_vibration

BANDS_HZ = ((0.0, 1000.0), (1000.0, 4000.0), (4000.0, 8000.0), (8000.0, 12800.0))
ENVELOPE_FEATURES = ("env_BPFO", "env_BPFI", "env_BSF", "env_FTF", "env_rms")


def time_features(x: np.ndarray) -> dict[str, float]:
    """时域统计特征（12 个）。"""
    abs_x = np.abs(x)
    abs_mean = float(np.mean(abs_x))
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(abs_x))
    sqrt_mean = float(np.mean(np.sqrt(abs_x)) ** 2)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "rms": rms,
        "peak": peak,
        "p2p": float(np.ptp(x)),
        "kurtosis": float(sp_kurtosis(x)),
        "skewness": float(sp_skew(x)),
        "crest_factor": peak / (rms + 1e-12),
        "clearance_factor": peak / (sqrt_mean + 1e-12),
        "shape_factor": rms / (abs_mean + 1e-12),
        "impulse_factor": peak / (abs_mean + 1e-12),
    }


def freq_features(x: np.ndarray, fs: float) -> dict[str, float]:
    """频域特征：重心/均方根/标准差频率、谱峭度、4 个频带能量占比。"""
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    power = spec**2
    total = float(np.sum(power)) + 1e-12
    fc = float(np.sum(freqs * power) / total)
    rmsf = float(np.sqrt(np.sum(freqs**2 * power) / total))
    rvf = float(np.sqrt(np.sum((freqs - fc) ** 2 * power) / total))
    spec_kurtosis = float(sp_kurtosis(power / total))
    feats = {"fc": fc, "rmsf": rmsf, "rvf": rvf, "spec_kurtosis": spec_kurtosis}
    for i, (lo, hi) in enumerate(BANDS_HZ):
        mask = (freqs >= lo) & (freqs < hi)
        feats[f"band{i}_ratio"] = float(np.sum(power[mask]) / total)
    return feats


def envelope_features(
    x: np.ndarray, fs: float, geometry: BearingGeometry, fr_hz: float,
    tol: float = 0.03,
) -> dict[str, float]:
    """希尔伯特包络谱特征：各故障特征频率处的谱峰幅值（容忍 ±tol 滑差）。"""
    envelope = np.abs(hilbert(x))
    envelope = envelope - np.mean(envelope)
    spec = np.abs(np.fft.rfft(envelope))
    freqs = np.fft.rfftfreq(len(envelope), d=1.0 / fs)
    feats: dict[str, float] = {"env_rms": float(np.sqrt(np.mean(envelope**2)))}
    for name, target in geometry.fault_frequencies(fr_hz).items():
        mask = np.abs(freqs - target) <= target * tol
        feats[f"env_{name}"] = float(np.max(spec[mask])) if mask.any() else 0.0
    return feats


def extract_channels(
    csv_path: str | Path, fs: float, geometry: BearingGeometry, fr_hz: float,
) -> dict[str, float]:
    """提取单个 CSV 两通道（_h 水平 / _v 垂直）的全部特征，共 50 维。"""
    data = load_vibration(csv_path)
    row: dict[str, float] = {}
    for ch, suffix in ((0, "h"), (1, "v")):
        x = data[:, ch]
        feats = time_features(x)
        feats.update(freq_features(x, fs))
        feats.update(envelope_features(x, fs, geometry, fr_hz))
        for k, v in feats.items():
            row[f"{k}_{suffix}"] = float(v)
    row["rms_mean"] = 0.5 * (row["rms_h"] + row["rms_v"])
    return row


def feature_columns() -> list[str]:
    """特征列名（与 extract_channels 输出一致，不含 rms_mean）。"""
    demo = time_features(np.zeros(16))
    demo.update(freq_features(np.zeros(16), 1.0))
    demo.update({k: 0.0 for k in ENVELOPE_FEATURES})
    return sorted(f"{k}_{s}" for s in ("h", "v") for k in demo)
