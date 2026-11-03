"""设备健康打分：把轴承特征表转换为 0–100 分、四级状态与告警。

打分口径是“可解释的规则加权”，不是新训练的模型：
  RMS 30% + 峭度 20% + 包络峰值 20% + 阶段 20% + 相对 RUL 10%
基线取每个轴承前 20% 文件的稳健中位数，避免单点冲击影响基线。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

LEVEL_NAMES = {0: "健康", 1: "注意", 2: "警告", 3: "危险"}
STAGE_NAMES = {0: "健康", 1: "退化", 2: "故障"}
ENV_COLUMNS = ("env_BPFO_v", "env_BPFI_v", "env_BSF_v", "env_FTF_v")
WEIGHTS = {"rms": 0.30, "kurtosis": 0.20, "envelope": 0.20, "stage": 0.20, "rul": 0.10}


@dataclass
class Baseline:
    rms: float
    kurtosis: float
    envelope: float
    files: int


@dataclass
class HealthPoint:
    condition: str
    bearing: str
    bearing_key: str
    file_idx: int
    stage: int
    rel_rul: float
    rms_mean: float
    kurtosis_v: float
    env_peak: float
    score: float
    level: int
    level_name: str
    trend: str
    rms_ratio: float
    env_ratio: float
    kurt_delta: float
    rms_score: float
    kurt_score: float
    env_score: float
    stage_score: float
    rul_score: float
    explanation: str
    alert_type: str = ""
    alert_message: str = ""


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _env_peak(row: pd.Series) -> float:
    return max(_num(row.get(column, 0.0)) for column in ENV_COLUMNS)


def build_baseline(group: pd.DataFrame) -> Baseline:
    n = max(5, int(len(group) * 0.20))
    n = min(n, len(group))
    head = group.head(n)
    rms = max(_num(head["rms_mean"].median()), 1e-6)
    kurtosis = _num(head["kurtosis_v"].median())
    env = max(_num(head[list(ENV_COLUMNS)].max(axis=1).median()), 1e-6)
    return Baseline(rms=rms, kurtosis=kurtosis, envelope=env, files=len(group))


def level_from_score(score: float) -> int:
    if score >= 85:
        return 0
    if score >= 70:
        return 1
    if score >= 50:
        return 2
    return 3


def _trend_from_scores(scores: list[float]) -> tuple[str, float | None]:
    if len(scores) < 6:
        return "平稳", None
    delta = scores[-1] - scores[-6]
    if delta <= -5:
        return "恶化", delta
    if delta >= 5:
        return "改善", delta
    return "平稳", delta


def _explain(
    baseline: Baseline,
    rms_mean: float,
    kurtosis_v: float,
    env_peak: float,
    stage: int,
    rel_rul: float,
    components: dict[str, float],
) -> str:
    impacts = {
        "RMS": WEIGHTS["rms"] * (100 - components["rms"]),
        "峭度": WEIGHTS["kurtosis"] * (100 - components["kurtosis"]),
        "包络": WEIGHTS["envelope"] * (100 - components["envelope"]),
        "阶段": WEIGHTS["stage"] * (100 - components["stage"]),
        "RUL": WEIGHTS["rul"] * (100 - components["rul"]),
    }
    parts = []
    for name, _ in sorted(impacts.items(), key=lambda item: item[1], reverse=True)[:3]:
        if name == "RMS":
            parts.append(
                f"RMS {components['rms']:.0f} 分（基线 {baseline.rms:.3f}，当前 {rms_mean:.3f}，"
                f"{rms_mean / baseline.rms:.1f} 倍）"
            )
        elif name == "峭度":
            parts.append(
                f"峭度 {components['kurtosis']:.0f} 分（较基线 +{max(0.0, kurtosis_v - baseline.kurtosis):.2f}）"
            )
        elif name == "包络":
            parts.append(
                f"包络 {components['envelope']:.0f} 分（基线 {baseline.envelope:.1f}，当前 {env_peak:.1f}，"
                f"{env_peak / baseline.envelope:.1f} 倍）"
            )
        elif name == "阶段":
            parts.append(f"阶段 {components['stage']:.0f} 分（{STAGE_NAMES.get(stage, '未知')}）")
        else:
            parts.append(f"RUL {components['rul']:.0f} 分（相对剩余寿命 {rel_rul:.2f}）")
    return "；".join(parts)


def _score_row(
    row: pd.Series,
    baseline: Baseline,
    recent_scores: list[float],
    previous: dict[str, Any] | None,
    seen_alerts: set[str],
) -> HealthPoint:
    stage = int(_num(row.get("stage"), 2))
    rel_rul = _clip01(_num(row.get("rel_rul"), 0.0))
    rms_mean = _num(row.get("rms_mean"))
    kurtosis_v = _num(row.get("kurtosis_v"))
    env_peak = _env_peak(row)

    rms_ratio = rms_mean / baseline.rms if baseline.rms > 0 else 1.0
    env_ratio = env_peak / baseline.envelope if baseline.envelope > 0 else 1.0
    kurt_delta = max(0.0, kurtosis_v - baseline.kurtosis)

    rms_score = 100.0 * _clip01(1.0 - max(0.0, rms_ratio - 1.0) / 2.0)
    kurt_score = 100.0 * _clip01(1.0 - kurt_delta / 8.0)
    env_score = 100.0 * _clip01(1.0 - max(0.0, env_ratio - 1.0) / 2.0)
    stage_score = {0: 100.0, 1: 75.0, 2: 45.0}.get(stage, 45.0)
    rul_score = 100.0 * _clip01(rel_rul)

    components = {
        "rms": rms_score,
        "kurtosis": kurt_score,
        "envelope": env_score,
        "stage": stage_score,
        "rul": rul_score,
    }
    score = (
        WEIGHTS["rms"] * rms_score
        + WEIGHTS["kurtosis"] * kurt_score
        + WEIGHTS["envelope"] * env_score
        + WEIGHTS["stage"] * stage_score
        + WEIGHTS["rul"] * rul_score
    )
    level = level_from_score(score)
    scores = recent_scores + [score]
    trend, delta = _trend_from_scores(scores)

    explanation = _explain(baseline, rms_mean, kurtosis_v, env_peak, stage, rel_rul, components)

    alert_messages: list[str] = []
    alert_type = ""
    if previous is not None:
        if level > int(previous.get("level", 0)):
            alert_type = "level_up"
            alert_messages.append(
                f"健康等级由 {LEVEL_NAMES[int(previous['level'])]} 升至 {LEVEL_NAMES[level]}"
            )
        if stage > int(previous.get("stage", 0)):
            alert_type = alert_type or "stage_up"
            alert_messages.append(
                f"健康阶段由 {STAGE_NAMES.get(int(previous['stage']), '未知')} 转为 {STAGE_NAMES.get(stage, '未知')}"
            )
        if rms_ratio >= 2.0 and float(previous.get("rms_ratio", 0.0)) < 2.0:
            alert_type = alert_type or "rms_high"
            alert_messages.append(f"振动 RMS 达到基线的 {rms_ratio:.1f} 倍")
        if env_ratio >= 2.0 and float(previous.get("env_ratio", 0.0)) < 2.0:
            alert_type = alert_type or "env_high"
            alert_messages.append(f"包络峰值达到基线的 {env_ratio:.1f} 倍")
        if delta is not None and delta <= -10:
            alert_type = alert_type or "score_drop"
            alert_messages.append(f"健康分在 5 个文件内下降 {-delta:.1f} 分")
        elif trend == "恶化" and previous.get("trend") != "恶化":
            alert_type = alert_type or "trend_down"
            alert_messages.append("健康分连续下降，进入恶化趋势")

    unique_messages = []
    for message in alert_messages:
        if message not in seen_alerts:
            seen_alerts.add(message)
            unique_messages.append(message)
    alert_messages = unique_messages
    if not alert_messages:
        alert_type = ""
    elif len(alert_messages) > 1:
        alert_type = alert_type or "multiple"

    return HealthPoint(
        condition=str(row.get("condition", "")),
        bearing=str(row.get("bearing", "")),
        bearing_key=str(row.get("bearing_key", "")),
        file_idx=int(_num(row.get("file_idx"), 0)),
        stage=stage,
        rel_rul=round(rel_rul, 4),
        rms_mean=round(rms_mean, 6),
        kurtosis_v=round(kurtosis_v, 6),
        env_peak=round(env_peak, 6),
        score=round(score, 2),
        level=level,
        level_name=LEVEL_NAMES[level],
        trend=trend,
        rms_ratio=round(rms_ratio, 4),
        env_ratio=round(env_ratio, 4),
        kurt_delta=round(kurt_delta, 4),
        rms_score=round(rms_score, 2),
        kurt_score=round(kurt_score, 2),
        env_score=round(env_score, 2),
        stage_score=round(stage_score, 2),
        rul_score=round(rul_score, 2),
        explanation=explanation,
        alert_type=alert_type,
        alert_message="；".join(alert_messages),
    )


def score_group(group: pd.DataFrame) -> list[HealthPoint]:
    group = group.sort_values("file_idx").reset_index(drop=True)
    baseline = build_baseline(group)
    points: list[HealthPoint] = []
    recent_scores: list[float] = []
    previous: dict[str, Any] | None = None
    seen_alerts: set[str] = set()
    for _, row in group.iterrows():
        point = _score_row(row, baseline, recent_scores, previous, seen_alerts)
        points.append(point)
        recent_scores.append(point.score)
        if len(recent_scores) > 12:
            recent_scores.pop(0)
        previous = asdict(point)
    return points


def score_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    required = {"condition", "bearing", "bearing_key", "file_idx", "stage", "rel_rul", "rms_mean", "kurtosis_v", *ENV_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"特征表缺少字段：{', '.join(sorted(missing))}")

    frames: list[pd.DataFrame] = []
    alerts_by_device: dict[str, list[dict[str, Any]]] = {}
    for bearing_key, group in df.groupby("bearing_key", sort=False):
        points = score_group(group)
        frames.append(pd.DataFrame([asdict(point) for point in points]))
        alerts = [
            {
                "id": f"{bearing_key}-{point.file_idx}",
                "fileIdx": point.file_idx,
                "level": point.level,
                "levelName": point.level_name,
                "type": point.alert_type,
                "message": point.alert_message,
            }
            for point in points
            if point.alert_message
        ]
        alerts_by_device[str(bearing_key)] = alerts

    scored = pd.concat(frames, ignore_index=True)
    return scored, alerts_by_device


def score_bearing(df: pd.DataFrame, query: str) -> list[HealthPoint]:
    query = query.strip()
    matches = df[
        df["bearing_key"].astype(str).str.contains(query, case=False, regex=False)
        | df["bearing"].astype(str).str.contains(query, case=False, regex=False)
    ]
    if matches.empty:
        return []
    bearing_key = str(matches.iloc[0]["bearing_key"])
    return score_group(df[df["bearing_key"].astype(str) == bearing_key])


def build_device_summary(
    scored: pd.DataFrame,
    alerts_by_device: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    devices = []
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for bearing_key, group in scored.groupby("bearing_key", sort=False):
        group = group.sort_values("file_idx")
        latest = group.iloc[-1]
        level = int(latest["level"])
        counts[level] += 1
        devices.append(
            {
                "id": str(bearing_key),
                "name": str(latest["bearing"]),
                "condition": str(latest["condition"]),
                "totalFiles": int(len(group)),
                "fileIdx": int(latest["file_idx"]),
                "score": float(latest["score"]),
                "level": level,
                "levelName": LEVEL_NAMES[level],
                "stage": int(latest["stage"]),
                "stageName": STAGE_NAMES.get(int(latest["stage"]), "未知"),
                "relRul": float(latest["rel_rul"]),
                "trend": str(latest["trend"]),
                "rmsRatio": float(latest["rms_ratio"]),
                "envRatio": float(latest["env_ratio"]),
                "alertCount": len(alerts_by_device.get(str(bearing_key), [])),
                "latestAlerts": alerts_by_device.get(str(bearing_key), [])[-5:],
            }
        )
    devices.sort(key=lambda item: (-item["level"], item["score"]))
    return {
        "generatedAt": pd.Timestamp.now().isoformat(timespec="seconds"),
        "summary": {
            "total": len(devices),
            "healthy": counts[0],
            "attention": counts[1],
            "warning": counts[2],
            "danger": counts[3],
        },
        "devices": devices,
    }