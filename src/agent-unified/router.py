"""统一智能体的意图路由：把一句自然语言分派到运维诊断、视觉质检或普通对话。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CSV_RE = re.compile(r"((?:\.{0,2}/)?(?:[\w\-.]+/)*[\w\-.]+\.csv)", re.IGNORECASE)
IMG_RE = re.compile(r"((?:\.{0,2}/)?(?:[\w\-.]+/)*[\w\-.]+\.(?:jpg|jpeg|png|bmp|tif|tiff))", re.IGNORECASE)
CONDITION_RE = re.compile(r"\d+(?:\.\d+)?Hz\d+(?:\.\d+)?kN", re.IGNORECASE)
MACHINE_RE = re.compile(r"(\d+)\s*号\s*(?:机台|机器|机组|线)")

VISION_WORDS = ("缺陷", "划痕", "划伤", "龟裂", "夹杂", "斑块", "麻点", "压入", "质检", "检测", "机台")
MAINT_WORDS = ("轴承", "振动", "寿命", "rul", "故障", "外圈", "内圈", "包络", "诊断")


@dataclass
class Route:
    tool: str
    text: str
    path: str | None = None
    machine: str | None = None
    condition: str | None = None


def extract_path(text: str) -> str | None:
    m = CSV_RE.search(text) or IMG_RE.search(text)
    return m.group(1) if m else None


def extract_machine(text: str) -> str | None:
    m = MACHINE_RE.search(text)
    return f"{m.group(1)}号机台" if m else None


def extract_condition(text: str) -> str | None:
    m = CONDITION_RE.search(text)
    return m.group(0) if m else None


def is_image(path: str | None) -> bool:
    return bool(path and Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


def route(text: str) -> Route:
    q = text.strip()
    low = q.lower()
    path = extract_path(q)
    machine = extract_machine(q)
    condition = extract_condition(q)

    if path and path.lower().endswith(".csv"):
        tool = "maintenance_report" if re.search(r"报告|report", low) else "maintenance_scan"
        return Route(tool, q, path=path, condition=condition)

    if path and is_image(path):
        return Route("inspect", q, path=path, machine=machine or "1号机台")

    if re.search(r"预警|告警|报警", low):
        return Route("alerts", q)
    if re.search(r"台账|检测状态|质检状态|各机台", low) and any(w in low for w in VISION_WORDS):
        return Route("status", q)
    if any(w in low for w in VISION_WORDS):
        if re.search(r"工艺|原因|为什么|怎么造成", low):
            return Route("rag", q)
        return Route("query", q, machine=machine)

    if re.search(r"报告|诊断报告", low) and any(w in low for w in MAINT_WORDS):
        return Route("maintenance_report", q, path=path, condition=condition)
    if any(w in low for w in MAINT_WORDS) and path:
        return Route("maintenance_scan", q, path=path, condition=condition)

    return Route("chat", q)
