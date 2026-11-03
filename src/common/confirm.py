"""离线模式确认：在线大模型不可用时，不自动降级，先征求用户同意。"""
from __future__ import annotations

import os


def confirm_offline(reason: str = "") -> bool:
    """返回 True 才允许使用离线规则引擎；非交互环境默认拒绝。"""
    if os.environ.get("AGENT_OFFLINE_CONFIRM", "").strip().lower() in {"1", "y", "yes", "true"}:
        print("[离线确认] AGENT_OFFLINE_CONFIRM=yes，使用离线规则引擎。")
        return True
    print("\n[离线确认] " + (reason or "在线大模型不可用") + "。")
    print("不会自动切换离线模式。")
    try:
        answer = input("是否改用离线规则引擎？[y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in {"y", "yes", "是", "1"}
