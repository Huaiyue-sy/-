"""诊断报告生成：结构化报告的渲染、DeepSeek 提示词构造、离线规则降级。"""
from __future__ import annotations

from common.confirm import confirm_offline
from diagnose import Diagnosis

SEVERITY_BY_STAGE = {0: "正常", 1: "轻度退化", 2: "严重/故障"}
ADVICE_BY_LOCATION = {
    "外圈故障": ["停机检查外圈滚道表面是否存在剥落/点蚀",
               "检查轴承座配合与径向载荷分布",
               "计划更换轴承外圈或整套轴承"],
    "内圈故障": ["检查内圈滚道与轴颈配合面",
               "测量轴的对中与不平衡量",
               "必要时更换轴承并检查安装过盈量"],
    "滚动体故障": ["检查滚动体表面损伤与润滑脂状态",
                 "取样做润滑脂铁谱分析",
                 "更换轴承并排查润滑周期是否过长"],
    "保持架故障": ["检查保持架磨损与引导间隙",
                 "排查轴向载荷与转速是否超限",
                 "更换轴承并复核选型"],
}


def diagnosis_facts(d: Diagnosis, condition: str) -> str:
    """把诊断结果转成给大模型的事实文本（也用于离线报告）。"""
    lines = [
        f"设备工况：{condition}（转速 {condition_rpm_text(condition)} rpm）",
        f"信号文件：{d.file_path}",
        f"孤立森林异常分：{d.anomaly_score:.4f}（越小越异常）",
        f"异常判定：{'是' if d.is_anomaly else '否'}",
    ]
    if d.is_anomaly:
        lines.append(f"分类模型判定健康阶段：{d.stage_name}")
        lines.append(f"疑似故障元件：{d.fault_location}（包络谱证据 "
                     + "、".join(f"{k}={v}" for k, v in d.fault_evidence.items()) + "）")
        if d.rel_rul is not None:
            lines.append(f"剩余寿命预测：{d.rul_note}（rel_rul={d.rel_rul:.3f}）")
    return "\n".join(lines)


def condition_rpm_text(condition: str) -> str:
    import re

    m = re.search(r"(\d+(?:\.\d+)?)Hz", condition)
    return f"{float(m.group(1)) * 60:.0f}" if m else "未知"


REPORT_SYSTEM_PROMPT = (
    "你是钢铁/机械行业的资深设备诊断工程师（AI 老师傅）。"
    "根据给定的振动诊断事实，输出一份结构化、专业的中文诊断报告，"
    "使用 Markdown，包含以下小节：\n"
    "1. **诊断结论**（一句话给出故障元件与严重程度）\n"
    "2. **证据链**（结合异常分、健康阶段、包络谱证据逐条说明）\n"
    "3. **处置建议**（分停机检修/计划检修/持续观察三档，给出具体动作）\n"
    "4. **剩余寿命与监测计划**（如无 RUL 数据则说明依据不足）\n"
    "事实必须全部来自输入，不得编造数值；语言简洁、面向产线工人可执行。"
)


def offline_report(d: Diagnosis, condition: str) -> str:
    """离线规则引擎生成的结构化报告（无 API key 时的降级路径）。"""
    facts = diagnosis_facts(d, condition)
    severity = SEVERITY_BY_STAGE.get(d.stage, "未知") if d.is_anomaly else "正常"
    location = d.fault_location or "—"
    advice = ADVICE_BY_LOCATION.get(location, ["按点检周期加密监测"])
    md = [
        "# 设备诊断报告（离线规则引擎生成）",
        "",
        "## 诊断结论",
        f"该轴承振动信号{'存在异常' if d.is_anomaly else '未见异常'}",
        f"严重程度：**{severity}**" + (f"，疑似故障元件：**{location}**" if d.is_anomaly else ""),
        "",
        "## 证据链",
        facts,
        "",
        "## 处置建议",
    ]
    if d.is_anomaly:
        md += [f"{i + 1}. {a}" for i, a in enumerate(advice)]
        if d.rel_rul is not None:
            md.append(f"{len(advice) + 1}. {d.rul_note}，据此安排检修窗口")
    else:
        md += ["1. 维持现有点检周期", "2. 持续采集数据，纳入趋势监控"]
    md += ["", "## 剩余寿命与监测计划"]
    if d.rel_rul is not None:
        md.append(f"RUL 模型输出 rel_rul={d.rel_rul:.3f}。{d.rul_note}。建议每分钟采集一次数据，"
                  "若异常分持续下降（变负），立即上报。")
    else:
        md.append("RUL 模型未加载或数据不足，暂无法给出剩余寿命结论。")
    md += ["", "> 说明：本报告由离线规则引擎生成，配置 DEEPSEEK_API_KEY 后可获得大模型版报告。"]
    return "\n".join(md)


def build_report(llm, d: Diagnosis, condition: str, allow_offline: bool | None = None) -> str:
    """优先大模型生成；失败时先确认，确认后才使用离线规则引擎。"""
    facts = diagnosis_facts(d, condition)
    reply = llm.chat(REPORT_SYSTEM_PROMPT, f"诊断事实如下：\n\n{facts}\n\n请生成诊断报告。")
    if reply:
        return f"# 设备诊断报告（{llm_backend_text(llm)} 生成）\n\n{reply}"
    if allow_offline is False:
        return "在线报告生成失败，已按你的要求不使用离线模式。"
    if allow_offline is True or confirm_offline("在线报告生成失败"):
        return offline_report(d, condition)
    return "已取消离线报告生成。"


def llm_backend_text(llm) -> str:
    return "DeepSeek API" if getattr(llm, "available", False) else "离线规则引擎"
