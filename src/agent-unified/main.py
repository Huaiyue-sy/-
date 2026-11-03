"""统一工业智能体：设备运维诊断 + 视觉质检 + 基础对话。

一个入口同时挂载两个原有智能体的工具，并支持自然语言路由：

    python3 src/agent-unified/main.py

示例：
    诊断 data/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1/123.csv
    生成报告 data/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1/123.csv
    检测 data/NEU-DET/IMAGES/scratches_10.jpg 3号机台
    3号机台最近有什么异常？
    划伤是什么工艺原因？
    项目目前的模型指标怎么样？
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "agent-maintenance"))
sys.path.insert(0, str(SRC / "agent-inspection"))

from common.bearing import BearingGeometry, condition_rpm  # noqa: E402
from common.neu import CLASS_CN  # noqa: E402

import alert  # noqa: E402
import query  # noqa: E402
import rag  # noqa: E402
import store  # noqa: E402
from chat import ChatSession  # noqa: E402
from detect_tool import VisionTool  # noqa: E402
from diagnose import MaintenanceAgent  # noqa: E402
from llm import DeepSeekClient, llm_backend_name  # noqa: E402
from report import build_report, diagnosis_facts  # noqa: E402
from router import is_image, route  # noqa: E402

CONDITION_RE = re.compile(r"\d+(?:\.\d+)?Hz\d+(?:\.\d+)?kN", re.IGNORECASE)


class UnifiedAgent:
    def __init__(self, args: argparse.Namespace):
        self.llm = DeepSeekClient()
        self.chat = ChatSession(self.llm)
        self.maint = MaintenanceAgent(args.features, args.classifier, args.rul, args.contamination)
        self.vision: VisionTool | None = None
        self.vision_error = ""
        try:
            self.vision = VisionTool(args.vision_model)
        except Exception as e:  # 模型缺失时仍允许运维诊断和对话
            self.vision_error = str(e)
        self.conn = store.connect(args.db)
        self.knowledge = rag.CraftRAG()

        print("== 工业智能体（运维诊断 + 视觉质检 + 对话） ==")
        print(f"大模型后端：{llm_backend_name(self.llm)}")
        print(f"运维模型：分类={'已加载' if self.maint.classifier is not None else '未加载'}，"
              f"RUL={'已加载' if self.maint.rul_model is not None else '未加载'}")
        if self.vision:
            acc = f"，测试集准确率 {self.vision.test_acc:.4f}" if self.vision.test_acc else ""
            print(f"视觉模型：{self.vision.model_path}{acc}")
        else:
            print(f"视觉模型：不可用（{self.vision_error.splitlines()[0]}）")
        print(f"记忆库：{args.db}（当前 {store.total(self.conn)} 条记录）")
        print("输入 /help 查看命令；也可以直接说“诊断 xxx.csv”“检测 xxx.jpg 3号机台”。\n")

    def close(self) -> None:
        self.conn.close()

    # ---------- 工具 ----------
    def _condition(self, path: str, explicit: str | None = None) -> str:
        if explicit:
            return explicit
        m = CONDITION_RE.search(path)
        return m.group(0) if m else "35Hz12kN"

    def scan(self, path: str, condition: str | None = None, report: bool = False) -> str:
        p = Path(path)
        if not p.exists():
            return f"文件不存在：{path}"
        cond = self._condition(path, condition)
        diag = self.maint.diagnose_file(str(p), cond, BearingGeometry())
        summary = (
            f"异常判定：{'异常' if diag.is_anomaly else '正常'}（异常分 {diag.anomaly_score:.4f}）\n"
            f"健康阶段：{diag.stage_name}\n"
            f"疑似故障元件：{diag.fault_location}\n"
            f"证据：{diag.fault_evidence}\n"
            f"{diag.rul_note}"
        )
        self.chat.remember_tool(f"运维诊断 {path}（{cond}）：\n{diagnosis_facts(diag, cond)}")
        if report:
            return build_report(self.llm, diag, cond)
        try:
            rpm = condition_rpm(cond)
            head = f"工况 {cond}（{rpm} rpm）"
        except Exception:
            head = f"工况 {cond}"
        return f"{head}\n{summary}"

    def inspect(self, target: str, machine: str = "1号机台") -> str:
        if self.vision is None:
            return f"视觉模型不可用：{self.vision_error}"
        p = Path(target)
        if p.is_file():
            images = [p]
        elif p.is_dir():
            images = self.vision.inspect_dir(p)
        else:
            return f"路径不存在：{target}"
        if not images:
            return f"{target} 下没有图片。"
        lines = [f"开始检测 {len(images)} 张图（记入 {machine}）…"]
        for image in images:
            cls, conf = self.vision.inspect(image)
            store.record(self.conn, machine, str(image), cls, conf)
            mark = "⚠" if conf < 0.5 else "✔"
            lines.append(f"  {mark} {image.name} -> {CLASS_CN.get(cls, cls)}（置信度 {conf:.3f}）")
        alerts = alert.find_alerts(self.conn, self.knowledge)
        if alerts:
            lines.extend(["", *alerts])
        self.chat.remember_tool(f"视觉质检 {target}（{machine}）：\n" + "\n".join(lines))
        return "\n".join(lines)

    def inspect_query(self, text: str) -> str:
        answer = query.answer(text, self.conn, self.knowledge)
        self.chat.remember_tool(f"质检查询：{text}\n{answer}")
        return answer

    def alerts(self) -> str:
        found = alert.find_alerts(self.conn, self.knowledge)
        return "\n\n".join(found) if found else "当前无预警。"

    def status(self) -> str:
        return alert.status_report(self.conn, self.knowledge)

    def rag_lookup(self, text: str) -> str:
        hits = self.knowledge.retrieve(text, top_k=2)
        if not hits:
            return "未检索到相关工艺文件。"
        return "\n".join(
            f"\n《{f.stem}》（相似度 {score:.2f}）：\n{self.knowledge.snippet(f, 350)}"
            for f, score in hits
        )

    # ---------- 分派 ----------
    def dispatch_text(self, text: str) -> str:
        plan = route(text)
        if plan.tool == "maintenance_scan":
            if plan.path:
                return self.scan(plan.path, plan.condition)
            return "请给出振动 CSV 路径，例如：诊断 data/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1/123.csv"
        if plan.tool == "maintenance_report":
            if plan.path:
                return self.scan(plan.path, plan.condition, report=True)
            return "请给出振动 CSV 路径，例如：生成报告 data/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1/123.csv"
        if plan.tool == "inspect":
            if plan.path:
                return self.inspect(plan.path, plan.machine or "1号机台")
            return "请给出图片或目录路径，例如：检测 data/NEU-DET/IMAGES/scratches_10.jpg 3号机台"
        if plan.tool == "query":
            return self.inspect_query(plan.text)
        if plan.tool == "alerts":
            return self.alerts()
        if plan.tool == "status":
            return self.status()
        if plan.tool == "rag":
            return self.rag_lookup(plan.text)
        return self.chat.ask(plan.text)

    # ---------- 命令 ----------
    def run_command(self, line: str) -> bool:
        cmd, _, rest = line.partition(" ")
        rest = rest.strip()
        if cmd in ("/quit", "/exit"):
            return False
        if cmd == "/help":
            print(__doc__)
            return True
        if cmd == "/scan":
            parts = rest.split()
            if not parts:
                print("用法：/scan <振动CSV> [工况，如 35Hz12kN]")
            else:
                print("\nAgent> " + self.scan(parts[0], parts[1] if len(parts) > 1 else None))
            return True
        if cmd == "/report":
            parts = rest.split()
            if not parts:
                print("用法：/report <振动CSV> [工况]")
            else:
                print("\nAgent> " + self.scan(parts[0], parts[1] if len(parts) > 1 else None, report=True))
            return True
        if cmd == "/inspect":
            parts = rest.split()
            if not parts:
                print("用法：/inspect <图片或目录> [机号，如 3号机台]")
            else:
                print("\nAgent> " + self.inspect(parts[0], parts[1] if len(parts) > 1 else "1号机台"))
            return True
        if cmd == "/query":
            print("\nAgent> " + (self.inspect_query(rest) if rest else "用法：/query <自然语言问题>"))
            return True
        if cmd == "/alerts":
            print("\nAgent> " + self.alerts())
            return True
        if cmd == "/status":
            print("\nAgent> " + self.status())
            return True
        if cmd == "/rag":
            print("\nAgent> " + (self.rag_lookup(rest) if rest else "用法：/rag <工艺问题>"))
            return True
        if cmd == "/chat":
            print("\nAgent> " + (self.chat.ask(rest) if rest else "用法：/chat <问题>"))
            return True
        if cmd == "/reset":
            self.chat.reset()
            print("Agent> 对话上下文已清空。")
            return True
        if cmd == "/history":
            lines = self.chat.history_lines()
            print("\nAgent> " + ("\n".join(lines) if lines else "暂无对话历史。"))
            return True
        print(f"未知命令：{cmd}，输入 /help 查看用法。")
        return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="results/week1/features.csv")
    ap.add_argument("--classifier", default="models/week1_classifier.joblib")
    ap.add_argument("--rul", default="models/week1_rul.joblib")
    ap.add_argument("--vision-model", default="models/resnet18_neu.pth")
    ap.add_argument("--db", default="results/agent-inspection/detections.db")
    ap.add_argument("--contamination", type=float, default=0.05)
    args = ap.parse_args()

    agent = UnifiedAgent(args)
    try:
        while True:
            try:
                line = input("你> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line.startswith("/"):
                if not agent.run_command(line):
                    break
            else:
                print("\nAgent> " + agent.dispatch_text(line) + "\n")
    finally:
        agent.close()


if __name__ == "__main__":
    main()
