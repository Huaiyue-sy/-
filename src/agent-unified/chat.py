"""统一智能体的基础对话模块：带上下文记忆，API 不可用时先确认再离线。"""
from __future__ import annotations

from common.confirm import confirm_offline

SYSTEM_PROMPT = """你是「工业数据分析与挖掘综合实训」项目的统一智能体，集成了设备运维诊断、视觉质检和普通对话能力。

可用工具：
- 运维诊断：/scan <振动CSV>、/report <振动CSV>，输出异常分、健康阶段、包络谱故障元件和 RUL。
- 视觉质检：/inspect <图片或目录> [机号]、/query <自然语言>、/alerts、/status、/rag <工艺问题>。
- 普通对话：项目背景、方法、指标、使用方法。

项目事实（回答时只能使用这些数值，不得编造）：
- XJTU-SY 轴承：9216 个振动 CSV、15 套全寿命数据；按轴承留出的三分类中 RandomForest 准确率 0.7820、macroF1 0.7202。
- RNN/LSTM 在跨轴承留出上准确率 0.8347，但全部判成“健康”，macroF1 仅 0.3033；1D-CNN 0.8169 / 0.3143。
- RUL：GradientBoosting 测试 MAE 0.2615、RMSE 0.3013、R² -0.0891；跨轴承 R² 为负。
- NEU-DET：ResNet18 六分类准确率 0.9944，EfficientNet-B0 0.9889；从零训练 ResNet18 为 0.8944。
- YOLOv8n 检测：mAP50 0.7703、mAP50-95 0.4362；crazing 类别最难（mAP50 0.418）。
- 消融实验最优：lr=1e-4 + CE+标签平滑 0.1，测试准确率 1.0000。
- 可视化智能体：TypeScript Web 应用，支持 10 个内置数据集、CSV 上传、自然语言生成 ECharts 图表。

回答要求：中文、简洁、工程化；不确定就说不知道；涉及操作时给出命令示例。"""


class ChatSession:
    def __init__(self, llm):
        self.llm = llm
        self.history: list[dict[str, str]] = []
        self.last_context: str | None = None

    def reset(self) -> None:
        self.history.clear()
        self.last_context = None

    def remember_tool(self, summary: str) -> None:
        """把最近一次工具结果带入后续对话上下文。"""
        self.last_context = summary.strip()

    def history_lines(self, limit: int = 8) -> list[str]:
        return [f"{'你' if m['role'] == 'user' else 'Agent'}：{m['content']}"
                for m in self.history[-limit:]]

    def ask(self, text: str) -> str:
        question = text.strip()
        if not question:
            return "请输入问题。"

        if getattr(self.llm, "available", False):
            messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
            if self.last_context:
                messages.append({"role": "system", "content": f"最近一次工具结果：\n{self.last_context}"})
            messages.extend(self.history[-8:])
            messages.append({"role": "user", "content": question})
            reply = self.llm.chat_messages(messages, temperature=0.4, max_tokens=900)
            if reply:
                self.history.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": reply},
                ])
                self.history = self.history[-12:]
                return reply

        if not confirm_offline("在线对话不可用"):
            return "已取消离线回答。"
        reply = self._offline(question)
        self.history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": reply},
        ])
        self.history = self.history[-12:]
        return reply

    def _offline(self, question: str) -> str:
        q = question.lower()
        if any(w in q for w in ("你好", "您好", "hi", "hello", "你是谁", "能做什么", "怎么用", "帮助")):
            return (
                "我是统一智能体，可以把运维诊断、视觉质检和项目问答放在一个入口里。\n"
                "可以直接说：\n"
                "  · 诊断 data/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1/123.csv\n"
                "  · 生成报告 data/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1/123.csv\n"
                "  · 检测 data/NEU-DET/IMAGES/scratches_10.jpg 3号机台\n"
                "  · 3号机台最近有什么异常？\n"
                "  · 划伤是什么工艺原因？"
            )
        if any(w in q for w in ("指标", "效果", "结果", "准确率", "f1", "map", "模型", "项目")):
            return (
                "项目关键指标：\n"
                "  · 轴承三分类（按轴承留出）：RandomForest 0.7820 / macroF1 0.7202；"
                "RNN/LSTM 0.8347 但全判健康，macroF1 0.3033。\n"
                "  · RUL：GradientBoosting 测试 MAE 0.2615，跨轴承 R² -0.0891。\n"
                "  · NEU-DET：ResNet18 0.9944，EfficientNet-B0 0.9889；从零 ResNet18 0.8944。\n"
                "  · YOLOv8n：mAP50 0.7703，mAP50-95 0.4362，crazing 最差（mAP50 0.418）。\n"
                "  · 消融最优：lr=1e-4 + CE+LS 0.1，测试准确率 1.0000。"
            )
        if any(w in q for w in ("轴承", "故障", "诊断", "寿命", "rul", "包络")):
            return (
                "运维诊断用法：/scan <振动CSV> 做异常检测、健康阶段、包络定位和 RUL；"
                "/report <CSV> 生成结构化报告。没有 CSV 路径时我无法凭空诊断。\n"
                "示例：/scan data/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1/123.csv"
            )
        if any(w in q for w in ("质检", "缺陷", "划伤", "龟裂", "yolo", "resnet")):
            return (
                "视觉质检用法：/inspect <图片或目录> [机号] 检测并入库，/query 查台账，"
                "/alerts 看连续缺陷预警，/rag 查工艺原因。\n"
                "示例：/inspect data/NEU-DET/IMAGES/scratches_10.jpg 3号机台"
            )
        if any(w in q for w in ("可视化", "图表", "typescript", "web", "echarts")):
            return (
                "可视化智能体在 agent-visualization/：React + Vite + ECharts 前端、Express 后端，"
                "支持 10 个内置数据集和 CSV 上传。启动：cd agent-visualization && npm run dev，"
                "打开 http://localhost:5173。"
            )
        base = "当前未配置有效的在线大模型，只能回答项目相关问题。"
        if self.last_context:
            base += f"\n最近一次工具结果：\n{self.last_context}"
        return base + "\n你可以问：项目指标、运维诊断用法、视觉质检用法、可视化智能体，或直接给出 CSV/图片路径让我执行工具。"
