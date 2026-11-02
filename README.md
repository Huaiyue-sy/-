# 工业数据分析与挖掘综合实训

> 基于 **XJTU-SY 轴承振动数据** 与 **NEU-DET 金属表面瑕疵图像** 的故障诊断、剩余寿命预测、缺陷分类检测与工业智能体实践。

项目覆盖「数据探索 → 特征工程 → 建模评估 → 调优消融 → 智能体集成」完整闭环，全部脚本可直接运行，实验数字均来自真实数据集的完整运行。

| 数据线 | 任务 | 代表方法 | 主要结果 |
| --- | --- | --- | --- |
| XJTU-SY 轴承 | 健康阶段三分类（按轴承留出） | 手工特征 + RandomForest | 准确率 **0.7820**，macroF1 **0.7202** |
| XJTU-SY 轴承 | 剩余寿命回归 | GradientBoosting / LSTM | 测试 MAE **0.2615**（GBoost） |
| NEU-DET 表面 | 六类瑕疵分类 | ResNet18 / EfficientNet-B0 迁移学习 | 准确率 **0.9944** / **0.9889** |
| NEU-DET 表面 | 缺陷检测 | YOLOv8n 预训练微调 | mAP50 **0.7703**，mAP50-95 **0.4362** |
| 智能体 | 设备运维诊断 / 视觉质检 | IsolationForest + 分类/RUL + RAG | 两个可离线演示的 CLI 智能体 |

| 缺陷检测（YOLOv8n） | 学习率 × 损失消融 |
| :---: | :---: |
| ![YOLO 预测](results/week2_3/yolo/predict/crazing_10.jpg) | ![消融热力图](results/week2_3/optimization_heatmap.png) |

## 项目结构

```
.
├── src/
│   ├── common/                  # 数据加载、50 维特征、DL/CV 训练公共件、中文绘图
│   ├── week1-bearing-ml/        # 探索 → 特征 → 相关性 → 分类 → RUL 回归
│   ├── week2-bearing-dl/        # RNN / LSTM / 1D-CNN / LSTM-RUL
│   ├── week2-3-surface-defect/  # NEU-DET 探索、ResNet/EfficientNet、YOLO、消融
│   ├── agent-maintenance/       # 设备运维诊断智能体
│   └── agent-inspection/        # 视觉质检智能体（SQLite 台账 + RAG + 预警）
├── scripts/smoke_test.sh        # 合成小数据全链路自检（18 个入口脚本）
├── config/                      # 轴承几何参数
├── docs/                        # 实训指南与规划思维导图
├── data/README.md               # 数据集布局与下载说明
├── results/                     # 报告图表与 YOLO 汇总（完整训练产物本地生成，不入库）
├── 实训手册/                     # 12 天每日记录
└── 实训报告/                     # 完整实训报告
```

## 快速开始

要求 Python 3.13（3.10+ 亦可）。依赖：numpy / scipy / scikit-learn / pandas / matplotlib / torch / torchvision / ultralytics。

```bash
git clone https://github.com/Huaiyue-sy/-.git
cd 工业数据分析与挖掘综合实训

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 数据集不随仓库分发，放置方式见 data/README.md
# XJTU-SY → data/XJTU-SY_Bearing_Datasets/
# NEU-DET  → data/NEU-DET/（分类）与 data/NEU-DET-yolo/（检测）

# 用合成小数据验证全部入口可运行（不会污染正式结果）
bash scripts/smoke_test.sh
```

## 复现实验

```bash
# 第 1 周：轴承特征工程与传统机器学习
.venv/bin/python src/week1-bearing-ml/01_explore.py
.venv/bin/python src/week1-bearing-ml/02_features.py
.venv/bin/python src/week1-bearing-ml/03_correlation.py
.venv/bin/python src/week1-bearing-ml/04_classify.py
.venv/bin/python src/week1-bearing-ml/05_rul_regression.py

# 第 2 周：RNN / LSTM / CNN 与 LSTM-RUL
.venv/bin/python src/week2-bearing-dl/01_verify_env.py
.venv/bin/python src/week2-bearing-dl/02_rnn_classifier.py
.venv/bin/python src/week2-bearing-dl/03_lstm_classifier.py
.venv/bin/python src/week2-bearing-dl/04_lstm_rul.py
.venv/bin/python src/week2-bearing-dl/05_cnn_classifier.py

# 第 2-3 周：NEU-DET 分类、检测与消融
.venv/bin/python src/week2-3-surface-defect/01_explore_neu.py --neu-dir data/NEU-DET
.venv/bin/python src/week2-3-surface-defect/02_resnet_classify.py
.venv/bin/python src/week2-3-surface-defect/03_efficientnet_classify.py
# 检测需 YOLO 布局：VOC 版先跑 convert_voc_to_yolo.py；yolov8n.pt 放项目根目录
.venv/bin/python src/week2-3-surface-defect/04_yolo_detect.py --neu-dir data/NEU-DET-yolo
.venv/bin/python src/week2-3-surface-defect/05_optimize.py
```

## 智能体演示

**设备运维诊断**（异常检测 → 健康阶段 → RUL → 包络谱定位 → 报告）：

```bash
.venv/bin/python src/agent-maintenance/main.py
/scan data/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1/123.csv
/report data/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1/123.csv
/ask 外圈故障应该怎么处理？
```

**视觉质检**（检测入库 → 自然语言查询 → 连续缺陷预警 → 工艺 RAG）：

```bash
.venv/bin/python src/agent-inspection/main.py
/inspect data/NEU-DET/IMAGES/scratches_10.jpg 3号机台
/inspect data/NEU-DET/IMAGES/scratches_11.jpg 3号机台
/inspect data/NEU-DET/IMAGES/scratches_12.jpg 3号机台
/query 最近哪种缺陷最多？
/alerts
/query 麻点的工艺原因是什么？
```

两个智能体均支持 **DeepSeek API 优先、离线规则降级**：设置 `DEEPSEEK_API_KEY` 时由大模型生成报告，无 key 或调用失败时自动切换到同结构的规则模板，断网也能完整演示。

> 智能体依赖 `models/` 下的分类、RUL 与 ResNet 权重（如 `week1_classifier.joblib`、`resnet18_neu.pth`）。这些文件体积较大且不入库，请先按上一节命令训练生成，或把本机 `models/` 目录复制过来。

## 评估口径

- 轴承任务默认 **按轴承留出**（Bearing1_4 / Bearing2_3 / Bearing3_2 测试），避免相邻文件泄漏；随机划分仅作对照。
- 图像任务分层 **8:1:1** 划分（seed=42）；分类看准确率与 macroF1，检测看 mAP50 / mAP50-95。
- 第 1 周同轴承 5 折 CV 可达 0.99，但跨轴承留出后 RF 为 0.7820；RNN/LSTM 的 0.8347 实为全部判成“健康”（macroF1 0.3033）。报告如实保留这些负面对照。

## 文档

- [实训报告](实训报告/实训报告.md)：需求分析、系统设计、编码实现、系统测试、智能体应用、问题与改进
- [实训手册](实训手册/)：12 天每日记录（目标、完成情况、问题与解决、次日计划）
- [数据说明](data/README.md)：两个数据集的下载、目录布局与几何参数
- [代码目录说明](src/README.md)：每个脚本的输入输出与用法

## 环境说明

本机实验环境为 macOS（Apple M5，PyTorch 2.15 CPU/MPS）。NVIDIA 机器可按实训指南安装 CUDA 版 PyTorch；所有脚本通过命令行参数指定数据与输出目录，不依赖固定路径。

## 致谢

- XJTU-SY 滚动轴承加速寿命试验数据集（西安交通大学）
- NEU-DET 热轧带钢表面缺陷数据集（东北大学）
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

本项目为课程实训作品，未附开源许可证；复用数据集与预训练权重时请遵循其原始许可。
