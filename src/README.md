# 代码目录

按三周任务推进，模块之间互不依赖，公共工具放 `common/`。
所有脚本均可直接 `python src/<目录>/<脚本>.py` 运行（脚本内部自行处理 sys.path）；数据目录与输出目录用参数指定，如 `--data-dir`（轴承线）、`--neu-dir`（NEU 线）、`--out`（输出）。

| 目录 | 对应进度 | 内容 |
| --- | --- | --- |
| `week1-bearing-ml/` | 第 1 周 | `01_explore` 数据探索与退化曲线 → `02_features` 时域/频域/包络谱特征（50 维）与健康阶段标注 → `03_correlation` 相关性分析与特征筛选（Top-20） → `04_classify` 随机森林/SVM/KNN/逻辑回归（health 与 condition 两任务、轴承留出与随机两种划分） → `05_rul_regression` 随机森林/梯度提升 RUL 回归 |
| `week2-bearing-dl/` | 第 2 周上半 | `01_verify_env` PyTorch 环境自检 → `02_rnn_classifier` RNN 健康阶段分类 → `03_lstm_classifier` LSTM（自动与 RNN 对比） → `04_lstm_rul` 滑动窗口 LSTM 剩余寿命回归 → `05_cnn_classifier` 1D-CNN（汇总 CNN vs LSTM vs 随机森林对比表） |
| `week2-3-surface-defect/` | 第 2 周下半～第 3 周 | `01_explore_neu` NEU-DET 探索（自动识别 classification/YOLO/VOC 布局） → `02_resnet_classify` ResNet18 迁移学习 → `03_efficientnet_classify` EfficientNet-B0 → `convert_voc_to_yolo` VOC 转 YOLO 标注 → `04_yolo_detect` YOLOv8n 缺陷检测 → `05_optimize` 学习率×损失函数（CE/标签平滑/Focal）消融 |
| `agent-maintenance/` | 智能体方向 | 设备运维诊断智能体：`anomaly` Isolation Forest 异常检测 → `diagnose` 编排特征提取/分类/RUL/包络谱定位 → `llm` DeepSeek 客户端（环境变量/本地 .env，离线自动降级） → `report` 结构化诊断报告 → `main` 对话入口（/scan /report /ask） |
| `agent-inspection/` | 智能体方向 | 视觉质检智能体：`detect_tool` ResNet 模型作工具 → `store` SQLite 检测台账 → `query` 自然语言统计查询 → `alert` 连续同类缺陷告警 → `rag` TF-IDF 工艺文件检索 → `main` 对话入口（/inspect /query /alerts /rag）；`craft_docs/` 为 6 类缺陷 + 应急规范的模拟工艺文件 |
| `common/` | 全程 | `bearing` 数据发现/加载/健康阶段标注/轴承几何与故障特征频率；`features` 特征计算；`neu` NEU-DET 布局识别与切分；`dl` PyTorch 训练工具；`cv` 图像分类流水线与 Focal Loss；`plotting` 中文绘图；`synthesize` 合成 fixture（冒烟测试用） |
