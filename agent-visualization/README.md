# 工业数据可视化智能体（TypeScript）

用自然语言分析工业数据：选择内置实训结果或上传 CSV，所有提问统一交给在线大模型理解。模型可以自由回答字段含义、指标解释、实验结论和工程建议，也可以在需要时返回受约束的图表规格，由 ECharts 渲染成交互式图表。除了表格分析，界面还支持附加金属表面图片，调用本地 YOLOv8n 检测六类表面缺陷，并由在线模型解读检测结果。

![界面预览](assets/demo.png)

## 功能

- **主界面 → 工作区**：主界面先选择数据、输入问题或附加图片，确认后才进入对话工作区；工作区左侧固定展示历史会话。
- **折叠式数据面板**：主界面和工作区的对话框顶部都有一个可展开/收起的数据面板，展开后选择 10 个内置数据集或上传 CSV，选择后自动收起。
- **YOLO 缺陷检测**：附加图片后界面自动切换为「表面缺陷检测模式」，调用本地 `best.pt` 输出缺陷框、类别计数与标注图，再由在线模型解读；检测期间不读取数据集，也可一键退出检测模式。
- **历史会话**：对话自动保存到浏览器 localStorage（最多 15 条），支持恢复、单条删除与清空，刷新或重开浏览器后仍在。
- **内置实训结果**：轴承特征、模型对比、RUL、ResNet/EfficientNet、YOLO、消融实验等 10 个数据集。
- **CSV 上传**：自动推断数值/类别/日期字段，支持任意列名与中文表头。
- **统一在线模型**：问候、字段说明、指标解读、实验结论、工程建议和图表请求全部由在线模型回答，不再用本地固定话术分流。
- **自由回答 + 可选图表**：模型自行判断是否需要可视化；需要时返回折线、柱状、散点、箱线、热力、饼图或直方图规格，不需要时只给文字回答，不会把输出限死为图表。
- **对话上下文**：前端把最近 8 轮对话带回服务端，支持“那 RMS 呢”“换成柱状图”这类追问。
- **纯在线模式**：未配置 key 或模型调用失败时直接返回错误，不提供离线规则引擎，也不会用本地模板冒充模型回答。网络超时或上游 5xx 会自动重试一次；图表规格不合法会自动让模型修正一次。
- **服务端保管密钥**：浏览器不接触 API key；只把字段画像、少量样本行和问题发给模型，不上传整份数据。
- **导出结果**：图表可导出 PNG，YOLO 标注图可导出 JPG。
- **可分享演示链接**：`http://localhost:5173/?q=画出 Bearing1_1 的 RMS 随文件序号变化` 自动执行一次提问。

## 快速开始

```bash
cd agent-visualization
npm install

# 必须配置在线大模型；未配置时提问会直接报错
cp .env.example .env
# 或直接复用项目根目录 .env 中的 DEEPSEEK_API_KEY

npm run dev
# 前端 http://localhost:5173
# 后端 http://localhost:8787
```

生产模式：

```bash
npm run build
npm start
# 打开 http://localhost:8787
```

要求 Node 20+（本机 Node 25），npm 10+。YOLO 缺陷检测默认读取项目根目录的 `.venv/bin/python` 与 `results/week2_3/yolo/train/weights/best.pt`；如果在本机其他位置，可通过 `VIZ_PYTHON`、`VIZ_YOLO_MODEL` 覆盖。修改后端代码后需重启 `npm run dev`。

## 环境变量

优先级：`agent-visualization/.env` > 项目根目录 `.env` > 进程环境变量。

| 变量 | 说明 | 默认 |
| --- | --- | --- |
| `VIZ_LLM_API_KEY` | 可视化智能体专用 key | 回退 `DEEPSEEK_API_KEY` |
| `VIZ_LLM_BASE_URL` | OpenAI 兼容地址 | `https://api.deepseek.com` |
| `VIZ_LLM_MODEL` | 模型名；当前可用 `deepseek-flash` / `deepseek-v4-pro` | `deepseek-chat`（兼容别名，实际映射 flash） |
| `PORT` | 后端端口 | `8787` |
| `VIZ_PYTHON` | YOLO 推理使用的 Python 解释器 | 项目根目录 `.venv/bin/python`，不存在则 `python3` |
| `VIZ_YOLO_MODEL` | YOLO 权重路径 | `results/week2_3/yolo/train/weights/best.pt` |
| `VIZ_YOLO_CONF` | 检测置信度阈值 | `0.25` |

任何 OpenAI 兼容服务都可以用，只要实现 `POST /chat/completions`。密钥只存在服务端，`.env` 已被 `.gitignore` 忽略。选择模型时：`deepseek-flash` 响应快，适合图表规划；`deepseek-v4-pro` 更适合长报告与复杂推理。

## 数据

- `agent-visualization/data/` 是随仓库分发的小型快照，保证克隆后即可演示。
- 如果本机存在 `results/week1/features.csv`，会优先使用完整 9216 行数据；否则自动使用 `bearing_features_sample.csv`（600 行分层抽样）。
- CSV 上传文件只保存在服务端内存（限制 25 MB、最多 5 万行），重启后清空。
- 缺陷检测图片限制 20 MB，写入系统临时目录，请求结束后立即清理，不落盘到项目目录。

内置数据集：

| 数据集 | 内容 |
| --- | --- |
| 轴承特征全量 | 56 列：工况、轴承、健康阶段、相对 RUL、时域/频域/包络特征 |
| 轴承模型对比 | RF / SVM / KNN / LR / CNN / LSTM / RNN 的准确率与 macroF1 |
| 四模型 CV / 留出指标 | 第 1 周四模型的 CV 与按轴承留出测试指标 |
| CNN / LSTM / RF | 第 2 周深度模型与随机森林同口径对比 |
| RUL 回归对比 | GBoost / RF / LSTM 的 MAE、RMSE、R² |
| 特征重要性排序 | 互信息与 ANOVA 排序 |
| ResNet / EfficientNet | NEU-DET 六分类预训练/从零对照 |
| YOLO 分类别指标 | 六类缺陷的 P / R / mAP50 / mAP50-95 |
| 学习率 × 损失消融 | 3 × 3 网格的测试准确率 |
| YOLO 总体指标 | mAP50、mAP50-95、mP、mR |

## YOLO 缺陷检测

附加图片后，界面自动切换到「表面缺陷检测模式」，数据集面板与数据相关问题建议会隐藏：

- 前端把图片与可选问题提交到 `POST /api/vision/detect`；
- 服务端把图片写入系统临时目录，通过 `execFile` 调用 `python/yolo_detect.py`（参数以数组传递，不经过 shell）；
- Ultralytics 加载 `best.pt`，输出检测框、类别、置信度、图像尺寸与耗时，并生成 Base64 JPEG 标注图；
- 服务端汇总类别计数后调用在线模型解读；模型失败时回退到模板化结论并返回 `warning`，检测本身不受影响；
- 结果面板展示标注图、类别计数、模型、阈值与图片尺寸，可导出 JPG；历史会话会记住这次检测结果。

支持类别：crazing（龟裂）、inclusion（夹杂）、patches（斑块）、pitted_surface（麻点）、rolled_in_scale（氧化铁皮压入）、scratches（划伤）。默认阈值 0.25，可用 `VIZ_YOLO_CONF` 调整。

## 架构

```text
web/（React + Vite + ECharts，home / workspace 双视图）
   │  POST /api/chat { datasetId, question, history }
   │  POST /api/vision/detect（multipart：file + question）
   ▼
server/index.ts (Express + TypeScript)
   ├── datasets.ts    CSV 读取、字段画像、内置数据、上传缓存
   ├── llm.ts         统一 Agent 调用：数据画像 + 对话历史 → 自由回答 + 可选图表规格（zod 校验）
   ├── chart.ts       聚合/分组/分位数/热力矩阵 → ECharts option
   └── vision.ts      调用 python/yolo_detect.py → 检测框、类别计数、Base64 标注图
```

前端把会话写入 localStorage（键 `viz-agent:sessions:v1`，最多 15 条）；每次提问或检测使用独立的会话令牌，切换数据集、新建对话或打开历史会话后，旧请求结果会被丢弃。

图表规格是一个受限 JSON：

```json
{
  "type": "line",
  "title": "RMS 均值 随 文件序号 变化",
  "x": "file_idx",
  "y": "rms_mean",
  "series": "stage",
  "agg": "mean",
  "filter": { "column": "bearing", "value": "Bearing1_1" },
  "sort": "asc",
  "limit": 300
}
```

在线模型返回 `{ "reply": "...", "chart": {...} }`：`reply` 是自由文字回答，`chart` 可以为 `null`。服务端会校验图表规格中的每个字段真实存在、y 必须是数值列，再做聚合与渲染；图表规格不合法时本次只返回文字回答，不会把整次请求判失败。智能体没有离线规则通道：未配置 key 返回 503，模型调用失败返回 502。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 服务状态、LLM 配置与 YOLO 就绪信息 |
| GET | `/api/datasets` | 内置数据集元信息 |
| GET | `/api/datasets/:id` | 单个数据集元信息 |
| POST | `/api/datasets/upload` | multipart `file=<csv>` |
| POST | `/api/chat` | `{ "datasetId": "...", "question": "...", "history": [...] }`；返回 `kind: "chat" \| "chart"`，未配置 503、调用失败 502 |
| POST | `/api/vision/detect` | multipart `file=<image>` + `question`；返回 `kind: "vision"`、`detections`、`counts`、`image`（data URL）、`model`、`conf`、`width`、`height`、`elapsedMs` |

## 示例问题

需要图表：

- 画出 Bearing1_1 的 RMS 随文件序号变化
- 按工况对比 RMS 分布
- 健康阶段与 RMS 的关系
- 对比各模型的测试准确率和 macroF1
- 对比 RUL 模型的测试 MAE 和 RMSE
- 画出六类缺陷的 mAP50 柱状图
- 画学习率×损失函数的测试准确率热力图
- 画互信息 Top 15 特征条形图

不需要图表（同样走在线模型）：

- 你好，介绍一下你能做什么
- 这份数据有哪些字段，分别是什么意思
- RUL 的 R² 为负说明什么
- 为什么 RNN/LSTM 准确率高但 macroF1 很低
- 六类缺陷里哪类最难检测，可能的原因是什么

缺陷检测（先在对话顶部附加图片）：

- 这张图主要是什么缺陷？
- 缺陷在哪些位置，数量是多少？
- 这些缺陷可能由什么工艺导致，该如何处理？

## 测试与类型检查

```bash
npm run typecheck
npm run build
```

本项目与主仓库的 Python 实训代码相互独立；可视化智能体只读取 `results/` 或自身 `data/` 中的 CSV，以及 `results/week2_3/yolo/train/weights/best.pt` 做推理，不修改训练产物。
