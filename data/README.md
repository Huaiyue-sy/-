# 数据集目录

`data/` 不入库（见 `.gitignore`）。两个数据集下载后按下述布局放置，各脚本的 `--data-dir` 参数默认指向这里的路径。

## 1. 西交大 XJTU-SY 轴承数据集

- 下载地址（百度网盘，见实训指南附录）：<https://pan.baidu.com/s/1OaY82azTXHBwjiCjA_jRcw>
- 参考论文：知网搜索《XJTU-SY滚动轴承加速寿命试验数据集解读》
- 官方教程（英文，Wang 等，MSSP）：<https://www.researchgate.net/publication/337421287_XJTU-SY_Rolling_Element_Bearing_Accelerated_Life_Test_Datasets_A_Tutorial>

### 期望布局

```
data/XJTU-SY_Bearing_Datasets/
├── 35Hz12kN/          # 2100rpm / 12kN
│   ├── Bearing1_1/
│   │   ├── 1.csv      # 每分钟采集一次，每次 1.28s
│   │   ├── 2.csv      # 32768 采样点 × 2 通道（水平/垂直）
│   │   └── ...csv     # 编号直到轴承失效
│   └── ...
├── 37.5Hz11kN/
└── 40Hz10kN/
```

- CSV 无表头、两列（部分整理版本带表头，加载器会自动识别）。
- 采样率 25.6 kHz；官方发布共 3 种工况、15 个轴承、9216 个 CSV（约 16 GB）。
  脚本支持 `--conditions`、`--bearings`、`--max-files` 过滤，数据量大时可只取部分。

### 轴承几何参数

包络谱特征（BPFO/BPFI/BSF/FTF 幅值）依赖轴承几何参数。试验轴承型号为 **LDK UER204**，
参数已按《XJTU-SY滚动轴承加速寿命试验数据集解读》（机械工程学报 2019）表 1 核实并写入
`src/common/bearing.py` 的 `BearingGeometry` 默认值：

| 参数 | 数值 |
| --- | --- |
| 滚珠个数 Z | 8 |
| 滚珠直径 d | 7.92 mm |
| 轴承中径 D | 34.55 mm |
| 接触角 α | 0° |

`config/xjtu_bearing_geometry.json` 内容与之一致；仅当使用其他型号轴承时才需要用
`--geometry-json` 覆盖：

```bash
python3 src/week1-bearing-ml/02_features.py --geometry-json config/xjtu_bearing_geometry.json
```

## 2. 东北大学 NEU-DET 金属表面瑕疵数据集

- 6 类缺陷：crazing（龟裂）、inclusion（夹杂）、patches（斑块）、pitted_surface（麻点）、rolled-in_scale（氧化铁皮压入）、scratches（划痕）
- 共 1800 张 200×200 灰度图，每类 300 张；检测版带矩形框标注
- 语雀介绍（见实训指南附录）：<https://www.yuque.com/lyliu/ifi7a4/ociycn3fgzfk17pw?singleDoc#>

### 期望布局

代码对**三类布局**都能识别（大小写不敏感），脚本用 `--neu-dir` 指向数据集根目录：

```
# 布局 A：分类用（每类一个文件夹）
data/NEU-DET/
├── crazing/*.jpg ... (6 类)

# 布局 B：检测用（YOLO 格式）
data/NEU-DET-yolo/
├── images/{train,valid,test}/*.jpg
└── labels/{train,valid,test}/*.txt   # YOLO 格式: class cx cy w h（归一化）

# 布局 C：VOC 格式（官方发布即为此布局：ANNOTATIONS/*.xml + IMAGES/*.jpg）
data/NEU-DET/
├── ANNOTATIONS/*.xml
└── IMAGES/*.jpg        # 文件名前缀即类别：crazing_1.jpg
```

官方 VOC 布局可直接跑分类脚本（按文件名前缀自动取类别标签）；检测脚本需先转换：

```bash
python3 src/week2-3-surface-defect/convert_voc_to_yolo.py --neu-dir data/NEU-DET --out data/NEU-DET-yolo
```

分类脚本（ResNet/EfficientNet）自动把图片按类别分层划分为 train/val/test（8:1:1，固定随机种子）；
检测脚本（YOLO）要求布局 B，若是布局 A/C 会给出明确报错提示。

## 3. 冒烟测试（无需真实数据）

`data/_smoke/` 由 `src/common/synthesize.py` 生成，是**极小的合成样例**，只用于
验证代码链路可运行，其结果**不得**写入实训手册或报告：

```bash
python3 src/common/synthesize.py --out data/_smoke
bash scripts/smoke_test.sh        # 全部入口脚本冒烟
```
