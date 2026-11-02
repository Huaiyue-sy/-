#!/usr/bin/env bash
# 全流程冒烟测试：用合成小数据跑通每个入口脚本，验证代码可运行。
# 产物全部落在 results/_smoke/ 与 models/_smoke/，不会污染正式实验结果。
# 这些合成数据的指标【不得】写入实训手册或报告。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"

SMOKE="$ROOT/data/_smoke"
OUT="$ROOT/results/_smoke"
MODEL="$ROOT/models/_smoke"

echo "========== 0. 生成合成冒烟数据 =========="
"$PY" src/common/synthesize.py --out "$SMOKE"

echo "========== 1. 第1周 轴承数据分析 =========="
"$PY" src/week1-bearing-ml/01_explore.py --data-dir "$SMOKE/XJTU-SY_Bearing_Datasets" --out "$OUT/week1"
"$PY" src/week1-bearing-ml/02_features.py --data-dir "$SMOKE/XJTU-SY_Bearing_Datasets" --out "$OUT/week1"
"$PY" src/week1-bearing-ml/03_correlation.py --features "$OUT/week1/features.csv" --out "$OUT/week1"
"$PY" src/week1-bearing-ml/04_classify.py --features "$OUT/week1/features.csv" --selected "$OUT/week1/selected_features.json" --out "$OUT/week1" --save-model "$MODEL/week1_classifier.joblib"
"$PY" src/week1-bearing-ml/05_rul_regression.py --features "$OUT/week1/features.csv" --selected "$OUT/week1/selected_features.json" --out "$OUT/week1" --save-model "$MODEL/week1_rul.joblib"

echo "========== 2. 第2周 深度学习（RNN/LSTM/RUL） =========="
"$PY" src/week2-bearing-dl/01_verify_env.py
"$PY" src/week2-bearing-dl/02_rnn_classifier.py --features "$OUT/week1/features.csv" --seq-len 64 --epochs 1 --out "$OUT/week2" --save-model "$MODEL/rnn_cls.pth"
"$PY" src/week2-bearing-dl/03_lstm_classifier.py --features "$OUT/week1/features.csv" --seq-len 64 --epochs 1 --out "$OUT/week2" --save-model "$MODEL/lstm_cls.pth"
"$PY" src/week2-bearing-dl/04_lstm_rul.py --features "$OUT/week1/features.csv" --selected "$OUT/week1/selected_features.json" --window 4 --epochs 1 --out "$OUT/week2" --save-model "$MODEL/lstm_rul.pth"
"$PY" src/week2-bearing-dl/05_cnn_classifier.py --features "$OUT/week1/features.csv" --seq-len 64 --epochs 1 --out "$OUT/week2" --save-model "$MODEL/cnn_cls.pth"

echo "========== 3. 第2-3周 NEU-DET 表面瑕疵 =========="
"$PY" src/week2-3-surface-defect/01_explore_neu.py --neu-dir "$SMOKE/NEU-DET" --out "$OUT/week2_3"
"$PY" src/week2-3-surface-defect/02_resnet_classify.py --neu-dir "$SMOKE/NEU-DET" --epochs 1 --img-size 64 --batch-size 16 --no-pretrained --out "$OUT/week2_3" --save-model "$MODEL/resnet18_neu.pth"
"$PY" src/week2-3-surface-defect/03_efficientnet_classify.py --neu-dir "$SMOKE/NEU-DET" --epochs 1 --img-size 64 --batch-size 16 --no-pretrained --out "$OUT/week2_3" --save-model "$MODEL/efficientnet_b0_neu.pth"
"$PY" src/week2-3-surface-defect/05_optimize.py --neu-dir "$SMOKE/NEU-DET" --epochs 1 --lrs 1e-2 --losses ce --img-size 64 --batch-size 16 --no-pretrained --out "$OUT/week2_3"
"$PY" src/week2-3-surface-defect/01_explore_neu.py --neu-dir "$SMOKE/NEU-DET-det" --out "$OUT/week2_3"

echo "========== 4. YOLO 检测（从零训练 1 epoch） =========="
"$PY" src/week2-3-surface-defect/04_yolo_detect.py --neu-dir "$SMOKE/NEU-DET-det" --epochs 1 --imgsz 320 --batch 8 --from-scratch --out "$OUT/week2_3"

echo "========== 5. 智能体一：设备运维诊断 =========="
printf '/scan %s\n/report %s\n/quit\n' \
  "$SMOKE/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1/12.csv" \
  "$SMOKE/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1/12.csv" \
  | "$PY" src/agent-maintenance/main.py --features "$OUT/week1/features.csv" --classifier "$MODEL/week1_classifier.joblib" --rul "$MODEL/week1_rul.joblib"

echo "========== 6. 智能体二：视觉质检 =========="
DB="$OUT/agent-inspection/detections.db"
rm -f "$DB"
printf '/inspect %s 3号机台\n/inspect %s 1号机台\n/query 最近哪种缺陷最多？\n/query 3号机台有没有异常趋势？\n/query 划伤出现了多少次？\n/status\n/quit\n' \
  "$SMOKE/NEU-DET/scratches" \
  "$SMOKE/NEU-DET/crazing" \
  | "$PY" src/agent-inspection/main.py --model "$MODEL/resnet18_neu.pth" --db "$DB"

echo ""
echo "冒烟测试全部通过。产物目录：$OUT、$MODEL（仅为链路验证，数字不可用于报告）。"
