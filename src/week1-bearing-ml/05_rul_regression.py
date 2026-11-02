"""第1周·第4天：基于回归的剩余寿命（RUL）预测。

以每文件特征预测相对剩余寿命 rel_rul ∈ (0,1]（1=刚上机，趋近 0=临近失效）。
回归模型：随机森林 / 梯度提升，按轴承留出测试，输出 MAE / RMSE / R²，
绘制预测-真值散点图，保存最优模型到 models/week1_rul.joblib。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.bearing import bearing_split_mask
from common.plotting import save_fig, setup_chinese_font

import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_models(seed: int) -> dict[str, Pipeline]:
    return {
        "RandomForest": Pipeline([("scaler", StandardScaler()),
                                  ("reg", RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1))]),
        "GradientBoosting": Pipeline([("scaler", StandardScaler()),
                                      ("reg", GradientBoostingRegressor(random_state=seed))]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="results/week1/features.csv")
    ap.add_argument("--selected", default="results/week1/selected_features.json")
    ap.add_argument("--out", default="results/week1")
    ap.add_argument("--save-model", default="models/week1_rul.joblib")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_chinese_font()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    if "rel_rul" not in df.columns:
        raise ValueError("特征表里没有 rel_rul 列，请用 02_features.py 重新生成。")
    sel_path = Path(args.selected)
    feats = json.loads(sel_path.read_text(encoding="utf-8"))["features"] \
        if sel_path.exists() else [c for c in df.columns if c not in
                                    ("condition", "bearing", "file_idx", "bearing_key", "stage", "rel_rul", "file_path")]
    X = df[feats].to_numpy()
    y = df["rel_rul"].to_numpy()

    tr_mask, te_mask = bearing_split_mask(df["bearing_key"].to_numpy())
    X_tr, X_te, y_tr, y_te = X[tr_mask], X[te_mask], y[tr_mask], y[te_mask]
    print(f"特征数={len(feats)} 样本数={len(y)} 训练={len(y_tr)} 测试={len(y_te)}（按轴承留出）")

    results = []
    cv = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    best_name, best_r2, best_model = None, -np.inf, None
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="理想预测线")
    for name, model in make_models(args.seed).items():
        cv_res = cross_validate(model, X_tr, y_tr, cv=cv,
                                scoring=("neg_mean_absolute_error", "r2"), n_jobs=-1)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        mae = mean_absolute_error(y_te, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        r2 = r2_score(y_te, y_pred)
        results.append({
            "模型": name,
            "CV MAE 均值": -cv_res["test_neg_mean_absolute_error"].mean(),
            "CV R2 均值": cv_res["test_r2"].mean(),
            "测试集 MAE": mae, "测试集 RMSE": rmse, "测试集 R2": r2,
        })
        print(f"{name:16s} 测试 MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f}")
        ax.scatter(y_te, y_pred, s=8, alpha=0.4, label=f"{name}（R²={r2:.3f}）")
        if r2 > best_r2:
            best_name, best_r2, best_model = name, r2, model

    ax.set_xlabel("真实 rel_rul")
    ax.set_ylabel("预测 rel_rul")
    ax.set_title("剩余寿命回归：预测值 vs 真值")
    ax.legend(fontsize=9)
    save_fig(fig, out / "rul_pred_vs_true.png")

    res_df = pd.DataFrame(results).sort_values("测试集 R2", ascending=False)
    res_df.to_csv(out / "rul_results.csv", index=False, encoding="utf-8-sig")
    print(f"结果表：{out / 'rul_results.csv'}")

    save_path = Path(args.save_model)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as f:
        pickle.dump({"model": best_model, "features": feats, "test_r2": best_r2}, f)
    print(f"最优模型 {best_name}（R²={best_r2:.4f}）已保存：{save_path}")


if __name__ == "__main__":
    main()
