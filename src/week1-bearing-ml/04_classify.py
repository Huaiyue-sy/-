"""第1周·第3天：传统机器学习分类（scikit-learn）。

任务（--task）：
  health    按 RMS 退化曲线打的健康阶段三分类（0=健康 1=退化 2=故障），默认
  condition 工况识别四分类（作为对照组）
划分方式（--split）：
  bearing   按轴承留出：每 4 个轴承留 1 个做测试集（工业上更严格，默认）
  random    分层随机划分
模型：随机森林 / SVM(RBF) / KNN / 逻辑回归，比较 5 折交叉验证与测试集指标，
保存最优模型到 models/week1_classifier.joblib。
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

STAGE_NAMES = {0: "健康", 1: "退化", 2: "故障"}


def make_models(seed: int) -> dict[str, Pipeline]:
    return {
        "RandomForest": Pipeline([("scaler", StandardScaler()),
                                  ("clf", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1))]),
        "SVM_RBF": Pipeline([("scaler", StandardScaler()),
                              ("clf", SVC(kernel="rbf", C=10, gamma="scale", random_state=seed))]),
        "KNN": Pipeline([("scaler", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=7))]),
        "LogisticRegression": Pipeline([("scaler", StandardScaler()),
                                        ("clf", LogisticRegression(max_iter=5000, random_state=seed))]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="results/week1/features.csv")
    ap.add_argument("--selected", default="results/week1/selected_features.json")
    ap.add_argument("--task", choices=("health", "condition"), default="health")
    ap.add_argument("--split", choices=("bearing", "random"), default="bearing")
    ap.add_argument("--models", nargs="*", default=["RandomForest", "SVM_RBF", "KNN", "LogisticRegression"])
    ap.add_argument("--out", default="results/week1")
    ap.add_argument("--save-model", default="models/week1_classifier.joblib")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_chinese_font()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    sel = json.loads(Path(args.selected).read_text(encoding="utf-8"))["features"] \
        if Path(args.selected).exists() else None
    feats = sel or [c for c in df.columns if c not in
                    ("condition", "bearing", "file_idx", "bearing_key", "stage", "rel_rul", "file_path")]
    X = df[feats].to_numpy()
    if args.task == "health":
        y = df["stage"].to_numpy()
        classes = [STAGE_NAMES[i] for i in range(3)]
    else:
        y = pd.factorize(df["condition"])[0]
        classes = pd.factorize(df["condition"])[1].tolist()
    print(f"任务={args.task} 特征数={len(feats)} 样本数={len(y)} 类别={classes}")

    if args.split == "bearing":
        tr_mask, te_mask = bearing_split_mask(df["bearing_key"].to_numpy())
        X_tr, X_te, y_tr, y_te = X[tr_mask], X[te_mask], y[tr_mask], y[te_mask]
        print(f"按轴承留出划分：训练 {tr_mask.sum()} / 测试 {te_mask.sum()}"
              f"（测试轴承：{sorted(set(df.loc[te_mask, 'bearing_key']))}）")
    else:
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=args.seed)
        print(f"分层随机划分：训练 {len(y_tr)} / 测试 {len(y_te)}")

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    best_name, best_f1 = None, -1.0
    for name in args.models:
        model = make_models(args.seed)[name]
        cv_res = cross_validate(model, X_tr, y_tr, cv=cv,
                                scoring=("accuracy", "f1_macro"),
                                n_jobs=-1)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_te, y_pred, average="macro", zero_division=0)
        results.append({
            "模型": name,
            "CV准确率均值": cv_res["test_accuracy"].mean(),
            "CV准确率 std": cv_res["test_accuracy"].std(),
            "CV macroF1 均值": cv_res["test_f1_macro"].mean(),
            "测试集准确率": acc,
            "测试集 macroP": p, "测试集 macroR": r, "测试集 macroF1": f1,
        })
        print(f"{name:18s} CV acc={cv_res['test_accuracy'].mean():.4f}±{cv_res['test_accuracy'].std():.4f} "
              f"测试 acc={acc:.4f} macroF1={f1:.4f}")
        cm = confusion_matrix(y_te, y_pred, labels=range(len(classes)))
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)), classes, rotation=45)
        ax.set_yticks(range(len(classes)), classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        ax.set_title(f"{name} 混淆矩阵")
        ax.set_xlabel("预测")
        ax.set_ylabel("真实")
        fig.colorbar(im, shrink=0.8)
        save_fig(fig, out / f"cm_{name}.png")
        if f1 > best_f1:
            best_name, best_f1, best_model = name, f1, model

    res_df = pd.DataFrame(results).sort_values("测试集 macroF1", ascending=False)
    res_df.to_csv(out / f"classify_{args.task}_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n结果表：{out / f'classify_{args.task}_results.csv'}")

    save_path = Path(args.save_model)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as f:
        pickle.dump({
            "model": best_model, "features": feats, "classes": classes,
            "task": args.task, "test_f1": best_f1,
        }, f)
    print(f"最优模型 {best_name}（macroF1={best_f1:.4f}）已保存：{save_path}")


if __name__ == "__main__":
    main()
