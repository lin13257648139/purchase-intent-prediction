# -*- coding: utf-8 -*-
"""
03_多模型训练与融合  (Modeling)
===============================================================
职责: 读取特征 → 训练 LGB/XGB/CatBoost/MLP → 等权软投票 → 评估图表
输入: features.npz + feature_names.json  (来自 02)
输出: 图8-图12.png + final_results.json  (→ 可视化图表/)
依赖: config.py / visualization.py
运行: python 03_多模型训练与Stacking.py
===============================================================
"""
from __future__ import annotations

import json
import sys
import time
import warnings

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from visualization import fp, FONT, FONT_B, FONT_S, FONT_M, FONT_L, FONT_XL, COLORS, save_fig, OUT
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED
from model_utils import equal_weight_soft_vote, evaluate_binary

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

t0 = time.time()

print("=" * 60)
print("  多模型训练与等权软投票")
print("=" * 60)

# ---------------------------------------------------------------------------
# 加载特征
# ---------------------------------------------------------------------------
data = np.load(OUT / "features.npz")
X_tr, X_va, y_tr, y_va = data["X_tr"], data["X_va"], data["y_tr"], data["y_va"]
with open(OUT / "feature_names.json", encoding="utf-8") as f:
    feat_names = json.load(f)["names"]
print(f"Train: {len(X_tr):,} | Val: {len(X_va):,} | Feats: {len(feat_names)}")


results: list[dict] = []
preds: dict[str, np.ndarray] = {}

# ---------------------------------------------------------------------------
# [1/4] LightGBM
# ---------------------------------------------------------------------------
print("\n[1/4] LightGBM")
import lightgbm as lgb

lgb_m = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight="balanced",
    random_state=RANDOM_SEED,
    verbose=0,
)
lgb_m.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    eval_metric="auc",
    callbacks=[lgb.callback.early_stopping(20), lgb.callback.log_evaluation(100)],
)
lgb_p = lgb_m.predict_proba(X_va)[:, 1]
preds["LightGBM"] = lgb_p
results.append(evaluate_binary("LightGBM", y_va, lgb_p))

# ---------------------------------------------------------------------------
# [2/4] XGBoost
# ---------------------------------------------------------------------------
print("\n[2/4] XGBoost")
import xgboost as xgb

pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
xgb_m = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    scale_pos_weight=pos_w,
    random_state=RANDOM_SEED,
    verbosity=0,
)
xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
xgb_p = xgb_m.predict_proba(X_va)[:, 1]
preds["XGBoost"] = xgb_p
results.append(evaluate_binary("XGBoost", y_va, xgb_p))

# ---------------------------------------------------------------------------
# [3/4] CatBoost
# ---------------------------------------------------------------------------
print("\n[3/4] CatBoost")
try:
    import catboost as cb

    cb_m = cb.CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=RANDOM_SEED,
        verbose=0,
    )
    cb_m.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=20)
    cb_p = cb_m.predict_proba(X_va)[:, 1]
    preds["CatBoost"] = cb_p
    results.append(evaluate_binary("CatBoost", y_va, cb_p))
except ImportError:
    print("  [跳过] 未安装 catboost")
except Exception as exc:
    print(f"  [跳过] CatBoost 训练失败: {exc}")

# ---------------------------------------------------------------------------
# [4/4] MLP
# ---------------------------------------------------------------------------
print("\n[4/4] MLP")
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_va_s = scaler.transform(X_va)
nn_m = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    alpha=0.001,
    batch_size=256,
    learning_rate="adaptive",
    learning_rate_init=0.001,
    max_iter=100,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=RANDOM_SEED,
)
nn_m.fit(X_tr_s, y_tr)
nn_p = nn_m.predict_proba(X_va_s)[:, 1]
preds["神经网络"] = nn_p
results.append(evaluate_binary("神经网络", y_va, nn_p))

# ---------------------------------------------------------------------------
# 等权软投票：权重不再由当前验证集指标决定
# ---------------------------------------------------------------------------
ENSEMBLE_NAME = "Ensemble等权融合"
print("\n[Ensemble] 等权概率软投票")
ens_p = equal_weight_soft_vote(preds)
results.append(evaluate_binary(ENSEMBLE_NAME, y_va, ens_p))

for r in results:
    print(
        f"  {r['name']:15s} AUC={r['auc']:.4f} F1={r['f1']:.4f} "
        f"Prec={r['precision']:.4f} Rec={r['recall']:.4f} "
        f"Acc={r['accuracy']:.4f}"
    )

# ---------------------------------------------------------------------------
# 图表输出
# ---------------------------------------------------------------------------
print("\n生成 ROC 曲线...")
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor(COLORS["bg"])
clrs = ["#4A90D9", "#E74C3C", "#8E44AD", "#F39C12", "#27AE60", "#2C3E50"]
for i, r in enumerate(results):
    name = r["name"]
    if name == ENSEMBLE_NAME:
        fpr, tpr, _ = roc_curve(y_va, ens_p)
        ax.plot(fpr, tpr, color="#2C3E50", linewidth=3.5, label=f"{name} (AUC={r['auc']:.4f})")
    elif name in preds:
        fpr, tpr, _ = roc_curve(y_va, preds[name])
        ax.plot(fpr, tpr, color=clrs[i % 6], linewidth=2.2, alpha=0.85, label=f"{name} (AUC={r['auc']:.4f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.25)
ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
ax.set_xlabel("假正率 (FPR)", fontproperties=fp(13))
ax.set_ylabel("真正率 (TPR)", fontproperties=fp(13))
ax.set_title("各模型 ROC 曲线对比", fontproperties=fp(18, True), pad=18, color=COLORS["text"])
ax.legend(prop=fp(11), loc="lower right", frameon=True, fancybox=True, shadow=True)
for label in ax.get_xticklabels():
    label.set_fontproperties(fp(10))
for label in ax.get_yticklabels():
    label.set_fontproperties(fp(10))
ax.set_facecolor(COLORS["bg"])
ax.grid(True, alpha=0.2, color=COLORS["grid"])
fig.tight_layout(pad=2)
save_fig(fig, "03_图8_ROC曲线对比.png", dpi=250)

print("  ✓ 图8")

print("生成混淆矩阵...")
ens_label = (ens_p >= 0.5).astype(int)
cm = confusion_matrix(y_va, ens_label)
fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(COLORS["bg"])
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", ax=ax,
    xticklabels=["预测: 不会购买", "预测: 会购买"],
    yticklabels=["实际: 不会购买", "实际: 会购买"],
    annot_kws={"fontsize": 18, "fontweight": "bold"},
    linewidths=1.5, linecolor="white",
    cbar_kws={"shrink": 0.8, "label": "样本数量"},
)
ax.set_title("等权软投票模型 · 混淆矩阵", fontproperties=fp(17, True), pad=18, color=COLORS["text"])
for label in ax.get_xticklabels():
    label.set_fontproperties(fp(12))
for label in ax.get_yticklabels():
    label.set_fontproperties(fp(12))
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.label.set_fontproperties(fp(11))
fig.tight_layout(pad=2)
save_fig(fig, "03_图9_混淆矩阵.png", dpi=250)

print("  ✓ 图9")

print("生成性能对比...")
fig, (ax_auc, ax_f1) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.patch.set_facecolor(COLORS["bg"])
mn = [r["name"] for r in results]
al = [r["auc"] for r in results]
fl = [r["f1"] for r in results]
x = np.arange(len(mn))


def _bar_value_labels(ax, bars, y_pad: float) -> None:
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + y_pad,
            f"{h:.4f}",
            ha="center",
            va="bottom",
            fontproperties=fp(9, True),
            color=COLORS["text"],
        )


b1 = ax_auc.bar(x, al, 0.55, color=COLORS["view"], edgecolor="white", linewidth=1.2)
auc_pad = max(0.008, (max(al) - min(al)) * 0.35)
ax_auc.set_ylim(min(al) - auc_pad, max(al) + auc_pad)
ax_auc.set_ylabel("AUC", fontproperties=fp(13))
ax_auc.set_title("各模型性能对比", fontproperties=fp(18, True), pad=18, color=COLORS["text"])
_bar_value_labels(ax_auc, b1, auc_pad * 0.25)
ax_auc.set_facecolor(COLORS["bg"])
ax_auc.grid(axis="y", alpha=0.2, color=COLORS["grid"])

b2 = ax_f1.bar(x, fl, 0.55, color=COLORS["cart"], edgecolor="white", linewidth=1.2)
f1_pad = max(0.05, max(fl) * 0.12)
ax_f1.set_ylim(0, max(fl) + f1_pad)
ax_f1.set_ylabel("F1-Score", fontproperties=fp(13))
_bar_value_labels(ax_f1, b2, f1_pad * 0.15)
ax_f1.set_xticks(x)
ax_f1.set_xticklabels(mn, rotation=20, ha="right", fontproperties=fp(11))
ax_f1.set_facecolor(COLORS["bg"])
ax_f1.grid(axis="y", alpha=0.2, color=COLORS["grid"])

for ax in (ax_auc, ax_f1):
    for label in ax.get_yticklabels():
        label.set_fontproperties(fp(10))
fig.tight_layout(pad=2)
save_fig(fig, "03_图10_模型性能对比.png", dpi=250)

print("  ✓ 图10")


def _localize_shap_labels(fig, xlabel: str, *, beeswarm: bool = False) -> None:
    """将 SHAP 图默认英文坐标轴/色条文字替换为中文。"""
    import matplotlib.text as mtext

    if fig.axes:
        fig.axes[0].set_xlabel(xlabel, fontproperties=fp(12), color=COLORS["text"])
    if not beeswarm:
        return

    _text_map = {"Feature value": "特征值", "High": "高", "Low": "低"}
    for ax in fig.axes[1:]:
        ylab = ax.get_ylabel()
        if ylab and "Feature" in ylab:
            ax.set_ylabel("特征值", fontproperties=fp(11), color=COLORS["text"])
    for text in fig.findobj(mtext.Text):
        mapped = _text_map.get(text.get_text().strip())
        if mapped:
            text.set_text(mapped)
            text.set_fontproperties(fp(int(text.get_fontsize() or 10)))


print("生成 SHAP...")
try:
    import shap

    n_shap = min(500, len(X_va))
    X_shap = X_va[:n_shap]
    explainer = shap.TreeExplainer(lgb_m)
    sv = explainer.shap_values(X_shap)
    if isinstance(sv, list):
        sv = sv[1]
    cn = {
        # 用户-商品交互
        "ui_acts": "用户-商品交互次数",
        "ui_view": "用户-商品浏览次数",
        "ui_fav": "用户-商品收藏次数",
        "ui_cart": "用户-商品加购次数",
        "ui_buy": "用户-商品购买次数",
        "ui_recency": "最近交互距今天数",
        "ui_duration": "用户-商品交互跨度",
        "ui_cart_rate": "用户-商品加购率",
        "ui_buy_rate": "用户-商品购买率",
        "ui_fav_rate": "用户-商品收藏率",
        # 用户行为
        "u_total": "用户总行为数",
        "u_view": "用户浏览次数",
        "u_fav": "用户收藏次数",
        "u_cart": "用户加购次数",
        "u_buy": "用户购买次数",
        "u_days": "用户活跃天数",
        "u_cats": "用户浏览品类数",
        "u_items": "用户浏览商品数",
        "u_hour_mean": "用户活跃时段均值",
        "u_hour_std": "用户活跃时段波动",
        "u_weekend_ratio": "用户周末行为占比",
        "u_dtotal": "用户衰减加权总行为",
        "u_dview": "用户衰减加权浏览",
        "u_dfav": "用户衰减加权收藏",
        "u_dcart": "用户衰减加权加购",
        "u_dbuy": "用户衰减加权购买",
        "u_1d_total": "用户近1日总行为",
        "u_1d_view": "用户近1日浏览",
        "u_1d_cart": "用户近1日加购",
        "u_1d_buy": "用户近1日购买",
        "u_3d_total": "用户近3日总行为",
        "u_3d_view": "用户近3日浏览",
        "u_3d_cart": "用户近3日加购",
        "u_3d_buy": "用户近3日购买",
        "u_7d_total": "用户近7日总行为",
        "u_7d_view": "用户近7日浏览",
        "u_7d_cart": "用户近7日加购",
        "u_7d_buy": "用户近7日购买",
        "u_buy_rate": "用户购买率",
        "u_cart_rate": "用户加购率",
        "u_fav_rate": "用户收藏率",
        "u_daily_avg": "用户日均行为",
        "u_fav_to_buy": "收藏→购买转化",
        "u_cart_to_buy": "加购→购买转化",
        # 商品
        "i_view": "商品浏览次数",
        "i_cart": "商品加购次数",
        "i_buy": "商品购买次数",
        "i_users": "商品触达用户数",
        "i_days": "商品活跃天数",
        "i_popularity": "商品综合热度",
        "i_buy_rate": "商品购买率",
        "i_cart_rate": "商品加购率",
        "i_daily_avg": "商品日均浏览",
        # 品类
        "c_view": "品类浏览次数",
        "c_cart": "品类加购次数",
        "c_buy": "品类购买次数",
        "c_users": "品类触达用户数",
        "c_items": "品类商品数",
        "c_buy_rate": "品类购买率",
        "c_cart_rate": "品类加购率",
        "c_div": "品类商品/用户比",
        # 交叉特征
        "cross_buy": "用户×商品购买",
        "cross_cart": "用户×商品加购",
        "cross_fav": "用户×商品收藏",
        "cross_ui": "用户×商品加购交叉",
        "cross_cat": "用户×品类加购",
        "cross_fav_buy": "收藏转化×商品购买率",
        "cross_cart_buy": "加购转化×商品购买率",
    }
    cnames = [cn.get(f, f) for f in feat_names]
    fig, ax = plt.subplots(figsize=(13, 11))
    fig.patch.set_facecolor(COLORS["bg"])
    _save_family = plt.rcParams['font.family']
    _save_sans = plt.rcParams['font.sans-serif'].copy()
    plt.rcParams['font.family'] = FONT.get_name()
    plt.rcParams['font.sans-serif'] = [FONT.get_name()] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    shap.summary_plot(sv, X_shap, feature_names=cnames, max_display=20, show=False, plot_size=None)
    fig = plt.gcf()
    fig.patch.set_facecolor(COLORS["bg"])
    _localize_shap_labels(fig, "SHAP 值（对模型输出的影响）", beeswarm=True)
    plt.title("SHAP 特征重要性分析 (蜂群图)", fontproperties=fp(17, True), pad=18, color=COLORS["text"])
    fig.tight_layout(pad=2)
    save_fig(fig, "03_图11_SHAP特征重要性.png", dpi=250)
    plt.rcParams['font.family'] = _save_family
    plt.rcParams['font.sans-serif'] = _save_sans
    
    fig, ax = plt.subplots(figsize=(13, 10))
    fig.patch.set_facecolor(COLORS["bg"])
    _save_family = plt.rcParams['font.family']
    _save_sans = plt.rcParams['font.sans-serif'].copy()
    plt.rcParams['font.family'] = FONT.get_name()
    plt.rcParams['font.sans-serif'] = [FONT.get_name()] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    shap.summary_plot(sv, X_shap, feature_names=cnames, plot_type="bar", max_display=20, show=False, plot_size=None)
    fig = plt.gcf()
    fig.patch.set_facecolor(COLORS["bg"])
    _localize_shap_labels(fig, "平均 |SHAP 值|（对模型输出影响幅度）")
    plt.title("SHAP 特征平均贡献度排名", fontproperties=fp(17, True), pad=18, color=COLORS["text"])
    fig.tight_layout(pad=2)
    save_fig(fig, "03_图12_SHAP贡献度排名.png", dpi=250)
    plt.rcParams['font.family'] = _save_family
    plt.rcParams['font.sans-serif'] = _save_sans
    
    print("  ✓ 图11 + 图12")
except Exception as exc:
    print(f"  SHAP 跳过: {exc}")

print(
    "\n"
    + classification_report(
        y_va,
        ens_label,
        target_names=["不会购买", "会购买"],
        zero_division=0,
    )
)
with open(OUT / "final_results.json", "w", encoding="utf-8") as f:
    json.dump(
        {"models": results, "feats": len(feat_names), "time": round(time.time() - t0, 1)},
        f, ensure_ascii=False, indent=2,
    )

print(f"\n{'=' * 60}")
best_result = max(results, key=lambda result: result["auc"])
print(
    f"  ✅ 已完成! 最佳 AUC: {best_result['auc']:.4f} "
    f"({best_result['name']}, {time.time() - t0:.0f}s)"
)
print(f"{'=' * 60}")
