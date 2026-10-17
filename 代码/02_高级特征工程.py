# -*- coding: utf-8 -*-
"""
02_高级特征工程  (Feature Engineering)
===============================================================
职责: 数据分块读取 -> 清洗 -> 全量增量聚合 -> 68维特征 -> 负采样 -> SMOTE
数据清洗入口: 本文件 Step2 循环中调用 data_utils.clean_behavior_chunk()
输入: 根目录 CSV
输出: features.npz + feature_names.json + 数据处理报告.json
依赖: config.py / data_utils.py
运行: python 02_高级特征工程.py
===============================================================
"""
from __future__ import annotations

import gc
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.stdout.reconfigure(encoding="utf-8")

from config import (
    DECAY_HALF_LIFE,
    MAX_NEG_SAMPLES,
    NEG_POS_RATIO,
    OUT_DIR,
    RANDOM_SEED,
    TIME_WINDOWS,
    TRAIN_DAY_MAX,
    VAL_SIZE,
)
from data_utils import (
    build_quality_report,
    check_data_files,
    clean_behavior_chunk,
    decay_weight,
    enrich_time_fields,
    iter_user_chunks,
    load_items,
    load_validation_labels,
    merge_sum_frames,
    save_quality_report,
)

t0 = time.time()
print("=" * 60)
print("  高级特征工程 · 全量分块处理")
print("=" * 60)

check_data_files()
items = load_items()
print(f"商品表: {len(items):,} 条 | 品类: {items['item_category'].nunique():,}")

# ---------------------------------------------------------------------------
# Step 1: 加载验证期标签 (day 18-19 购买)
# ---------------------------------------------------------------------------
print("\n[Step 1] 加载验证期购买标签...")
val_buys, val_items = load_validation_labels()
print(f"  验证期购买对: {len(val_buys):,} | 涉及商品: {len(val_items):,}")

# ---------------------------------------------------------------------------
# Step 2: 全量分块增量聚合 (day <= TRAIN_DAY_MAX)
# ---------------------------------------------------------------------------
print(f"\n[Step 2] 全量分块聚合 (特征窗口 day <= {TRAIN_DAY_MAX})...")

max_day = TRAIN_DAY_MAX
total_stats = {"raw": 0, "invalid_behavior": 0, "missing_category": 0, "filled_category": 0, "valid": 0, "chunks": 0}

# 分块聚合容器
user_sum_parts: list[pd.DataFrame] = []
user_decay_parts: list[pd.DataFrame] = []
user_window_parts: dict[int, list[pd.DataFrame]] = {w: [] for w in TIME_WINDOWS}
item_parts: list[pd.DataFrame] = []
cat_parts: list[pd.DataFrame] = []
ui_count_parts: list[pd.DataFrame] = []
ui_day_min_parts: list[pd.DataFrame] = []
ui_day_max_parts: list[pd.DataFrame] = []

train_rows = 0

for chunk_idx, raw_chunk in enumerate(iter_user_chunks(), start=1):
    chunk = enrich_time_fields(raw_chunk)
    if chunk["day"].min() > TRAIN_DAY_MAX:
        break

    chunk = chunk[chunk["day"] <= TRAIN_DAY_MAX]
    if chunk.empty:
        continue


    # === 数据清洗 (data_utils.clean_behavior_chunk) ===
    # 职责: 过滤非法行为 / 补全品类缺失值 / 质量统计
    chunk, qstats = clean_behavior_chunk(chunk, items)
    for k in total_stats:
        if k in qstats:
            total_stats[k] += qstats[k]
    total_stats["chunks"] += 1
    train_rows += len(chunk)

    # --- 用户基础统计 ---
    user_sum_parts.append(
        chunk.groupby("user_id", sort=False).agg(
            u_total=("behavior_type", "count"),
            u_view=("behavior_type", lambda x: (x == 1).sum()),
            u_fav=("behavior_type", lambda x: (x == 2).sum()),
            u_cart=("behavior_type", lambda x: (x == 3).sum()),
            u_buy=("behavior_type", lambda x: (x == 4).sum()),
            u_days=("day", "nunique"),
            u_cats=("item_category", "nunique"),
            u_items=("item_id", "nunique"),
            u_hour_sum=("hour", "sum"),
            u_hour_sq_sum=("hour", lambda x: (x ** 2).sum()),
            u_hour_cnt=("hour", "count"),
            u_weekend_sum=("is_weekend", "sum"),
        ).reset_index()
    )

    # --- 时间衰减加权 ---
    w = decay_weight(chunk["day"].values, max_day, DECAY_HALF_LIFE)
    chunk = chunk.assign(
        w=w,
        w_view=w * (chunk["behavior_type"].values == 1),
        w_fav=w * (chunk["behavior_type"].values == 2),
        w_cart=w * (chunk["behavior_type"].values == 3),
        w_buy=w * (chunk["behavior_type"].values == 4),
    )
    user_decay_parts.append(
        chunk.groupby("user_id", sort=False).agg(
            u_dtotal=("w", "sum"),
            u_dview=("w_view", "sum"),
            u_dfav=("w_fav", "sum"),
            u_dcart=("w_cart", "sum"),
            u_dbuy=("w_buy", "sum"),
        ).reset_index()
    )

    # --- 多时间窗口 ---
    for ws in TIME_WINDOWS:
        recent = chunk[chunk["day"] >= max_day - ws + 1]
        if recent.empty:
            continue
        user_window_parts[ws].append(
            recent.groupby("user_id", sort=False).agg(
                **{
                    f"u_{ws}d_total": ("behavior_type", "count"),
                    f"u_{ws}d_view": ("behavior_type", lambda x: (x == 1).sum()),
                    f"u_{ws}d_cart": ("behavior_type", lambda x: (x == 3).sum()),
                    f"u_{ws}d_buy": ("behavior_type", lambda x: (x == 4).sum()),
                }
            ).reset_index()
        )

    # --- 商品 / 品类 ---
    item_parts.append(
        chunk.groupby("item_id", sort=False).agg(
            i_view=("behavior_type", lambda x: (x == 1).sum()),
            i_cart=("behavior_type", lambda x: (x == 3).sum()),
            i_buy=("behavior_type", lambda x: (x == 4).sum()),
            i_users=("user_id", "nunique"),
            i_days=("day", "nunique"),
        ).reset_index()
    )
    cat_parts.append(
        chunk.groupby("item_category", sort=False).agg(
            c_view=("behavior_type", lambda x: (x == 1).sum()),
            c_cart=("behavior_type", lambda x: (x == 3).sum()),
            c_buy=("behavior_type", lambda x: (x == 4).sum()),
            c_users=("user_id", "nunique"),
            c_items=("item_id", "nunique"),
        ).reset_index()
    )

    # --- 用户-商品交互 ---
    ui_count_parts.append(
        chunk.groupby(["user_id", "item_id"], sort=False).agg(
            ui_acts=("behavior_type", "count"),
            ui_view=("behavior_type", lambda x: (x == 1).sum()),
            ui_fav=("behavior_type", lambda x: (x == 2).sum()),
            ui_cart=("behavior_type", lambda x: (x == 3).sum()),
            ui_buy=("behavior_type", lambda x: (x == 4).sum()),
        ).reset_index()
    )
    ui_day_min_parts.append(
        chunk.groupby(["user_id", "item_id"], sort=False).agg(ui_first_day=("day", "min")).reset_index()
    )
    ui_day_max_parts.append(
        chunk.groupby(["user_id", "item_id"], sort=False).agg(ui_last_day=("day", "max")).reset_index()
    )

    if chunk_idx % 5 == 0:
        print(
            f"  已处理 {chunk_idx} 块 | "
            f"清洗: 过滤非法 {total_stats['invalid_behavior']:,} / "
            f"补全品类 {total_stats['filled_category']:,} | "
            f"有效记录: {train_rows:,}"
        )

print(f"  聚合完成: {train_rows:,} 条有效记录 | {total_stats['chunks']} 块")
print(
    f"  清洗总计: 过滤非法行为 {total_stats['invalid_behavior']:,} | "
    f"补全品类缺失 {total_stats['filled_category']:,} | "
    f"原始记录 {total_stats['raw']:,}"
)

# ---------------------------------------------------------------------------
# Step 3: 合并分块聚合结果
# ---------------------------------------------------------------------------
print("\n[Step 3] 合并分块聚合 & 派生特征...")

uf = merge_sum_frames(user_sum_parts, ["user_id"])
uf["u_hour_mean"] = uf["u_hour_sum"] / uf["u_hour_cnt"].replace(0, 1)
uf["u_hour_var"] = (uf["u_hour_sq_sum"] / uf["u_hour_cnt"].replace(0, 1)) - uf["u_hour_mean"] ** 2
uf["u_hour_std"] = np.sqrt(uf["u_hour_var"].clip(lower=0))
uf["u_weekend_ratio"] = uf["u_weekend_sum"] / uf["u_hour_cnt"].replace(0, 1)
uf = uf.drop(columns=["u_hour_sum", "u_hour_sq_sum", "u_hour_cnt", "u_hour_var", "u_weekend_sum"])

ud = merge_sum_frames(user_decay_parts, ["user_id"])
uf = uf.merge(ud, on="user_id", how="left")

for ws in TIME_WINDOWS:
    if user_window_parts[ws]:
        uw = merge_sum_frames(user_window_parts[ws], ["user_id"])
        uf = uf.merge(uw, on="user_id", how="left")

# 转化漏斗派生
uf["u_buy_rate"] = uf["u_buy"] / uf["u_view"].replace(0, 1)
uf["u_cart_rate"] = uf["u_cart"] / uf["u_view"].replace(0, 1)
uf["u_fav_rate"] = uf["u_fav"] / uf["u_view"].replace(0, 1)
uf["u_daily_avg"] = uf["u_total"] / uf["u_days"].replace(0, 1)
uf["u_fav_to_buy"] = uf["u_buy"] / uf["u_fav"].replace(0, 1)
uf["u_cart_to_buy"] = uf["u_buy"] / uf["u_cart"].replace(0, 1)
print(f"  用户特征: {uf.shape[1]} 列 | {len(uf):,} 用户")

itf = merge_sum_frames(item_parts, ["item_id"])
itf = itf.merge(items, on="item_id", how="left")
itf["item_category"] = itf["item_category"].fillna(-1).astype("int16")
itf["i_popularity"] = itf["i_view"] + itf["i_cart"] * 3 + itf["i_buy"] * 10
itf["i_buy_rate"] = itf["i_buy"] / itf["i_view"].replace(0, 1)
itf["i_cart_rate"] = itf["i_cart"] / itf["i_view"].replace(0, 1)
itf["i_daily_avg"] = itf["i_view"] / itf["i_days"].replace(0, 1)
print(f"  商品特征: {itf.shape[1]} 列 | {len(itf):,} 商品")

cf = merge_sum_frames(cat_parts, ["item_category"])
cf["c_buy_rate"] = cf["c_buy"] / cf["c_view"].replace(0, 1)
cf["c_cart_rate"] = cf["c_cart"] / cf["c_view"].replace(0, 1)
cf["c_div"] = cf["c_items"] / cf["c_users"].replace(0, 1)
print(f"  品类特征: {cf.shape[1]} 列")

uif = merge_sum_frames(ui_count_parts, ["user_id", "item_id"])
ui_min = pd.concat(ui_day_min_parts).groupby(["user_id", "item_id"], sort=False)["ui_first_day"].min().reset_index()
ui_max = pd.concat(ui_day_max_parts).groupby(["user_id", "item_id"], sort=False)["ui_last_day"].max().reset_index()
uif = uif.merge(ui_min, on=["user_id", "item_id"], how="left").merge(ui_max, on=["user_id", "item_id"], how="left")
uif["ui_recency"] = max_day - uif["ui_last_day"]
uif["ui_duration"] = uif["ui_last_day"] - uif["ui_first_day"]
uif["ui_cart_rate"] = uif["ui_cart"] / uif["ui_acts"].replace(0, 1)
uif["ui_buy_rate"] = uif["ui_buy"] / uif["ui_acts"].replace(0, 1)
uif["ui_fav_rate"] = uif["ui_fav"] / uif["ui_acts"].replace(0, 1)
print(f"  用户-商品对: {len(uif):,}")

del user_sum_parts, user_decay_parts, user_window_parts, item_parts, cat_parts
del ui_count_parts, ui_day_min_parts, ui_day_max_parts, ud
gc.collect()

# ---------------------------------------------------------------------------
# Step 4: 构建训练样本 & 交叉特征（仅对正负样本 merge，避免全量打标）
# ---------------------------------------------------------------------------
print("\n[Step 4] 构建训练样本...")

val_df = pd.DataFrame(list(val_buys.keys()), columns=["user_id", "item_id"])
pos = uif.merge(val_df, on=["user_id", "item_id"], how="inner")
pos["label"] = 1

neg_pool = uif.merge(val_df, on=["user_id", "item_id"], how="left", indicator=True)
neg_pool = neg_pool[neg_pool["_merge"] == "left_only"].drop(columns=["_merge"])
neg_n = min(len(pos) * NEG_POS_RATIO, MAX_NEG_SAMPLES, len(neg_pool))
neg = neg_pool.sample(n=neg_n, random_state=RANDOM_SEED) if neg_n > 0 else pd.DataFrame()
neg["label"] = 0

uif_total_pairs = len(uif)
neg_pool_size = len(neg_pool)
val_label_pairs = len(val_buys)

train_samp = pd.concat([pos, neg], ignore_index=True)
print(f"  全量用户-商品对: {uif_total_pairs:,} | 负样本池: {neg_pool_size:,}")
print(f"  正样本: {len(pos):,} | 负样本(采样): {len(neg):,} | 建模合计: {len(train_samp):,}")

train_samp = train_samp.merge(uf, on="user_id", how="left")
train_samp = train_samp.merge(itf, on="item_id", how="left")
train_samp = train_samp.merge(cf, on="item_category", how="left")

train_samp["cross_buy"] = train_samp["u_buy_rate"] * train_samp["i_buy_rate"]
train_samp["cross_cart"] = train_samp["u_cart_rate"] * train_samp["i_cart_rate"]
train_samp["cross_fav"] = train_samp["u_fav_rate"] * train_samp["i_cart_rate"]
train_samp["cross_ui"] = train_samp["ui_cart"] * train_samp["u_cart_rate"]
train_samp["cross_cat"] = train_samp["u_cart_rate"] * train_samp["c_cart_rate"]
train_samp["cross_fav_buy"] = train_samp["u_fav_to_buy"] * train_samp["i_buy_rate"]
train_samp["cross_cart_buy"] = train_samp["u_cart_to_buy"] * train_samp["i_buy_rate"]

del uif, uf, itf, cf, neg_pool
gc.collect()

# ---------------------------------------------------------------------------
# Step 5: 特征矩阵 & 先划分后 SMOTE
# ---------------------------------------------------------------------------
print("\n[Step 5] 特征矩阵构建 & 训练/验证划分...")

EXCLUDE_COLS = {
    "user_id", "item_id", "label", "item_geohash", "item_category",
    "level_0", "index", "ui_first_day", "ui_last_day",
}
NUMERIC_KINDS = ("int8", "int16", "int32", "int64", "float16", "float32", "float64")

feat_names = [
    c for c in train_samp.columns
    if c not in EXCLUDE_COLS and str(train_samp[c].dtype) in NUMERIC_KINDS
]
train_samp[feat_names] = train_samp[feat_names].fillna(0)

X = train_samp[feat_names].values.astype(np.float32)
y = train_samp["label"].values.astype(np.int32)
groups = train_samp["user_id"].values

# 按用户分组划分，避免同一用户同时出现在训练集和验证集；
# 随后仅对训练集 SMOTE，验证分布保持不变。
splitter = GroupShuffleSplit(
    n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_SEED,
)
train_idx, val_idx = next(splitter.split(X, y, groups=groups))
X_tr, X_va = X[train_idx], X[val_idx]
y_tr, y_va = y[train_idx], y[val_idx]
if np.unique(y_tr).size < 2 or np.unique(y_va).size < 2:
    raise RuntimeError("分组切分未能在训练集和验证集中同时保留正负样本")
print(f"  划分后 Train: {len(X_tr):,} | Val: {len(X_va):,}")
print(
    f"  用户数 Train: {np.unique(groups[train_idx]).size:,} | "
    f"Val: {np.unique(groups[val_idx]).size:,}"
)
print(f"  Train 正样本率: {y_tr.mean():.4f} | Val 正样本率: {y_va.mean():.4f}")

if y_tr.sum() >= 6:
    from imblearn.over_sampling import SMOTE

    k = min(5, int(y_tr.sum()) - 1)
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
    X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
    print(f"  SMOTE(仅训练集): {len(y_tr):,} 条 | 正样本: {y_tr.sum():,}")

# ---------------------------------------------------------------------------
# Step 6: 持久化
# ---------------------------------------------------------------------------
np.savez_compressed(
    OUT_DIR / "features.npz",
    X_tr=X_tr, X_va=X_va, y_tr=y_tr, y_va=y_va,
    feature_names=np.array(feat_names, dtype="U50"),
)
with open(OUT_DIR / "feature_names.json", "w", encoding="utf-8") as f:
    json.dump({"names": feat_names, "count": len(feat_names), "time_sec": round(time.time() - t0, 1)}, f, ensure_ascii=False, indent=2)

report = build_quality_report(
    total_stats=total_stats,
    train_rows=train_rows,
    max_day=max_day,
    uif_total_pairs=uif_total_pairs,
    neg_pool_size=neg_pool_size,
    val_label_pairs=val_label_pairs,
    pos_count=len(pos),
    neg_count=len(neg),
    n_features=len(feat_names),
    elapsed=time.time() - t0,
)
save_quality_report(report)

print(f"\n{'=' * 60}")
print(f"  完成 | 特征: {len(feat_names)} 维 | 耗时: {time.time() - t0:.0f}s")
print(f"  输出: features.npz | 数据处理报告.json")
print(f"{'=' * 60}")
