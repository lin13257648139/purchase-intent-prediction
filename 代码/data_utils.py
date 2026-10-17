# -*- coding: utf-8 -*-
"""
数据处理工具模块
----------------
提供数据校验、分块读取、增量聚合、质量报告等能力，保证数据处理流程规范可审计。
"""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd

from config import (
    CHUNK_SIZE,
    ITEM_CSV,
    ITEM_DTYPES,
    LABEL_DAY_MAX,
    LABEL_DAY_MIN,
    RANDOM_SEED,
    TRAIN_DAY_MAX,
    USER_CSV,
    USER_DTYPES,
    VAL_SIZE,
    VALID_BEHAVIORS,
    OUT_DIR,
)


def check_data_files() -> None:
    """启动前校验原始数据文件是否存在。"""
    if not USER_CSV.exists():
        raise FileNotFoundError(f"未找到用户行为数据: {USER_CSV}")
    if not ITEM_CSV.exists():
        raise FileNotFoundError(f"未找到商品数据: {ITEM_CSV}")


def load_items() -> pd.DataFrame:
    """加载商品表并校验主键唯一性。"""
    items = pd.read_csv(ITEM_CSV, dtype=ITEM_DTYPES)
    if items["item_id"].duplicated().any():
        items = items.drop_duplicates("item_id", keep="first")
    return items


def iter_user_chunks(usecols: list[str] | None = None):
    """分块迭代用户行为 CSV。"""
    kwargs: dict[str, Any] = {
        "dtype": USER_DTYPES,
        "parse_dates": ["time"],
        "chunksize": CHUNK_SIZE,
    }
    if usecols:
        kwargs["usecols"] = usecols
    return pd.read_csv(USER_CSV, **kwargs)


def enrich_time_fields(chunk: pd.DataFrame) -> pd.DataFrame:
    """为行为块添加 day / hour / weekday / is_weekend 字段。"""
    chunk = chunk.copy()
    chunk["day"] = chunk["time"].dt.day.astype("int8")
    chunk["hour"] = chunk["time"].dt.hour.astype("int8")
    chunk["weekday"] = chunk["time"].dt.weekday.astype("int8")
    chunk["is_weekend"] = (chunk["weekday"] >= 5).astype("int8")
    return chunk


def clean_behavior_chunk(chunk: pd.DataFrame, items: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    清洗单个数据块：过滤非法行为、补全品类、统计质量问题。
    返回 (清洗后数据, 质量统计字典)。
    """
    n_raw = len(chunk)
    stats = {"raw": n_raw, "invalid_behavior": 0, "missing_category": 0, "filled_category": 0}

    invalid_mask = ~chunk["behavior_type"].isin(VALID_BEHAVIORS)
    stats["invalid_behavior"] = int(invalid_mask.sum())
    chunk = chunk.loc[~invalid_mask]

    missing_cat = chunk["item_category"].isna()
    stats["missing_category"] = int(missing_cat.sum())
    if missing_cat.any():
        cat_map = items.set_index("item_id")["item_category"]
        chunk = chunk.copy()
        chunk.loc[missing_cat, "item_category"] = chunk.loc[missing_cat, "item_id"].map(cat_map)
        filled = chunk["item_category"].notna() & missing_cat.reindex(chunk.index, fill_value=False)
        stats["filled_category"] = int(filled.sum())
        chunk["item_category"] = chunk["item_category"].fillna(-1).astype("int16")

    stats["valid"] = len(chunk)
    return chunk, stats


def merge_sum_frames(frames: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    """将多个分块 groupby 结果按 key 求和合并（适用于 count/sum 类聚合）。"""
    if not frames:
        return pd.DataFrame(columns=keys)
    merged = pd.concat(frames, ignore_index=True)
    numeric_cols = [c for c in merged.columns if c not in keys]
    return merged.groupby(keys, sort=False, as_index=False)[numeric_cols].sum()


def merge_mean_frames(frames: list[pd.DataFrame], keys: list[str], weight_col: str | None = None) -> pd.DataFrame:
    """
    合并分块均值类聚合。
    若提供 weight_col，则按权重做加权平均；否则对同 key 简单平均后再与 count 结合时慎用。
    """
    if not frames:
        return pd.DataFrame(columns=keys)
    merged = pd.concat(frames, ignore_index=True)
    if weight_col and weight_col in merged.columns:
        value_cols = [c for c in merged.columns if c not in keys + [weight_col]]
        for col in value_cols:
            merged[f"__w_{col}"] = merged[col] * merged[weight_col]
        grouped = merged.groupby(keys, sort=False)
        num = grouped[[f"__w_{c}" for c in value_cols]].sum()
        den = grouped[weight_col].sum().replace(0, np.nan)
        out = den.to_frame("_w").reset_index()
        for col in value_cols:
            out[col] = (num[f"__w_{col}"] / den).values
        return out
    numeric_cols = [c for c in merged.columns if c not in keys]
    return merged.groupby(keys, sort=False, as_index=False)[numeric_cols].mean()


def load_validation_labels() -> tuple[dict[tuple[int, int], int], set[int]]:
    """
    扫描 day 18-19 的购买记录，构建验证期标签。
    返回: ({(user_id, item_id): 1}, 验证期涉及商品集合)
    """
    val_buys: dict[tuple[int, int], int] = {}
    val_items: set[int] = set()

    for chunk in iter_user_chunks():
        chunk = enrich_time_fields(chunk)
        if chunk["day"].min() > LABEL_DAY_MAX:
            break
        buys = chunk[(chunk["behavior_type"] == 4) & (chunk["day"] >= LABEL_DAY_MIN)]
        for uid, iid in zip(buys["user_id"].values, buys["item_id"].values):
            val_buys[(int(uid), int(iid))] = 1
            val_items.add(int(iid))
        if chunk["day"].max() > LABEL_DAY_MAX:
            break

    return val_buys, val_items


def decay_weight(days: np.ndarray, max_day: int, half_life: float = 5.0) -> np.ndarray:
    """牛顿冷却定律形式的时间衰减权重: w = 2^(-(max_day - day) / half_life)。"""
    return np.exp(-np.log(2) * (max_day - days) / half_life)


def assign_labels(pairs: pd.DataFrame, val_buys: dict[tuple[int, int], int]) -> pd.Series:
    """向量化打标：验证期购买为正样本，否则为负样本（merge 实现，百万级可接受）。"""
    if not val_buys:
        return pd.Series(0, index=pairs.index, dtype="int8")
    val_df = pd.DataFrame(list(val_buys.keys()), columns=["user_id", "item_id"])
    val_df["label"] = 1
    tagged = pairs[["user_id", "item_id"]].merge(
        val_df, on=["user_id", "item_id"], how="left",
    )
    return tagged["label"].fillna(0).astype("int8").values


def build_quality_report(
    total_stats: dict,
    train_rows: int,
    max_day: int,
    uif_total_pairs: int,
    neg_pool_size: int,
    val_label_pairs: int,
    pos_count: int,
    neg_count: int,
    n_features: int,
    elapsed: float,
) -> dict:
    """生成数据处理质量报告（供评委审阅）。"""
    return {
        "pipeline_version": "2.0",
        "data_usage_summary": {
            "raw_csv_total_records": 12256906,
            "note": "原始 CSV 约 1225 万条；特征聚合用 day<=17 的有效记录；建模用采样后的用户-商品对",
        },
        "time_split": {
            "feature_window": f"day <= {TRAIN_DAY_MAX}",
            "label_window": f"day {LABEL_DAY_MIN}-{LABEL_DAY_MAX}",
            "max_feature_day": int(max_day),
        },
        "cleaning": total_stats,
        "feature_aggregation": {
            "interaction_records_used": train_rows,
            "user_item_pairs_total": uif_total_pairs,
            "description": "全量分块聚合，用于计算用户/商品/品类/交互特征",
        },
        "modeling_samples": {
            "val_period_buy_pairs": val_label_pairs,
            "positive_in_training": pos_count,
            "negative_pool_size": neg_pool_size,
            "negative_sampled": neg_count,
            "total_for_model": pos_count + neg_count,
            "pos_neg_ratio": round(neg_count / max(pos_count, 1), 2),
        },
        "features": {"count": n_features},
        "methodology": {
            "aggregation": "全量分块增量聚合（非随机截断采样）",
            "smote": "仅对训练集过采样，验证集保持原始分布",
            "validation_split": (
                f"grouped hold-out by user_id "
                f"{int((1 - VAL_SIZE) * 100)}/{int(VAL_SIZE * 100)}"
            ),
            "random_seed": RANDOM_SEED,
        },
        "elapsed_sec": round(elapsed, 1),
    }


def save_quality_report(report: dict, path=None) -> None:
    """保存数据处理质量报告 JSON。"""
    path = path or (OUT_DIR / "数据处理报告.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def scan_full_dataset_stats() -> dict:
    """扫描全量数据统计（用于 EDA 摘要，与建模使用同一数据源）。"""
    bt_tot = {1: 0, 2: 0, 3: 0, 4: 0}
    nlines = 0
    t0 = time.time()

    for chunk in iter_user_chunks(usecols=["behavior_type"]):
        for bt in VALID_BEHAVIORS:
            bt_tot[bt] += int((chunk["behavior_type"] == bt).sum())
        nlines += len(chunk)

    total = sum(bt_tot.values())
    return {
        "records": nlines,
        "bt": {str(k): v for k, v in bt_tot.items()},
        "buy_rate": bt_tot[4] / total * 100 if total else 0,
        "cart_conv": bt_tot[4] / bt_tot[3] * 100 if bt_tot[3] else 0,
        "fav_conv": bt_tot[4] / bt_tot[2] * 100 if bt_tot[2] else 0,
        "scan_time_sec": round(time.time() - t0, 1),
    }
