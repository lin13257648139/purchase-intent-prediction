# -*- coding: utf-8 -*-
"""
项目全局配置模块
----------------
统一管理路径、数据划分、采样参数，避免硬编码绝对路径，保证代码可移植与可复现。
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径：自动定位项目根目录（本文件位于 项目/代码/config.py）
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent

DATA_DIR = Path(os.environ.get("TIANCHI_DATA_DIR", WORKSPACE_DIR)).expanduser().resolve()
USER_CSV = DATA_DIR / "tianchi_mobile_recommend_train_user.csv"
ITEM_CSV = DATA_DIR / "tianchi_mobile_recommend_train_item.csv"

OUT_DIR = PROJECT_DIR / "可视化图表"
REPORT_DIR = PROJECT_DIR / "报告"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 数据读取 dtype（减少内存、加速 IO）
# ---------------------------------------------------------------------------
USER_DTYPES = {
    "user_id": "int32",
    "item_id": "int32",
    "behavior_type": "int8",
    "item_category": "int16",
}

ITEM_DTYPES = {
    "item_id": "int32",
    "item_category": "int16",
}

# 行为类型编码：1浏览 2收藏 3加购 4购买
BEHAVIOR_LABELS = {1: "浏览", 2: "收藏", 3: "加购", 4: "购买"}
VALID_BEHAVIORS = frozenset(BEHAVIOR_LABELS.keys())

# ---------------------------------------------------------------------------
# 时间划分（天池数据：前17天特征，18-19天作为预测标签）
# ---------------------------------------------------------------------------
TRAIN_DAY_MAX = 17          # 特征窗口：day <= 17
LABEL_DAY_MIN = 18          # 标签窗口：day >= 18
LABEL_DAY_MAX = 19

# ---------------------------------------------------------------------------
# 特征工程 / 采样参数
# ---------------------------------------------------------------------------
CHUNK_SIZE = 300_000
DECAY_HALF_LIFE = 5         # 时间衰减半衰期（天）
TIME_WINDOWS = (1, 3, 7)    # 多时间窗口（天）
NEG_POS_RATIO = 15          # 负样本 : 正样本 上限
MAX_NEG_SAMPLES = 300_000
RANDOM_SEED = 42
VAL_SIZE = 0.2

# ---------------------------------------------------------------------------
# 字体（Windows 常见路径，不存在时 matplotlib 回退默认字体）
# ---------------------------------------------------------------------------
FONT_SIMHEI = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_MSYH = Path(r"C:\Windows\Fonts\msyh.ttc")


def resolve_font(preferred: Path | None = None) -> Path | None:
    """返回可用的中文字体路径。"""
    candidates = [preferred, FONT_SIMHEI, FONT_MSYH]
    for path in candidates:
        if path and path.exists():
            return path
    return None
