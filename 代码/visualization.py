# -*- coding: utf-8 -*-
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUT_DIR, resolve_font

_font = resolve_font()
_FONT_PATH = str(_font) if _font else None

if _FONT_PATH:
    fm.fontManager.addfont(_FONT_PATH)

def fp(size=12, bold=False):
    kw = {"size": size, "weight": "bold" if bold else "normal"}
    if _FONT_PATH:
        return fm.FontProperties(fname=_FONT_PATH, **kw)
    return fm.FontProperties(**kw)

FONT   = fp(12)
FONT_B = fp(12, True)
FONT_S = fp(10)
FONT_M = fp(12)
FONT_L = fp(14)
FONT_XL = fp(16)

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

COLORS = {
    "view": "#4A90D9",
    "fav":  "#8E44AD",
    "cart": "#F39C12",
    "buy":  "#E74C3C",
    "bg":   "#FAFBFC",
    "grid": "#E8ECF1",
    "text": "#2C3E50",
    "l1": "#EBF5FB",
    "l2": "#D6EAF8",
    "l3": "#AED6F1",
    "l4": "#5DADE2",
    "l5": "#2E86C1",
    "g0": "#ECEFF1",
    "g1": "#CFD8DC",
    "g2": "#90A4AE",
    "g3": "#546E7A",
    "g4": "#263238",
    "green": "#27AE60",
    "dark":  "#2C3E50",
}

OUT = OUT_DIR

def save_fig(fig, name, dpi=250):
    fig.savefig(OUT / name, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
