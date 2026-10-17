# -*- coding: utf-8 -*-
"""
全流程运行入口
--------------
按顺序执行 EDA → 特征工程 → 模型训练 → 竞赛看板，任一步失败会提示手动排查。
"""
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent
STEPS = [
    ("01_数据探索与深度分析.py", "[1/4] EDA 深度分析", "图表"),
    ("02_高级特征工程.py", "[2/4] 高级特征工程", "特征"),
    ("03_多模型训练与Stacking.py", "[3/4] 多模型训练 + SHAP", "模型"),
    ("04_竞赛级综合看板.py", "[4/4] 竞赛级综合看板", "主图 + 决策看板"),
]

print("="*65)
print("  互联网平台采购意向预测 - 全流程运行")
print("  模型: LGB + XGB + CatBoost + MLP + Stacking")
print("  创新: 时间衰减 | SMOTE | SHAP可解释 | RFM分层")
print("="*65)
print()

for script,desc,what in STEPS:
    path=BASE/script
    if not path.exists():
        print(f"[SKIP] {path.name} (not found)")
        continue
    print(f"{desc}...")
    t0=time.time()
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(BASE),
        capture_output=False,
    )
    elapsed=time.time()-t0
    if result.returncode==0:
        print(f"  [OK] {desc} ({elapsed:.0f}s)")
    else:
        print(f"  [FAIL] {desc} (exit {result.returncode})")
        print(f"  Run manually: python {path}")
    print()

print("="*65)
print("  All done! Check: 可视化图表/ for results")
print("="*65)
