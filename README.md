# 互联网平台采购意向预测

基于天池移动推荐数据，构建从数据探索、特征工程、多模型训练到交互式决策看板的完整采购意向预测流程。

> 公开版说明：仓库不包含原始数据、训练缓存、逐样本预测结果，以及带有队员个人信息的 Word 报告。历史结果来自原竞赛实验；公开代码已加强验证方式，重跑后的指标可能与历史快照不同。

## 项目亮点

- 处理约 **1,225 万条**用户行为记录，覆盖 **31 万件商品**和 **991 个类目**
- 从用户、商品、类目、交互和时间维度构建 **68 个特征**
- 对比 LightGBM、XGBoost、CatBoost 与 MLP，并提供软投票融合
- 使用按用户分组的训练/验证切分，降低同一用户同时出现在两侧造成的信息泄漏
- 输出特征重要性、ROC/PR 曲线、混淆矩阵、模型对比和业务洞察看板

历史实验中，单模型最高验证 AUC 为 **0.9107（CatBoost）**，原始融合结果为 **0.9145**。原始融合权重由同一验证集 AUC 决定，因此该数字可能偏乐观，不应直接作为独立测试集成绩。详细说明见 [MODEL_CARD.md](MODEL_CARD.md)。

## 在线可读成果

- [交互式决策系统](报告/决策系统.html)
- [比赛报告（Markdown）](报告/比赛报告.md)
- [模型对比图](可视化图表/04_图2_模型对比.png)
- [ROC 曲线](可视化图表/03_图8_ROC曲线对比.png)
- [特征重要性](可视化图表/03_图11_SHAP特征重要性.png)

## 技术路线

    行为日志
      → 时间切分与标签构造
      → 候选用户-商品对生成
      → 用户/商品/类目/交互/时间特征
      → 按用户分组划分训练集与验证集
      → 仅对训练集执行 SMOTE
      → 多模型训练与等权软投票
      → 指标评估、可解释分析与决策看板

## 项目结构

    互联网平台采购意向预测/
    ├─ 代码/
    │  ├─ 00_全流程运行.py
    │  ├─ 01_数据探索与深度分析.py
    │  ├─ 02_高级特征工程.py
    │  ├─ 03_多模型训练与Stacking.py
    │  ├─ 04_竞赛级综合看板.py
    │  ├─ config.py
    │  ├─ data_utils.py
    │  ├─ model_utils.py
    │  └─ visualization.py
    ├─ 报告/
    │  ├─ 比赛报告.md
    │  └─ 决策系统.html
    ├─ 可视化图表/
    ├─ tests/
    ├─ MODEL_CARD.md
    └─ requirements.txt

03_多模型训练与Stacking.py 是为兼容原项目保留的历史文件名；公开版实现的是等权软投票，并非严格意义上的 Stacking。

## 数据准备

数据来自天池的 [移动推荐算法数据集](https://tianchi.aliyun.com/dataset/46)，请自行遵守数据集许可与平台规则。下载后准备：

    tianchi_mobile_recommend_train_user.csv
    tianchi_mobile_recommend_train_item.csv

默认情况下，代码会在项目上级目录查找数据。也可通过环境变量指定目录：

    $env:TIANCHI_DATA_DIR = "D:\\datasets\\tianchi-mobile"

## 安装与运行

推荐使用 Python 3.10 或 3.11：

    python -m venv .venv
    .\\.venv\\Scripts\\Activate.ps1
    python -m pip install -r requirements.txt
    python "代码\\00_全流程运行.py"

也可以逐阶段执行：

    python "代码\\01_数据探索与深度分析.py"
    python "代码\\02_高级特征工程.py"
    python "代码\\03_多模型训练与Stacking.py"
    python "代码\\04_竞赛级综合看板.py"

运行单元测试：

    python -m pip install -r requirements-dev.txt
    python -m pytest -q

## 当前限制与后续方向

- 历史指标来自验证集，不等同于竞赛隐藏测试集成绩
- 公开版分组验证逻辑需要在完整数据上重新训练，才能形成可复现的新基线
- 可进一步采用时间外推验证、真正的 OOF Stacking、概率校准及阈值成本优化
- 原始数据体量较大，建议使用 Parquet、Polars 或 DuckDB 优化数据读取

本项目适合作为数据科学课程设计、推荐系统二分类基线和个人作品集案例。
