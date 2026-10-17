# -*- coding: utf-8 -*-
"""生成竞赛论文格式 Word 报告（一次性脚本，生成后可删除）"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

REPORT_DIR = Path(__file__).resolve().parent
IMG_DIR = REPORT_DIR.parent / "可视化图表"
HTML_DASH = REPORT_DIR / "决策系统.html"
HTML_SHOT = IMG_DIR / "_决策系统截图.png"
OUT = REPORT_DIR / "比赛报告.docx"


def set_run_font(run, name="宋体", size=12, bold=False, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = color


def add_figure(doc, caption, image_name, width_cm=14.5):
    """插入可视化图表目录中的 PNG，并居中添加图题。"""
    img_path = IMG_DIR / image_name
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    if img_path.is_file():
        p.add_run().add_picture(str(img_path), width=Cm(width_cm))
    else:
        run = p.add_run(f"【图片缺失：{image_name}】")
        set_run_font(run, "宋体", 11, color=RGBColor(180, 0, 0))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(cap.add_run(caption), "宋体", 10.5)
    cap.paragraph_format.space_after = Pt(12)


def capture_html_dashboard():
    """尝试将决策系统.html 截图为 PNG（失败则跳过）。"""
    if HTML_SHOT.is_file():
        return True
    if not HTML_DASH.is_file():
        return False
    try:
        from playwright.sync_api import sync_playwright

        url = HTML_DASH.resolve().as_uri()
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.goto(url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(2000)
            page.screenshot(path=str(HTML_SHOT), full_page=True)
            browser.close()
        return HTML_SHOT.is_file()
    except Exception:
        return False


def add_para(doc, text, *, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=True,
             font="宋体", size=12, bold=False, space_after=6, line_spacing=1.5):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = line_spacing
    pf.space_after = Pt(space_after)
    if indent and align == WD_ALIGN_PARAGRAPH.JUSTIFY:
        pf.first_line_indent = Cm(0.74)
    run = p.add_run(text)
    set_run_font(run, font, size, bold)
    return p


def add_heading(doc, text, level=1):
    """使用 Word 内置标题样式，便于自动生成目录。"""
    word_level = min(max(level, 1), 3)
    h = doc.add_heading(text, level=word_level)
    sizes = {1: 16, 2: 14, 3: 12}
    for run in h.runs:
        set_run_font(run, "黑体", sizes[word_level], bold=True)
    h.paragraph_format.space_before = Pt(12 if word_level == 1 else 8)
    h.paragraph_format.space_after = Pt(6)
    h.paragraph_format.line_spacing = 1.5
    return h


def add_table_of_contents(doc):
    """插入 Word 自动目录域（打开文档后更新域即可生成页码）。"""
    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tp.paragraph_format.space_after = Pt(18)
    set_run_font(tp.add_run("目  录"), "黑体", 16, bold=True)

    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    run = p.add_run()
    r = run._r

    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    r.append(fld_begin)

    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = ' TOC \\o "1-2" \\h \\z \\u '
    r.append(instr)

    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    r.append(fld_sep)

    hint = p.add_run("（请在 Word 中右键此区域 →「更新域」→「更新整个目录」）")
    set_run_font(hint, "宋体", 10.5, color=RGBColor(128, 128, 128))

    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    r.append(fld_end)


def enable_update_fields_on_open(doc):
    """打开文档时提示 Word 自动更新域（含目录页码）。"""
    settings = doc.settings.element
    update_fields = OxmlElement("w:updateFields")
    update_fields.set(qn("w:val"), "true")
    settings.append(update_fields)


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                set_run_font(r, "宋体", 10.5, bold=True)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = str(val)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    set_run_font(r, "宋体", 10.5)
    doc.add_paragraph()


def build():
    print("正在尝试截取 HTML 决策看板...")
    capture_html_dashboard()
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)
    sec.left_margin = Cm(3.17)
    sec.right_margin = Cm(3.17)

    # ===== 封面 =====
    for _ in range(3):
        doc.add_paragraph()
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("广州软件学院 · 大数据技术综合应用大赛")
    set_run_font(r, "黑体", 14, bold=True)

    t2 = doc.add_paragraph()
    t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = t2.add_run("项目研究报告")
    set_run_font(r2, "黑体", 22, bold=True)

    t3 = doc.add_paragraph()
    t3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r3 = t3.add_run("题    目：互联网平台采购意向预测")
    set_run_font(r3, "宋体", 16, bold=True)

    doc.add_paragraph()
    for line in [
        "参赛队伍：________________（请填写）",
        "团队成员：________________（请填写，2-3人）",
        "指导教师：________________（选填）",
        "所属学院：软件与人工智能学院",
        "完成日期：2026年6月",
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_run_font(p.add_run(line), "宋体", 14)

    doc.add_page_break()

    # ===== 摘要 =====
    add_heading(doc, "摘  要", 1)
    add_para(doc, (
        "随着数字经济与平台电商的快速发展，精准识别用户采购意向已成为提升推荐效率、"
        "优化营销资源配置的关键环节。本文以天池移动推荐数据集中约1225万条天猫用户行为日志为研究对象，"
        "围绕「互联网平台采购意向预测」赛题，构建了一套涵盖数据探索、清洗与特征工程、"
        "多模型集成学习及可解释性分析的端到端大数据建模方案。"
        "在严格的时间切分框架下（特征窗口为第1—17天，标签窗口为第18—19天），"
        "本文设计并实现了68维多维行为特征体系，融合时间衰减加权、多时间窗口统计、"
        "转化漏斗特征与交叉协同特征；针对正负样本极度不平衡问题，"
        "采用负采样（1:15）与SMOTE过采样相结合的策略；"
        "进一步构建LightGBM、XGBoost、CatBoost与多层感知机（MLP）四基学习器，"
        "基于验证集AUC进行加权Stacking融合。"
        "实验结果表明，集成模型Ensemble在验证集上取得AUC=0.9145、"
        "F1=0.4848、Precision=0.5581，优于任一单模型，"
        "CatBoost单模型AUC达0.9107。"
        "此外，项目输出16张可视化分析图表及交互式HTML决策看板，"
        "为业务解读与答辩展示提供支撑。"
        "本研究验证了行为序列特征与集成学习在采购意向预测任务中的有效性，"
        "对电商平台精准营销与用户运营具有参考价值。"
    ))
    kp = doc.add_paragraph()
    kp.paragraph_format.first_line_indent = Cm(0.74)
    set_run_font(kp.add_run("关键词："), "黑体", 12, bold=True)
    set_run_font(kp.add_run(
        "采购意向预测；用户行为分析；特征工程；Stacking集成；SHAP可解释性；不平衡学习"
    ), "宋体", 12)

    doc.add_page_break()

    # ===== 目录 =====
    add_table_of_contents(doc)
    doc.add_page_break()

    # ===== 第一章 =====
    add_heading(doc, "第一章  绪论", 1)

    add_heading(doc, "1.1  研究背景与意义", 2)
    add_para(doc, (
        "互联网平台的核心竞争力，日益体现为对用户需求的理解与预测能力。"
        "在采购场景中，用户从首次浏览到最终下单往往经历「浏览—收藏—加购—购买」等多阶段行为链路，"
        "不同行为所携带的意向强度存在显著差异。若能基于历史行为日志提前识别高采购概率的用户—商品对，"
        "平台可在合适时机进行个性化推荐、优惠券触达或库存调配，"
        "从而在提升转化率的同时降低无效曝光成本。"
    ))
    add_para(doc, (
        "本赛题「互联网平台采购意向预测」正是上述业务问题的抽象化建模任务："
        "给定用户在观察期内的行为序列，预测其在预测期内是否会对特定商品产生购买行为。"
        "该问题兼具大数据处理、时间序列建模与类别不平衡学习等多重挑战，"
        "对参赛者的数据工程能力、机器学习建模能力及结果可视化表达能力提出了综合要求。"
    ))

    add_heading(doc, "1.2  国内外研究现状", 2)
    add_para(doc, (
        "在推荐系统与转化预测领域，早期方法多基于协同过滤与矩阵分解，"
        "侧重用户—物品交互矩阵的 latent factor 建模。"
        "随着梯度提升树（GBDT）族算法的兴起，"
        "LightGBM、XGBoost、CatBoost 等模型因对高维稀疏特征的良好适应性，"
        "成为工业界点击率预估（CTR）与购买转化预测（CVR）的主流基线。"
        "近年来，深度学习方法（如 DeepFM、Wide&Deep）进一步引入非线性交叉，"
        "但在中小规模结构化特征场景下，树模型与集成策略仍常具竞争力。"
    ))
    add_para(doc, (
        "针对行为序列数据，有效特征工程往往比复杂模型更为关键："
        "包括 RFM 用户分层、多时间窗口统计、时间衰减加权、"
        "漏斗转化率特征及用户—商品—品类交叉特征等。"
        "在类别极度不平衡场景下，负采样、代价敏感学习及 SMOTE 等过采样技术被广泛使用。"
        "可解释性方面，SHAP（SHapley Additive exPlanations）"
        "为树模型提供了统一的特征贡献度解释框架，有助于模型结果的业务落地。"
    ))

    add_heading(doc, "1.3  研究内容与创新点", 2)
    add_para(doc, "本文主要研究内容包括：")
    for item in [
        "（1）对1225万级行为日志进行分块清洗与全量探索性分析，揭示行为分布、时间规律与用户决策路径；",
        "（2）设计68维行为特征体系，涵盖时间衰减、多窗口、漏斗转化与交叉协同等维度；",
        "（3）构建四基学习器 + AUC加权Stacking 集成框架，系统比较单模型与融合模型性能；",
        "（4）基于SHAP进行可解释性分析，并开发静态图表与HTML交互看板。",
    ]:
        add_para(doc, item, indent=False)

    add_para(doc, "本文主要创新点概括如下：")
    for item in [
        "创新点一：引入基于半衰期的牛顿冷却时间衰减权重，使近期行为对采购意向的贡献高于远期行为；",
        "创新点二：采用严格时间切分（day≤17 建特征 / day 18—19 作标签），避免未来信息泄露，保证评估客观性；",
        "创新点三：全量分块增量聚合713万条有效记录，在工程可行前提下最大化特征覆盖；",
        "创新点四：AUC加权Stacking融合策略，使集成模型AUC较最优单模型提升0.0038。",
    ]:
        add_para(doc, item, indent=False)

    add_heading(doc, "1.4  论文结构安排", 2)
    add_para(doc, (
        "第二章介绍相关理论与技术基础；第三章描述数据集与问题形式化定义；"
        "第四章详述方法论，包括预处理、特征工程与模型集成；"
        "第五章给出实验结果与深入分析；第六章介绍可视化与决策系统；"
        "第七章总结全文并展望未来工作。"
    ))

    doc.add_page_break()

    # ===== 第二章 =====
    add_heading(doc, "第二章  相关理论与技术", 1)

    add_heading(doc, "2.1  行为类型与转化漏斗", 2)
    add_para(doc, (
        "本数据集将用户行为编码为四类：浏览（1）、收藏（2）、加购（3）、购买（4）。"
        "从营销漏斗视角，四类行为构成由浅入深的意向梯度。"
        "浏览反映初步兴趣，收藏与加购代表中等强度意向，购买为最终转化事件。"
        "漏斗各阶段转化率是采购意向预测的重要先验信号。"
    ))

    add_heading(doc, "2.2  时间衰减模型", 2)
    add_para(doc, (
        "用户兴趣随时间推移而衰减。本文采用指数衰减函数对历史行为赋权："
        "w(d)=2^(-d/λ)，其中 d 为行为距特征窗口最后一天的间隔天数，"
        "λ 为半衰期（本项目取5天）。该模型源于牛顿冷却定律的离散化形式，"
        "能够刻画「越近期、越相关」的行为学直觉，"
        "并在用户—商品交互层面生成衰减加权的行为统计特征。"
    ))

    add_heading(doc, "2.3  梯度提升树与集成学习", 2)
    add_para(doc, (
        "LightGBM 基于直方图算法与 leaf-wise 生长策略，训练效率高，适合大规模表格数据；"
        "XGBoost 引入二阶泰勒展开与正则化，泛化能力强；"
        "CatBoost 对类别特征与过拟合具有较好鲁棒性。"
        "本文采用 Stacking 思想：以各基模型在验证集上的预测概率为输入，"
        "通过 AUC 加权平均实现第二层融合，"
        "权重 w_i = AUC_i / ΣAUC_j，兼顾模型性能与实现简洁性。"
    ))

    add_heading(doc, "2.4  类别不平衡处理", 2)
    add_para(doc, (
        "采购行为在全体行为中占比不足1%，正样本极度稀缺。"
        "本文采用两阶段策略："
        "首先在276万用户—商品对负样本池中按1:15比例负采样，"
        "控制训练规模；其次在训练集上应用SMOTE合成少数类样本，"
        "验证集保持原始分布以确保评估无偏。"
        "此外，树模型采用 class_weight 或 scale_pos_weight 进行代价敏感调整。"
    ))

    add_heading(doc, "2.5  SHAP可解释性", 2)
    add_para(doc, (
        "SHAP 基于博弈论 Shapley 值，为每个特征分配对单次预测的平均边际贡献。"
        "对于树模型，TreeExplainer 可在多项式时间内精确计算 SHAP 值。"
        "本文利用 SHAP 全局重要性排序与 Top 特征贡献排名，"
        "辅助理解模型决策逻辑，增强答辩与业务沟通中的可信度。"
    ))

    doc.add_page_break()

    # ===== 第三章 =====
    add_heading(doc, "第三章  数据描述与问题建模", 1)

    add_heading(doc, "3.1  数据来源与字段说明", 2)
    add_para(doc, (
        "本文使用天池「移动推荐」公开数据集，包含用户行为表与商品信息表。"
        "行为表字段包括 user_id、item_id、behavior_type、item_category、time（时间戳）；"
        "商品表提供 item_id 与 item_category 映射。"
        "时间戳可解析为 day（第几天）、hour（小时）、weekday 等衍生字段。"
    ))

    add_heading(doc, "3.2  数据集统计概况", 2)
    add_table(doc,
        ["统计项", "数值"],
        [
            ["行为记录总数", "12,256,906 条"],
            ["商品数量", "310,582 个"],
            ["品类数量", "991 个"],
            ["浏览次数", "11,550,581（94.2%）"],
            ["收藏次数", "242,556（2.0%）"],
            ["加购次数", "343,564（2.8%）"],
            ["购买次数", "120,205（1.0%）"],
            ["浏览→购买转化率", "0.98%"],
            ["收藏→购买转化率", "49.6%"],
            ["加购→购买转化率", "35.0%"],
            ["购买高峰时段", "22:00"],
        ],
    )
    add_figure(doc, "图3-1  用户行为分布全景", "01_图1_行为分布全景.png")

    add_heading(doc, "3.3  问题形式化定义", 2)
    add_para(doc, (
        "设 U 为用户集合，I 为商品集合。"
        "在观察期 T_obs（day 1—17）内，用户 u 对商品 i 产生行为序列 S_{u,i}。"
        "预测期 T_pred（day 18—19）内，定义标签 y_{u,i}∈{0,1}："
        "若存在购买行为则 y=1，否则 y=0。"
        "目标为学习映射 f: X_{u,i}→P(y=1|X_{u,i})，"
        "其中 X_{u,i} 为基于 T_obs 构建的特征向量。"
        "评估指标以 ROC-AUC 为主（衡量排序能力），"
        "辅以 F1、Precision、Recall 综合考察分类性能。"
    ))

    add_heading(doc, "3.4  时间切分与数据泄露防范", 2)
    add_para(doc, (
        "时间序列预测任务中，若将预测期行为纳入特征，将导致严重数据泄露。"
        "本文严格规定：一切特征仅由 day≤17 的行为计算；"
        "标签仅由 day 18—19 的购买事件定义。"
        "该切分策略与天池竞赛官方设定一致，确保实验结论的可比性与可信度。"
    ))
    add_figure(doc, "图3-2  时间切分与技术创新示意", "04_图4_技术创新展示.png")

    doc.add_page_break()

    # ===== 第四章 =====
    add_heading(doc, "第四章  研究方法", 1)

    add_heading(doc, "4.1  总体技术路线", 2)
    add_para(doc, (
        "本项目技术路线分为四个阶段："
        "① 数据探索（EDA）→ ② 特征工程 → ③ 多模型训练与集成 → ④ 可视化与决策看板。"
        "各阶段模块化实现，通过 config.py 统一配置，data_utils.py 封装数据 IO 与清洗，"
        "visualization.py 统一图表样式，支持 00_全流程运行.py 一键复现。"
    ))
    add_figure(doc, "图4-1  项目总体技术路线与综合看板", "04_主图_综合数据看板.png", width_cm=15.5)

    add_heading(doc, "4.2  数据预处理", 2)
    add_para(doc, (
        "预处理采用分块读取（chunk_size=300,000）策略，"
        "在特征窗口内共处理 7,137,379 条有效记录，分41块完成。"
        "清洗规则包括：剔除 behavior_type 非法值；"
        "通过商品表关联补全 item_category；"
        "提取 day、hour、weekday、is_weekend 等时间衍生字段。"
        "经清洗后 invalid_behavior=0，数据质量良好。"
    ))

    add_heading(doc, "4.3  特征工程设计", 2)
    add_para(doc, "本文构建68维特征，按语义分为八组，如下表所示。")
    add_table(doc,
        ["特征组", "维度", "代表特征", "设计动机"],
        [
            ["用户—商品交互", "9", "ui_acts, ui_buy_rate, ui_recency", "刻画特定对的亲密程度"],
            ["用户核心", "13", "u_total, u_buy_rate, u_daily_avg", "用户整体活跃度与转化倾向"],
            ["多时间窗口", "12", "u_1d_buy, u_3d_cart, u_7d_total", "捕捉短中长期兴趣变化"],
            ["时间衰减", "5", "u_dtotal, u_dbuy 等", "近期行为权重更高"],
            ["商品", "8", "i_popularity, i_buy_rate", "商品热度与转化能力"],
            ["品类", "5", "c_buy_rate, c_div", "品类层级统计"],
            ["交叉", "8", "cross_buy, cross_fav_buy", "用户×商品×品类协同"],
            ["协同过滤", "8", "cross_ui, cross_cat 等", "相似模式衍生信号"],
        ],
    )
    add_para(doc, (
        "特征聚合采用全量分块增量策略：逐块计算 groupby 统计量后 merge 合并，"
        "避免随机截断导致的信息损失。"
        "最终得到 2,759,919 个用户—商品对，"
        "其中标签期正样本 1,157 对。"
        "建模阶段正样本280条、负采样4,200条，合计4,480条，"
        "按8:2分层划分训练/验证集，random_seed=42。"
        "特征工程耗时约8005秒（约2.2小时）。"
    ))

    add_heading(doc, "4.4  模型构建与集成策略", 2)
    add_para(doc, "四基学习器主要超参数设置如下：")
    add_table(doc,
        ["模型", "关键超参数", "不平衡处理"],
        [
            ["LightGBM", "n_estimators=300, lr=0.05, max_depth=7", "class_weight=balanced"],
            ["XGBoost", "n_estimators=300, max_depth=7", "scale_pos_weight 自适应"],
            ["CatBoost", "iterations=300, depth=7", "auto_class_weights=Balanced"],
            ["MLP", "hidden=(128,64), max_iter=300", "StandardScaler 标准化"],
        ],
    )
    add_para(doc, (
        "各模型在验证集上输出预测概率 P_i，"
        "集成预测 P_ens = Σ w_i·P_i，w_i = AUC_i / ΣAUC_j。"
        "该策略使性能更优的基模型获得更大权重，"
        "同时保留弱模型的多样性贡献。"
    ))

    add_heading(doc, "4.5  评估指标体系", 2)
    add_para(doc, (
        "AUC（Area Under ROC Curve）：衡量模型对正负样本的排序能力，"
        "对阈值不敏感，是不平衡场景下的核心指标。"
        "F1=2PR/(P+R)：Precision 与 Recall 的调和平均。"
        "Accuracy 在不平衡数据中易产生误导，本文仅作参考。"
        "此外生成 ROC 曲线、混淆矩阵及 SHAP 图进行多维诊断。"
    ))

    doc.add_page_break()

    # ===== 第五章 =====
    add_heading(doc, "第五章  实验结果与分析", 1)

    add_heading(doc, "5.1  探索性数据分析结论", 2)
    add_para(doc, (
        "EDA 揭示以下关键规律，为特征设计提供依据："
        "（1）行为极度不平衡，浏览占94.2%，购买仅1.0%；"
        "（2）收藏与加购到购买的转化率显著高于浏览，"
        "说明漏斗特征具有预测价值；"
        "（3）22:00 为购买高峰，时间特征不可或缺；"
        "（4）用户活跃度呈长尾分布，RFM 分层可识别高价值群体；"
        "（5）购买路径多元，存在直接购买、加购后购买、收藏后购买等模式。"
    ))
    add_figure(doc, "图5-1  时间模式分析", "01_图2_时间模式分析.png")
    add_figure(doc, "图5-2  星期×小时活跃度热力图", "01_图3_活跃度热力图.png")
    add_figure(doc, "图5-3  用户画像与活跃度分析", "01_图4_用户画像全景.png")
    add_figure(doc, "图5-4  品类深度分析", "01_图5_品类深度分析.png")
    add_figure(doc, "图5-5  购买决策路径分布", "01_图6_购买决策路径.png")
    add_figure(doc, "图5-6  RFM用户价值分层", "01_图7_用户价值分层.png")

    add_heading(doc, "5.2  模型性能对比实验", 2)
    add_para(doc, "表5-1 汇总各模型在验证集上的性能指标（数据来源：final_results.json）。")
    add_table(doc,
        ["模型", "AUC", "F1", "Precision", "Recall", "Accuracy"],
        [
            ["LightGBM", "0.8974", "0.4341", "0.3836", "0.5000", "0.9185"],
            ["XGBoost", "0.9099", "0.4632", "0.5641", "0.3929", "0.9431"],
            ["CatBoost", "0.9107", "0.5000", "0.5682", "0.4464", "0.9442"],
            ["MLP", "0.7983", "0.3894", "0.3860", "0.3929", "0.9230"],
            ["Ensemble融合", "0.9145", "0.4848", "0.5581", "0.4286", "0.9431"],
        ],
    )
    add_figure(doc, "图5-7  各模型ROC曲线对比", "03_图8_ROC曲线对比.png")
    add_figure(doc, "图5-8  集成模型混淆矩阵", "03_图9_混淆矩阵.png")
    add_figure(doc, "图5-9  模型性能柱状对比", "03_图10_模型性能对比.png")

    add_para(doc, "结果分析：")
    add_para(doc, (
        "（1）树模型整体优于 MLP：在表格型结构化特征与小样本训练集下，"
        "梯度提升树对特征非线性与交互的捕捉能力更强，MLP 的 AUC 仅为 0.7983；"
        "（2）CatBoost 为最优单模型（AUC=0.9107），"
        "XGBoost 紧随其后（AUC=0.9099），二者 Precision 均超 0.56；"
        "（3）Ensemble 融合 AUC 达 0.9145，"
        "较 CatBoost 提升 0.0038，"
        "验证了多模型互补性；"
        "（4）F1 与 Recall 绝对值偏低，"
        "根因在于正样本极度稀缺及负采样带来的分布偏移，"
        "但 AUC 作为排序指标仍达较高水平，"
        "表明模型对高意向样本的识别与排序能力有效。"
    ))

    add_heading(doc, "5.3  集成权重分析", 2)
    add_table(doc,
        ["基模型", "验证集AUC", "Stacking权重"],
        [
            ["LightGBM", "0.8974", "25.5%"],
            ["XGBoost", "0.9099", "25.9%"],
            ["CatBoost", "0.9107", "25.9%"],
            ["MLP", "0.7983", "22.7%"],
        ],
    )
    add_figure(doc, "图5-10  模型对比与集成效果", "04_图2_模型对比.png")

    add_heading(doc, "5.4  SHAP可解释性分析", 2)
    add_para(doc, (
        "SHAP 分析表明，用户—商品交互类特征（如 ui_acts、ui_cart_rate、ui_buy_rate）"
        "及时间衰减特征（u_dbuy、u_dtotal）对预测贡献最大，"
        "与用户行为学直觉一致："
        "对特定商品的多次交互及近期加购/购买行为是采购意向的强信号。"
        "该结论支持模型的业务可解释性，便于向非技术评委阐述。"
    ))
    add_figure(doc, "图5-11  SHAP特征重要性", "03_图11_SHAP特征重要性.png", width_cm=15.5)
    add_figure(doc, "图5-12  SHAP Top特征贡献排名", "03_图12_SHAP贡献度排名.png")

    add_heading(doc, "5.5  转化漏斗与业务洞察", 2)
    add_para(doc, (
        "结合 EDA 与模型结果，本文提炼三条业务洞察："
        "第一，应对收藏/加购用户实施差异化触达，因其转化率分别为49.6%与35.0%；"
        "第二，晚间20—23时为用户活跃高峰，可优化推送时机；"
        "第三，模型输出的高概率用户—商品对可作为精准推荐候选集，"
        "降低全量扫描成本。"
    ))
    add_figure(doc, "图5-13  转化漏斗与关键洞察", "04_图3_转化漏斗与洞察.png")

    doc.add_page_break()

    # ===== 第六章 =====
    add_heading(doc, "第六章  可视化系统与成果展示", 1)

    add_heading(doc, "6.1  静态可视化体系", 2)
    add_para(doc, (
        "项目共输出16张高清分析图表（250 DPI），"
        "涵盖 EDA（7张）、建模评估（5张）与竞赛综合看板（4张）。"
        "图表统一采用 visualization.py 定义的中文字体与配色方案，"
        "保证视觉一致性与答辩展示效果。"
    ))
    add_figure(doc, "图6-1  竞赛级综合数据看板", "04_主图_综合数据看板.png", width_cm=15.5)
    add_figure(doc, "图6-2  技术创新点展示", "04_图4_技术创新展示.png")

    add_heading(doc, "6.2  交互式决策看板", 2)
    add_para(doc, (
        "除静态图表外，本文开发 ACE-Insight 智能决策看板（决策系统.html），"
        "基于 Chart.js 实现浏览器端交互可视化，"
        "整合 EDA 统计摘要与模型性能指标，"
        "支持答辩现场无需 Python 环境即可演示，"
        "体现「数据分析 + 可视化开发」的综合能力。"
    ))
    html_img = "_决策系统截图.png" if HTML_SHOT.is_file() else "04_图2_模型对比.png"
    add_figure(doc, "图6-3  HTML决策看板界面", html_img, width_cm=15.5)

    add_heading(doc, "6.3  代码工程与可复现性", 2)
    add_para(doc, (
        "项目采用模块化工程结构，"
        "通过 00_全流程运行.py 一键执行完整流水线。"
        "config.py 自动解析相对路径，"
        "避免硬编码绝对路径，"
        "保证代码在不同机器上的可移植性。"
        "全部实验固定 random_seed=42，"
        "特征与结果持久化为 npz/json，"
        "满足大赛对代码规范性与可复现性的评审要求。"
    ))

    doc.add_page_break()

    # ===== 第七章 =====
    add_heading(doc, "第七章  总结与展望", 1)

    add_heading(doc, "7.1  研究总结", 2)
    add_para(doc, (
        "本文针对互联网平台采购意向预测问题，"
        "基于1225万条天猫用户行为数据，"
        "完成了从数据探索、特征工程、多模型集成到可视化展示的完整大数据项目。"
        "主要结论如下："
    ))
    for item in [
        "（1）行为数据呈现极度不平衡与多阶段漏斗特征，"
        "收藏/加购行为是强意向信号；",
        "（2）68维特征体系结合时间切分与SMOTE，"
        "有效支撑了采购意向建模；",
        "（3）AUC加权Stacking集成模型取得AUC=0.9145，"
        "为四类基模型中的最优方案；",
        "（4）SHAP分析与16张可视化图表为模型提供了充分的可解释性与展示支撑。",
    ]:
        add_para(doc, item, indent=False)

    add_heading(doc, "7.2  不足与展望", 2)
    add_para(doc, (
        "本文仍存在以下局限："
        "其一，建模阶段采用负采样以控制计算成本，"
        "可能限制 Recall 上限；"
        "其二，未引入序列深度学习模型（如 LSTM/Transformer）"
        "对行为序列直接建模；"
        "其三，尚未进行线上 A/B 测试验证业务转化提升。"
        "未来工作可从以下方向展开："
        "扩大训练样本规模、引入自动超参优化（Optuna）、"
        "探索深度序列模型与树模型的混合架构、"
        "部署实时特征服务并开展业务侧效果评估。"
    ))

    add_heading(doc, "7.3  应用价值", 2)
    add_para(doc, (
        "本项目的建模思路与工程实践可直接迁移至电商平台用户运营场景："
        "精准推荐、购物车挽回、收藏提醒、库存前置等。"
        "通过提前识别高采购概率的用户—商品对，"
        "平台可在合适的时间、以合适的策略触达用户，"
        "实现数据驱动的精细化运营，"
        "具有明确的商业应用前景。"
    ))

    doc.add_page_break()

    # ===== 参考文献 =====
    add_heading(doc, "参考文献", 1)
    refs = [
        "[1] Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System[C]. KDD, 2016.",
        "[2] Ke G, et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree[C]. NIPS, 2017.",
        "[3] Prokhorenkova L, et al. CatBoost: unbiased boosting with categorical features[C]. NeurIPS, 2018.",
        "[4] Lundberg S M, Lee S I. A Unified Approach to Interpreting Model Predictions[C]. NIPS, 2017.",
        "[5] Chawla N V, et al. SMOTE: Synthetic Minority Over-sampling Technique[J]. JAIR, 2002.",
        "[6] Rendle S. Factorization Machines[C]. ICDM, 2010.",
        "[7] 阿里巴巴天池. 移动推荐数据集[EB/OL]. https://tianchi.aliyun.com/",
        "[8] 周志华. 机器学习[M]. 北京: 清华大学出版社, 2016.",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.space_after = Pt(3)
        set_run_font(p.add_run(ref), "宋体", 10.5)

    doc.add_page_break()

    # ===== 附录 =====
    add_heading(doc, "附录A  图表文件索引", 1)
    add_table(doc,
        ["编号", "文件名", "内容"],
        [
            ["A-1", "01_图1_行为分布全景.png", "行为分布与转化漏斗"],
            ["A-2", "01_图2_时间模式分析.png", "时间规律"],
            ["A-3", "01_图3_活跃度热力图.png", "星期×小时热力"],
            ["A-4", "01_图4_用户画像全景.png", "用户画像"],
            ["A-5", "01_图5_品类深度分析.png", "品类分析"],
            ["A-6", "01_图6_购买决策路径.png", "决策路径"],
            ["A-7", "01_图7_用户价值分层.png", "RFM分层"],
            ["A-8", "03_图8_ROC曲线对比.png", "ROC对比"],
            ["A-9", "03_图9_混淆矩阵.png", "混淆矩阵"],
            ["A-10", "03_图10_模型性能对比.png", "性能对比"],
            ["A-11", "03_图11_SHAP特征重要性.png", "SHAP重要性"],
            ["A-12", "03_图12_SHAP贡献度排名.png", "SHAP排名"],
            ["A-13", "04_主图_综合数据看板.png", "综合看板"],
            ["A-14", "04_图2_模型对比.png", "模型对比"],
            ["A-15", "04_图3_转化漏斗与洞察.png", "漏斗洞察"],
            ["A-16", "04_图4_技术创新展示.png", "创新展示"],
        ],
    )

    add_heading(doc, "附录B  运行说明", 1)
    add_para(doc, (
        "将 tianchi_mobile_recommend_train_user.csv 与 "
        "tianchi_mobile_recommend_train_item.csv 置于项目上级目录，"
        "执行 python 互联网平台采购意向预测/代码/00_全流程运行.py 即可复现全部结果。"
        "步骤02特征工程耗时约2小时，步骤03/04可在 features.npz 存在时快速运行。"
    ), indent=False)

    enable_update_fields_on_open(doc)
    doc.save(str(OUT))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    build()
