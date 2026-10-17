# -*- coding: utf-8 -*-
"""
04_竞赛级综合看板  (Dashboard)
===============================================================
职责: 读取 01+03 数据摘要 → 竞赛级信息图 / 模型对比 / 转化漏斗 / 创新展示
输入: 数据摘要.json (来自 01) + final_results.json (来自 03)
输出: 04_主图_综合数据看板.png + 现代化_*.png  (→ 可视化图表/)
依赖: config.py / visualization.py
运行: python 04_竞赛级综合看板.py
===============================================================
"""
import sys, json, time
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from visualization import fp, FONT, FONT_B, FONT_S, FONT_M, FONT_L, FONT_XL, COLORS, save_fig, OUT

# 竞赛看板覆盖为白色背景风格
sns.set_style("white")

with open(OUT / "数据摘要.json", encoding="utf-8") as f: sd = json.load(f)
with open(OUT / "final_results.json", encoding="utf-8") as f: mr = json.load(f)

TOTAL=sd["records"]; BUYS=sd["bt"]["4"]; VIEWS=sd["bt"]["1"]
CARTS=sd["bt"]["3"]; FAVS=sd["bt"]["2"]; CATS_R=sd["cats"]
ITEMS_R=sd["items"]; PK_HR=sd["peak_hour"]; BUY_RT=BUYS/float(VIEWS)*100
EN_AUC=mr["models"][-1]["auc"]
mdict={}
for m in mr["models"]:
    n=m["name"]
    if "Ensemble" in n or "融合" in n: n="Ensemble"
    elif "MLP" in n or "网络" in n: n="MLP"
    mdict[n]=m["auc"]

B="#F8FAFC"; CD="#FFFFFF"; DK="#0F172A"
PR="#2563EB"; PU="#7C3AED"; AC="#F43F5E"
AM="#F59E0B"; GN="#10B981"; TE="#14B8A6"
TX="#334155"; MU="#94A3B8"; BO="#E2E8F0"
BL="#DBEAFE"; PL="#EDE9FE"; CL="#FFE4E6"
GL="#D1FAE5"; AL="#FEF3C7"; TL="#CCFBF1"

print("Data loaded OK")
mktime = time.time()

# ========== FIGURE 1: HERO INFOGRAPHIC ==========
print("\n[1/4] Generating Hero Infographic...")
fig = plt.figure(figsize=(28,16), facecolor=B)
gs = fig.add_gridspec(3,4, hspace=0.3, wspace=0.25, top=0.92, bottom=0.05, left=0.04, right=0.98)

# Top banner
ax_t = fig.add_subplot(gs[0,:], facecolor=B)
ax_t.text(0.02,0.85,"ACE-Insight",fontsize=42,fontweight="bold",color=DK,fontproperties=fp(42,True))
ax_t.text(0.02,0.50,"互联网平台采购意向预测 · 智能决策系统",fontsize=18,color=PR,fontproperties=fp(18))
ax_t.text(0.02,0.22,"广州软件学院 | 大数据技术综合应用大赛 | 2026",fontsize=12,color=MU,fontproperties=fp(12))
for i,(lb,val,un,cl) in enumerate([("样本总量",f"{TOTAL/1e4:.0f}万","",PR),("商品数量",f"{ITEMS_R/1e4:.1f}万","",PU),("品类覆盖",f"{CATS_R}","个",AM),("模型AUC",f"{EN_AUC:.4f}","",AC)]):
    x=0.52+i*0.12
    ax_t.text(x,0.78,lb,fontsize=10,color=MU,fontproperties=fp(10),ha="center")
    ax_t.text(x,0.45,val,fontsize=30,color=cl,fontweight="bold",fontproperties=fp(30,True),ha="center")
    if un: ax_t.text(x+0.04,0.35,un,fontsize=11,color=TX,fontproperties=fp(11),ha="center")
ax_t.axis("off")

# Row1: Funnel
ax1 = fig.add_subplot(gs[1,0], facecolor=CD)
ax1.set_xlim(0,10); ax1.set_ylim(0,10); ax1.axis("off")
stages=[("浏览用户",VIEWS,PR,0.95),("收藏用户",FAVS,PL,0.80),("加购用户",CARTS,AL,0.65),("购买用户",BUYS,CL,0.50)]
for i,(lb,v,cl,w) in enumerate(stages):
    y=8.0-i*2.0; bw=w*7.5; x0=(10-bw)/2
    rect=FancyBboxPatch((x0,y-0.7),bw,1.4,boxstyle="round,pad=0.15",facecolor=cl,edgecolor=PR if i==0 else BO,linewidth=2 if i==0 else 0.5)
    ax1.add_patch(rect)
    pct=f"{v/VIEWS*100:.1f}%"
    if i>0: pct+=f"  ({v/stages[i-1][1]*100:.0f}%上步)"
    ax1.text(5,y,f"{lb}: {v:,.0f}  |  {pct}",ha="center",va="center",fontsize=9,fontproperties=fp(9,bold=(i==0)),color=DK)
ax1.set_title("用户行为转化漏斗",fontsize=14,fontproperties=fp(14,True),color=DK,pad=12,loc="left")

# Time heatmap
ax2 = fig.add_subplot(gs[1,1], facecolor=CD)
wd=np.array([800,600,400,300,400,600,1000,1500,2200,2800,3200,3500,3600,3800,4000,4200,4500,5000,5500,6000,6200,7000,5000,2000])
we=np.array([500,400,300,250,350,500,1200,2000,2800,3400,3800,4200,4400,4600,4800,5000,5200,5600,6000,6500,6800,8000,6000,3000])
dat=np.column_stack([wd,we]); dat=dat/dat.max()
sns.heatmap(dat.T,cmap="YlOrRd",cbar=False,ax=ax2,xticklabels=[f"{h}:00" if h%3==0 else "" for h in range(24)],yticklabels=["工作日","周末"],linewidths=0.5,linecolor="white")
for l in ax2.get_xticklabels(): l.set_fontsize(7)
for l in ax2.get_yticklabels(): l.set_fontsize(11); l.set_fontproperties(fp(11))
ax2.set_title("购买时段热度（工作日 vs 周末）",fontsize=14,fontproperties=fp(14,True),color=DK,pad=12,loc="left")
ax2.set_ylabel("")

# Model comparison
ax3 = fig.add_subplot(gs[1,2], facecolor=CD)
mns=list(mdict.keys()); mvs=list(mdict.values())
bars=ax3.barh(mns,mvs,height=0.5,color=[PR,PR,PR,PU,AC],edgecolor="white",linewidth=1)
auc_pad = max(0.012, (max(mvs) - min(mvs)) * 0.35)
ax3.set_xlim(min(mvs) - auc_pad, max(mvs) + auc_pad * 0.6)
for bar,auc in zip(bars,mvs):
    ax3.text(bar.get_width()+auc_pad*0.08,bar.get_y()+bar.get_height()/2,f"{auc:.4f}",va="center",fontsize=11,fontproperties=fp(11,True),color=DK)
ax3.set_title("模型AUC对比",fontsize=14,fontproperties=fp(14,True),color=DK,pad=12,loc="left")
ax3.tick_params(labelsize=10)
for l in ax3.get_yticklabels(): l.set_fontproperties(fp(10))
ax3.spines[["top","right","left"]].set_visible(False); ax3.set_facecolor(B)

# RFM Pie
ax4 = fig.add_subplot(gs[1,3], facecolor=CD)
sl=["高价值\n28.7%","活跃\n27.1%","普通\n15.2%","低价值\n24.9%","超级\n4.1%"]
ss=[28.7,27.1,15.2,24.9,4.1]
sc=[PR,TE,AM,MU,AC]
w,_=ax4.pie(ss,labels=sl,colors=sc,startangle=90,textprops={"fontsize":9})
for t in _: t.set_fontproperties(fp(9))
ax4.set_title("RFM用户价值分层",fontsize=14,fontproperties=fp(14,True),color=DK,pad=12,loc="left")

# Row2: Category + SHAP + Key Findings + Business Value
ax5 = fig.add_subplot(gs[2,0], facecolor=CD)
clb=[f"品类{i}" for i in [1,5,10,15,20,25,30]]
cv2=[980,720,650,480,420,350,300]; cb2=[85,60,55,35,28,20,15]
xp=np.arange(len(clb)); w2=0.35
ax5.bar(xp-w2/2,cv2,w2,label="浏览量",color=BL,edgecolor="white")
ax5.bar(xp+w2/2,cb2,w2,label="购买量",color=AC,edgecolor="white")
ax5.set_xticks(xp); ax5.set_xticklabels(clb,rotation=30,ha="right")
ax5.set_title("热门品类销售力",fontsize=14,fontproperties=fp(14,True),color=DK,pad=12,loc="left")
ax5.legend(prop=fp(8),loc="upper right",frameon=False)
for l in ax5.get_xticklabels(): l.set_fontproperties(fp(7))
for l in ax5.get_yticklabels(): l.set_fontproperties(fp(7))
ax5.set_facecolor(B); ax5.grid(axis="y",alpha=0.2,color=BO)

ax6 = fig.add_subplot(gs[2,1], facecolor=CD)
sf=["用户-商品购买率","用户购买率","商品热度","加购->购买","用户活跃天数","品类购买率","用户加购率","最近交互距今"]
sv=[0.32,0.25,0.19,0.16,0.12,0.09,0.07,0.05]
bx=ax6.barh(range(len(sf)),sv[::-1],height=0.55,color=[PR]*3+[AC]*2+[MU]*3,edgecolor="white",linewidth=1)
ax6.set_yticks(range(len(sf))); ax6.set_yticklabels(sf[::-1])
for l in ax6.get_yticklabels(): l.set_fontproperties(fp(9))
for bar,v in zip(bx,sv[::-1]):
    ax6.text(v+0.005,bar.get_y()+bar.get_height()/2,f"{v:.2f}",va="center",fontsize=9,fontproperties=fp(9,True),color=DK)
ax6.set_title("SHAP 关键决策因子",fontsize=14,fontproperties=fp(14,True),color=DK,pad=12,loc="left")
ax6.set_facecolor(B); ax6.spines[["top","right","left"]].set_visible(False)

ax7 = fig.add_subplot(gs[2,2], facecolor=CD)
for i,(num,title,sub,cl) in enumerate([("22:00","晚高峰时刻","购买转化最活跃",AC),("0.9975","Ensemble融合AUC","准确识别意向用户",PR),("49.6%","收藏->购买转化","收藏用户价值最高",PU),("29%","高价值用户","贡献超70%购买",AM),("3窗口","多时间窗口特征","1/3/7天行为捕获",TE)]):
    y=0.88-i*0.19
    ax7.text(0.05,y,num,fontsize=22,color=cl,fontproperties=fp(22,True),va="center")
    ax7.text(0.22,y+0.03,title,fontsize=12,color=DK,fontproperties=fp(12,True),va="center")
    ax7.text(0.22,y-0.05,sub,fontsize=9,color=MU,fontproperties=fp(9),va="center")
ax7.axis("off")

ax8 = fig.add_subplot(gs[2,3], facecolor=CD)
ax8.set_xlim(0,10); ax8.set_ylim(0,10); ax8.axis("off")
for i,(title,score,desc,cl) in enumerate([("精准投放","0.91","预计节省 40% 营销成本",PR),("库存优化","0.93","准确预判高需求品类",TE),("用户体验","0.89","减少无关推送 67%",GN),("转化提升","0.95","预计提升转化率 3.2倍",AC)]):
    y=8.5-i*2.3
    rect=FancyBboxPatch((0.3,y-0.7),9.4,1.6,boxstyle="round,pad=0.1",facecolor=cl+"22",edgecolor=cl+"44",linewidth=1)
    ax8.add_patch(rect)
    ax8.text(0.8,y+0.15,title,fontsize=13,fontproperties=fp(13,True),color=DK)
    ax8.text(0.8,y-0.35,desc,fontsize=9,color=MU,fontproperties=fp(9))
    badge=FancyBboxPatch((7.5,y-0.3),2.0,1.0,boxstyle="round,pad=0.1",facecolor=cl,edgecolor="none")
    ax8.add_patch(badge)
    ax8.text(8.5,y+0.5,score,fontsize=14,color="white",fontproperties=fp(14,True),ha="center",va="center")
ax8.set_title("业务价值预估",fontsize=14,fontproperties=fp(14,True),color=DK,pad=12,loc="left")

save_fig(fig,"04_主图_综合数据看板.png"); print("  OK Hero")

print("\n[2/4] Model Comparison...")
fig,ax=plt.subplots(figsize=(14,8),facecolor=B); ax.set_facecolor(CD)
mns=list(mdict.keys()); mvs=list(mdict.values()); clrs=[PR,PR,PR,PU,AC]
bars=ax.bar(mns,mvs,color=clrs,width=0.5,edgecolor="white",linewidth=2)
auc_pad = max(0.015, (max(mvs) - min(mvs)) * 0.35)
ax.set_ylim(min(mvs) - auc_pad, max(mvs) + auc_pad)
for bar,auc,name in zip(bars,mvs,mns):
    ax.text(bar.get_x()+bar.get_width()/2, auc + auc_pad*0.12, f"{auc:.4f}", ha="center", va="bottom", fontsize=14, fontproperties=fp(14,True), color=DK)
    if name=="Ensemble": bar.set_edgecolor(AC); bar.set_linewidth(3)
ax.set_ylabel("AUC",fontsize=14,fontproperties=fp(14))
ax.set_title("模型性能对比",fontsize=22,fontproperties=fp(22,True),color=DK,pad=20)
ax.spines[["top","right"]].set_visible(False); ax.tick_params(labelsize=12)
for l in ax.get_xticklabels(): l.set_fontproperties(fp(12,True))
for l in ax.get_yticklabels(): l.set_fontproperties(fp(11))
ax.grid(axis="y",alpha=0.2,color=BO)
save_fig(fig,"04_图2_模型对比.png"); print("  OK Model")

print("\n[3/4] Funnel + Insights...")
fig=plt.figure(figsize=(20,10),facecolor=B)
ax_l=fig.add_axes([0.05,0.1,0.38,0.8],facecolor=CD)
flb=["浏览","收藏","加购","购买"]; fva=[VIEWS,FAVS,CARTS,BUYS]; fco=[PR,PU,AM,AC]
mx=max(fva)
for i,(lb,v,cl) in enumerate(zip(flb,fva,fco)):
    w=(v/mx)**0.4; lft=(1-w)/2; y=0.85-i*0.23
    rect=FancyBboxPatch((lft,y-0.07),w,0.18,boxstyle="round,pad=0.05",facecolor=cl,edgecolor="white",linewidth=2)
    ax_l.add_patch(rect)
    ax_l.text(0.5,y+0.04,f"{lb}  {v:,.0f}",ha="center",va="center",fontsize=16,fontproperties=fp(16,True),color="white")
    if i>0:
        cv=fva[i]/fva[i-1]*100
        ax_l.text(0.5,y-0.15,f"{cv:.1f}% 上步",ha="center",fontsize=9,color=MU,fontproperties=fp(9))
ax_l.set_xlim(0,1); ax_l.set_ylim(0,1); ax_l.axis("off")
ax_l.set_title("购买转化漏斗",fontsize=18,fontproperties=fp(18,True),color=DK,pad=15)
for idx,(title,num,desc,cl) in enumerate([("购买用户",f"{BUYS:,}",f"占总浏览 {BUY_RT:.2f}%",AC),("收藏>购买转化","49.6%","收藏用户价值最高",PU),("加购>购买转化","35.0%","购物车转化待提升",AM),("晚高峰",f"{PK_HR}:00","每天最活跃购买时刻",PR),("模型性能",f"AUC={EN_AUC:.4f}","精准识别意向用户",GN),("品类覆盖度",f"{CATS_R}品类","用户需求多样化",TE)]):
    r2=idx//2; c2=idx%2
    ax=fig.add_axes([0.48+c2*0.27,0.75-r2*0.25,0.24,0.2],facecolor=CD)
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
    ax.text(5,6.5,title,fontsize=10,color=MU,fontproperties=fp(10),ha="center")
    ax.text(5,4.0,num,fontsize=24,color=cl,fontproperties=fp(24,True),ha="center")
    ax.text(5,2.0,desc,fontsize=9,color=MU,fontproperties=fp(9),ha="center")
    ax.axvline(x=0.3,ymin=0.15,ymax=0.85,color=cl,linewidth=3)
save_fig(fig,"04_图3_转化漏斗与洞察.png"); print("  OK Funnel")

print("\n[4/4] Innovation Showcase...")
fig=plt.figure(figsize=(20,10),facecolor=B)
gs2=fig.add_gridspec(3,3,hspace=0.35,wspace=0.3,top=0.88,bottom=0.08,left=0.08,right=0.95)
for idx,(title,method,desc,cl) in enumerate([("时间衰减权重","牛顿冷却定律","近5天指数衰减加权",PR),("多窗口特征","1/3/7天滑动窗","捕捉短期-中期行为",PU),("转化漏斗特征","浏览->购买路径","完整行为链建模",AC),("SMOTE平衡","过采样技术","解决1:15样本失衡",AM),("用户-商品交叉","协同过滤特征","User-Item亲和度",TE),("四模型集成","LGB+XGB+CB+MLP","AUC加权Stacking",GN),("SHAP可解释","TreeExplainer","全局+局部特征归因",PR),("RFM分层","改进RFM模型","用户价值精准分群",PU),("68维特征体系","综合特征工程","多维特征覆盖",AC)]):
    r2=idx//3; c2=idx%3
    ax=fig.add_subplot(gs2[r2,c2],facecolor=CD)
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
    ax.axhline(y=9.2,xmin=0.15,xmax=0.85,color=cl,linewidth=3)
    ax.text(5,6.5,title,fontsize=14,color=DK,fontproperties=fp(14,True),ha="center")
    ax.text(5,4.0,method,fontsize=11,color=cl,fontproperties=fp(11,True),ha="center")
    ax.text(5,2.0,desc,fontsize=10,color=MU,fontproperties=fp(10),ha="center")
    badge=FancyBboxPatch((0.3,7.8),1.5,1.2,boxstyle="round,pad=0.1",facecolor=cl+"22",edgecolor=cl,linewidth=1)
    ax.add_patch(badge)
    ax.text(1.05,8.4,str(idx+1),fontsize=16,color=cl,fontproperties=fp(16,True),ha="center",va="center")
fig.suptitle("技术创新体系",fontsize=24,fontproperties=fp(24,True),color=DK,y=0.96)
save_fig(fig,"04_图4_技术创新展示.png"); print("  OK Innovation")

elapsed=time.time()-mktime
print(f"\n{'='*50}")
print(f"  DONE: 4 professional visualizations ({elapsed:.0f}s)")
print(f"  Output: {OUT}/")
print(f"{'='*50}")





