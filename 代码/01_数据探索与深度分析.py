# -*- coding: utf-8 -*-
"""
01_数据探索与深度分析  (EDA)
===============================================================
职责: 数据加载 → 统计分析 → 7 组专业级可视化图表
输出: 图1-图7.png + 数据摘要.json  (→ 可视化图表/)
依赖: config.py / data_utils.py / visualization.py
运行: python 01_数据探索与深度分析.py
===============================================================
"""
import gc
import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from config import ITEM_CSV, USER_CSV, USER_DTYPES
from data_utils import check_data_files, load_items, scan_full_dataset_stats
from visualization import fp, FONT, FONT_B, FONT_S, FONT_M, FONT_L, FONT_XL, COLORS, save_fig, OUT

check_data_files()
OUT.mkdir(parents=True, exist_ok=True)
UPATH = USER_CSV
IPATH = ITEM_CSV

BM = {1:"浏览",2:"收藏",3:"加购",4:"购买"}
BC = {1:COLORS["view"],2:COLORS["fav"],3:COLORS["cart"],4:COLORS["buy"]}
BO = [1,2,3,4]
t0 = time.time()

def fmtM(x,p): return f"{x/1e6:.1f}M" if x>=1e6 else f"{x/1e3:.0f}K" if x>=1e4 else f"{x:.0f}"

print("="*65)
print("    互联网平台采购意向预测 — 深度数据分析")
print("="*65)

# ===== 1. Data Load =====
print("\n[1/8] 数据加载中...")
items = load_items()
ncat = int(items["item_category"].nunique())
print(f"  商品: {len(items):,}  品类: {ncat:,}")

dt = USER_DTYPES
sample=pd.read_csv(UPATH,nrows=500000,dtype=dt,parse_dates=["time"])
sample["hour"]=sample["time"].dt.hour.astype("int8")
sample["day"]=sample["time"].dt.day.astype("int8")
sample["weekday"]=sample["time"].dt.weekday.astype("int8")

bt_tot={1:0,2:0,3:0,4:0};nlines=0
for chunk in pd.read_csv(UPATH,usecols=["behavior_type"],chunksize=500000,dtype={"behavior_type":"int8"}):
    for bt in [1,2,3,4]:bt_tot[bt]+=int((chunk["behavior_type"]==bt).sum())
    nlines+=len(chunk)
total=sum(bt_tot.values())
print(f"  总记录: {nlines:,}  购买率: {bt_tot[4]/total*100:.2f}%")
print(f"  转化: 收藏→购买 {bt_tot[4]/bt_tot[2]*100:.1f}%  加购→购买 {bt_tot[4]/bt_tot[3]*100:.1f}%")

# ===== FIGURE 1: Behavior Overview =====
print("\n[2/8] 图1: 行为分布全景")
fig,axes=plt.subplots(1,3,figsize=(24,7.5))
fig.patch.set_facecolor(COLORS["bg"])
# 1a
ax=axes[0]
bars=ax.bar([BM[i] for i in BO],[bt_tot[i] for i in BO],color=list(BC.values()),edgecolor="white",linewidth=1.5,width=0.55)
ax.set_title("用户行为类型分布",fontproperties=fp(16,True),pad=18,color=COLORS["text"])
for bar,bt in zip(bars,BO):
    v=bt_tot[bt];p=v/total*100
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+total*0.003,f"{v:,}\n({p:.1f}%)",ha="center",va="bottom",fontproperties=fp(10,False))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmtM))
ax.set_ylim(0,max(bt_tot.values())*1.15);ax.tick_params(labelsize=10)
for label in ax.get_xticklabels():label.set_fontproperties(FONT_M)
for label in ax.get_yticklabels():label.set_fontproperties(FONT_S)
ax.set_facecolor(COLORS["bg"]);ax.grid(axis="y",alpha=0.3,color=COLORS["grid"])

# 1b: Funnel
ax=axes[1]
fv=[bt_tot[2],bt_tot[3],bt_tot[4]];fc=[COLORS["fav"],COLORS["cart"],COLORS["buy"]];fl=["收藏","加购","购买"]
max_f=max(fv)
for i,(l,v,c) in enumerate(zip(fl,fv,fc)):
    w=v/max_f;left=(1-w)/2
    ax.barh(i,w,left=left,height=0.5,color=c,edgecolor="white",linewidth=2)
    pct=f"({v/fv[0]*100:.1f}%)" if i==0 else f"({v/fv[0]*100:.1f}%)\n↓" if i<2 else f"({v/fv[0]*100:.1f}%)"
    ax.text(0.5,i,f"{l}\n{v:,}\n{pct}",ha="center",va="center",fontproperties=fp(11,True),color="white")
ax.set_xlim(0,1.35);ax.set_title("高价值行为转化漏斗",fontproperties=fp(16,True),pad=18,color=COLORS["text"]);ax.axis("off")

# 1c: Conversion
ax=axes[2]
conv_d=[("浏览→购买",bt_tot[4]/bt_tot[1]*100,COLORS["view"]),("收藏→购买",bt_tot[4]/bt_tot[2]*100,COLORS["fav"]),("加购→购买",bt_tot[4]/bt_tot[3]*100,COLORS["cart"])]
bars=ax.barh([c[0] for c in conv_d],[c[1] for c in conv_d],color=[c[2] for c in conv_d],edgecolor="white",linewidth=1.5,height=0.45)
ax.set_title("各路径购买转化率",fontproperties=fp(16,True),pad=18,color=COLORS["text"])
for bar,(l,v,c) in zip(bars,conv_d):
    ax.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,f"{v:.2f}%",ha="left",va="center",fontproperties=fp(12,True),color=COLORS["text"])
for label in ax.get_yticklabels():label.set_fontproperties(FONT_M)
for label in ax.get_xticklabels():label.set_fontproperties(FONT_S)
ax.set_xlim(0,max([c[1] for c in conv_d])*1.4)
ax.set_facecolor(COLORS["bg"]);ax.grid(axis="x",alpha=0.3,color=COLORS["grid"])
fig.tight_layout(pad=2.5);save_fig(fig,"01_图1_行为分布全景.png");print("  ✓ 图1")

# ===== FIGURE 2: Time Pattern =====
print("\n[3/8] 图2: 时间模式分析")
hourly=sample.groupby("hour")["behavior_type"].value_counts().unstack(fill_value=0)
fig,axes=plt.subplots(2,1,figsize=(18,11))
fig.patch.set_facecolor(COLORS["bg"])
# 2a
ax=axes[0]
for bt in BO:
    ax.plot(hourly.index,hourly[bt],marker="o",linewidth=3,color=BC[bt],label=BM[bt],markersize=6,markeredgecolor="white",markeredgewidth=1)
    ax.fill_between(hourly.index,hourly[bt],alpha=0.08,color=BC[bt])
peak=int(hourly[4].idxmax())
ax.axvline(x=peak,color=COLORS["buy"],linestyle="--",alpha=0.5,linewidth=1.5)
ax.annotate(f"购买峰值 {peak}:00",xy=(peak,hourly[4].max()),xytext=(peak+3,hourly[4].max()*1.08),fontsize=12,color="#C0392B",arrowprops=dict(arrowstyle="->",color="#C0392B",lw=2),fontproperties=FONT_M)
ax.set_title("各行为类型的小时分布规律",fontproperties=fp(16,True),pad=18,color=COLORS["text"])
ax.legend(prop=FONT_M,loc="upper left",frameon=True,fancybox=True)
ax.set_xticks(range(0,24));ax.set_xticklabels([f"{h}:00" for h in range(24)],rotation=45)
for label in ax.get_xticklabels():label.set_fontproperties(FONT_S)
for label in ax.get_yticklabels():label.set_fontproperties(FONT_S)
ax.set_facecolor(COLORS["bg"]);ax.grid(True,alpha=0.25,color=COLORS["grid"])

# 2b
ax=axes[1]
bp=sample[sample["behavior_type"]==4]["hour"].value_counts().sort_index()
bcolors=[]
for h in bp.index:
    if 20<=h or h<=1:bcolors.append("#C0392B")
    elif 18<=h<=19 or 2<=h<=3:bcolors.append("#E74C3C")
    elif 10<=h<=17:bcolors.append("#F1948A")
    else:bcolors.append("#F5B7B1")
ax.bar(bp.index,bp.values,color=bcolors,edgecolor="white",linewidth=1,width=0.8)
ax.annotate(f"购买高峰\n{peak}:00时段",xy=(peak,bp[peak]),xytext=(peak+5,bp[peak]*1.1),fontsize=12,fontweight="bold",color="#C0392B",arrowprops=dict(arrowstyle="->",color="#922B21",lw=2),bbox=dict(boxstyle="round,pad=0.4",facecolor="#FADBD8",edgecolor="#E74C3C",alpha=0.9),fontproperties=FONT_M)
ax.set_title("购买行为时段热度分析 (红色越深=购买越活跃)",fontproperties=fp(16,True),pad=18,color=COLORS["text"])
ax.set_xlabel("时段（小时）",fontproperties=FONT_L);ax.set_ylabel("购买次数",fontproperties=FONT_L)
ax.set_xticks(range(0,24));ax.set_xticklabels([f"{h}:00" for h in range(24)],rotation=45)
for label in ax.get_xticklabels():label.set_fontproperties(FONT_S)
for label in ax.get_yticklabels():label.set_fontproperties(FONT_S)
ax.set_facecolor(COLORS["bg"]);ax.grid(axis="y",alpha=0.25,color=COLORS["grid"])
fig.tight_layout(pad=2.5);save_fig(fig,"01_图2_时间模式分析.png");print("  ✓ 图2")

# ===== FIGURE 3: Heatmap =====
print("\n[4/8] 图3: 用户活跃度热力图")
pivot=sample.pivot_table(index="hour",columns="weekday",values="behavior_type",aggfunc="count")
fig,ax=plt.subplots(figsize=(15,9))
fig.patch.set_facecolor(COLORS["bg"])
sns.heatmap(pivot,cmap="RdYlGn",annot=True,fmt=".0f",xticklabels=["周一","周二","周三","周四","周五","周六","周日"],ax=ax,linewidths=1.5,linecolor="white",cbar_kws={"label":"行为次数","shrink":0.85},annot_kws={"fontsize":8,"fontweight":"bold"})
ax.set_title("用户活跃度热力图 (小时 × 星期)",fontproperties=fp(18,True),pad=20,color=COLORS["text"])
ax.set_xlabel("星期",fontproperties=FONT_L);ax.set_ylabel("小时",fontproperties=FONT_L)
ax.set_yticklabels([f"{h}:00" for h in range(24)],rotation=0)
for label in ax.get_xticklabels():label.set_fontproperties(FONT_M)
for label in ax.get_yticklabels():label.set_fontproperties(FONT_S)
cbar=ax.collections[0].colorbar;cbar.ax.yaxis.label.set_fontproperties(FONT_M)
fig.tight_layout(pad=2.5);save_fig(fig,"01_图3_活跃度热力图.png");print("  ✓ 图3")

# ===== FIGURE 4: User Profile =====
print("\n[5/8] 图4: 用户画像全景")
ua=sample.groupby("user_id").agg(total=("behavior_type","count"),buy=("behavior_type",lambda x:(x==4).sum()),fav=("behavior_type",lambda x:(x==2).sum()),cart=("behavior_type",lambda x:(x==3).sum()),cats=("item_category","nunique")).reset_index()
ua["buy_rate"]=ua["buy"]/ua["total"].replace(0,1);ua["has_buy"]=(ua["buy"]>0).astype(int)
fig,axes=plt.subplots(2,3,figsize=(24,14))
fig.patch.set_facecolor(COLORS["bg"])

# 4a
ax=axes[0,0]
ax.hist(ua["total"],bins=np.logspace(0,np.log10(ua["total"].max()),60),color=COLORS["view"],edgecolor="white",alpha=0.9,linewidth=0.5)
ax.set_xscale("log");ax.set_title("用户活跃度分布 (对数)",fontproperties=fp(14,True),pad=12,color=COLORS["text"])
ax.set_xlabel("总行为次数",fontproperties=FONT_M);ax.set_ylabel("用户数",fontproperties=FONT_M)
for label in ax.get_xticklabels():label.set_fontproperties(FONT_S)
for label in ax.get_yticklabels():label.set_fontproperties(FONT_S)
ax.set_facecolor(COLORS["bg"]);ax.grid(axis="y",alpha=0.25,color=COLORS["grid"])
ax.annotate(f"中位数: {ua['total'].median():.0f}次\n均值: {ua['total'].mean():.0f}次",xy=(0.04,0.88),xycoords="axes fraction",fontsize=10,bbox=dict(boxstyle="round",facecolor="#EBF5FB",alpha=0.95),fontproperties=FONT_S)

# 4b
ax=axes[0,1];bu=ua[ua["buy"]>0]
ax.hist(bu["buy"],bins=35,color=COLORS["buy"],edgecolor="white",alpha=0.9,linewidth=0.5)
ax.set_title(f"购买频次分布 (购买用户: {len(bu):,})",fontproperties=fp(14,True),pad=12,color=COLORS["text"])
ax.set_xlabel("购买次数",fontproperties=FONT_M);ax.set_ylabel("用户数",fontproperties=FONT_M)
for label in ax.get_xticklabels():label.set_fontproperties(FONT_S)
for label in ax.get_yticklabels():label.set_fontproperties(FONT_S)
ax.set_facecolor(COLORS["bg"]);ax.grid(axis="y",alpha=0.25,color=COLORS["grid"])
ax.annotate(f"1次: {(bu['buy']==1).sum()/len(bu)*100:.0f}%\n≥3次: {(bu['buy']>=3).sum()/len(bu)*100:.0f}%",xy=(0.62,0.88),xycoords="axes fraction",fontsize=10,bbox=dict(boxstyle="round",facecolor="#FDEDEC",alpha=0.95),fontproperties=FONT_S)

# 4c
ax=axes[0,2];ua["level"]=pd.cut(ua["total"],bins=[0,5,20,50,100,500,10000],labels=["1-5","6-20","21-50","51-100","101-500","500+"])
lc=ua.groupby("level",observed=True)["has_buy"].mean()*100
gc=[COLORS["l1"],COLORS["l2"],COLORS["l3"],"#3498DB",COLORS["l4"],"#1A5276"]
bars=ax.bar(range(len(lc)),lc.values,color=gc,edgecolor="white",linewidth=1.5)
ax.set_xticks(range(len(lc)));ax.set_xticklabels(lc.index,rotation=30)
ax.set_title("活跃度 → 购买转化率",fontproperties=fp(14,True),pad=12,color=COLORS["text"])
ax.set_ylabel("购买转化率 (%)",fontproperties=FONT_M)
for label in ax.get_xticklabels():label.set_fontproperties(FONT_S)
for label in ax.get_yticklabels():label.set_fontproperties(FONT_S)
for bar,v in zip(bars,lc.values):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.4,f"{v:.1f}%",ha="center",fontproperties=fp(10,True),color="#1A5276")
ax.set_facecolor(COLORS["bg"]);ax.grid(axis="y",alpha=0.25,color=COLORS["grid"])

# 4d
ax=axes[1,0];ua2=ua[ua["total"]>=5].copy();ua2["cart_rate"]=ua2["cart"]/ua2["total"]
ax.scatter(ua2["cart_rate"],ua2["buy_rate"],c=COLORS["view"],alpha=0.2,s=4,edgecolors="none")
ax.set_xlabel("用户加购率",fontproperties=FONT_M);ax.set_ylabel("用户购买率",fontproperties=FONT_M)
ax.set_title("加购率 vs 购买率",fontproperties=fp(14,True),pad=12,color=COLORS["text"])
for label in ax.get_xticklabels():label.set_fontproperties(FONT_S)
for label in ax.get_yticklabels():label.set_fontproperties(FONT_S)
ax.set_facecolor(COLORS["bg"]);ax.grid(alpha=0.2,color=COLORS["grid"])
corr=ua2["cart_rate"].corr(ua2["buy_rate"])
ax.annotate(f"r = {corr:.3f}",xy=(0.62,0.12),xycoords="axes fraction",fontsize=11,bbox=dict(boxstyle="round",facecolor="#EBF5FB",alpha=0.95),fontproperties=FONT_M)

# 4e
ax=axes[1,1];j=np.random.randn(len(ua))*0.05
ax.scatter(ua["cats"],ua["has_buy"]*1+j,c=COLORS["view"],alpha=0.15,s=3,edgecolors="none")
ax.set_xlabel("浏览品类数",fontproperties=FONT_M);ax.set_ylabel("是否购买",fontproperties=FONT_M)
ax.set_title("品类多样性 vs 购买",fontproperties=fp(14,True),pad=12,color=COLORS["text"])
ax.set_yticks([0,1]);ax.set_yticklabels(["未购买","已购买"])
for label in ax.get_xticklabels():label.set_fontproperties(FONT_S)
for label in ax.get_yticklabels():label.set_fontproperties(FONT_M)
ax.set_facecolor(COLORS["bg"]);ax.grid(alpha=0.2,color=COLORS["grid"])
bmn=ua[ua["has_buy"]==1]["cats"].mean();nmn=ua[ua["has_buy"]==0]["cats"].mean()
ax.axvline(x=bmn,color=COLORS["buy"],ls="--",alpha=0.7,lw=1.5);ax.axvline(x=nmn,color=COLORS["g2"],ls="--",alpha=0.7,lw=1.5)
ax.annotate(f"购买用户均值: {bmn:.1f}",xy=(bmn+0.5,0.95),fontsize=9,color=COLORS["buy"],fontproperties=FONT_S)
ax.annotate(f"未购买用户均值: {nmn:.1f}",xy=(nmn+0.5,0.85),fontsize=9,color=COLORS["g2"],fontproperties=FONT_S)

# 4f
ax=axes[1,2];uf=[sample[sample["behavior_type"]==bt]["user_id"].nunique() for bt in BO]
ul=["浏览用户","收藏用户","加购用户","购买用户"];uc=list(BC.values())
for i,(l,v,c) in enumerate(zip(ul,uf,uc)):
    w=v/max(uf);left=(1-w)/2
    ax.barh(i,w,left=left,height=0.5,color=c,edgecolor="white",linewidth=2)
    pct=f" ({v/uf[0]*100:.1f}%)" if i>0 else ""
    ax.text(0.5,i,f"{l}: {v:,}{pct}",ha="center",va="center",fontproperties=fp(11,True),color="white")
ax.set_xlim(0,1.35);ax.set_title("用户转化漏斗",fontproperties=fp(14,True),pad=12,color=COLORS["text"]);ax.axis("off")

fig.tight_layout(pad=2.5);save_fig(fig,"01_图4_用户画像全景.png");print("  ✓ 图4")

# ===== FIGURE 5: Category =====
print("\n[6/8] 图5: 品类深度分析")
cs=sample.groupby("item_category").agg(views=("behavior_type",lambda x:(x==1).sum()),carts=("behavior_type",lambda x:(x==3).sum()),buys=("behavior_type",lambda x:(x==4).sum())).reset_index()
cs["conv"]=cs["buys"]/cs["views"].replace(0,1)*100;cs["cart_conv"]=cs["carts"]/cs["views"].replace(0,1)*100
cs["score"]=cs["views"]+cs["carts"]*3+cs["buys"]*10;top=cs.nlargest(15,"score")
fig,axes=plt.subplots(1,2,figsize=(21,8.5));fig.patch.set_facecolor(COLORS["bg"])
# 5a
ax=axes[0];x=range(len(top));w=0.26
ax.bar([i-w for i in x],top["views"],w,label="浏览",color=COLORS["view"],edgecolor="white",linewidth=0.8)
ax.bar(x,top["carts"],w,label="加购",color=COLORS["cart"],edgecolor="white",linewidth=0.8)
ax.bar([i+w for i in x],top["buys"],w,label="购买",color=COLORS["buy"],edgecolor="white",linewidth=0.8)
ax.set_xticks(x);ax.set_xticklabels([f"品类{c}" for c in top["item_category"]],rotation=45,ha="right")
ax.set_title("热门品类 Top 15 行为分布",fontproperties=fp(16,True),pad=18,color=COLORS["text"])
ax.set_ylabel("行为次数",fontproperties=FONT_L);ax.legend(prop=FONT_M,loc="upper right",frameon=True,fancybox=True)
for label in ax.get_xticklabels():label.set_fontproperties(FONT_S)
for label in ax.get_yticklabels():label.set_fontproperties(FONT_S)
ax.set_facecolor(COLORS["bg"]);ax.grid(axis="y",alpha=0.25,color=COLORS["grid"])
# 5b
ax=axes[1];cs2=cs[cs["views"]>=10]
sc=ax.scatter(cs2["cart_conv"],cs2["conv"],c=cs2["score"],s=np.clip(cs2["score"]/cs2["score"].max()*280,8,280),cmap="YlOrRd",alpha=0.6,edgecolors="#2C3E50",linewidth=0.3)
ax.set_xlabel("加购转化率 (%)",fontproperties=FONT_L);ax.set_ylabel("购买转化率 (%)",fontproperties=FONT_L)
ax.set_title("品类转化率分析 (气泡大小=综合热度)",fontproperties=fp(16,True),pad=18,color=COLORS["text"])
for label in ax.get_xticklabels():label.set_fontproperties(FONT_S)
for label in ax.get_yticklabels():label.set_fontproperties(FONT_S)
cbar=plt.colorbar(sc,ax=ax,shrink=0.85);cbar.set_label("综合热度评分",fontproperties=FONT_M)
for i,(_,row) in enumerate(cs2.nlargest(3,"conv").iterrows()):
    ax.annotate(f"品类{int(row['item_category'])}",(row["cart_conv"],row["conv"]),xytext=(7,7),textcoords="offset points",fontsize=8,color="#922B21",fontweight="bold",fontproperties=FONT_S)
ax.set_facecolor(COLORS["bg"]);ax.grid(alpha=0.25,color=COLORS["grid"])
fig.tight_layout(pad=2.5);save_fig(fig,"01_图5_品类深度分析.png");print("  ✓ 图5")

# ===== FIGURE 6: Purchase Path =====
print("\n[7/8] 图6: 购买决策路径")
buyers=set(sample[sample["behavior_type"]==4]["user_id"].unique())
bdata=sample[sample["user_id"].isin(buyers)].copy()
paths=bdata.groupby("user_id")["behavior_type"].apply(list).reset_index()
def cp(bh):
    hf=2 in bh;hc=3 in bh
    if hf and hc:return "收藏+加购→购买"
    elif hf:return "收藏→购买"
    elif hc:return "加购→购买"
    else:return "直接购买"
paths["路径"]=paths["behavior_type"].apply(cp);pc=paths["路径"].value_counts()
fig,ax=plt.subplots(figsize=(9,9));fig.patch.set_facecolor(COLORS["bg"])
wedges,texts,autotexts=ax.pie(pc.values,labels=pc.index,autopct="%1.1f%%",colors=[COLORS["view"],COLORS["fav"],COLORS["cart"],COLORS["buy"]],startangle=90,wedgeprops=dict(edgecolor="white",linewidth=2.5,width=0.4),pctdistance=0.78,labeldistance=1.1)
for t in autotexts:t.set_fontsize(13);t.set_fontweight("bold")
for t in texts:t.set_fontsize(13);t.set_fontproperties(FONT_M)
ax.set_title("购买用户决策路径分布",fontproperties=fp(16,True),pad=20,color=COLORS["text"])
ax.text(0,0,f"购买用户\n{len(buyers):,}人",ha="center",va="center",fontproperties=fp(13,True),color=COLORS["text"])
fig.tight_layout(pad=2);save_fig(fig,"01_图6_购买决策路径.png");print("  ✓ 图6")

# ===== FIGURE 7: RFM =====
print("\n[8/8] 图7: 用户价值分层")
btdf=sample.copy();btdf["day"]=btdf["time"].dt.day;max_day=btdf["day"].max()
rfm=btdf.groupby("user_id").agg(R=("day",lambda x:max_day-x.max()),F=("behavior_type","count"),buy_c=("behavior_type",lambda x:(x==4).sum()),cart_c=("behavior_type",lambda x:(x==3).sum())).reset_index()
rfm["score"]=rfm["buy_c"]*10+rfm["cart_c"]*3;rfm["段"]=pd.cut(rfm["score"],bins=[0,3,10,30,100,100000],labels=["低价值","普通","活跃","高价值","超级"])
sc=rfm["段"].value_counts()
sd=[];rfc=[COLORS["g1"],COLORS["l1"],COLORS["view"],COLORS["l4"],"#1A5276"]
for s in ["低价值","普通","活跃","高价值","超级"]:
    cnt=sc.get(s,0);ab=rfm[rfm["段"]==s]["buy_c"].mean() if cnt>0 else 0
    sd.append({"段":s,"用户数":cnt,"占比":cnt/len(rfm)*100,"均购":ab})
sd=pd.DataFrame(sd)
fig,axes=plt.subplots(1,2,figsize=(18,7.5));fig.patch.set_facecolor(COLORS["bg"])
ax=axes[0];bars=ax.bar(sd["段"],sd["用户数"],color=rfc,edgecolor="white",linewidth=1.5)
ax.set_title("用户价值分层 (RFM改编模型)",fontproperties=fp(16,True),pad=18,color=COLORS["text"])
ax.set_ylabel("用户数量",fontproperties=FONT_L)
for label in ax.get_xticklabels():label.set_fontproperties(FONT_M)
for label in ax.get_yticklabels():label.set_fontproperties(FONT_S)
for bar,cnt,pct in zip(bars,sd["用户数"],sd["占比"]):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+len(rfm)*0.005,f"{cnt:,}人 ({pct:.1f}%)",ha="center",fontproperties=fp(10,True),color=COLORS["text"])
ax.set_facecolor(COLORS["bg"]);ax.grid(axis="y",alpha=0.25,color=COLORS["grid"])
ax=axes[1];bars=ax.bar(sd["段"],sd["均购"],color=rfc,edgecolor="white",linewidth=1.5)
ax.set_title("各层级平均购买次数",fontproperties=fp(16,True),pad=18,color=COLORS["text"])
ax.set_ylabel("平均购买次数",fontproperties=FONT_L)
for label in ax.get_xticklabels():label.set_fontproperties(FONT_M)
for label in ax.get_yticklabels():label.set_fontproperties(FONT_S)
for bar,v in zip(bars,sd["均购"]):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.03,f"{v:.2f}",ha="center",fontproperties=fp(11,True),color=COLORS["text"])
ax.set_facecolor(COLORS["bg"]);ax.grid(axis="y",alpha=0.25,color=COLORS["grid"])
fig.tight_layout(pad=2.5);save_fig(fig,"01_图7_用户价值分层.png");print("  ✓ 图7")

# Summary
sm={"records":nlines,"items":len(items),"cats":ncat,"bt":{str(k):v for k,v in bt_tot.items()},"buy_rate":bt_tot[4]/total*100,"cart_conv":bt_tot[4]/bt_tot[3]*100,"fav_conv":bt_tot[4]/bt_tot[2]*100,"peak_hour":int(peak),"time":round(time.time()-t0,1)}
with open(OUT/"数据摘要.json","w",encoding="utf-8") as f:json.dump(sm,f,ensure_ascii=False,indent=2)
print(f"\n{'='*65}")
print(f"  ✅ 全部完成! 7组专业级图表 + 数据摘要")
print(f"  ⏱️  耗时: {time.time()-t0:.0f}s")
print(f"{'='*65}")
