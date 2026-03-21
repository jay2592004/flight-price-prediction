"""
src/eda.py
Full Exploratory Data Analysis pipeline.
Generates 16 plots saved to reports/ as PNG files.
Also saves reports/insights.json so the Streamlit app can display
text insights without re-running the pipeline.
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

from .config import REPORTS_DIR

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8FBFF",
})
C_BLUE   = "#1565C0"
C_GOLD   = "#FFB300"
C_GREEN  = "#0D9B6E"
C_RED    = "#C62828"
C_PURPLE = "#7B1FA2"
C_ORANGE = "#E65100"
PALETTE  = [C_BLUE, C_GREEN, C_ORANGE, C_RED, C_PURPLE, C_GOLD, "#00695C", "#37474F"]

CLASS_ORDER = ["Economy", "Premium Economy", "Business", "First"]
CLASS_COLORS = ["#90CAF9", "#42A5F5", C_BLUE, C_GOLD]


def _save(name: str):
    path = os.path.join(REPORTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def _fmt_inr(x, _):
    return f"₹{x/1000:.0f}K"


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_price_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Raw distribution
    axes[0].hist(df["Price"], bins=80, color=C_BLUE, edgecolor="white", alpha=0.85)
    axes[0].axvline(df["Price"].median(), color=C_RED, linestyle="--", linewidth=2,
                    label=f"Median ₹{df['Price'].median():,.0f}")
    axes[0].axvline(df["Price"].mean(), color=C_GOLD, linestyle="--", linewidth=2,
                    label=f"Mean ₹{df['Price'].mean():,.0f}")
    axes[0].set_title("Price Distribution", fontweight="bold", fontsize=13)
    axes[0].set_xlabel("Price (₹)"); axes[0].set_ylabel("Count")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))
    axes[0].legend(fontsize=9)

    # Log-transformed
    axes[1].hist(np.log1p(df["Price"]), bins=80, color=C_GREEN, edgecolor="white", alpha=0.85)
    axes[1].set_title("Log-Transformed Price\n(near-normal after transform)", fontweight="bold", fontsize=13)
    axes[1].set_xlabel("log(1 + Price)"); axes[1].set_ylabel("Count")

    # Box plot by class
    df.boxplot(column="Price", by="Class", ax=axes[2],
               positions=range(len(CLASS_ORDER)),
               widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#E3F2FD"),
               medianprops=dict(color=C_RED, linewidth=2))
    axes[2].set_xticklabels(CLASS_ORDER, rotation=15, ha="right")
    axes[2].set_title("Price Range by Class", fontweight="bold", fontsize=13)
    axes[2].set_xlabel(""); axes[2].set_ylabel("Price (₹)")
    axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))
    plt.suptitle("")
    plt.tight_layout()
    return _save("01_price_distribution.png")


def plot_price_by_class(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    class_stats = df.groupby("Class")["Price"].agg(["mean","median","std"]).reindex(CLASS_ORDER)
    x = range(len(CLASS_ORDER))
    bars = axes[0].bar(CLASS_ORDER, class_stats["mean"], color=CLASS_COLORS,
                       edgecolor="white", width=0.6)
    for bar, (_, row) in zip(bars, class_stats.iterrows()):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+300,
                     f"₹{bar.get_height():,.0f}", ha="center", fontsize=9.5, fontweight="bold")
    axes[0].errorbar(x, class_stats["mean"], yerr=class_stats["std"],
                     fmt="none", color="#333", capsize=6, linewidth=1.5)
    axes[0].set_title("Mean Price ± Std Dev by Cabin Class", fontweight="bold", fontsize=13)
    axes[0].set_ylabel("Price (₹)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))

    # Volume
    cnt = df["Class"].value_counts().reindex(CLASS_ORDER)
    axes[1].pie(cnt.values, labels=CLASS_ORDER, autopct="%1.1f%%",
                colors=CLASS_COLORS, startangle=90,
                wedgeprops={"edgecolor":"white","linewidth":2})
    axes[1].set_title("Flight Volume by Cabin Class", fontweight="bold", fontsize=13)

    plt.tight_layout()
    return _save("02_price_by_class.png")


def plot_price_by_airline(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    a_med = df.groupby("Airline")["Price"].median().sort_values(ascending=True)
    colors = [C_GOLD if v == a_med.max() else C_BLUE for v in a_med.values]
    axes[0].barh(a_med.index, a_med.values, color=colors, edgecolor="white", height=0.7)
    for i, v in enumerate(a_med.values):
        axes[0].text(v+100, i, f"₹{v:,.0f}", va="center", fontsize=8.5)
    axes[0].set_title("Median Price by Airline", fontweight="bold", fontsize=13)
    axes[0].set_xlabel("Median Price (₹)")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))

    # Tier comparison
    if "Airline_Tier" in df.columns:
        tier_data = df.groupby("Airline_Tier")["Price"].median().reindex(["budget","mid","premium"])
        axes[1].bar(["Budget","Mid-tier","Premium"],
                    tier_data.values,
                    color=["#90CAF9", C_BLUE, C_GOLD], edgecolor="white", width=0.5)
        for i, v in enumerate(tier_data.values):
            axes[1].text(i, v+400, f"₹{v:,.0f}", ha="center", fontsize=12, fontweight="bold")
        axes[1].set_title("Median Price by Airline Tier", fontweight="bold", fontsize=13)
        axes[1].set_ylabel("Median Price (₹)")
        axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))

    plt.tight_layout()
    return _save("03_price_by_airline.png")


def plot_price_vs_distance(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sample = df.sample(min(8000, len(df)), random_state=42)
    pal = dict(zip(CLASS_ORDER, CLASS_COLORS))
    for cls in CLASS_ORDER:
        sub = sample[sample["Class"]==cls]
        axes[0].scatter(sub["Distance_km"], sub["Price"],
                        alpha=0.18, s=8, label=cls, color=pal[cls])
    corr = df["Distance_km"].corr(df["Price"])
    # Regression line
    m, b, r, p, _ = stats.linregress(df["Distance_km"].values, df["Price"].values)
    xs = np.linspace(df["Distance_km"].min(), df["Distance_km"].max(), 100)
    axes[0].plot(xs, m*xs+b, color=C_RED, linewidth=2, linestyle="--", label=f"Trend (r={corr:.3f})")
    axes[0].set_title(f"Price vs Distance  (r = {corr:.3f})", fontweight="bold", fontsize=13)
    axes[0].set_xlabel("Distance (km)"); axes[0].set_ylabel("Price (₹)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))
    axes[0].legend(markerscale=4, fontsize=9)

    # Distance bins
    df["Dist_Bin"] = pd.cut(df["Distance_km"],
                            bins=[0,1000,3000,6000,10000,20000],
                            labels=["<1K","1–3K","3–6K","6–10K","10K+"])
    db = df.groupby("Dist_Bin", observed=True)["Price"].median()
    bars = axes[1].bar(db.index.astype(str), db.values,
                       color=PALETTE[:5], edgecolor="white", width=0.6)
    for bar in bars:
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                     f"₹{bar.get_height():,.0f}", ha="center", fontsize=9, fontweight="bold")
    axes[1].set_title("Median Price by Distance Band", fontweight="bold", fontsize=13)
    axes[1].set_xlabel("Distance (km)"); axes[1].set_ylabel("Median Price (₹)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))

    plt.tight_layout()
    return _save("04_price_vs_distance.png")


def plot_booking_window(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bins   = [0,7,14,30,60,90,180,365]
    labels = ["1–7\n(Last min)","8–14","15–30","31–60","61–90","91–180","181–365"]
    df["BW_Bin"] = pd.cut(df["Days_Until_Departure"], bins=bins, labels=labels)
    bw_med  = df.groupby("BW_Bin", observed=True)["Price"].median()
    bw_mean = df.groupby("BW_Bin", observed=True)["Price"].mean()

    axes[0].plot(range(len(bw_med)), bw_med.values, "o-",
                 color=C_BLUE, linewidth=2.5, markersize=9,
                 markerfacecolor="white", markeredgewidth=2.5, label="Median")
    axes[0].plot(range(len(bw_mean)), bw_mean.values, "s--",
                 color=C_GOLD, linewidth=1.5, markersize=7, label="Mean")
    axes[0].fill_between(range(len(bw_med)), bw_med.values, alpha=0.1, color=C_BLUE)
    axes[0].set_xticks(range(len(labels))); axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_title("Price vs Booking Lead Time", fontweight="bold", fontsize=13)
    axes[0].set_xlabel("Days Until Departure (bins)"); axes[0].set_ylabel("Price (₹)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))
    axes[0].legend()
    for i, (med, mean) in enumerate(zip(bw_med.values, bw_mean.values)):
        axes[0].annotate(f"₹{med/1000:.0f}K", (i, med),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=7.5, color=C_BLUE)

    # Volume per bin
    bw_cnt = df.groupby("BW_Bin", observed=True).size()
    axes[1].bar(range(len(bw_cnt)), bw_cnt.values,
                color=[C_RED if i==0 else C_BLUE for i in range(len(bw_cnt))],
                edgecolor="white", width=0.7)
    axes[1].set_xticks(range(len(labels))); axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_title("Booking Volume per Lead-Time Window", fontweight="bold", fontsize=13)
    axes[1].set_xlabel("Days Until Departure"); axes[1].set_ylabel("Number of Bookings")

    plt.tight_layout()
    return _save("05_booking_window.png")


def plot_seasonality(df: pd.DataFrame):
    import calendar as cal
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    m_avg  = df.groupby("Journey_Month")["Price"].mean()
    m_med  = df.groupby("Journey_Month")["Price"].median()
    m_lbl  = [cal.month_abbr[m] for m in m_avg.index]
    peak   = [3,4,5,12,1]
    bar_colors = [C_RED if m in peak else C_BLUE for m in m_avg.index]

    axes[0].bar(m_lbl, m_avg.values, color=bar_colors, edgecolor="white", width=0.7, label="Mean")
    axes[0].plot(m_lbl, m_med.values, "o-", color=C_GOLD, linewidth=2,
                 markersize=7, label="Median")
    axes[0].axhline(m_avg.mean(), color="#888", linestyle="--", linewidth=1.2,
                    label=f"Overall avg ₹{m_avg.mean():,.0f}")
    axes[0].set_title("Price by Month  (Red = Peak season)", fontweight="bold", fontsize=13)
    axes[0].set_ylabel("Price (₹)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))
    axes[0].legend(fontsize=9)

    # Weekday
    wd_avg = df.groupby("Journey_Weekday")["Price"].mean().reindex(range(7))
    wd_lbl = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    wc     = [C_RED if i>=5 else C_BLUE for i in range(7)]
    bars   = axes[1].bar(wd_lbl, wd_avg.values, color=wc, edgecolor="white", width=0.65)
    for bar in bars:
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+80,
                     f"₹{bar.get_height():,.0f}", ha="center", fontsize=9, fontweight="bold")
    axes[1].set_title("Avg Price by Day of Week  (Red = Weekend)", fontweight="bold", fontsize=13)
    axes[1].set_ylabel("Avg Price (₹)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))

    plt.tight_layout()
    return _save("06_seasonality.png")


def plot_stops_analysis(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    stops_order = ["non-stop","1 stop","2 stops"]
    s_med = df.groupby("Total_Stops")["Price"].median().reindex(stops_order)
    bars  = axes[0].bar(stops_order, s_med.values,
                        color=[C_GOLD, C_BLUE, "#90CAF9"], edgecolor="white", width=0.5)
    for bar in bars:
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                     f"₹{bar.get_height():,.0f}", ha="center", fontsize=12, fontweight="bold")
    axes[0].set_title("Median Price by Stops", fontweight="bold", fontsize=13)
    axes[0].set_ylabel("Median Price (₹)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))

    # Class × Stops heatmap
    pivot = df.groupby(["Class","Total_Stops"])["Price"].mean().unstack().reindex(CLASS_ORDER)
    sns.heatmap(pivot, annot=True, fmt=",.0f", cmap="Blues", ax=axes[1],
                linewidths=0.5, linecolor="white", annot_kws={"size":9})
    axes[1].set_title("Mean Price (₹) — Class × Stops", fontweight="bold", fontsize=13)

    plt.tight_layout()
    return _save("07_stops_analysis.png")


def plot_brd_macrofactors(df: pd.DataFrame):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # SAF Zone
    ax1 = fig.add_subplot(gs[0, 0])
    sm  = df.groupby("SAF_Zone")["Price"].agg(["median","mean"])
    ax1.bar(["Zone 0\nNo mandate","Zone 1\nVoluntary","Zone 2\nEU Mandatory"],
            [sm.loc[0,"median"],sm.loc[1,"median"],sm.loc[2,"median"]],
            color=[C_GREEN,C_ORANGE,C_RED], edgecolor="white", width=0.55)
    ax1.set_title("Median Price by SAF Zone", fontweight="bold")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))

    # Env Tier
    ax2 = fig.add_subplot(gs[0, 1])
    em  = df.groupby("Env_Surcharge_Tier")["Price"].median()
    ax2.bar([f"Tier {i}" for i in sorted(em.index)], em.values,
            color=[C_GREEN,C_BLUE,C_ORANGE,C_RED][:len(em)], edgecolor="white", width=0.55)
    ax2.set_title("Median Price by Env Surcharge Tier", fontweight="bold")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))

    # Fleet Age scatter
    ax3 = fig.add_subplot(gs[0, 2])
    s   = df.sample(min(4000,len(df)), random_state=42)
    ax3.scatter(s["Fleet_Age_Years"], s["Price"],
                alpha=0.15, color=C_PURPLE, s=8)
    m, b, r, *_ = stats.linregress(df["Fleet_Age_Years"], df["Price"])
    xs = np.linspace(df["Fleet_Age_Years"].min(), df["Fleet_Age_Years"].max(), 100)
    ax3.plot(xs, m*xs+b, color=C_RED, linewidth=2, linestyle="--", label=f"r={r:.3f}")
    ax3.set_title("Fleet Age vs Price", fontweight="bold")
    ax3.set_xlabel("Fleet Age (years)"); ax3.set_ylabel("Price (₹)")
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))
    ax3.legend(fontsize=9)

    # Restricted airspace
    ax4 = fig.add_subplot(gs[1, 0])
    rm  = df.groupby("Is_Restricted_Airspace")["Price"].agg(["median","mean"])
    x   = [0, 1]
    ax4.bar(["Normal route","Restricted\n(+9% reroute)"],
            [rm.loc[0,"median"],rm.loc[1,"median"]],
            color=[C_BLUE,C_RED], edgecolor="white", width=0.4)
    for i, v in enumerate([rm.loc[0,"median"],rm.loc[1,"median"]]):
        ax4.text(i, v+200, f"₹{v:,.0f}", ha="center", fontsize=11, fontweight="bold")
    ax4.set_title("Restricted Airspace Impact", fontweight="bold")
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))

    # Geo Risk Score
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(s["Geo_Risk_Score"], s["Price"], alpha=0.18, color=C_ORANGE, s=8)
    ax5.set_title("Geo Risk Score vs Price", fontweight="bold")
    ax5.set_xlabel("Geo Risk Score (0–1)"); ax5.set_ylabel("Price (₹)")
    ax5.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))

    # Fuel Price Index
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(s["Fuel_Price_Index"], s["Price"], alpha=0.18, color=C_GREEN, s=8)
    ax6.set_title("Fuel Price Index vs Ticket Price", fontweight="bold")
    ax6.set_xlabel("Fuel Price Index (USD/bbl)"); ax6.set_ylabel("Price (₹)")
    ax6.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))

    plt.suptitle("BRD Phase-2 Macro-Factor Analysis", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    return _save("08_brd_macrofactors.png")


def plot_feature_correlation(df: pd.DataFrame):
    num_cols = [c for c in [
        "Price","Distance_km","Days_Until_Departure","Journey_Month","Journey_Weekday",
        "Is_Weekend","SAF_Zone","Env_Surcharge_Tier","Fleet_Age_Years",
        "Is_Restricted_Airspace","Geo_Risk_Score","Fuel_Price_Index",
        "Seat_Availability","Layover_Hours"
    ] if c in df.columns]

    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax,
                linewidths=0.5, annot_kws={"size":8},
                cbar_kws={"shrink":0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save("09_correlation_heatmap.png")


def plot_seat_availability_fuel(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    s = df.sample(min(6000,len(df)), random_state=42)

    sc1 = axes[0].scatter(s["Seat_Availability"], s["Price"],
                          c=s["Distance_km"], cmap="Blues",
                          alpha=0.3, s=10, vmin=0, vmax=15000)
    plt.colorbar(sc1, ax=axes[0], label="Distance (km)")
    axes[0].set_title("Seat Availability vs Price\n(colour = distance)", fontweight="bold")
    axes[0].set_xlabel("Seat Availability (0=full, 1=empty)")
    axes[0].set_ylabel("Price (₹)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))

    sc2 = axes[1].scatter(s["Fuel_Price_Index"], s["Price"],
                          c=s["Distance_km"], cmap="Oranges",
                          alpha=0.3, s=10, vmin=0, vmax=15000)
    plt.colorbar(sc2, ax=axes[1], label="Distance (km)")
    axes[1].set_title("Fuel Price Index vs Ticket Price\n(colour = distance)", fontweight="bold")
    axes[1].set_xlabel("Fuel Price Index (USD/bbl)")
    axes[1].set_ylabel("Price (₹)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))

    plt.tight_layout()
    return _save("10_availability_fuel.png")


def plot_source_destination(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    src_med = df.groupby("Source")["Price"].median().sort_values(ascending=True)
    axes[0].barh(src_med.index, src_med.values, color=C_BLUE, edgecolor="white", height=0.7)
    axes[0].set_title("Median Fare by Source City", fontweight="bold", fontsize=13)
    axes[0].set_xlabel("Median Price (₹)")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))

    dst_med = df.groupby("Destination")["Price"].median().sort_values(ascending=True)
    axes[1].barh(dst_med.index, dst_med.values, color=C_PURPLE, edgecolor="white", height=0.7)
    axes[1].set_title("Median Fare by Destination City", fontweight="bold", fontsize=13)
    axes[1].set_xlabel("Median Price (₹)")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))

    plt.tight_layout()
    return _save("11_source_destination.png")


def plot_class_distance_heatmap(df: pd.DataFrame):
    df["Dist_Band"] = pd.cut(df["Distance_km"],
                             bins=[0,1500,4000,8000,20000],
                             labels=["Short\n<1.5K","Medium\n1.5–4K",
                                     "Long\n4–8K","Ultra\n>8K"])
    pivot = df.groupby(["Class","Dist_Band"],observed=True)["Price"].mean().unstack().reindex(CLASS_ORDER)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=",.0f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, linecolor="white", annot_kws={"size":10})
    ax.set_title("Mean Price (₹) — Cabin Class × Distance Band",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    return _save("12_class_distance_heatmap.png")


def plot_layover_aircraft(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    lh_bins = pd.cut(df["Layover_Hours"],
                     bins=[-1,0,4,8,14],
                     labels=["Non-stop","1–4 hrs","5–8 hrs","9+ hrs"])
    lh_med = df.groupby(lh_bins, observed=True)["Price"].median()
    axes[0].bar(lh_med.index.astype(str), lh_med.values,
                color=[C_GOLD,C_BLUE,C_BLUE,"#90CAF9"], edgecolor="white", width=0.6)
    for i, v in enumerate(lh_med.values):
        axes[0].text(i, v+200, f"₹{v:,.0f}", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_title("Median Price by Layover Duration", fontweight="bold", fontsize=13)
    axes[0].set_ylabel("Median Price (₹)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))

    ac_med = df.groupby("Aircraft_Type")["Price"].median()
    axes[1].bar(["Narrow-body\n(<4,000 km)","Wide-body\n(>4,000 km)"],
                [ac_med.get("narrow-body",0), ac_med.get("wide-body",0)],
                color=[C_BLUE, C_GOLD], edgecolor="white", width=0.5)
    for i, v in enumerate([ac_med.get("narrow-body",0),ac_med.get("wide-body",0)]):
        axes[1].text(i, v+200, f"₹{v:,.0f}", ha="center", fontsize=12, fontweight="bold")
    axes[1].set_title("Median Price by Aircraft Type", fontweight="bold", fontsize=13)
    axes[1].set_ylabel("Median Price (₹)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))

    plt.tight_layout()
    return _save("13_layover_aircraft.png")


def plot_price_over_time(df: pd.DataFrame):
    df2 = df.copy()
    df2["Journey_Date_dt"] = pd.to_datetime(df2["Journey_Date"], dayfirst=True, errors="coerce")
    df2 = df2.dropna(subset=["Journey_Date_dt"])
    df2["YearMonth"] = df2["Journey_Date_dt"].dt.to_period("M")

    monthly = df2.groupby("YearMonth")["Price"].agg(["median","mean","std"]).reset_index()
    monthly["YearMonth_str"] = monthly["YearMonth"].astype(str)
    x = range(len(monthly))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, monthly["median"], "o-", color=C_BLUE, linewidth=2.5,
            markersize=6, label="Median")
    ax.plot(x, monthly["mean"], "s--", color=C_GOLD, linewidth=1.5,
            markersize=5, label="Mean")
    ax.fill_between(x,
                    monthly["mean"]-monthly["std"],
                    monthly["mean"]+monthly["std"],
                    alpha=0.1, color=C_BLUE, label="±1 Std Dev")
    ax.set_xticks(x[::2])
    ax.set_xticklabels(monthly["YearMonth_str"].iloc[::2], rotation=45, ha="right", fontsize=8)
    ax.set_title("Monthly Price Trend (2026–2027)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Month"); ax.set_ylabel("Price (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))
    ax.legend()
    plt.tight_layout()
    return _save("14_price_over_time.png")


def plot_outlier_analysis(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Z-score outliers
    z = np.abs(stats.zscore(df["Price"]))
    outliers = df[z > 3]
    normal   = df[z <= 3]
    axes[0].hist(normal["Price"], bins=60, color=C_BLUE,  edgecolor="white", alpha=0.7, label=f"Normal ({len(normal):,})")
    axes[0].hist(outliers["Price"], bins=20, color=C_RED, edgecolor="white", alpha=0.9, label=f"Outliers z>3 ({len(outliers):,})")
    axes[0].set_title("Price Distribution with Outlier Flags", fontweight="bold", fontsize=13)
    axes[0].set_xlabel("Price (₹)"); axes[0].set_ylabel("Count")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))
    axes[0].legend()

    # IQR per class
    for i, cls in enumerate(CLASS_ORDER):
        sub = df[df["Class"]==cls]["Price"]
        q1, q3 = sub.quantile(0.25), sub.quantile(0.75)
        iqr = q3 - q1
        axes[1].barh(i, iqr, left=q1, color=CLASS_COLORS[i], edgecolor="white", height=0.5)
        axes[1].text(q3+50, i, f"IQR ₹{iqr:,.0f}", va="center", fontsize=9)
    axes[1].set_yticks(range(len(CLASS_ORDER)))
    axes[1].set_yticklabels(CLASS_ORDER)
    axes[1].set_title("Price IQR (Q1 → Q3) by Class", fontweight="bold", fontsize=13)
    axes[1].set_xlabel("Price (₹)")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))

    plt.tight_layout()
    return _save("15_outlier_analysis.png")


def plot_feature_importance_eda(df: pd.DataFrame):
    """Correlation of each numeric feature with Price — quick EDA importance."""
    num_cols = [c for c in [
        "Distance_km","Days_Until_Departure","Journey_Month","Journey_Weekday",
        "Is_Weekend","SAF_Zone","Env_Surcharge_Tier","Fleet_Age_Years",
        "Is_Restricted_Airspace","Geo_Risk_Score","Fuel_Price_Index",
        "Seat_Availability","Layover_Hours"
    ] if c in df.columns]

    corr_with_price = df[num_cols+["Price"]].corr()["Price"].drop("Price").sort_values()

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [C_RED if v < 0 else C_BLUE for v in corr_with_price.values]
    bars   = ax.barh(corr_with_price.index, corr_with_price.values,
                     color=colors, edgecolor="white", height=0.7)
    ax.axvline(0, color="#333", linewidth=0.8)
    for bar, v in zip(bars, corr_with_price.values):
        ax.text(v + (0.003 if v>=0 else -0.003), bar.get_y()+bar.get_height()/2,
                f"{v:.3f}", va="center", ha="left" if v>=0 else "right", fontsize=9)
    ax.set_title("Pearson Correlation with Price\n(numeric features only — EDA pass)",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel("Correlation coefficient r")
    plt.tight_layout()
    return _save("16_feature_importance_eda.png")


# ── Master function ───────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> dict:
    """
    Run all 16 EDA plots and save to reports/.
    Returns insights dict also saved to reports/insights.json.
    """
    print("  Running EDA — saving 16 plots to reports/...")

    paths = {}
    paths["price_distribution"]   = plot_price_distribution(df)
    paths["price_by_class"]       = plot_price_by_class(df)
    paths["price_by_airline"]     = plot_price_by_airline(df)
    paths["price_vs_distance"]    = plot_price_vs_distance(df)
    paths["booking_window"]       = plot_booking_window(df)
    paths["seasonality"]          = plot_seasonality(df)
    paths["stops_analysis"]       = plot_stops_analysis(df)
    paths["brd_macrofactors"]     = plot_brd_macrofactors(df)
    paths["correlation_heatmap"]  = plot_feature_correlation(df)
    paths["availability_fuel"]    = plot_seat_availability_fuel(df)
    paths["source_destination"]   = plot_source_destination(df)
    paths["class_distance"]       = plot_class_distance_heatmap(df)
    paths["layover_aircraft"]     = plot_layover_aircraft(df)
    paths["price_over_time"]      = plot_price_over_time(df)
    paths["outlier_analysis"]     = plot_outlier_analysis(df)
    paths["feature_importance_eda"]= plot_feature_importance_eda(df)

    for name in paths:
        print(f"    ✓  {os.path.basename(paths[name])}")

    # ── Build insights dict (read by Streamlit) ──────────────────────────────
    eco_mean  = df[df["Class"]=="Economy"]["Price"].mean()
    fst_mean  = df[df["Class"]=="First"]["Price"].mean()
    lm_mean   = df[df["Days_Until_Departure"]<=7]["Price"].mean()
    adv_mean  = df[df["Days_Until_Departure"]>90]["Price"].mean()
    dist_corr = df["Distance_km"].corr(df["Price"])
    top_airline = df.groupby("Airline")["Price"].median().idxmax()
    low_airline = df.groupby("Airline")["Price"].median().idxmin()
    peak_month  = df.groupby("Journey_Month")["Price"].mean().idxmax()
    off_month   = df.groupby("Journey_Month")["Price"].mean().idxmin()
    import calendar as cal
    wknd_mean   = df[df["Is_Weekend"]==1]["Price"].mean()
    wkdy_mean   = df[df["Is_Weekend"]==0]["Price"].mean()
    saf2_med    = df[df["SAF_Zone"]==2]["Price"].median()
    saf0_med    = df[df["SAF_Zone"]==0]["Price"].median()
    restr_med   = df[df["Is_Restricted_Airspace"]==1]["Price"].median()
    norm_med    = df[df["Is_Restricted_Airspace"]==0]["Price"].median()

    insights = {
        "total_rows":         int(len(df)),
        "total_columns":      int(len(df.columns)),
        "price_min":          int(df["Price"].min()),
        "price_max":          int(df["Price"].max()),
        "price_mean":         int(df["Price"].mean()),
        "price_median":       int(df["Price"].median()),
        "price_std":          int(df["Price"].std()),

        "economy_mean":       int(eco_mean),
        "first_mean":         int(fst_mean),
        "first_vs_economy":   round(fst_mean/eco_mean, 2),

        "lastminute_mean":    int(lm_mean),
        "advance_mean":       int(adv_mean),
        "lastminute_premium": round(lm_mean/adv_mean, 2),

        "distance_corr":      round(dist_corr, 3),
        "top_airline":        top_airline,
        "low_airline":        low_airline,
        "peak_month":         cal.month_name[int(peak_month)],
        "off_peak_month":     cal.month_name[int(off_month)],
        "weekend_premium_pct": round((wknd_mean/wkdy_mean-1)*100, 1),

        "saf_zone2_premium_pct":  round((saf2_med/saf0_med-1)*100, 1) if saf0_med else 0,
        "restricted_premium_pct": round((restr_med/norm_med-1)*100, 1) if norm_med else 0,

        "plots": {k: os.path.basename(v) for k, v in paths.items()},
    }

    insights_path = os.path.join(REPORTS_DIR, "insights.json")
    with open(insights_path, "w") as f:
        json.dump(insights, f, indent=2)

    print(f"  ✓  insights.json saved to reports/")
    return insights
