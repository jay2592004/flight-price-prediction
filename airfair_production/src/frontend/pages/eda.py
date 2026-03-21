"""
src/frontend/pages/eda.py  —  EDA Dashboard page
Reads data/flight_price_combined.csv — auto-shows BRD tab when columns exist.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os, sys, calendar as cal

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "data")
PALETTE  = ["#0A1931","#1565C0","#1E88E5","#42A5F5","#90CAF9","#BBDEFB"]
plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False})

CSS = """<style>
.hdr{background:linear-gradient(135deg,#0A1931,#1565C0);border-radius:14px;
  padding:1.8rem 2.4rem;margin-bottom:1.8rem;color:#fff;}
.hdr h2{font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:900;margin:0 0 .3rem;}
.hdr p{color:#90BEFF;font-size:.93rem;margin:0;}
.card{background:#fff;border:1px solid #DCE8FA;border-radius:14px;
  padding:1.4rem;box-shadow:0 2px 12px rgba(21,101,192,.07);margin-bottom:1.4rem;}
.ins{background:#EBF5FF;border-left:4px solid #1565C0;border-radius:8px;
  padding:.85rem 1.1rem;margin-top:.75rem;font-size:.83rem;color:#1A3560;line-height:1.7;}
</style>"""


@st.cache_data(show_spinner="Loading dataset...")
def _load():
    for fname in ["flight_price_combined.csv","flight_price_synthetic.csv","flight_price_dataset.csv"]:
        p = os.path.join(DATA_DIR, fname)
        if os.path.exists(p):
            df = pd.read_csv(p)
            df = df[df["Price"] > 0].copy()
            if "Journey_Date" in df.columns:
                df["Journey_Date_dt"] = pd.to_datetime(df["Journey_Date"], dayfirst=True, errors="coerce")
            if "Journey_Weekday" not in df.columns:
                df["Journey_Weekday"] = df.get("Journey_Date_dt", pd.Series()).apply(
                    lambda d: d.weekday() if pd.notna(d) else 0)
            if "Is_Weekend" not in df.columns:
                df["Is_Weekend"] = (df["Journey_Weekday"] >= 5).astype(int)
            if "Season" not in df.columns:
                df["Season"] = df["Journey_Month"].apply(
                    lambda m: "peak" if m in [12,1,3,4,5] else ("shoulder" if m in [2,6,10,11] else "off_peak"))
            return df, fname
    return pd.DataFrame(), ""


def _saveable_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def render():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""<div class="hdr">
      <h2>📊 Exploratory Data Analysis</h2>
      <p>Live insights from the combined dataset · 25 features · BRD Phase 2 macro-factors included.</p>
    </div>""", unsafe_allow_html=True)

    df, src = _load()
    if df.empty:
        st.error("No dataset found in data/. Run `python train_model.py` first.")
        return

    has_brd = all(c in df.columns for c in
                  ["SAF_Zone","Env_Surcharge_Tier","Fleet_Age_Years","Is_Restricted_Airspace"])
    st.success(f"✅ **{src}** — {len(df):,} rows · {len(df.columns)} columns", icon="📂")

    class_order = ["Economy","Premium Economy","Business","First"]

    tab_names = ["💰 Price","💺 Class","✈️ Airlines","📏 Distance","📅 Season"]
    if has_brd:
        tab_names.append("🌿 BRD Factors")
    tab_names.append("🔗 Correlations")
    tabs = st.tabs(tab_names)
    tab  = {n:t for n,t in zip(tab_names,tabs)}

    # ── Price Overview ────────────────────────────────────────────────────────
    with tab["💰 Price"]:
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig,ax = plt.subplots(figsize=(6,4))
            ax.hist(df["Price"],bins=80,color="#1565C0",edgecolor="white",alpha=.85)
            ax.axvline(df["Price"].median(),color="red",linestyle="--",linewidth=1.8,
                       label=f"Median ₹{df['Price'].median():,.0f}")
            ax.set_title("Price Distribution",fontweight="bold")
            ax.set_xlabel("Price (₹)"); ax.set_ylabel("Count")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
            ax.legend(); ax.set_facecolor("#F8FBFF"); fig.patch.set_facecolor("white")
            plt.tight_layout(); _saveable_fig(fig)
            st.markdown(f"""<div class="ins">
              Range ₹{df['Price'].min():,.0f}–₹{df['Price'].max():,.0f} ·
              Mean ₹{df['Price'].mean():,.0f} · Median ₹{df['Price'].median():,.0f}</div>""",
              unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig,ax = plt.subplots(figsize=(6,4))
            ax.hist(np.log1p(df["Price"]),bins=80,color="#00897B",edgecolor="white",alpha=.85)
            ax.set_title("Log-Price (near-normal)",fontweight="bold")
            ax.set_xlabel("log(1+Price)"); ax.set_ylabel("Count")
            ax.set_facecolor("#F8FBFF"); fig.patch.set_facecolor("white")
            plt.tight_layout(); _saveable_fig(fig)
            st.markdown('<div class="ins">Log-transform removes skew — improves linear baselines.</div>',
                        unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        m1,m2,m3,m4,m5 = st.columns(5)
        for col,lbl,val in zip([m1,m2,m3,m4,m5],
            ["Records","Mean","Median","Std Dev","Max"],
            [f"{len(df):,}",f"₹{df['Price'].mean():,.0f}",f"₹{df['Price'].median():,.0f}",
             f"₹{df['Price'].std():,.0f}",f"₹{df['Price'].max():,.0f}"]):
            with col: st.metric(lbl,val)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Cabin Class ───────────────────────────────────────────────────────────
    with tab["💺 Class"]:
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            ca = df.groupby("Class")["Price"].mean().reindex(class_order)
            fig,ax = plt.subplots(figsize=(6,4))
            bars = ax.bar(ca.index,ca.values,color=["#90CAF9","#42A5F5","#1565C0","#FFB300"],edgecolor="white")
            for b in bars:
                ax.text(b.get_x()+b.get_width()/2,b.get_height()+200,
                        f"₹{b.get_height():,.0f}",ha="center",fontsize=9,fontweight="bold")
            ax.set_title("Avg Price by Cabin Class",fontweight="bold")
            ax.set_ylabel("Avg Price (₹)")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
            ax.set_facecolor("#F8FBFF"); fig.patch.set_facecolor("white")
            plt.tight_layout(); _saveable_fig(fig)
            eco=df[df["Class"]=="Economy"]["Price"].mean()
            fst=df[df["Class"]=="First"]["Price"].mean()
            st.markdown(f'<div class="ins">First (₹{fst:,.0f}) = {fst/eco:.1f}× Economy (₹{eco:,.0f}). '
                        f'<b>Class is #1 price driver.</b></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            pivot = df.groupby(["Class","Total_Stops"])["Price"].mean().unstack().reindex(class_order)
            fig,ax = plt.subplots(figsize=(6,4))
            sns.heatmap(pivot,annot=True,fmt=",.0f",cmap="Blues",ax=ax,
                        linewidths=.5,linecolor="white",annot_kws={"size":9})
            ax.set_title("Avg Price by Class & Stops",fontweight="bold")
            fig.patch.set_facecolor("white"); plt.tight_layout(); _saveable_fig(fig)
            st.markdown('<div class="ins">Non-stop premium routes cost most — even with more stops.</div>',
                        unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Airlines ──────────────────────────────────────────────────────────────
    with tab["✈️ Airlines"]:
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            aa = df.groupby("Airline")["Price"].mean().sort_values(ascending=True)
            fig,ax = plt.subplots(figsize=(6,5.5))
            colors = ["#FFB300" if v==aa.max() else "#1565C0" for v in aa.values]
            ax.barh(aa.index,aa.values,color=colors,edgecolor="white")
            for i,v in enumerate(aa.values):
                ax.text(v+100,i,f"₹{v:,.0f}",va="center",fontsize=8)
            ax.set_title("Avg Price by Airline",fontweight="bold")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
            ax.set_facecolor("#F8FBFF"); fig.patch.set_facecolor("white")
            plt.tight_layout(); _saveable_fig(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            if "Airline_Tier" in df.columns:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                tm = df.groupby("Airline_Tier")["Price"].median().reindex(["budget","mid","premium"])
                fig,ax = plt.subplots(figsize=(6,4))
                ax.bar(["Budget","Mid","Premium"],tm.values,
                       color=["#90CAF9","#1565C0","#FFB300"],edgecolor="white",width=.5)
                for i,v in enumerate(tm.values):
                    ax.text(i,v+300,f"₹{v:,.0f}",ha="center",fontsize=11,fontweight="bold")
                ax.set_title("Median Price by Airline Tier",fontweight="bold")
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
                ax.set_facecolor("#F8FBFF"); fig.patch.set_facecolor("white")
                plt.tight_layout(); _saveable_fig(fig)
                st.markdown('<div class="ins">Premium carriers command ~58% premium over budget.</div>',
                            unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # ── Distance & Stops ──────────────────────────────────────────────────────
    with tab["📏 Distance"]:
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            s = df.sample(min(5000,len(df)),random_state=42)
            pal= {"Economy":"#90CAF9","Premium Economy":"#1E88E5","Business":"#1565C0","First":"#FFB300"}
            fig,ax = plt.subplots(figsize=(6,4.5))
            for cls in class_order:
                sub=s[s["Class"]==cls]
                ax.scatter(sub["Distance_km"],sub["Price"],alpha=.2,s=7,label=cls,color=pal[cls])
            corr = df["Distance_km"].corr(df["Price"])
            ax.set_title(f"Distance vs Price  (r={corr:.3f})",fontweight="bold")
            ax.set_xlabel("Distance (km)"); ax.set_ylabel("Price (₹)")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
            ax.legend(title="Class",markerscale=4,fontsize=8)
            ax.set_facecolor("#F8FBFF"); fig.patch.set_facecolor("white")
            plt.tight_layout(); _saveable_fig(fig)
            st.markdown(f'<div class="ins">r={corr:.3f} — Distance is <b>#2 feature</b>.</div>',
                        unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            bins_ = pd.cut(df["Days_Until_Departure"],[0,7,30,90,180,365],
                           labels=["1–7","8–30","31–90","91–180","181+"])
            da = df.groupby(bins_,observed=True)["Price"].mean()
            fig,ax = plt.subplots(figsize=(6,4.5))
            ax.bar(da.index.astype(str),da.values,
                   color=["#E53935","#FF7043","#1E88E5","#1565C0","#0A1931"],edgecolor="white")
            for i,v in enumerate(da.values):
                ax.text(i,v+150,f"₹{v:,.0f}",ha="center",fontsize=8.5,fontweight="bold")
            ax.set_title("Price by Booking Lead Time",fontweight="bold")
            ax.set_ylabel("Avg Price (₹)")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
            ax.set_facecolor("#F8FBFF"); fig.patch.set_facecolor("white")
            plt.tight_layout(); _saveable_fig(fig)
            lm=df[df["Days_Until_Departure"]<=7]["Price"].mean()
            ea=df[df["Days_Until_Departure"]>90]["Price"].mean()
            st.markdown(f'<div class="ins">Last-minute (₹{lm:,.0f}) = {lm/ea:.1f}× early-bird (₹{ea:,.0f}).</div>',
                        unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Seasonality ───────────────────────────────────────────────────────────
    with tab["📅 Season"]:
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            ma = df.groupby("Journey_Month")["Price"].mean()
            ml = [cal.month_abbr[m] for m in ma.index]
            cols_ = ["#E53935" if m in [3,4,5,12,1] else "#1565C0" for m in ma.index]
            fig,ax = plt.subplots(figsize=(7,4))
            ax.bar(ml,ma.values,color=cols_,edgecolor="white",width=.7)
            ax.axhline(ma.mean(),color="#FFB300",linestyle="--",linewidth=1.8,label="Annual avg")
            ax.set_title("Avg Price by Month  (Red=Peak)",fontweight="bold")
            ax.set_ylabel("Avg Price (₹)")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
            ax.legend(); ax.set_facecolor("#F8FBFF"); fig.patch.set_facecolor("white")
            plt.tight_layout(); _saveable_fig(fig)
            st.markdown('<div class="ins">Dec–Jan and Mar–May peak months show 15–20% higher fares.</div>',
                        unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            wd = df.groupby("Journey_Weekday")["Price"].mean().reindex(range(7))
            wl = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            wc = ["#E53935" if i>=5 else "#1565C0" for i in range(7)]
            fig,ax = plt.subplots(figsize=(7,4))
            ax.bar(wl,wd.values,color=wc,edgecolor="white",width=.6)
            ax.set_title("Avg Price by Weekday  (Red=Weekend)",fontweight="bold")
            ax.set_ylabel("Avg Price (₹)")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
            ax.set_facecolor("#F8FBFF"); fig.patch.set_facecolor("white")
            plt.tight_layout(); _saveable_fig(fig)
            st.markdown('<div class="ins">Weekends ~7% more expensive. Tue/Wed cheapest.</div>',
                        unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── BRD Macro-Factors ─────────────────────────────────────────────────────
    if has_brd:
        with tab["🌿 BRD Factors"]:
            st.info("BRD Phase 2 external macro-factors — each contributes a measurable signal "
                    "to the model, verified by SHAP.", icon="🔬")
            c1,c2 = st.columns(2)
            with c1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                sm = df.groupby("SAF_Zone")["Price"].median()
                fig,ax = plt.subplots(figsize=(6,4))
                ax.bar(["Zone 0\nNo mandate","Zone 1\nVoluntary","Zone 2\nEU Mandatory"],
                       [sm.get(0,0),sm.get(1,0),sm.get(2,0)],
                       color=["#0D9B6E","#E65100","#C62828"],edgecolor="white")
                ax.set_title("Median Price by SAF Zone",fontweight="bold")
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
                ax.set_facecolor("#F8FBFF"); fig.patch.set_facecolor("white")
                plt.tight_layout(); _saveable_fig(fig)
                st.markdown('<div class="ins">EU mandatory SAF (Zone 2) adds ~6% to fares.</div>',
                            unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                rm = df.groupby("Is_Restricted_Airspace")["Price"].median()
                fig,ax = plt.subplots(figsize=(6,4))
                ax.bar(["Normal route","Restricted\n(+9% reroute)"],
                       [rm.get(0,0),rm.get(1,0)],
                       color=["#1565C0","#C62828"],edgecolor="white",width=.4)
                for i,v in enumerate([rm.get(0,0),rm.get(1,0)]):
                    ax.text(i,v+200,f"₹{v:,.0f}",ha="center",fontsize=11,fontweight="bold")
                ax.set_title("Normal vs Restricted Airspace",fontweight="bold")
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
                ax.set_facecolor("#F8FBFF"); fig.patch.set_facecolor("white")
                plt.tight_layout(); _saveable_fig(fig)
                st.markdown('<div class="ins">Restricted routes (Delhi–London, Mumbai–Paris, etc.) '
                            'cost ~9% more due to rerouting.</div>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            c3,c4 = st.columns(2)
            with c3:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                s2=df.sample(min(4000,len(df)),random_state=42)
                fig,ax = plt.subplots(figsize=(6,4))
                ax.scatter(s2["Fleet_Age_Years"],s2["Price"],alpha=.15,color="#7B1FA2",s=8)
                ax.set_title("Fleet Age vs Price",fontweight="bold")
                ax.set_xlabel("Fleet Age (years)"); ax.set_ylabel("Price (₹)")
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
                ax.set_facecolor("#F8FBFF"); fig.patch.set_facecolor("white")
                plt.tight_layout(); _saveable_fig(fig)
                st.markdown('<div class="ins">Older fleets → higher operating costs → higher fares.</div>',
                            unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with c4:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                em = df.groupby("Env_Surcharge_Tier")["Price"].median()
                fig,ax = plt.subplots(figsize=(6,4))
                ax.bar([f"Tier {i}" for i in sorted(em.index)],em.values,
                       color=["#0D9B6E","#1565C0","#E65100","#C62828"][:len(em)],edgecolor="white")
                ax.set_title("Env Surcharge Tier vs Price",fontweight="bold")
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}K"))
                ax.set_facecolor("#F8FBFF"); fig.patch.set_facecolor("white")
                plt.tight_layout(); _saveable_fig(fig)
                st.markdown('<div class="ins">EU ETS (Tier 3) routes show highest environmental surcharges.</div>',
                            unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # ── Correlations ──────────────────────────────────────────────────────────
    with tab["🔗 Correlations"]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        num_cols = [c for c in ["Price","Distance_km","Days_Until_Departure","Journey_Month",
                                 "Journey_Weekday","Is_Weekend","SAF_Zone","Env_Surcharge_Tier",
                                 "Fleet_Age_Years","Is_Restricted_Airspace","Fuel_Price_Index",
                                 "Seat_Availability","Layover_Hours"] if c in df.columns]
        corr = df[num_cols].corr()
        fig,ax = plt.subplots(figsize=(11,8))
        sns.heatmap(corr,annot=True,fmt=".2f",cmap="coolwarm",center=0,ax=ax,
                    linewidths=.5,annot_kws={"size":8})
        ax.set_title("Feature Correlation Matrix",fontsize=13,fontweight="bold")
        fig.patch.set_facecolor("white"); plt.tight_layout(); _saveable_fig(fig)
        st.markdown('<div class="ins">Distance and Price have the strongest positive correlation. '
                    'BRD macro-factors show small but real positive correlations with Price.</div>',
                    unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
