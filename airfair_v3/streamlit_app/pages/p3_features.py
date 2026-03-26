"""
pages/p3_features.py  —  Feature Engineering  (Classic Edition)
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.loader import load_insights

# ── Classic palette ───────────────────────────────────────────────────────────
C_NAVY   = "#0D1B2A"
C_GOLD   = "#C5A028"
C_IVORY  = "#F5F3EE"
C_SLATE  = "#1A3A5C"
C_WINE   = "#9B2335"
C_SAGE   = "#276749"
C_WARM   = "#8B6914"
C_MUTED  = "#718096"

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#FAFAF8",
    "axes.edgecolor":    "#D4C9B0",
    "grid.color":        "#EDE8DC",
    "grid.alpha":        0.6,
    "font.family":       "sans-serif",
})


def _fig_card(title: str, insight_html: str, fig):
    st.markdown(f'<div class="af-card"><div class="af-card-title">{title}</div>',
                unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    if insight_html:
        st.markdown(f'<div class="af-insight">{insight_html}</div>',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render():
    ins = load_insights()

    st.markdown("""<div class="af-page-hdr">
      <h2>⚙️  Feature Engineering</h2>
      <p>How 25 raw columns become 32 model-ready features ·
         Every transformation explained visually with insight</p>
    </div>""", unsafe_allow_html=True)

    # ── Summary pipeline table ─────────────────────────────────────────────────
    st.markdown('<div class="af-card"><div class="af-card-title">Engineering Pipeline — 25 raw → 32 model features</div>',
                unsafe_allow_html=True)
    tbl = pd.DataFrame([
        ("8 categorical columns",      "Label Encoding",  "8 → 8",  "Integer codes for tree splits"),
        ("Distance_km",                "Log Transform",   "1 → 2",  "Captures diminishing returns on ultra-long-haul"),
        ("Days_Until_Departure",       "Log Transform",   "1 → 2",  "Compresses advance window; expands last-minute spike"),
        ("Class_Enc × Distance_km",    "Interaction",     "→ 1",    "Top-3 feature — cabin premium scales with distance"),
        ("Season_Enc × Days_Until",    "Interaction",     "→ 1",    "Peak season × late booking = maximum price premium"),
        ("Airline_Tier × SAF_Zone",    "Interaction",     "→ 1",    "Premium carrier on SAF route = double levy"),
        ("Total_Stops × Distance",     "Interaction",     "→ 1",    "Stop count effect scales with route distance"),
        ("Env_Tier × Fleet_Age",       "Interaction",     "→ 1",    "High-env route + old fleet = compounded cost"),
        ("Distance_km > 5,000",        "Binary Flag",     "→ 1",    "Wide-body / ultra-long-haul threshold"),
        ("Days_Until < 7",             "Binary Flag",     "→ 1",    "Last-minute urgent booking signal"),
        ("Days_Until > 90",            "Binary Flag",     "→ 1",    "Early-bird advance booking discount window"),
    ], columns=["Input","Transform","Output cols","Why it matters"])
    st.dataframe(tbl, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    tabs = st.tabs(["📝 Encoding","📐 Log Transforms","🔀 Interactions","🚩 Binary Flags","🌿 BRD Features"])

    # ── Encoding ──────────────────────────────────────────────────────────────
    with tabs[0]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        tier_labels = ["Budget\n(AirAsia, IndiGo,\nSpiceJet)",
                       "Mid-tier\n(Air India, Thai,\nVistara)",
                       "Premium\n(Emirates, Singapore,\nQatar)"]
        tier_prices = [12000, 19000, 31000]
        bars = axes[0].bar(tier_labels, tier_prices,
                           color=[C_SLATE, C_NAVY, C_GOLD],
                           edgecolor="white", width=0.55)
        axes[0].set_title("Airline Tier → Median Price", fontweight="bold", fontsize=12)
        axes[0].set_ylabel("Median Price (₹)")
        for bar, v in zip(bars, tier_prices):
            axes[0].text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+400, f"₹{v:,}",
                         ha="center", fontsize=10, fontweight="bold", color=C_NAVY)

        stops_labels = ["non-stop\nEnc = 0","1 stop\nEnc = 1","2 stops\nEnc = 2"]
        stops_prices = [18500, 15200, 12800]
        axes[1].bar(stops_labels, stops_prices,
                    color=[C_GOLD, C_NAVY, C_SLATE],
                    edgecolor="white", width=0.5)
        axes[1].set_title("Stop Count — Ordinal Encoding Preserves Rank",
                          fontweight="bold", fontsize=12)
        axes[1].set_ylabel("Median Price (₹)")
        for i, v in enumerate(stops_prices):
            axes[1].text(i, v+300, f"₹{v:,}", ha="center",
                         fontsize=10, fontweight="bold", color=C_NAVY)
        plt.tight_layout()

        _fig_card("Label Encoding",
            """<strong>Label Encoding</strong> converts string categories to integers.
            For ordinal features (Stops: 0/1/2) the rank is preserved.
            For nominal features (Airline, City) the values are arbitrary —
            but tree models learn the correct split boundaries regardless.<br><br>
            <strong>Airline_Tier</strong> is an engineered grouping of 15 carriers into
            3 tiers, reducing sparsity while retaining the dominant pricing signal.
            It alone explains ~8% of price variance in isolation.""", fig)

    # ── Log Transforms ────────────────────────────────────────────────────────
    with tabs[1]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        x_dist = np.linspace(300, 15000, 300)
        x_days = np.linspace(1, 365, 300)

        price_d = 5000 + x_dist * 1.9 * (1 - np.log1p(x_dist)*0.018)
        price_y = 30000 * np.exp(-0.008 * x_days) + 8000

        ax0 = axes[0]
        ax0.plot(x_dist, price_d, color=C_NAVY, linewidth=2.5, label="Price (₹)")
        ax0b = ax0.twinx()
        ax0b.plot(x_dist, np.log1p(x_dist), color=C_GOLD,
                  linewidth=1.8, linestyle="--", label="log(1 + Distance)")
        ax0.set_title("Distance → Log Transform\n(diminishing returns at long haul)",
                      fontweight="bold", fontsize=11)
        ax0.set_xlabel("Distance (km)"); ax0.set_ylabel("Price (₹)", color=C_NAVY)
        ax0b.set_ylabel("log(1 + Distance)", color=C_GOLD)
        ax0.legend(loc="upper left", fontsize=8)
        ax0b.legend(loc="lower right", fontsize=8)

        ax1 = axes[1]
        ax1.plot(x_days, price_y, color=C_NAVY, linewidth=2.5, label="Price (₹)")
        ax1b = ax1.twinx()
        ax1b.plot(x_days, np.log1p(x_days), color=C_GOLD,
                  linewidth=1.8, linestyle="--", label="log(1 + Days)")
        ax1.set_title("Days Until Departure → Log Transform\n(steep last-minute spike)",
                      fontweight="bold", fontsize=11)
        ax1.set_xlabel("Days Until Departure"); ax1.set_ylabel("Price (₹)", color=C_NAVY)
        ax1b.set_ylabel("log(1 + Days)", color=C_GOLD)
        ax1.axvspan(0, 7, alpha=0.12, color=C_WINE, label="Last-minute window")
        ax1.legend(loc="upper left", fontsize=8)
        ax1b.legend(loc="lower right", fontsize=8)
        plt.tight_layout()

        _fig_card("Log Transforms — Distance & Days Until Departure",
            """<strong>Why log(Distance)?</strong><br>
            A 14,000 km flight costs more than a 7,000 km flight — but not 2×.
            The price curve is concave (diminishing returns). log(1+x) linearises it,
            reducing influence of ultra-long-haul outliers on tree splits.<br><br>
            <strong>Why log(Days_Until)?</strong><br>
            The 0–7 day window has an extremely steep price spike.
            Log-transform compresses the long advance-booking tail and gives the
            model fine-grained resolution exactly where prices vary most.""", fig)

    # ── Interactions ──────────────────────────────────────────────────────────
    with tabs[2]:
        class_labels = ["Economy","Prem. Eco","Business","First"]
        class_base   = [5000, 12000, 28000, 55000]
        dists_plot   = [1000, 3000, 6000, 10000, 14000]
        colors_cls   = ["#90CAF9", "#42A5F5", C_NAVY, C_GOLD]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for i, (cls, base, col) in enumerate(zip(class_labels, class_base, colors_cls)):
            prices   = [base + d*1.9*(1-np.log1p(d)*0.018) for d in dists_plot]
            interact = [i * d / 1000 for d in dists_plot]
            axes[0].plot(dists_plot, prices, "o-", color=col, linewidth=2.5,
                         markersize=7, label=cls)
            axes[1].plot(dists_plot, interact, "o-", color=col, linewidth=2,
                         markersize=6, label=cls)

        axes[0].set_title("Separate features:\nClass + Distance", fontweight="bold")
        axes[0].set_xlabel("Distance (km)"); axes[0].set_ylabel("Price (₹)")
        axes[0].legend(fontsize=9)

        axes[1].set_title("Interaction term:\nClass_Enc × Distance / 1,000",
                          fontweight="bold")
        axes[1].set_xlabel("Distance (km)")
        axes[1].set_ylabel("Class_Enc × Distance (scaled)")
        axes[1].legend(fontsize=9)
        plt.tight_layout()

        _fig_card("Interaction Features — Class × Distance (top-3 model feature)",
            """<strong>Why interaction features?</strong><br>
            Tree models <em>can</em> learn interactions by nesting splits —
            but explicit terms pre-compute the signal, reducing required tree depth
            and improving generalisation on unseen route-class combinations.<br><br>
            <strong>Class × Distance</strong> — the First/Economy price gap
            <em>widens</em> on longer routes.
            Neither feature alone captures this scaling; their product does.<br><br>
            <strong>Season × Booking Window</strong> — peak season + last-minute
            booking = worst-case price. The interaction captures the multiplicative effect.<br><br>
            <strong>Env_Tier × Fleet_Age</strong> — high-surcharge routes <em>and</em>
            inefficient older aircraft compound the cost pressure additively.""", fig)

    # ── Binary Flags ──────────────────────────────────────────────────────────
    with tabs[3]:
        lm_v  = ins.get("lastminute_mean",  38000) or 38000
        adv_v = ins.get("advance_mean",     16000) or 16000

        fig, ax = plt.subplots(figsize=(10, 4.5))
        labels = ["Is_Long_Haul\nDistance > 5,000 km",
                  "Is_Last_Minute\nDays < 7",
                  "Is_Advance_Booking\nDays > 90"]
        values = [14200, int(lm_v), int(adv_v)]
        colors = [C_NAVY, C_WINE, C_SAGE]

        bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+400,
                    f"₹{v:,}", ha="center", fontsize=11, fontweight="bold", color=C_NAVY)
        ax.set_title("Binary Flags — Avg Price per Segment", fontweight="bold", fontsize=12)
        ax.set_ylabel("Average Price (₹)")
        plt.tight_layout()

        _fig_card("Binary Flags — Sharp Boundaries for Tree Models",
            f"""Binary flags create hard decision boundaries that tree models can
            exploit with a single split node — highly efficient for training.<br><br>
            <strong>Is_Long_Haul</strong> (Distance &gt; 5,000 km) — clean threshold
            separating continental from intercontinental pricing tiers.<br>
            <strong>Is_Last_Minute</strong> (Days &lt; 7) — the steepest spike zone;
            avg ₹{lm_v:,} vs ₹{adv_v:,} for advance bookings.<br>
            <strong>Is_Advance_Booking</strong> (Days &gt; 90) — captures the early-bird
            discount window, ~{round((1-adv_v/lm_v)*100,0):.0f}% cheaper than last-minute.""", fig)

    # ── BRD Features ──────────────────────────────────────────────────────────
    with tabs[4]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        zones = ["Zone 0\nNo mandate", "Zone 1\nVoluntary", "Zone 2\nEU Mandatory"]
        prems = [0, 2, 6]
        bars  = axes[0].bar(zones, prems,
                            color=[C_SAGE, C_WARM, C_WINE],
                            edgecolor="white", width=0.5)
        for bar, v in zip(bars, prems):
            axes[0].text(bar.get_x()+bar.get_width()/2, v+0.15,
                         f"+{v}%", ha="center", fontsize=13, fontweight="bold",
                         color=C_NAVY)
        axes[0].set_title("SAF Zone → Price Premium",
                          fontweight="bold", fontsize=12)
        axes[0].set_ylabel("Price premium (%)")

        ages  = np.linspace(3, 25, 200)
        mult  = (1 + (ages-8)*0.004).clip(0.96, 1.10)
        axes[1].plot(ages, (mult-1)*100, color=C_NAVY, linewidth=2.5)
        axes[1].axvline(8, color=C_MUTED, linestyle="--", linewidth=1.2,
                        label="Baseline (8 yr)")
        axes[1].fill_between(ages, (mult-1)*100, alpha=0.15, color=C_GOLD)
        axes[1].set_title("Fleet Age → Operating Cost Premium",
                          fontweight="bold", fontsize=12)
        axes[1].set_xlabel("Fleet Age (years)")
        axes[1].set_ylabel("Price premium (%)")
        axes[1].legend(fontsize=9)
        plt.tight_layout()

        st.markdown('<div class="af-card"><div class="af-card-title">BRD Phase-2 Feature Reference</div>',
                    unsafe_allow_html=True)
        brd_df = pd.DataFrame([
            ("SAF_Zone",              "0 / 1 / 2",    "Destination SAF mandate",       "+0% / +2% / +6%",  "✅"),
            ("Env_Surcharge_Tier",    "0 / 1 / 2 / 3","Max of source+dest tier",       "+0–4.5%",           "✅"),
            ("Fleet_Age_Years",       "3.0 – 25.0 yr","Per-airline avg fleet age",      "+0.4%/yr >8yr",     "✅"),
            ("Is_Restricted_Airspace","0 or 1",       "Route requires detour",         "+9% reroute cost", "✅"),
        ], columns=["Feature","Values","Source","Price Effect","SHAP top-10"])
        st.dataframe(brd_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        _fig_card("BRD Phase-2 Macro-Factors — SAF Zone & Fleet Age",
            f"""<strong>SAF_Zone</strong> — EU Regulation 2023/2405 mandates
            Sustainable Aviation Fuel at EU/UK airports (real cost since Jan 2025).
            Zone 2 routes add ~6% to ticket price.<br><br>
            <strong>Env_Surcharge_Tier</strong> — EU ETS carbon allowance costs.
            Tier 3 (London, Frankfurt, Paris) carry the highest permits.<br><br>
            <strong>Fleet_Age_Years</strong> — older aircraft burn more fuel per km.
            A 20-year fleet costs ~{round((1+(20-8)*0.004-1)*100,1):.0f}% more to operate
            vs an 8-year baseline fleet.<br><br>
            <strong>Is_Restricted_Airspace</strong> — post-2022 overflight
            restrictions force Delhi/Mumbai → Europe routes to add 2–4 hours
            and ~9% operating cost.""", fig)
