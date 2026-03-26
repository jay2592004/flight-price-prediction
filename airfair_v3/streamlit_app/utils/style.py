"""
utils/style.py
Clean, readable design system.
Fonts: Inter (system fallback stack) + Georgia for display headings.
No external font dependency — works fully offline.
"""

GLOBAL_CSS = """
<style>
/* ── Font stack — readable system fonts, no external CDN ────── */
html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont,
               'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  font-size: 15px;
  line-height: 1.6;
  background-color: #F5F3EE;
  color: #1A1A2E;
  -webkit-font-smoothing: antialiased;
}

/* display / serif headings */
.af-display {
  font-family: Georgia, 'Times New Roman', Times, serif;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: #0D1B2A !important;
  border-right: 1px solid #1E3A5F;
}
[data-testid="stSidebar"] * { color: #CBD5E0 !important; }
[data-testid="stSidebar"] label {
  color: #718096 !important;
  font-size: .72rem !important;
  font-weight: 600 !important;
  text-transform: uppercase;
  letter-spacing: .09em;
}
[data-testid="stSidebar"] .stRadio > div > label {
  padding: .5rem .75rem;
  border-radius: 6px;
  font-size: .9rem !important;
  font-weight: 500 !important;
  letter-spacing: 0;
  text-transform: none !important;
  transition: background .15s;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
  background: rgba(255,255,255,.07) !important;
}

/* ── Main container ──────────────────────────────────────── */
.main .block-container {
  background: #F5F3EE;
  padding-top: 1.4rem;
}

/* ── Primary button ──────────────────────────────────────── */
.stButton > button {
  background: #1A3A5C !important;
  color: #F5F3EE !important;
  border: 1px solid #C5A028 !important;
  border-radius: 6px !important;
  padding: .55rem 1.6rem !important;
  font-family: inherit !important;
  font-weight: 600 !important;
  font-size: .9rem !important;
  letter-spacing: .02em !important;
  width: 100% !important;
  transition: background .18s, color .18s !important;
}
.stButton > button:hover {
  background: #C5A028 !important;
  color: #0D1B2A !important;
  border-color: #C5A028 !important;
}

/* ── Form inputs ─────────────────────────────────────────── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
  background: #FFFFFF !important;
  border: 1px solid #D4C9B0 !important;
  border-radius: 6px !important;
  color: #1A1A2E !important;
  font-family: inherit !important;
  font-size: .9rem !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus {
  border-color: #C5A028 !important;
  box-shadow: 0 0 0 2px rgba(197,160,40,.18) !important;
}

/* ── Hero banner ─────────────────────────────────────────── */
.af-hero {
  background: #0D1B2A;
  padding: 2.4rem 2.8rem 2rem;
  margin: -1.4rem -1rem 1.8rem -1rem;
  border-bottom: 3px solid #C5A028;
  position: relative;
  overflow: hidden;
}
.af-hero::before {
  content: '';
  position: absolute; top: 0; right: 0; bottom: 0;
  width: 38%;
  background: radial-gradient(ellipse at right center,
    rgba(197,160,40,.1) 0%, transparent 70%);
  pointer-events: none;
}
.af-hero-eyebrow {
  font-size: .7rem; font-weight: 700;
  color: #C5A028; letter-spacing: .2em;
  text-transform: uppercase; margin-bottom: .5rem;
}
.af-hero h1 {
  font-family: Georgia, 'Times New Roman', serif;
  font-size: 2.6rem; font-weight: 700;
  color: #F5F3EE; margin: 0 0 .45rem; line-height: 1.15;
}
.af-hero p {
  color: #8FA8C0; font-size: .92rem;
  margin: 0; max-width: 580px;
}

/* ── KPI strip ───────────────────────────────────────────── */
.af-kpi-strip {
  display: flex;
  background: #FFFFFF;
  border: 1px solid #D4C9B0;
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 1.8rem;
}
.af-kpi {
  flex: 1; padding: 1rem 1.2rem;
  border-right: 1px solid #D4C9B0;
  text-align: center;
}
.af-kpi:last-child { border-right: none; }
.af-kpi-value {
  font-family: Georgia, serif;
  font-size: 1.75rem; font-weight: 700;
  color: #0D1B2A; line-height: 1;
}
.af-kpi-value.gold { color: #C5A028; }
.af-kpi-label {
  font-size: .67rem; font-weight: 600;
  color: #718096; text-transform: uppercase;
  letter-spacing: .08em; margin-top: .3rem;
}
.af-kpi-sub { font-size: .64rem; color: #A0AEC0; margin-top: .1rem; }

/* ── Section heading ─────────────────────────────────────── */
.af-section-title {
  font-family: Georgia, serif;
  font-size: 1.35rem; font-weight: 700;
  color: #0D1B2A;
  padding-bottom: .4rem;
  border-bottom: 2px solid #C5A028;
  display: inline-block;
  margin: 0 0 1.1rem;
}

/* ── Form panel ──────────────────────────────────────────── */
.af-panel {
  background: #FFFFFF;
  border: 1px solid #D4C9B0;
  border-radius: 8px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 1.1rem;
}
.af-panel-title {
  font-size: .7rem; font-weight: 700;
  color: #718096; letter-spacing: .12em;
  text-transform: uppercase; margin-bottom: .9rem;
  padding-bottom: .45rem; border-bottom: 1px solid #EDE8DC;
}

/* ── Dynamic travel time display ────────────────────────── */
.af-traveltime {
  background: #F0F8FF;
  border: 1px solid #B8D4E8;
  border-radius: 6px;
  padding: .55rem 1rem;
  font-size: .85rem; color: #1A3A5C;
  font-weight: 600;
  margin-top: .5rem;
  display: flex; align-items: center; gap: .5rem;
}
.af-traveltime-label {
  font-size: .68rem; color: #718096;
  font-weight: 400; margin-left: auto;
}

/* ── Distance result box ─────────────────────────────────── */
.af-dist-result {
  background: #FBF7EC;
  border: 1px solid #C5A028;
  border-radius: 6px;
  padding: .55rem 1rem;
  font-size: .82rem; color: #744210;
  margin-top: .5rem;
}

/* ── Prediction result card ──────────────────────────────── */
.af-result {
  background: #0D1B2A;
  border: 1px solid #C5A028;
  border-radius: 8px;
  padding: 1.8rem 1.8rem 1.5rem;
  text-align: center;
  margin-top: 1.1rem;
}
.af-result-eyebrow {
  font-size: .68rem; font-weight: 700;
  color: #8FA8C0; letter-spacing: .16em;
  text-transform: uppercase; margin-bottom: .45rem;
}
.af-result-price {
  font-family: Georgia, serif;
  font-size: 3.2rem; font-weight: 700;
  color: #C5A028; line-height: 1;
}
.af-result-range {
  font-size: .78rem; color: #8FA8C0; margin-top: .35rem;
}
.af-result-badge {
  display: inline-block;
  background: rgba(197,160,40,.14);
  color: #C5A028;
  border: 1px solid rgba(197,160,40,.35);
  border-radius: 4px;
  padding: .18rem .85rem;
  font-size: .7rem; font-weight: 600;
  letter-spacing: .04em; margin-top: .45rem;
}
.af-result-model {
  font-size: .7rem; color: #4A5568; margin-top: .5rem;
}

/* ── Booking summary table ───────────────────────────────── */
.af-summary {
  background: #FFFFFF;
  border: 1px solid #D4C9B0;
  border-radius: 8px;
  padding: 1.1rem 1.3rem;
  font-size: .85rem;
}
.af-summary-row {
  display: flex; justify-content: space-between;
  align-items: baseline;
  border-bottom: 1px solid #F0EBE0;
  padding: .22rem 0;
}
.af-summary-row:last-child { border-bottom: none; }
.af-summary-key { color: #718096; font-weight: 500; }
.af-summary-val { color: #1A1A2E; font-weight: 600; text-align: right; }
.af-summary-val.restricted { color: #9B2335; }
.af-summary-val.normal     { color: #276749; }
.af-summary-val.gold       { color: #B7860B; }
.af-summary-val.muted      { color: #718096; font-weight: 400; }

/* ── Travel tip ──────────────────────────────────────────── */
.af-tip {
  background: #FBF7EC;
  border-left: 3px solid #C5A028;
  border-radius: 0 6px 6px 0;
  padding: .7rem .95rem;
  margin-top: .75rem;
  font-size: .81rem; color: #744210;
  line-height: 1.75;
}

/* ── Feature / driver cards ──────────────────────────────── */
.af-fcard {
  background: #FFFFFF;
  border: 1px solid #D4C9B0;
  border-radius: 8px;
  padding: .95rem 1.05rem;
  border-left: 3px solid #1A3A5C;
  margin-bottom: .75rem;
}
.af-fcard-title {
  font-size: .87rem; font-weight: 700;
  color: #0D1B2A; margin-bottom: .25rem;
}
.af-fcard-desc { font-size: .79rem; color: #4A5568; line-height: 1.55; }

/* ── Page header (non-hero pages) ────────────────────────── */
.af-page-hdr {
  background: #0D1B2A;
  padding: 1.5rem 2rem;
  margin: -1.4rem -1rem 1.8rem -1rem;
  border-bottom: 3px solid #C5A028;
  color: #F5F3EE;
}
.af-page-hdr h2 {
  font-family: Georgia, serif;
  font-size: 1.65rem; font-weight: 700;
  margin: 0 0 .2rem; color: #F5F3EE;
}
.af-page-hdr p { color: #8FA8C0; font-size: .86rem; margin: 0; }

/* ── Content cards ───────────────────────────────────────── */
.af-card {
  background: #FFFFFF;
  border: 1px solid #D4C9B0;
  border-radius: 8px;
  padding: 1.3rem;
  margin-bottom: 1.1rem;
}
.af-card-title {
  font-family: Georgia, serif;
  font-size: 1rem; font-weight: 700;
  color: #0D1B2A; margin-bottom: .7rem;
  padding-bottom: .45rem;
  border-bottom: 1px solid #EDE8DC;
}

/* ── Insight block ───────────────────────────────────────── */
.af-insight {
  background: #F5F3EE;
  border-left: 3px solid #1A3A5C;
  border-radius: 0 6px 6px 0;
  padding: .75rem .95rem;
  margin-top: .65rem;
  font-size: .81rem; color: #2D3748;
  line-height: 1.8;
}
.af-insight strong { color: #0D1B2A; }

/* ── Model metric strip ──────────────────────────────────── */
.af-metric-strip {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
  gap: .65rem;
  margin-bottom: 1.4rem;
}
.af-metric {
  background: #FFFFFF;
  border: 1px solid #D4C9B0;
  border-radius: 8px;
  padding: .9rem 1rem;
  text-align: center;
}
.af-metric-val {
  font-family: Georgia, serif;
  font-size: 1.55rem; font-weight: 700;
  color: #C5A028; line-height: 1;
}
.af-metric-label {
  font-size: .67rem; font-weight: 600;
  color: #718096; text-transform: uppercase;
  letter-spacing: .07em; margin-top: .25rem;
}

/* ── Tabs ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  border-bottom: 2px solid #D4C9B0;
}
.stTabs [data-baseweb="tab"] {
  font-size: .82rem; font-weight: 600;
  color: #718096;
  padding: .45rem 1rem;
  border-radius: 5px 5px 0 0;
}
.stTabs [aria-selected="true"] {
  color: #0D1B2A !important;
  border-bottom: 2px solid #C5A028 !important;
  background: rgba(197,160,40,.05) !important;
}

/* ── Inline code ─────────────────────────────────────────── */
code {
  font-family: 'Courier New', Courier, monospace;
  background: #EDE8DC;
  color: #1A1A2E;
  padding: .1rem .32rem;
  border-radius: 3px;
  font-size: .82em;
}

/* ── Dataframe ───────────────────────────────────────────── */
.stDataFrame {
  border: 1px solid #D4C9B0 !important;
  border-radius: 6px !important;
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #F5F3EE; }
::-webkit-scrollbar-thumb { background: #C5A028; border-radius: 3px; }
</style>
"""
