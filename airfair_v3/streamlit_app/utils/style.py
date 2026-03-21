"""
utils/style.py
Shared CSS injected once at app startup.
"""

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0A1931 0%, #0D2550 100%);
}
[data-testid="stSidebar"] * { color: #C8DEFF !important; }
[data-testid="stSidebar"] label {
  color: #90B8F0 !important; font-size: .76rem !important;
  text-transform: uppercase; letter-spacing: .07em;
}

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, #1565C0, #1E88E5) !important;
  color: #fff !important; border: none !important;
  border-radius: 10px !important; padding: .65rem 2rem !important;
  font-weight: 600 !important; width: 100% !important;
  box-shadow: 0 4px 14px rgba(21,101,192,.3) !important;
}

/* Hero banner */
.hero {
  background: linear-gradient(135deg, #0A1931 0%, #1565C0 55%, #0D2550 100%);
  border-radius: 18px; padding: 2.4rem 3rem; margin-bottom: 1.8rem;
  position: relative; overflow: hidden;
}
.hero::after {
  content: '✈'; position: absolute; right: 2.5rem; top: 50%;
  transform: translateY(-50%) rotate(15deg); font-size: 7rem; opacity: .07;
}
.hero h1 {
  font-family: 'Playfair Display', serif; font-size: 2.8rem;
  font-weight: 900; color: #fff; margin: 0 0 .35rem;
}
.hero p { color: #90BEFF; font-size: 1rem; margin: 0; }

/* KPI cards */
.kpi {
  background: #fff; border: 1px solid #DCE8FA; border-radius: 14px;
  padding: 1.25rem 1.4rem; text-align: center;
  box-shadow: 0 2px 14px rgba(21,101,192,.07);
}
.kpi .v {
  font-family: 'Playfair Display', serif; font-size: 2.1rem;
  font-weight: 700; color: #0A1931; line-height: 1;
}
.kpi .l { font-size: .75rem; color: #64748B; text-transform: uppercase;
           letter-spacing: .08em; margin-top: .25rem; }
.kpi .s { font-size: .68rem; color: #94A3B8; margin-top: .12rem; }

/* Panel (form background) */
.panel {
  background: #F8FBFF; border: 1px solid #DCE8FA;
  border-radius: 16px; padding: 1.7rem 2rem; margin-bottom: 1.4rem;
}

/* Prediction result */
.result {
  background: linear-gradient(135deg, #0A1931, #1565C0);
  border-radius: 16px; padding: 2rem 2.4rem; text-align: center;
  color: #fff; margin-top: 1.4rem;
  box-shadow: 0 8px 30px rgba(21,101,192,.35);
}
.result .rl { font-size: .82rem; letter-spacing: .12em;
              text-transform: uppercase; color: #90BEFF; margin-bottom: .4rem; }
.result .rv {
  font-family: 'Playfair Display', serif; font-size: 3.4rem;
  font-weight: 900; color: #FFB300; line-height: 1;
}
.result .rs { font-size: .82rem; color: #90BEFF; margin-top: .45rem; }
.badge {
  display: inline-block; background: rgba(0,176,255,.15); color: #00B0FF;
  border: 1px solid rgba(0,176,255,.3); border-radius: 20px;
  padding: .18rem .85rem; font-size: .75rem; font-weight: 600; margin-top: .45rem;
}

/* Travel tip */
.tip {
  background: #FFF8E6; border-left: 4px solid #FFB300; border-radius: 8px;
  padding: .85rem 1.1rem; margin-top: .9rem; font-size: .84rem; color: #7A5C00;
}

/* Section heading */
.sec {
  font-family: 'Playfair Display', serif; font-size: 1.45rem; font-weight: 700;
  color: #0A1931; padding-bottom: .45rem; border-bottom: 2px solid #1565C0;
  display: inline-block; margin: 0 0 1.1rem;
}

/* EDA card */
.card {
  background: #fff; border: 1px solid #DCE8FA; border-radius: 14px;
  padding: 1.4rem; box-shadow: 0 2px 12px rgba(21,101,192,.07); margin-bottom: 1.4rem;
}

/* Insight block */
.insight {
  background: #EBF5FF; border-left: 4px solid #1565C0; border-radius: 8px;
  padding: .85rem 1.1rem; margin-top: .75rem; font-size: .85rem;
  color: #1A3560; line-height: 1.75;
}

/* Feature card */
.fcard {
  background: #F0F6FF; border-radius: 12px; padding: 1.1rem 1.3rem;
  border-left: 4px solid #1565C0; margin-bottom: .9rem;
}
.fcard h4 { color: #0A1931; margin: 0 0 .35rem; font-size: .93rem; }
.fcard p  { color: #64748B; font-size: .81rem; margin: 0; line-height: 1.5; }

/* Page header */
.page-hdr {
  background: linear-gradient(135deg, #0A1931, #1565C0); border-radius: 14px;
  padding: 1.8rem 2.4rem; margin-bottom: 1.8rem; color: #fff;
}
.page-hdr h2 {
  font-family: 'Playfair Display', serif; font-size: 1.9rem;
  font-weight: 900; margin: 0 0 .3rem;
}
.page-hdr p { color: #90BEFF; font-size: .93rem; margin: 0; }
</style>
"""
