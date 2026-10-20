# =============================================================================
# UTS INTEGRASI DATA MINING & BUSINESS INTELLIGENCE
# app.py — REVISI 10 (sesuai soal UAS: Tahap 1–6 + Luaran)
# Author: MARIA BIMOLO (51230019)
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from collections import Counter
from itertools import combinations
import base64
import html as htmlmod
import math
import os
import warnings
warnings.filterwarnings("ignore")

from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

JENIS_UJIAN = "UTS"   # ganti ke "UTS" jika perlu

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title=f"BI Analytics System — {JENIS_UJIAN} | Maria Bimolo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
.stApp, [data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg, #ff4d8d, #7c3aed, #2563eb, #06b6d4,
                                #10b981, #facc15, #fb923c, #f43f5e) !important;
    background-size: 500% 500% !important;
    animation: gradientBG 18s ease infinite;
}
@keyframes gradientBG {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
[data-testid="stHeader"] { background: transparent !important; }

.block-container { font-size: 18px; }
.block-container p, .block-container li { font-size: 18px; line-height: 1.7; }
.block-container h1 { font-size: 36px; }
.block-container h2 { font-size: 32px; }
.block-container h3 { font-size: 28px; }
.block-container h4 { font-size: 23px; }
[data-testid="stWidgetLabel"] p { font-size: 17px !important; font-weight: 700 !important; color: #1e293b !important; }
.stButton button p { font-size: 17px !important; font-weight: 700 !important; }
[data-testid="stMetricValue"] { font-size: 36px !important; }
[data-testid="stMetricLabel"] p { font-size: 16px !important; }
[data-testid="stCaptionContainer"] p { font-size: 15.5px !important; }
button[data-baseweb="tab"] p { font-size: 18px !important; font-weight: 800 !important; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(79,70,229,0.97), rgba(192,38,211,0.97) 55%, rgba(244,63,94,0.97)) !important;
    backdrop-filter: blur(12px);
}
section[data-testid="stSidebar"] * { color: #f8fafc; }
section[data-testid="stSidebar"] p { font-size: 16.5px; }
section[data-testid="stSidebar"] .stRadio label p { font-size: 17.5px !important; color: #f8fafc !important; }
section[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #facc15, #fb923c) !important;
    border: none !important;
}
section[data-testid="stSidebar"] .stButton button * { color: #1e293b !important; font-weight: 800; }

[data-testid="stExpandSidebarButton"], [data-testid="collapsedControl"] { margin-left: 100px; }

.block-container {
    background: rgba(255,255,255,0.94);
    border-radius: 24px;
    margin-top: 1rem;
    padding: 2.5rem 3rem;
    box-shadow: 0 25px 70px rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.7);
}
h1, h2, h3, h4 { color: #1e293b; }

[data-testid="stPlotlyChart"] {
    border-radius: 18px; overflow: hidden;
    box-shadow: 0 6px 20px rgba(124,58,237,0.18); border: 1px solid #e9d5ff;
}

.fixed-logo {
    position: fixed; top: 12px; left: 14px; width: 82px; z-index: 999992;
    background: #ffffff; border-radius: 18px; padding: 6px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.35); border: 2px solid rgba(255,255,255,0.9);
    pointer-events: none;
}
.fixed-logo img { width: 100%; display: block; border-radius: 12px; }
.fixed-logo .txt { font-weight: 900; color: #7c3aed; font-size: 15px; text-align: center; padding: 14px 0; }
.fixed-logo.cover {
    top: 18px; left: 20px; width: 138px; padding: 10px 10px 12px;
    border-radius: 22px; text-align: center;
}
.fixed-logo.cover img { border-radius: 14px; }
.fixed-logo .uni-name {
    margin-top: 9px; font-weight: 900; font-size: 14.5px; line-height: 1.25;
    letter-spacing: 1px; text-transform: uppercase;
    background: linear-gradient(135deg, #7c3aed, #db2777);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; color: transparent;
}

.page-header {
    background: linear-gradient(135deg, #7c3aed, #db2777 50%, #f97316);
    padding: 25px 35px; border-radius: 18px; margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(124,58,237,0.4);
}
.page-header h1 { color: #fff !important; margin: 0; font-size: 34px; }
.page-header p  { color: #ffe4e6 !important; margin: 8px 0 0; font-size: 17px; }
.page-header.big { padding: 46px 54px; border-radius: 24px; margin-bottom: 32px;
    box-shadow: 0 14px 38px rgba(124,58,237,0.45); }
.page-header.big h1 { font-size: 48px !important; }
.page-header.big p  { font-size: 22px !important; margin-top: 14px; }

.kpi-card {
    background: white; padding: 22px; border-radius: 16px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    border-left: 8px solid #c33764; height: 100%; transition: transform .2s;
}
.kpi-card:hover { transform: translateY(-4px); }
.kpi-card .lbl { font-size: 14px; color: #64748b; font-weight: 700;
    text-transform: uppercase; letter-spacing: .5px; }
.kpi-card .val { font-size: 32px; font-weight: 800; color: #1e293b; margin-top: 4px; }
.kpi-card .sub { font-size: 14px; color: #94a3b8; margin-top: 4px; }
.kpi-card.k1 { border-left-color:#c33764; background: linear-gradient(135deg,#ffffff 60%,#fce7f3 100%); }
.kpi-card.k2 { border-left-color:#7597de; background: linear-gradient(135deg,#ffffff 60%,#e0e7ff 100%); }
.kpi-card.k3 { border-left-color:#10b981; background: linear-gradient(135deg,#ffffff 60%,#d1fae5 100%); }
.kpi-card.k4 { border-left-color:#ffb347; background: linear-gradient(135deg,#ffffff 60%,#fef3c7 100%); }
.kpi-card.k5 { border-left-color:#8b5cf6; background: linear-gradient(135deg,#ffffff 60%,#ede9fe 100%); }
.kpi-card.k6 { border-left-color:#ef4444; background: linear-gradient(135deg,#ffffff 60%,#fee2e2 100%); }

.alert-success { background: #d1fae5; color: #065f46; padding: 16px 18px; font-size: 17.5px;
    border-radius: 12px; border-left: 5px solid #10b981; margin-bottom: 14px; }
.alert-info { background: #dbeafe; color: #1e40af; padding: 16px 18px; font-size: 17.5px;
    border-radius: 12px; border-left: 5px solid #3b82f6; margin-bottom: 14px; }
.alert-warn { background: #fef3c7; color: #92400e; padding: 16px 18px; font-size: 17.5px;
    border-radius: 12px; border-left: 5px solid #f59e0b; margin-bottom: 14px; }
.insight-box { background: #f0fdf4; padding: 16px 18px; border-radius: 0 12px 12px 0; font-size: 17.5px;
    border-left: 5px solid #10b981; margin-bottom: 12px; }
.reco-card { background: white; padding: 24px; border-radius: 16px; margin-bottom: 18px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08); border-top: 5px solid #c33764; }
.reco-card p, .reco-card li { font-size: 17.5px; }

.q-card { background: white; padding: 18px 24px; border-radius: 16px; margin: 20px 0 10px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08); border-left: 10px solid #c33764; }
.q-card .badge { display:inline-block; color:white; font-weight:700; font-size:14px;
    padding: 4px 14px; border-radius: 20px; margin-bottom: 8px; }
.q-card .txt { font-size: 19px; font-weight: 700; color: #1e293b; }
.q1 { border-left-color:#c33764; } .q1 .badge { background:#c33764; }
.q2 { border-left-color:#7597de; } .q2 .badge { background:#7597de; }
.q3 { border-left-color:#ffb347; } .q3 .badge { background:#ffb347; }

.pretty-wrap { overflow-x: auto; border-radius: 16px; margin: 12px 0 24px;
    box-shadow: 0 8px 25px rgba(195,55,100,0.2); border: 1px solid #fce7f3; background: white; }
table.pretty { width: 100%; border-collapse: collapse; font-size: 17px; background: white; }
table.pretty th { background: linear-gradient(135deg, #7c3aed, #db2777); color: #ffffff;
    padding: 16px 18px; font-size: 16px; font-weight: 800; text-align: center;
    white-space: nowrap; letter-spacing: .3px; position: sticky; top: 0; }
table.pretty td { padding: 14px 18px; border-bottom: 1px solid #fce7f3; color: #1e293b; }
table.pretty td.num { text-align: center; font-weight: 700; font-variant-numeric: tabular-nums; }
table.pretty td.txt { text-align: left; font-weight: 600; }
table.pretty tr:nth-child(even) td { background-color: #fdf2f8; }
table.pretty tr:hover td { background-color: #fce7f3 !important; }
table.pretty tr.hl td { background-color: #fef9c3 !important; font-weight: 800; }
table.pretty tr:last-child td { border-bottom: none; }
.pill { color: #ffffff; padding: 6px 16px; border-radius: 20px; font-weight: 800;
    font-size: 15px; white-space: nowrap; display: inline-block; }

.cover-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.99), rgba(253,242,248,0.99));
    border-radius: 32px; padding: 50px 55px 40px;
    box-shadow: 0 35px 90px rgba(0,0,0,0.4); text-align: center;
    max-width: 920px; margin: 10px auto 30px auto; border-top: 12px solid #db2777;
    position: relative; overflow: hidden;
}
.cover-card::before {
    content: ""; position: absolute; top: -90px; right: -90px; width: 260px; height: 260px;
    background: radial-gradient(circle, #facc15 0%, rgba(250,204,21,0) 70%); opacity: .7;
}
.cover-card::after {
    content: ""; position: absolute; bottom: -100px; left: -100px; width: 300px; height: 300px;
    background: radial-gradient(circle, #06b6d4 0%, rgba(6,182,212,0) 70%); opacity: .55;
}
.cover-card > * { position: relative; z-index: 2; }
.cover-ribbon {
    display: inline-block; padding: 8px 26px; border-radius: 30px; margin-bottom: 22px;
    background: linear-gradient(135deg, #7c3aed, #db2777, #f97316); color: #fff;
    font-weight: 800; font-size: 16px; letter-spacing: 1.5px;
    box-shadow: 0 6px 18px rgba(124,58,237,0.45);
}
.cover-title {
    font-size: 64px; font-weight: 900; line-height: 1.1;
    background: linear-gradient(135deg, #7c3aed, #db2777, #f97316);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; color: transparent; margin: 14px 0 12px;
}
.cover-subtitle { font-size: 23px; color: #334155; margin-bottom: 18px; font-weight: 600; line-height: 1.5; }
.chip-row { display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; margin: 10px 0 8px; }
.chip { padding: 10px 22px; border-radius: 30px; font-weight: 800; font-size: 16.5px; color: #fff;
    box-shadow: 0 6px 16px rgba(0,0,0,0.2); }
.chip.c1 { background: linear-gradient(135deg,#ec4899,#f43f5e); }
.chip.c2 { background: linear-gradient(135deg,#3b82f6,#06b6d4); }
.chip.c3 { background: linear-gradient(135deg,#10b981,#84cc16); }
.photo-wrap {
    width: 190px; height: 190px; border-radius: 50%; margin: 0 auto 22px; padding: 7px;
    background: conic-gradient(#ec4899, #facc15, #10b981, #06b6d4, #7c3aed, #ec4899);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 14px 40px rgba(124,58,237,0.5);
}
.photo-inner {
    width: 176px; height: 176px; border-radius: 50%; background: #fce7f3;
    display: flex; align-items: center; justify-content: center;
    font-size: 66px; color: #c33764; font-weight: 800; overflow: hidden;
}
.photo-inner img { width: 100%; height: 100%; object-fit: cover; }
.id-card {
    background: linear-gradient(135deg, #fdf2f8 0%, #ede9fe 100%);
    border-radius: 20px; padding: 26px 34px; margin: 26px auto 10px;
    max-width: 560px; text-align: left; border-left: 8px solid #7c3aed;
    box-shadow: 0 8px 22px rgba(124,58,237,0.18);
}
.id-card .row { display:flex; margin: 10px 0; font-size: 21px; }
.id-card .row .key { width: 170px; color:#64748b; font-weight: 700; }
.id-card .row .val { color:#1e293b; font-weight: 800; }

.login-title { text-align:center; color:#ffffff; font-size: 56px; font-weight: 900; margin: 6px 0 6px;
    text-shadow: 0 4px 18px rgba(0,0,0,0.45); }
.login-sub { text-align:center; color:#ffffff; font-size: 23px; font-weight: 600; margin-bottom: 26px;
    text-shadow: 0 2px 10px rgba(0,0,0,0.4); }
.login-icon { text-align:center; font-size: 84px; filter: drop-shadow(0 6px 14px rgba(0,0,0,0.4)); }
[data-testid="stForm"] {
    background: rgba(255,255,255,0.97); border-radius: 26px; padding: 34px 36px;
    box-shadow: 0 25px 60px rgba(0,0,0,0.4); border: none;
}
[data-testid="stForm"] [data-testid="stWidgetLabel"] p { font-size: 22px !important; }
[data-testid="stForm"] input { font-size: 21px !important; padding: 14px 14px !important; }
[data-testid="stForm"] button { min-height: 3.6rem; }
[data-testid="stForm"] button p { font-size: 23px !important; }

.brand-box {
    background: linear-gradient(135deg, #facc15, #fb923c, #f43f5e);
    padding: 18px; border-radius: 14px; margin-bottom: 18px; text-align: center;
}
.brand-box h2 { color: #1e293b !important; margin: 0; font-size: 24px !important; }
.brand-box p { color: #3b0764 !important; margin: 6px 0 0; font-size: 14px !important; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


def inject_stage_css(stage):
    if stage in ("cover", "login"):
        st.markdown("""
        <style>
        .block-container {
            background: transparent !important; box-shadow: none !important;
            border: none !important;
        }
        .stButton button { min-height: 3.6rem; border-radius: 16px; }
        .stButton button p { font-size: 23px !important; }
        section[data-testid="stSidebar"] { display: none; }
        </style>
        """, unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
for key, default in {
    "stage": "cover", "logged_in": False, "user": None,
    "df_raw": None, "df_clean": None, "prep_log": [],
    "apriori_result": None, "kmeans_result": None,
    "nav_page": "📤 Upload Data", "goto_page": None, "last_upload": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# =============================================================================
# KONSTANTA
# =============================================================================
COLUMNS = ['Transaction_ID', 'Transaction_Date', 'Customer_ID', 'Region', 'Customer_Type',
           'Product_ID', 'Product_Name', 'Category', 'Quantity', 'Unit_Price',
           'Discount_Rate', 'Line_Total', 'Payment_Method']

REGION_LIST   = ['Kupang', 'Insana', 'Naibenu', 'Atambua', 'Bikomi', 'Kefamenanu', 'Miomaffo', 'Biboki']
CATEGORY_LIST = ['Sembako', 'Minuman', 'Makanan', 'Perawatan', 'Rumah Tangga', 'Bayi',
                 'Elektronik', 'Protein', 'ATK']
PAYMENT_LIST  = ['Tunai', 'Transfer', 'Debit', 'QRIS']
CTYPE_LIST    = ['Member', 'Non-Member']

IDENTITAS = {"Nama": "MARIA BIMOLO", "NPM": "51230019", "Mata Kuliah": f"{JENIS_UJIAN} Business Intelligence"}

PALETTE = ["#c33764", "#7597de", "#10b981", "#ffb347", "#8b5cf6",
           "#ef4444", "#14b8a6", "#f97316", "#3b82f6", "#ec4899"]

SEGMENT_COLORS = {
    "Loyal / High-Value": "#10b981",
    "Potensial": "#7597de",
    "Pelanggan Baru / Pasif": "#ffb347",
    "Berisiko Tidak Kembali": "#ef4444",
}

STRATEGIES = {
    "Loyal / High-Value": {
        "aktif": "Pertahankan dengan program loyalty (poin, tier, birthday reward), early access produk baru.",
        "kpi": "Retention rate ≥ 85%, AOV tetap stabil."},
    "Potensial": {
        "aktif": "Upsell produk premium, personalisasi promo via WhatsApp/email.",
        "kpi": "Frekuensi beli naik ≥ 1× dalam 60 hari."},
    "Pelanggan Baru / Pasif": {
        "aktif": "Onboarding: diskon first repeat purchase, newsletter edukasi produk.",
        "kpi": "Repeat purchase rate ≥ 20% dalam 30 hari."},
    "Berisiko Tidak Kembali": {
        "aktif": "Win-back campaign: voucher diskon besar, survei kepuasan untuk tahu penyebab.",
        "kpi": "Reaktivasi ≥ 10%, penurunan churn ≥ 15%."},
}

# =============================================================================
# UTIL
# =============================================================================
def fmt(n):
    try:
        return f"{n:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(n)

def fmt_int(n):
    try:
        return f"{int(round(n)):,}".replace(",", ".")
    except Exception:
        return str(n)

def rp(v):
    return "Rp " + fmt_int(v)

def n1(v):
    return f"{v:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")

def pct1(v):
    return f"{v:.1f}%".replace(".", ",")

def fmt_jt(v):
    return f"{v/1e6:.1f}".replace(".", ",") + " jt"

def kpi_card(cls, label, value, sub=""):
    return (f'<div class="kpi-card {cls}"><div class="lbl">{label}</div>'
            f'<div class="val">{value}</div><div class="sub">{sub}</div></div>')

def insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

def style_fig(fig, height=None):
    fig.update_layout(
        paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0.75)",
        font=dict(color="#1e293b", size=15), title_font=dict(size=21),
        legend=dict(font=dict(size=15)), margin=dict(l=30, r=30, t=60, b=40),
    )
    fig.update_xaxes(automargin=True, tickfont=dict(size=15), title_font=dict(size=16))
    fig.update_yaxes(automargin=True, tickfont=dict(size=15), title_font=dict(size=16))
    if height:
        fig.update_layout(height=height)
    return fig

def show(fig, height=None):
    st.plotly_chart(style_fig(fig, height), use_container_width=True, theme=None)

def style_fig_big(fig, height=None):
    fig.update_layout(
        paper_bgcolor="#eef2ff", plot_bgcolor="#fdf4ff",
        font=dict(color="#0f172a", size=18), legend=dict(font=dict(size=18)),
        margin=dict(l=20, r=40, t=50, b=60), hoverlabel=dict(font_size=17),
    )
    axis_style = dict(automargin=True, tickfont=dict(size=17, color="#0f172a"),
                      title_font=dict(size=19, color="#1e293b"),
                      gridcolor="#e9d5ff", linecolor="#a78bfa", showline=True)
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    if height:
        fig.update_layout(height=height)
    return fig

def show_big(fig, height=None):
    st.plotly_chart(style_fig_big(fig, height), use_container_width=True, theme=None)

def page_header(title, subtitle="", big=False):
    cls = "page-header big" if big else "page-header"
    st.markdown(f'<div class="{cls}"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def need_data():
    st.info("📭 Belum ada data. Silakan upload file Excel di menu **Upload Data**.")

def need_clean():
    st.warning("⚠️ Jalankan **Preprocessing** dulu (menu 🧹 Preprocessing).")

def df_to_csv_bytes(d):
    return d.to_csv(index=False).encode("utf-8")

def df_to_xlsx_bytes(d):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        d.to_excel(w, index=False, sheet_name="Data_Clean")
    return buf.getvalue()

# =============================================================================
# LOGO
# =============================================================================
@st.cache_data(show_spinner=False)
def get_logo_html(cover=False):
    cls = "fixed-logo cover" if cover else "fixed-logo"
    name_html = '<div class="uni-name">Universitas<br>Timor</div>' if cover else ""
    for fname in ["logo_unimor.jpeg", "logo_unimor.png", "logo_unimor.jpg"]:
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = "png" if fname.endswith("png") else "jpeg"
            return (f'<div class="{cls}"><img src="data:image/{ext};base64,{b64}" alt="Logo Unimor">'
                    f'{name_html}</div>')
    return f'<div class="{cls}"><div class="txt">UNIMOR</div>{name_html}</div>'

def render_fixed_logo(cover=False):
    st.markdown(get_logo_html(cover), unsafe_allow_html=True)

# =============================================================================
# TABEL CANTIK
# =============================================================================
def _default_fmt(v):
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, (float, np.floating)):
        return fmt_int(v) if float(v).is_integer() else fmt(v)
    if isinstance(v, (int, np.integer)):
        return fmt_int(v)
    return str(v)

def html_table(df, formats=None, bars=None, badge=None, highlight=None, max_height=None):
    formats = formats or {}
    bars = bars or {}
    cols = list(df.columns)
    numeric = {c: pd.api.types.is_numeric_dtype(df[c]) for c in cols}
    maxes = {c: pd.to_numeric(df[c], errors="coerce").max() for c in bars}

    head = "".join(f"<th>{htmlmod.escape(str(c))}</th>" for c in cols)
    rows = []
    for idx, row in df.iterrows():
        tds = []
        for c in cols:
            v = row[c]
            if not isinstance(v, (list, tuple, set)) and pd.isna(v):
                txt = "-"
            elif c in formats:
                txt = formats[c](v)
            else:
                txt = _default_fmt(v)
            txt = htmlmod.escape(str(txt))
            style = ""
            if c in bars and numeric[c] and maxes[c] and maxes[c] > 0 and not pd.isna(v):
                p = max(float(v) / float(maxes[c]) * 100, 0)
                style = f"background:linear-gradient(90deg,{bars[c]}55 {p:.0f}%,transparent {p:.0f}%);"
            if badge and c == badge:
                color = SEGMENT_COLORS.get(v, "#64748b")
                txt = f'<span class="pill" style="background:{color}">{txt}</span>'
            cls = "num" if numeric[c] else "txt"
            tds.append(f'<td class="{cls}" style="{style}">{txt}</td>')
        rcls = "hl" if (highlight is not None and idx in highlight) else ""
        rows.append(f'<tr class="{rcls}">{"".join(tds)}</tr>')
    wrap_style = f"max-height:{max_height}px; overflow-y:auto;" if max_height else ""
    return (f'<div class="pretty-wrap" style="{wrap_style}"><table class="pretty">'
            f'<thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')

def show_table(df, **kw):
    st.markdown(html_table(df, **kw), unsafe_allow_html=True)

# =============================================================================
# PEMBERSIHAN DATA (log terstruktur: tindakan + alasan + jumlah)
# =============================================================================
def canon_text(series, valid):
    mapping = {v.lower(): v for v in valid}
    def f(x):
        if pd.isna(x): return np.nan
        t = str(x).strip()
        if t == "" or t.lower() in ("nan", "none", "nat"): return np.nan
        return mapping.get(t.lower(), t.title())
    return series.apply(f)

def canon_by_freq(series):
    s = series.astype(object).where(series.notna(), np.nan)
    s = s.apply(lambda x: " ".join(str(x).split()) if pd.notna(x) else np.nan)
    low = s.str.lower()
    tmp = pd.DataFrame({"o": s, "l": low}).dropna()
    if tmp.empty: return s
    m = tmp.groupby("l")["o"].agg(lambda x: x.value_counts().index[0])
    return low.map(m).where(s.notna(), np.nan)

def impute_from_product_id(df, col):
    ref = df.dropna(subset=[col, "Product_ID"]).groupby("Product_ID")[col].agg(lambda s: s.mode().iloc[0])
    mask = df[col].isna()
    before = int(mask.sum())
    df.loc[mask, col] = df.loc[mask, "Product_ID"].map(ref)
    return before - int(df[col].isna().sum())

def iqr_stats(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return dict(q1=0, q3=0, lo=0, hi=0, n_out=0, n_ext=0, ext_lo=0, ext_hi=0)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    ext_lo, ext_hi = q1 - 3 * iqr, q3 + 3 * iqr
    return dict(q1=q1, q3=q3, lo=lo, hi=hi, ext_lo=ext_lo, ext_hi=ext_hi,
                n_out=int(((s < lo) | (s > hi)).sum()),
                n_ext=int(((s < ext_lo) | (s > ext_hi)).sum()))

def clean_pipeline(df_in, log=None):
    def L(tahap, tindakan, alasan, terdampak, dibuang, sisa):
        if log is not None:
            log.append({"Tahap": tahap, "Tindakan": tindakan, "Alasan / Justifikasi": alasan,
                        "Baris Terdampak": terdampak, "Baris Dibuang": dibuang, "Sisa Baris": sisa})

    df = df_in.copy()
    start = len(df)
    L("0", "Salin Data_Raw (data asli tidak ditimpa)", "Menjaga keaslian data & memungkinkan perbandingan sebelum–sesudah",
      0, 0, start)

    for c in ["Transaction_ID", "Customer_ID", "Product_ID"]:
        df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
        df[c] = df[c].apply(lambda x: str(x).strip() if pd.notna(x) and str(x).strip() not in ("", "nan", "None") else np.nan)
    txt_cols = ["Region", "Category", "Customer_Type", "Payment_Method", "Product_Name"]
    before_txt = df[txt_cols].astype(object).copy()
    df["Region"] = canon_text(df["Region"], REGION_LIST)
    df["Category"] = canon_text(df["Category"], CATEGORY_LIST)
    df["Customer_Type"] = canon_text(df["Customer_Type"], CTYPE_LIST)
    df["Payment_Method"] = canon_text(df["Payment_Method"], PAYMENT_LIST)
    df["Product_Name"] = canon_by_freq(df["Product_Name"])
    changed = int((before_txt.fillna("") != df[txt_cols].astype(object).fillna("")).any(axis=1).sum())
    L("1", "Normalisasi teks (huruf besar/kecil, spasi, ejaan nama produk)",
      "Teks tidak konsisten (mis. 'qris' vs 'QRIS') akan dihitung sebagai kategori berbeda dan merusak Apriori/agregasi",
      changed, 0, len(df))

    n_name = impute_from_product_id(df, "Product_Name")
    n_cat = impute_from_product_id(df, "Category")
    L("2", "Imputasi Product_Name & Category dari Product_ID (modus)",
      "Satu Product_ID selalu menunjuk satu produk, sehingga nilai kosong dapat diisi dari baris lain tanpa menebak",
      n_name + n_cat, 0, len(df))

    before = len(df)
    df = df.drop_duplicates()
    L("3", "Hapus duplikasi persis", "Baris identik 100% = pencatatan ganda; jika dibiarkan, penjualan & support Apriori menggelembung",
      before - len(df), before - len(df), len(df))

    before = len(df)
    df = df.dropna(subset=["Transaction_ID", "Customer_ID", "Product_ID", "Product_Name", "Category"])
    L("4", "Hapus baris tanpa ID transaksi / pelanggan / produk / nama produk / kategori",
      "Kolom kunci tidak bisa diimputasi dengan benar; tanpa ID, baris tidak dapat masuk ke basket maupun RFM",
      before - len(df), before - len(df), len(df))

    n_bad_date = int(pd.to_datetime(df["Transaction_Date"], errors="coerce").isna().sum())
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Transaction_Date"])
    L("5", "Transformasi Transaction_Date → datetime; hapus tanggal tidak valid",
      "Recency & tren bulanan butuh tanggal valid; tanggal tidak terbaca tidak bisa ditebak",
      n_bad_date, before - len(df), len(df))

    before = len(df)
    df = df[(df["Quantity"] > 0) & (df["Quantity"] < 999)]
    L("6", "Hapus Quantity ≤ 0, kosong, atau ≥ 999",
      "Jumlah barang negatif/nol tidak mungkin untuk penjualan; 999+ adalah kode salah input, bukan pembelian riil",
      before - len(df), before - len(df), len(df))

    before = len(df)
    df = df[(df["Unit_Price"] > 0) & (df["Unit_Price"] < 999999)]
    L("7", "Hapus Unit_Price ≤ 0, kosong, atau ≥ 999.999",
      "Harga nol/negatif tidak wajar; harga sangat besar adalah placeholder salah input",
      before - len(df), before - len(df), len(df))

    before = len(df)
    df = df[(df["Discount_Rate"] >= 0) & (df["Discount_Rate"] <= 0.5)]
    L("8", "Hapus Discount_Rate di luar [0 ; 0,5]",
      "Diskon negatif / di atas 50% tidak sesuai kebijakan toko dan dianggap salah input",
      before - len(df), before - len(df), len(df))

    wrong = int((~np.isclose(df["Line_Total"].fillna(-1),
                             df["Quantity"] * df["Unit_Price"] * (1 - df["Discount_Rate"]), atol=1)).sum())
    df["Line_Total"] = df["Quantity"] * df["Unit_Price"] * (1 - df["Discount_Rate"])
    L("9", "Hitung ulang Line_Total = Quantity × Unit_Price × (1 − Discount_Rate)",
      "Line_Total asli ada yang tidak konsisten dengan komponennya; rumus menjamin nilai uang benar untuk Monetary",
      wrong, 0, len(df))

    n_fill = int((df[["Region", "Customer_Type", "Payment_Method"]].isna()).sum().sum())
    for c in ["Region", "Customer_Type", "Payment_Method"]:
        df[c] = df[c].fillna("Tidak Diketahui")
    L("10", "Isi Region / Customer_Type / Payment_Method kosong dengan 'Tidak Diketahui'",
      "Kolom deskriptif non-kunci; baris dipertahankan agar penjualan tidak hilang dan nilai kosong tetap terlacak",
      n_fill, 0, len(df))

    ex_q = iqr_stats(df["Quantity"]); ex_l = iqr_stats(df["Line_Total"])
    df["Outlier_Flag"] = ((df["Quantity"] > ex_q["ext_hi"]) | (df["Line_Total"] > ex_l["ext_hi"]))
    L("11", "Tandai nilai ekstrem (> Q3 + 3×IQR pada Quantity / Line_Total) di kolom Outlier_Flag",
      "Pembelian grosir bisa sah secara bisnis, jadi TIDAK dibuang; hanya ditandai agar bisa dianalisis terpisah. "
      "StandardScaler pada K-Means mengurangi pengaruh skala",
      int(df["Outlier_Flag"].sum()), 0, len(df))

    bs = df.groupby("Transaction_ID")["Product_ID"].nunique().reset_index()
    bs.columns = ["Transaction_ID", "Basket_Size"]
    df = df.merge(bs, on="Transaction_ID", how="left")
    L("12", "Tambah fitur Basket_Size (jumlah produk unik per transaksi)",
      "Dibutuhkan untuk KPI rata-rata basket size dan menilai peluang cross-selling",
      len(df), 0, len(df))
    return df.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def build_eda_data(df):
    return clean_pipeline(df, log=None)

def quality_stats(df):
    q_bad = int(((df["Quantity"] <= 0) | (df["Quantity"] >= 999)).sum())
    return {
        "Jumlah baris": len(df), "Jumlah kolom": df.shape[1],
        "Baris duplikat": int(df.duplicated().sum()),
        "Sel kosong (missing)": int(df.isna().sum().sum()),
        "Baris yang memuat missing": int(df.isna().any(axis=1).sum()),
        "Quantity ≤ 0 atau ≥ 999": q_bad,
        "Unit_Price ≤ 0": int((df["Unit_Price"] <= 0).sum()),
        "Discount_Rate di luar [0 ; 0,5]": int(((df["Discount_Rate"] < 0) | (df["Discount_Rate"] > 0.5)).sum()),
        "Transaksi unik": int(df["Transaction_ID"].nunique()),
        "Pelanggan unik": int(df["Customer_ID"].nunique()),
        "Produk unik": int(df["Product_ID"].nunique()),
    }

# =============================================================================
# HELPER ANALISIS (di-cache agar cepat)
# =============================================================================
@st.cache_data(show_spinner=False)
def top_pairs_cached(df, n=10):
    pair_counter = Counter()
    for _, grp in df.groupby("Transaction_ID")["Product_Name"]:
        items = sorted(set(grp.dropna().unique()))
        if len(items) >= 2:
            for a, b in combinations(items, 2):
                pair_counter[(a, b)] += 1
    return pair_counter.most_common(n)

@st.cache_data(show_spinner=False)
def basket_matrix(df):
    basket = df.pivot_table(index="Transaction_ID", columns="Product_Name",
                            values="Quantity", aggfunc="sum", fill_value=0)
    return basket > 0

@st.cache_data(show_spinner=False)
def run_apriori_cached(df, min_support, min_conf):
    bb = basket_matrix(df)
    n_trans = len(bb)
    if n_trans == 0:
        return None, None, 0
    freq = apriori(bb, min_support=min_support, use_colnames=True)
    if len(freq) == 0:
        return None, None, n_trans
    freq["length"] = freq["itemsets"].apply(len)
    try:
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    except TypeError:
        rules = association_rules(freq, num_itemsets=n_trans, metric="confidence", min_threshold=min_conf)
    except ValueError:
        rules = pd.DataFrame()
    if len(rules) == 0:
        return freq, rules, n_trans
    rules = rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    return freq, rules, n_trans

def default_apriori(df):
    """Apriori otomatis (dipakai Dashboard/Rekomendasi bila menu Apriori belum dijalankan)."""
    for ms in (0.01, 0.005, 0.002):
        freq, rules, n = run_apriori_cached(df, ms, 0.30)
        if rules is not None and len(rules) > 0:
            return freq, rules, n, ms, 0.30
    return None, None, 0, 0.002, 0.30

def build_rfm(df):
    snapshot = df["Transaction_Date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("Customer_ID").agg(
        Recency=("Transaction_Date", lambda x: (snapshot - x.max()).days),
        Frequency=("Transaction_ID", "nunique"),
        Monetary=("Line_Total", "sum"),
    ).reset_index()
    rfm = rfm[rfm["Monetary"] > 0].reset_index(drop=True)
    return rfm, snapshot

def _sil(X, labels):
    if len(set(labels)) < 2:
        return 0.0
    ss = 5000 if len(X) > 5000 else None
    return float(silhouette_score(X, labels, sample_size=ss, random_state=42))

def run_kmeans_core(rfm, k):
    X = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    out = rfm.copy()
    out["Cluster"] = labels
    return out, _sil(X, labels)

def label_clusters(profile, rfm):
    """Label berdasarkan bukti: bandingkan rata-rata klaster dengan rata-rata seluruh pelanggan."""
    r_avg, f_avg, m_avg = rfm["Recency"].mean(), rfm["Frequency"].mean(), rfm["Monetary"].mean()
    labels, basis = {}, {}
    for _, row in profile.iterrows():
        recent = row["Recency_Mean"] <= r_avg
        f_hi = row["Frequency_Mean"] > f_avg
        m_hi = row["Monetary_Mean"] > m_avg
        if recent and f_hi and m_hi: lab = "Loyal / High-Value"
        elif not recent: lab = "Berisiko Tidak Kembali"
        elif f_hi or m_hi: lab = "Potensial"
        else: lab = "Pelanggan Baru / Pasif"
        c = int(row["Cluster"])
        labels[c] = lab
        basis[c] = (f"Recency {'lebih rendah' if recent else 'lebih tinggi'} dari rata-rata ({n1(r_avg)} hari), "
                    f"frekuensi {'di atas' if f_hi else 'di bawah'} rata-rata ({n1(f_avg)}×), "
                    f"monetary {'di atas' if m_hi else 'di bawah'} rata-rata ({rp(m_avg)})")
    return labels, basis

@st.cache_data(show_spinner=False)
def elbow_cached(rfm, max_k):
    X = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    ks, inertias, sils = list(range(2, max_k + 1)), [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lab = km.fit_predict(X)
        inertias.append(float(km.inertia_))
        sils.append(_sil(X, lab))
    return {"ks": ks, "inertias": inertias, "sils": sils}

@st.cache_data(show_spinner=False)
def segment_customers(df, k):
    rfm, snapshot = build_rfm(df)
    if len(rfm) < 4:
        return None
    k = int(min(max(k, 2), len(rfm) - 1))
    out, sil = run_kmeans_core(rfm, k)
    profile = out.groupby("Cluster").agg(
        Jumlah_Pelanggan=("Customer_ID", "count"), Recency_Mean=("Recency", "mean"),
        Frequency_Mean=("Frequency", "mean"), Monetary_Mean=("Monetary", "mean"),
        Monetary_Median=("Monetary", "median"), Monetary_Total=("Monetary", "sum"),
    ).reset_index()
    labels, basis = label_clusters(profile, rfm)
    profile["Label"] = profile["Cluster"].map(labels)
    profile["Dasar"] = profile["Cluster"].map(basis)
    out["Label"] = out["Cluster"].map(labels)
    return out, profile, float(sil), snapshot, k

def current_k():
    r = st.session_state.kmeans_result
    return r["k"] if r else 4

def priority_segments(profile):
    """Segmen prioritas + alasan berbasis angka."""
    tot_m = profile["Monetary_Total"].sum()
    tot_c = profile["Jumlah_Pelanggan"].sum()
    items = []
    loyal = profile[profile["Label"] == "Loyal / High-Value"].sort_values("Monetary_Total", ascending=False)
    risk = profile[profile["Label"] == "Berisiko Tidak Kembali"].sort_values("Monetary_Total", ascending=False)
    if len(loyal):
        r = loyal.iloc[0]
        items.append(("Prioritas 1", r["Label"], int(r["Cluster"]),
                      f"hanya {pct1(r['Jumlah_Pelanggan']/tot_c*100)} pelanggan tetapi menyumbang "
                      f"{pct1(r['Monetary_Total']/tot_m*100)} total belanja → kehilangan segmen ini paling mahal, jadi wajib dipertahankan."))
    if len(risk):
        r = risk.iloc[0]
        items.append(("Prioritas 2", r["Label"], int(r["Cluster"]),
                      f"{fmt_int(r['Jumlah_Pelanggan'])} pelanggan lama tidak belanja (recency {n1(r['Recency_Mean'])} hari) "
                      f"dengan nilai historis {rp(r['Monetary_Total'])} → potensi pendapatan yang bisa diselamatkan lewat win-back."))
    pot = profile[profile["Label"] == "Potensial"].sort_values("Monetary_Total", ascending=False)
    if len(pot):
        r = pot.iloc[0]
        items.append((f"Prioritas {len(items)+1}", r["Label"], int(r["Cluster"]),
                      f"masih aktif (recency {n1(r['Recency_Mean'])} hari) dan berpeluang naik kelas ke Loyal lewat upselling."))
    if not items:
        r = profile.sort_values("Monetary_Total", ascending=False).iloc[0]
        items.append(("Prioritas 1", r["Label"], int(r["Cluster"]),
                      f"kontributor belanja terbesar ({pct1(r['Monetary_Total']/tot_m*100)})."))
    return items

def explain_rule(row, n_trans):
    a, c = row["antecedents"], row["consequents"]
    sup, conf, lift = row["support"], row["confidence"], row["lift"]
    base = sup / conf if conf > 0 else 0
    n_both = int(round(sup * n_trans))
    arah = "saling melengkapi (asosiasi positif)" if lift > 1 else ("independen" if abs(lift - 1) < 1e-9 else "cenderung saling menggantikan")
    return (f"<b>Support {sup*100:.2f}%</b> → {fmt_int(n_both)} dari {fmt_int(n_trans)} transaksi memuat <i>{a}</i> dan <i>{c}</i> sekaligus.<br>"
            f"<b>Confidence {conf*100:.1f}%</b> → dari semua transaksi yang membeli <i>{a}</i>, {conf*100:.1f}% juga membeli <i>{c}</i>.<br>"
            f"<b>Lift {lift:.2f}</b> → pembelian <i>{c}</i> {lift:.2f}× lebih mungkin terjadi bila <i>{a}</i> sudah dibeli "
            f"dibanding pembelian acak (peluang dasar {base*100:.1f}%). Hubungan: {arah}.")

def lift_heatmap(rules, max_items=12):
    r = rules[~rules["antecedents"].str.contains(", ") & ~rules["consequents"].str.contains(", ")]
    if r.empty: return None
    items = pd.concat([r["antecedents"], r["consequents"]]).value_counts().head(max_items).index
    r = r[r["antecedents"].isin(items) & r["consequents"].isin(items)]
    if r.empty: return None
    piv = r.pivot_table(index="antecedents", columns="consequents", values="lift", aggfunc="max")
    fig = px.imshow(piv, color_continuous_scale="Plasma", aspect="auto", text_auto=".2f",
                    labels=dict(x="Maka membeli", y="Jika membeli", color="Lift"),
                    title="Heatmap Lift antar Produk")
    return fig

def plot_rules_network(rules, top_n=20):
    if rules is None or len(rules) == 0:
        return None
    top_rules = rules.head(top_n).copy()
    nodes = set()
    for _, row in top_rules.iterrows():
        nodes.add(row["antecedents"]); nodes.add(row["consequents"])
    nodes = list(nodes)
    n = len(nodes)
    angles = [2 * math.pi * i / max(n, 1) for i in range(n)]
    pos = {node: (math.cos(a), math.sin(a)) for node, a in zip(nodes, angles)}
    edge_x, edge_y, annotations = [], [], []
    for _, row in top_rules.iterrows():
        x0, y0 = pos[row["antecedents"]]; x1, y1 = pos[row["consequents"]]
        width = 1 + (row["lift"] / top_rules["lift"].max()) * 4
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
        annotations.append(dict(x=(x0 + x1) / 2, y=(y0 + y1) / 2, ax=x0, ay=y0, axref='x', ayref='y',
                                xref='x', yref='y', showarrow=True, arrowhead=2, arrowsize=1.5,
                                arrowwidth=width, arrowcolor='#7c3aed', opacity=0.6))
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#cbd5e1"), hoverinfo="none", mode="lines")
    node_trace = go.Scatter(
        x=[pos[nm][0] for nm in nodes], y=[pos[nm][1] for nm in nodes], mode="markers+text",
        text=nodes, textposition="bottom center", textfont=dict(size=14, color="#1e293b"),
        marker=dict(size=30, color=[PALETTE[i % len(PALETTE)] for i in range(n)], line=dict(width=3, color="white")),
        hoverinfo="text", hovertext=[f"<b>{nm}</b>" for nm in nodes])
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=dict(text=f"Network Graph — Top {min(top_n, len(rules))} Rules (berdasarkan Lift)", font=dict(size=20)),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
        annotations=annotations, margin=dict(l=20, r=20, t=60, b=20), height=600)
    return fig

def render_elbow(res):
    ks, inertias, sils = res["ks"], res["inertias"], res["sils"]
    best_k = ks[int(np.argmax(sils))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers", name="Inertia (Elbow)",
                             line=dict(color="#c33764", width=4), marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=ks, y=sils, mode="lines+markers", name="Silhouette Score", yaxis="y2",
                             line=dict(color="#10b981", width=4), marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=[best_k], y=[max(sils)], mode="markers", name="k terbaik (silhouette)", yaxis="y2",
                             marker=dict(size=24, color="#ffb347", symbol="star", line=dict(width=2, color="white"))))
    fig.update_layout(title="Elbow Method & Silhouette Score", xaxis=dict(title="Jumlah K", dtick=1),
                      yaxis=dict(title="Inertia"), yaxis2=dict(title="Silhouette", overlaying="y", side="right"),
                      legend=dict(orientation="h", y=-0.2))
    show(fig, 440)
    tbl = pd.DataFrame({"K": ks, "Inertia": inertias, "Silhouette": sils})
    tbl["Catatan"] = ["⭐ Silhouette tertinggi" if k == best_k else "" for k in ks]
    show_table(tbl, formats={"K": lambda v: f"k = {int(v)}", "Inertia": lambda v: f"{v:.2f}",
                             "Silhouette": lambda v: f"{v:.4f}"},
               bars={"Silhouette": "#10b981", "Inertia": "#c33764"},
               highlight=set(tbl.index[tbl["K"] == best_k]))
    return best_k

# =============================================================================
# COVER & LOGIN
# =============================================================================
def render_cover():
    photo_html = ""
    for fname in ["mery.jpeg", "mery.jpg", "mery.png", "foto.jpg", "foto.png"]:
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = "png" if fname.endswith("png") else "jpeg"
            photo_html = f'<img src="data:image/{ext};base64,{b64}" alt="Foto">'
            break
    if not photo_html:
        photo_html = "MB"
    ribbon = "UJIAN AKHIR SEMESTER" if JENIS_UJIAN == "UAS" else "UJIAN TENGAH SEMESTER"
    st.markdown(f"""
    <div style="max-width: 940px; margin: 0 auto;">
        <div class="cover-card">
            <div class="cover-ribbon">🎓 {ribbon} • UNIMOR</div>
            <div class="photo-wrap"><div class="photo-inner">{photo_html}</div></div>
            <div class="cover-title">BI Analytics System</div>
            <div class="cover-subtitle">
                {JENIS_UJIAN} Integrasi Data Mining &amp; Business Intelligence
            </div>
            <div class="chip-row">
                <span class="chip c1">🛒 Market Basket Analysis</span>
                <span class="chip c2">🎯 Customer Segmentation</span>
                <span class="chip c3">📊 Dashboard BI</span>
            </div>
            <div class="id-card">
                <div class="row"><div class="key">Nama</div><div class="val">: {IDENTITAS['Nama']}</div></div>
                <div class="row"><div class="key">NPM</div><div class="val">: {IDENTITAS['NPM']}</div></div>
                <div class="row"><div class="key">Mata Kuliah</div><div class="val">: {IDENTITAS['Mata Kuliah']}</div></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.4, 1])
    with c2:
        if st.button("Lanjut ke Login →", use_container_width=True, type="primary"):
            st.session_state.stage = "login"
            st.rerun()

def render_login():
    st.markdown(f"""
    <div style="margin-top: 20px;">
        <div class="login-icon">🔐</div>
        <div class="login-title">Login ke Sistem</div>
        <div class="login-sub">BI Analytics System — {JENIS_UJIAN} Business Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Masukkan username")
            password = st.text_input("Password", type="password", placeholder="Masukkan password")
            submit = st.form_submit_button("Masuk →", use_container_width=True, type="primary")
            if submit:
                if username == "admin" and password == "admin123":
                    st.session_state.logged_in = True
                    st.session_state.stage = "app"
                    st.session_state.user = {"username": username, "role": "Administrator"}
                    st.rerun()
                else:
                    st.error("❌ Username atau password salah!")
        if st.button("← Kembali ke Cover", use_container_width=True):
            st.session_state.stage = "cover"
            st.rerun()

# =============================================================================
# SIDEBAR  (urutan menu mengikuti Tahap 1–6 pada soal)
# =============================================================================
P_UPLOAD = "📤 Upload Data"
P_BU = "🎯 Business Understanding"
P_EDA = "🔍 EDA & Insight"
P_PREP = "🧹 Preprocessing"
P_APR = "🔗 Apriori (MBA)"
P_KM = "🎯 K-Means (Segmentasi)"
P_DASH = "📊 Dashboard BI"
P_REC = "💡 Rekomendasi"
P_LAP = "📑 Laporan & Luaran"
NAV_PAGES = [P_UPLOAD, P_BU, P_EDA, P_PREP, P_APR, P_KM, P_DASH, P_REC, P_LAP]

def render_sidebar():
    if st.session_state.goto_page:
        st.session_state.nav_page = st.session_state.goto_page
        st.session_state.goto_page = None
    with st.sidebar:
        st.markdown('<div style="height:100px;"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="brand-box"><h2>📊 BI Analytics</h2><p>{JENIS_UJIAN} Business Intelligence</p></div>',
                    unsafe_allow_html=True)
        st.markdown(f"👤 **{st.session_state.user['username']}**  \n_{st.session_state.user['role']}_")
        st.markdown(f"🎓 **{IDENTITAS['Nama']}** · {IDENTITAS['NPM']}")
        st.markdown("---")
        page = st.radio("📌 Menu Navigasi", NAV_PAGES, key="nav_page", label_visibility="collapsed")
        st.markdown("---")
        if st.session_state.df_raw is not None:
            st.success(f"✅ Data_Raw: **{len(st.session_state.df_raw):,}** baris")
        else:
            st.info("⚪ Belum ada data")
        if st.session_state.df_clean is not None:
            st.success(f"✅ Data_Clean: **{len(st.session_state.df_clean):,}** baris")
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.stage = "login"
            st.session_state.user = None
            st.rerun()
        return page

# =============================================================================
# PAGE: UPLOAD
# =============================================================================
def render_upload():
    page_header("📤 Upload Data Excel",
                "Unggah file dataset transaksi toko (sheet Data_Raw) untuk memulai analisis", big=True)
    uploaded = st.file_uploader("Pilih file Excel", type=["xlsx", "xls"], help="File harus memiliki sheet 'Data_Raw'")
    go_next = False
    if uploaded is not None:
        sig = (uploaded.name, uploaded.size)
        if st.session_state.last_upload != sig:
            with st.spinner("Membaca file Excel..."):
                try:
                    file_bytes = uploaded.read()
                    xls = pd.ExcelFile(BytesIO(file_bytes))
                    sheet_names = xls.sheet_names
                    sheet = next((s for s in sheet_names if "data_raw" in s.lower()), None)
                    if sheet is None:
                        sheet = next((s for s in sheet_names if "raw" in s.lower()), sheet_names[-1])
                    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet)
                    col_map = {}
                    for c in df.columns:
                        for target in COLUMNS:
                            if str(c).strip().lower() == target.lower():
                                col_map[c] = target
                                break
                    df = df.rename(columns=col_map)
                    for c in COLUMNS:
                        if c not in df.columns:
                            df[c] = None
                    for numcol in ["Quantity", "Unit_Price", "Discount_Rate", "Line_Total"]:
                        df[numcol] = pd.to_numeric(df[numcol], errors="coerce")
                    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
                    for tcol in ["Transaction_ID", "Customer_ID", "Region", "Customer_Type",
                                 "Product_ID", "Product_Name", "Category", "Payment_Method"]:
                        df[tcol] = (df[tcol].astype(str).str.strip()
                                    .replace({"nan": np.nan, "None": np.nan, "NaT": np.nan, "": np.nan}))
                    st.session_state.df_raw = df
                    st.session_state.df_clean = None
                    st.session_state.prep_log = []
                    st.session_state.apriori_result = None
                    st.session_state.kmeans_result = None
                    st.session_state.last_upload = sig
                    go_next = True
                except Exception as e:
                    st.error(f"❌ Gagal membaca file: {e}")
        else:
            st.markdown(f'<div class="alert-success">✅ File <b>{uploaded.name}</b> sudah dimuat '
                        f'({len(st.session_state.df_raw):,} baris).</div>', unsafe_allow_html=True)
    if go_next:
        st.session_state.goto_page = P_BU
        st.rerun()

    if st.session_state.df_raw is not None:
        st.markdown("---")
        st.subheader(f"📋 Seluruh Data Raw (Total {len(st.session_state.df_raw):,} baris)")
        st.dataframe(st.session_state.df_raw, use_container_width=True, height=600)
        st.download_button("⬇ Unduh Data Raw (CSV)", data=df_to_csv_bytes(st.session_state.df_raw),
                           file_name="data_raw.csv", mime="text/csv")

# =============================================================================
# PAGE: BUSINESS UNDERSTANDING  (Tahap 1 — Soal 1)
# =============================================================================
def render_business():
    page_header("🎯 Business Understanding", "Tahap 1 · Soal 1 — Memahami permasalahan bisnis dan merumuskan pertanyaan analitis")

    st.subheader("a. Permasalahan Utama Perusahaan")
    problems = pd.DataFrame({
        "No": [1, 2, 3, 4, 5],
        "Permasalahan": [
            "Data hanya dipakai untuk laporan penjualan rutin",
            "Pola pembelian pelanggan belum dipahami",
            "Kombinasi produk yang sering dibeli bersama belum diketahui",
            "Pelanggan belum dikelompokkan berdasarkan perilaku belanja",
            "Kualitas data transaksi belum terjamin",
        ],
        "Bukti pada narasi studi kasus": [
            "“manajemen hanya menggunakan data untuk laporan penjualan rutin”",
            "“belum memanfaatkan data secara optimal untuk memahami pola pembelian”",
            "“mengidentifikasi kombinasi produk yang sering dibeli bersama”",
            "“mengelompokkan pelanggan berdasarkan perilaku belanja, menyusun strategi pemasaran yang tepat”",
            "missing values, data duplikat, nilai tidak wajar, format tidak konsisten",
        ],
        "Dampak bagi bisnis": [
            "Keputusan bersifat reaktif, bukan berbasis pola",
            "Promosi tidak tepat waktu & tidak tepat sasaran",
            "Peluang cross-selling / bundling & penataan rak terlewat",
            "Promosi massal yang boros; pelanggan bernilai tinggi tidak diistimewakan",
            "Angka penjualan & hasil model bisa menyesatkan",
        ],
    })
    show_table(problems)

    st.subheader("b. Pertanyaan Bisnis yang Dijawab Lewat Analisis Data Transaksi")
    qs = pd.DataFrame({
        "Kode": ["Q1", "Q2", "Q3", "Q4", "Q5"],
        "Pertanyaan Bisnis": [
            "Produk apa saja yang sering dibeli bersama dalam satu transaksi?",
            "Segmen pelanggan mana yang paling bernilai dan strategi pemasaran apa yang tepat untuk tiap segmen?",
            "Kapan penjualan meningkat dan kategori/produk apa yang paling berkontribusi?",
            "Pelanggan mana yang berisiko tidak kembali dan bagaimana cara mempertahankannya?",
            "Apakah kualitas data cukup baik untuk dipakai mengambil keputusan?",
        ],
        "Metode": ["Apriori (Market Basket)", "RFM + K-Means", "EDA & tren bulanan", "RFM + K-Means (Recency tinggi)", "EDA & Preprocessing"],
        "Keputusan yang didukung": [
            "Bundling, cross-selling, penempatan rak",
            "Program loyalty, upselling, personalisasi promo",
            "Waktu promosi & perencanaan stok",
            "Kampanye win-back",
            "Perbaikan proses pencatatan data",
        ],
    })
    show_table(qs)

    st.subheader("c. Bagaimana Hasil Analisis Membantu Manajemen")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="reco-card" style="border-top-color:#10b981;">
        <h4>📈 Meningkatkan Penjualan</h4>
        <ul>
          <li><b>Apriori</b> → bundling & cross-selling menaikkan nilai keranjang (basket size / AOV).</li>
          <li><b>Tren bulanan</b> → promosi & stok difokuskan pada bulan puncak.</li>
          <li><b>Kategori & produk terlaris</b> → prioritas stok dan penempatan produk.</li>
          <li><b>Dashboard</b> → manajemen memantau KPI tanpa menunggu laporan manual.</li>
        </ul></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="reco-card" style="border-top-color:#7597de;">
        <h4>🤝 Mempertahankan Pelanggan</h4>
        <ul>
          <li><b>RFM + K-Means</b> → pelanggan dibagi menjadi segmen berdasarkan perilaku nyata.</li>
          <li><b>Loyal / High-Value</b> → diberi loyalty program agar tidak pindah toko.</li>
          <li><b>Berisiko tidak kembali</b> → dihubungi lewat win-back sebelum hilang total.</li>
          <li><b>Potensial & Baru</b> → didorong naik kelas lewat upselling dan promo repeat purchase.</li>
        </ul></div>""", unsafe_allow_html=True)

    st.markdown('<div class="alert-info">➡️ Lanjutkan ke menu <b>EDA & Insight</b> (Soal 2), lalu <b>Preprocessing</b> (Soal 3–4).</div>',
                unsafe_allow_html=True)

# =============================================================================
# PAGE: EDA  (Tahap 1 — Soal 2)
# =============================================================================
def render_eda():
    page_header("🔍 EDA & Insight", "Tahap 1 · Soal 2 — Eksplorasi data: struktur, statistik, kualitas, visualisasi, temuan")
    if st.session_state.df_raw is None:
        need_data(); return
    df = st.session_state.df_raw
    dv = build_eda_data(df)

    # ---------- 2a ----------
    st.subheader("1️⃣ Struktur Dataset (Soal 2a)")
    dmin, dmax = df["Transaction_Date"].min(), df["Transaction_Date"].max()
    span = (dmax - dmin).days if pd.notna(dmin) and pd.notna(dmax) else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("k1", "Jumlah Baris", fmt_int(len(df)), "data raw"), unsafe_allow_html=True)
    c2.markdown(kpi_card("k2", "Jumlah Kolom", str(df.shape[1]), "atribut"), unsafe_allow_html=True)
    c3.markdown(kpi_card("k3", "Tanggal Awal", dmin.strftime("%d-%m-%Y") if pd.notna(dmin) else "-", "transaksi pertama"), unsafe_allow_html=True)
    c4.markdown(kpi_card("k4", "Tanggal Akhir", dmax.strftime("%d-%m-%Y") if pd.notna(dmax) else "-", f"rentang {fmt_int(span)} hari"), unsafe_allow_html=True)
    st.markdown("#### Tipe data tiap kolom")
    dt = pd.DataFrame({
        "Kolom": df.columns, "Tipe Data": [str(t) for t in df.dtypes],
        "Terisi": df.notna().sum().values, "Missing": df.isna().sum().values,
        "Nilai Unik": df.nunique().values,
        "Contoh Nilai": [str(df[c].dropna().iloc[0])[:30] if df[c].notna().any() else "-" for c in df.columns],
    })
    show_table(dt, bars={"Missing": "#ef4444"})

    # ---------- 2b ----------
    st.subheader("2️⃣ Statistik Deskriptif (Soal 2b)")
    valid_q = df.loc[df["Quantity"].between(1, 998), "Quantity"]
    n_tx = df["Transaction_ID"].nunique()
    s1, s2, s3 = st.columns(3)
    s1.markdown(kpi_card("k1", "Jumlah Transaksi", fmt_int(n_tx), "Transaction_ID unik"), unsafe_allow_html=True)
    s2.markdown(kpi_card("k2", "Pelanggan Unik", fmt_int(df["Customer_ID"].nunique()), "Customer_ID unik"), unsafe_allow_html=True)
    s3.markdown(kpi_card("k3", "Produk Unik", fmt_int(df["Product_ID"].nunique()), "Product_ID unik"), unsafe_allow_html=True)
    s4, s5, s6 = st.columns(3)
    s4.markdown(kpi_card("k4", "Rata-rata Jumlah Pembelian", n1(valid_q.mean()), "unit per baris (Quantity valid)"), unsafe_allow_html=True)
    s5.markdown(kpi_card("k5", "Rata-rata Item / Transaksi", n1(len(df) / max(n_tx, 1)), "baris per transaksi"), unsafe_allow_html=True)
    s6.markdown(kpi_card("k6", "Total Penjualan (raw)", rp(df["Line_Total"].sum()), "sebelum pembersihan"), unsafe_allow_html=True)
    desc = df[["Quantity", "Unit_Price", "Discount_Rate", "Line_Total"]].describe().T.reset_index()
    desc.columns = ["Variabel", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    st.caption("Ringkasan statistik variabel numerik (data raw, apa adanya).")
    show_table(desc)

    # ---------- 2c ----------
    st.subheader("3️⃣ Kualitas Data: Missing, Duplikat, Outlier (Soal 2c)")
    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi_card("k6", "Baris Duplikat", fmt_int(df.duplicated().sum()), "identik di semua kolom"), unsafe_allow_html=True)
    k2.markdown(kpi_card("k4", "Sel Missing", fmt_int(df.isna().sum().sum()), "pada seluruh kolom"), unsafe_allow_html=True)
    k3.markdown(kpi_card("k5", "Baris Memuat Missing", fmt_int(df.isna().any(axis=1).sum()), "≥ 1 sel kosong"), unsafe_allow_html=True)
    st.markdown("#### Missing values per kolom")
    miss = df.isna().sum().reset_index()
    miss.columns = ["Kolom", "Jumlah Missing"]
    miss["Persen"] = miss["Jumlah Missing"] / max(len(df), 1) * 100
    miss = miss[miss["Jumlah Missing"] > 0].sort_values("Jumlah Missing", ascending=False)
    if len(miss):
        show_table(miss, formats={"Persen": lambda v: f"{v:.2f}%".replace(".", ",")},
                   bars={"Jumlah Missing": "#ef4444", "Persen": "#ffb347"})
    else:
        st.success("✅ Tidak ada missing values.")

    st.markdown("#### Nilai tidak wajar (aturan bisnis)")
    out_df = pd.DataFrame({
        "Pemeriksaan": ["Quantity ≤ 0", "Quantity ≥ 999", "Unit_Price ≤ 0", "Discount_Rate di luar [0 ; 0,5]"],
        "Jumlah Baris": [int((df["Quantity"] <= 0).sum()), int((df["Quantity"] >= 999).sum()),
                         int((df["Unit_Price"] <= 0).sum()),
                         int(((df["Discount_Rate"] < 0) | (df["Discount_Rate"] > 0.5)).sum())],
    })
    show_table(out_df, bars={"Jumlah Baris": "#ef4444"})

    st.markdown("#### Outlier statistik (metode IQR)")
    rows = []
    for col in ["Quantity", "Unit_Price", "Line_Total"]:
        s = iqr_stats(df[col])
        rows.append({"Variabel": col, "Q1": s["q1"], "Q3": s["q3"], "Batas Bawah (1,5×IQR)": s["lo"],
                     "Batas Atas (1,5×IQR)": s["hi"], "Jumlah Outlier": s["n_out"], "Ekstrem (3×IQR)": s["n_ext"]})
    show_table(pd.DataFrame(rows), bars={"Jumlah Outlier": "#ef4444", "Ekstrem (3×IQR)": "#f97316"})
    b1, b2 = st.columns(2)
    with b1:
        fig = px.box(df[df["Quantity"] > 0], y="Quantity", points="outliers", color_discrete_sequence=["#c33764"],
                     title="Boxplot Quantity")
        show_big(fig, 420)
    with b2:
        fig = px.box(df[df["Unit_Price"] > 0], y="Unit_Price", points="outliers", color_discrete_sequence=["#7597de"],
                     title="Boxplot Unit_Price")
        show_big(fig, 420)
    insight("<b>Kesimpulan kualitas data:</b> dataset memuat missing value, duplikat, nilai tidak wajar dan outlier, "
            "sehingga <b>wajib dibersihkan</b> di menu Preprocessing sebelum Apriori dan K-Means.")

    # ---------- 2d : pertanyaan bisnis berbasis grafik ----------
    st.subheader("4️⃣ Visualisasi Awal & Jawaban Pertanyaan Bisnis (Soal 2d)")
    st.caption("Visualisasi memakai data yang dibersihkan otomatis (duplikat & nilai tidak wajar dibuang).")

    st.markdown('<div class="q-card q1"><span class="badge">Q1</span><div class="txt">Produk apa saja yang sering dibeli bersama dalam satu transaksi?</div></div>', unsafe_allow_html=True)
    top_pairs = top_pairs_cached(dv, 10)
    if top_pairs:
        pair_df = pd.DataFrame([{"Pasangan Produk": f"{a}  +  {b}", "Frekuensi": c} for (a, b), c in top_pairs])
        fig = px.bar(pair_df, x="Frekuensi", y="Pasangan Produk", orientation="h", color="Frekuensi",
                     color_continuous_scale="Magma", text="Frekuensi", title="Top 10 Pasangan Produk yang Sering Dibeli Bersama")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        show_big(fig, 520)
        t1 = top_pairs[0]
        insight(f"<b>Jawaban Q1:</b> pasangan terbanyak adalah <b>{t1[0][0]}</b> + <b>{t1[0][1]}</b> ({fmt_int(t1[1])} transaksi). "
                "Dasar untuk bundling dan penempatan rak berdekatan (dibuktikan lebih lanjut dengan Apriori).")
    else:
        st.info("Tidak cukup transaksi multi-produk.")

    st.markdown('<div class="q-card q2"><span class="badge">Q2</span><div class="txt">Segmen pelanggan mana yang paling bernilai (RFM)?</div></div>', unsafe_allow_html=True)
    rfm, _ = build_rfm(dv)
    if len(rfm) > 0:
        q1_, q2_ = rfm["Monetary"].quantile([0.33, 0.66])
        rfm["Segmen"] = rfm["Monetary"].apply(lambda m: "High-Value" if m >= q2_ else ("Potensial" if m >= q1_ else "Low-Value"))
        cmap = {"High-Value": "#10b981", "Potensial": "#7597de", "Low-Value": "#ef4444"}
        fig = px.scatter(rfm, x="Recency", y="Monetary", color="Segmen", size="Frequency", color_discrete_map=cmap,
                         hover_data=["Customer_ID", "Frequency"], title="Gambaran Awal RFM: Recency vs Monetary")
        show_big(fig, 500)
        high = rfm[rfm["Segmen"] == "High-Value"]
        insight(f"<b>Jawaban Q2 (awal):</b> {fmt_int(len(high))} pelanggan High-Value (sepertiga belanja teratas) "
                f"dengan rata-rata belanja <b>{rp(high['Monetary'].mean())}</b> dan recency <b>{n1(high['Recency'].mean())} hari</b>. "
                "Segmentasi final memakai K-Means (menu K-Means).")

    st.markdown('<div class="q-card q3"><span class="badge">Q3</span><div class="txt">Kapan penjualan meningkat dan kategori apa paling berkontribusi?</div></div>', unsafe_allow_html=True)
    dfm = dv.copy()
    dfm["Bulan"] = dfm["Transaction_Date"].dt.to_period("M").dt.to_timestamp()
    monthly = dfm.groupby("Bulan")["Line_Total"].sum().reset_index()
    if len(monthly) > 0:
        fig = px.line(monthly, x="Bulan", y="Line_Total", markers=True, color_discrete_sequence=["#c33764"],
                      labels={"Line_Total": "Total Penjualan (Rp)"}, title="Tren Penjualan Bulanan")
        fig.update_traces(mode="lines+markers+text", text=[fmt_jt(v) for v in monthly["Line_Total"]],
                          textposition="top center", textfont=dict(size=15, color="#7c3aed"), line=dict(width=4))
        fig.update_xaxes(dtick="M1", tickformat="%b %Y", tickangle=-30)
        show_big(fig, 460)
        best = monthly.loc[monthly["Line_Total"].idxmax()]
        cat_data = dv.groupby("Category")["Line_Total"].sum().reset_index().sort_values("Line_Total", ascending=False)
        fig2 = px.bar(cat_data, x="Category", y="Line_Total", color="Category", color_discrete_sequence=PALETTE,
                      text=[fmt_jt(v) for v in cat_data["Line_Total"]], title="Kontribusi Penjualan per Kategori",
                      labels={"Line_Total": "Penjualan (Rp)", "Category": "Kategori"})
        fig2.update_traces(textposition="outside", cliponaxis=False)
        fig2.update_layout(showlegend=False)
        fig2.update_xaxes(tickangle=-35)
        show_big(fig2, 480)
        insight(f"<b>Jawaban Q3:</b> penjualan tertinggi pada <b>{best['Bulan'].strftime('%B %Y')}</b> ({rp(best['Line_Total'])}). "
                f"Kategori paling berkontribusi: <b>{cat_data.iloc[0]['Category']}</b> ({rp(cat_data.iloc[0]['Line_Total'])}).")

    # ---------- 2e ----------
    st.subheader("5️⃣ Temuan Awal — minimal 3 (Soal 2e)")
    cat_sum = dv.groupby("Category")["Line_Total"].sum().sort_values(ascending=False)
    known = dv[dv["Customer_Type"].isin(CTYPE_LIST)]
    member_share = (known["Customer_Type"] == "Member").mean() * 100 if len(known) else 0
    pay_count = dv["Payment_Method"].value_counts().reset_index()
    pay_count.columns = ["Metode Pembayaran", "Jumlah Baris Transaksi"]
    avg_basket = dv.drop_duplicates("Transaction_ID")["Basket_Size"].mean()
    reg = dv.groupby("Region")["Line_Total"].sum().reset_index().sort_values("Line_Total", ascending=False)

    with st.expander(f"Temuan 1 — Kategori {cat_sum.index[0]} menyumbang penjualan terbesar", expanded=True):
        st.markdown(f"**Fakta:** {cat_sum.index[0]} = **{rp(cat_sum.iloc[0])}** "
                    f"({pct1(cat_sum.iloc[0] / cat_sum.sum() * 100)} dari total). **Makna:** stok & promosi harus diprioritaskan di kategori ini.")
        fig = px.pie(cat_sum.reset_index(), names="Category", values="Line_Total", hole=0.45,
                     color_discrete_sequence=PALETTE, title="Porsi Penjualan per Kategori")
        fig.update_traces(textinfo="percent+label", textfont=dict(size=15))
        show_big(fig, 440)
    with st.expander(f"Temuan 2 — Pelanggan Member = {member_share:.1f}% dari baris transaksi (tipe diketahui)", expanded=True):
        st.markdown("**Makna:** perlu strategi konversi Non-Member menjadi Member.")
        ct = known["Customer_Type"].value_counts().reset_index()
        ct.columns = ["Tipe", "Jumlah"]
        fig = px.pie(ct, names="Tipe", values="Jumlah", hole=0.45, color="Tipe",
                     color_discrete_map={"Member": "#10b981", "Non-Member": "#ffb347"}, title="Member vs Non-Member")
        fig.update_traces(textinfo="percent+label", textfont=dict(size=16))
        show_big(fig, 420)
    with st.expander(f"Temuan 3 — Metode pembayaran terbanyak: {pay_count.iloc[0, 0]}; rata-rata basket size {avg_basket:.2f}", expanded=True):
        st.markdown("**Makna:** basket kecil berarti ruang cross-selling terbuka; metode dominan layak diberi insentif.")
        fig = px.bar(pay_count, x="Metode Pembayaran", y="Jumlah Baris Transaksi", color="Metode Pembayaran",
                     color_discrete_sequence=PALETTE, text="Jumlah Baris Transaksi", title="Distribusi Metode Pembayaran")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False)
        show_big(fig, 400)
    with st.expander(f"Temuan 4 — Wilayah {reg.iloc[0]['Region']} memberi penjualan tertinggi", expanded=False):
        fig = px.bar(reg, x="Region", y="Line_Total", color="Region", color_discrete_sequence=PALETTE,
                     text=[fmt_jt(v) for v in reg["Line_Total"]], title="Penjualan per Wilayah",
                     labels={"Line_Total": "Penjualan (Rp)"})
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False)
        show_big(fig, 420)

# =============================================================================
# PAGE: PREPROCESSING  (Tahap 2 — Soal 3 & 4)
# =============================================================================
def run_preprocessing():
    log = []
    df = clean_pipeline(st.session_state.df_raw, log)
    st.session_state.df_clean = df
    st.session_state.prep_log = log
    st.session_state.apriori_result = None
    st.session_state.kmeans_result = None

def render_preprocessing():
    page_header("🧹 Data Preprocessing", "Tahap 2 · Soal 3 (pembersihan & transformasi) dan Soal 4 (feature engineering)")
    if st.session_state.df_raw is None:
        need_data(); return

    st.markdown("""
    **Alur:** salin data (Data_Raw tidak ditimpa) → tangani missing, duplikat, data tidak valid → transformasi format →
    tangani nilai ekstrem → bandingkan sebelum/sesudah → bentuk RFM, Basket_Size, matriks Apriori & K-Means → dokumentasi.
    """)
    if st.button("▶ Jalankan Preprocessing", type="primary"):
        with st.spinner("Membersihkan data..."):
            run_preprocessing()
        st.success("✅ Preprocessing selesai!")

    if st.session_state.df_clean is None:
        return
    raw, clean = st.session_state.df_raw, st.session_state.df_clean
    log = pd.DataFrame(st.session_state.prep_log)

    st.markdown("---")
    st.subheader("📋 Soal 3a–3d · Log Penanganan Data + Alasan Pemilihan Metode")
    show_table(log, formats={"Baris Terdampak": fmt_int, "Baris Dibuang": lambda v: f"-{fmt_int(v)}" if v else "0",
                             "Sisa Baris": fmt_int},
               bars={"Baris Dibuang": "#ef4444", "Sisa Baris": "#7597de"})

    st.subheader("🔄 Soal 3c · Transformasi Format")
    tr = pd.DataFrame({
        "Atribut": ["Transaction_Date", "Quantity", "Unit_Price", "Discount_Rate", "Line_Total", "Region / Category / Customer_Type / Payment_Method", "Basket_Size", "Outlier_Flag"],
        "Tipe Sebelum": [str(raw["Transaction_Date"].dtype), str(raw["Quantity"].dtype), str(raw["Unit_Price"].dtype),
                         str(raw["Discount_Rate"].dtype), str(raw["Line_Total"].dtype), "object (tidak baku)", "(belum ada)", "(belum ada)"],
        "Tipe Sesudah": [str(clean["Transaction_Date"].dtype), str(clean["Quantity"].dtype), str(clean["Unit_Price"].dtype),
                         str(clean["Discount_Rate"].dtype), str(clean["Line_Total"].dtype), "object (baku)", str(clean["Basket_Size"].dtype), str(clean["Outlier_Flag"].dtype)],
        "Keterangan": ["Teks → datetime agar bisa dihitung Recency & tren bulanan", "Numerik valid (> 0)", "Numerik valid (> 0)",
                       "Rentang 0–0,5", "Dihitung ulang dari komponen", "Huruf & ejaan diseragamkan", "Fitur baru", "Penanda nilai ekstrem"],
    })
    show_table(tr)

    st.subheader("⚠️ Soal 3d · Nilai Ekstrem dengan Konteks Bisnis")
    qx, lx = iqr_stats(clean["Quantity"]), iqr_stats(clean["Line_Total"])
    st.markdown(f"""
    Nilai ekstrem didefinisikan sebagai **> Q3 + 3×IQR**: Quantity > **{fmt_int(qx['ext_hi'])}** unit atau Line_Total > **{rp(lx['ext_hi'])}**.
    Ditemukan **{fmt_int(clean['Outlier_Flag'].sum())}** baris ({pct1(clean['Outlier_Flag'].mean()*100)}).
    Dalam toko ritel, pembelian dalam jumlah besar (grosir/acara) adalah transaksi sah, sehingga **tidak dihapus** — hanya ditandai di kolom
    `Outlier_Flag`. Nilai yang jelas salah input (Quantity ≤ 0 / ≥ 999, harga ≤ 0, diskon di luar 0–50%) sudah dihapus pada tahap 6–8.
    """)

    st.subheader("📊 Soal 3e · Perbandingan Sebelum vs Sesudah Preprocessing")
    qs_r, qs_c = quality_stats(raw), quality_stats(clean)
    comp = pd.DataFrame({"Indikator": list(qs_r.keys()),
                         "Sebelum (Data_Raw)": [fmt_int(v) for v in qs_r.values()],
                         "Sesudah (Data_Clean)": [fmt_int(v) for v in qs_c.values()],
                         "Selisih": [("+" if qs_c[k] - qs_r[k] > 0 else "") + fmt_int(qs_c[k] - qs_r[k]) for k in qs_r]})
    extra = pd.DataFrame({
        "Indikator": ["Total penjualan (Rp)", "Rentang tanggal"],
        "Sebelum (Data_Raw)": [rp(raw["Line_Total"].sum()), f"{raw['Transaction_Date'].min():%d-%m-%Y} s/d {raw['Transaction_Date'].max():%d-%m-%Y}"],
        "Sesudah (Data_Clean)": [rp(clean["Line_Total"].sum()), f"{clean['Transaction_Date'].min():%d-%m-%Y} s/d {clean['Transaction_Date'].max():%d-%m-%Y}"],
        "Selisih": [rp(clean["Line_Total"].sum() - raw["Line_Total"].sum()), "-"],
    })
    show_table(pd.concat([comp, extra], ignore_index=True))
    diff = len(raw) - len(clean)
    c1, c2 = st.columns(2)
    c1.markdown(kpi_card("k2", "Data_Raw (Sebelum)", f"{fmt_int(len(raw))} baris", "data asli dari Excel"), unsafe_allow_html=True)
    c2.markdown(kpi_card("k1", "Data_Clean (Sesudah)", f"{fmt_int(len(clean))} baris",
                         f"<span style='color:#ef4444;font-weight:bold;'>-{fmt_int(diff)} baris ({pct1(diff/max(len(raw),1)*100)}) dibuang</span>"),
                unsafe_allow_html=True)
    fig = go.Figure()
    for name, vals, col in [("Sebelum", [qs_r["Baris duplikat"], qs_r["Sel kosong (missing)"], qs_r["Quantity ≤ 0 atau ≥ 999"]], "#ef4444"),
                            ("Sesudah", [qs_c["Baris duplikat"], qs_c["Sel kosong (missing)"], qs_c["Quantity ≤ 0 atau ≥ 999"]], "#10b981")]:
        fig.add_trace(go.Bar(name=name, x=["Duplikat", "Sel missing", "Quantity tidak wajar"], y=vals, marker_color=col, text=vals, textposition="outside"))
    fig.update_layout(barmode="group", title="Masalah Kualitas Data: Sebelum vs Sesudah")
    show_big(fig, 420)

    # ================= SOAL 4 =================
    st.markdown("---")
    st.header("⚙️ Soal 4 · Feature Engineering & Persiapan Analisis")

    st.subheader("4a · Fitur RFM per Pelanggan")
    rfm, snapshot = build_rfm(clean)
    st.markdown(f"**Recency** = jarak hari dari transaksi terakhir ke tanggal acuan (**{snapshot.date()}**, = tanggal terakhir + 1 hari); "
                f"**Frequency** = jumlah transaksi unik; **Monetary** = total Line_Total. Terbentuk **{fmt_int(len(rfm))} pelanggan**.")
    show_table(rfm.head(10), formats={"Monetary": rp}, bars={"Monetary": "#10b981"})

    st.subheader("4b · Basket_Size (jumlah produk berbeda per transaksi)")
    bs = clean.drop_duplicates("Transaction_ID")["Basket_Size"]
    dist = bs.value_counts().sort_index().reset_index()
    dist.columns = ["Basket_Size", "Jumlah Transaksi"]
    dist["Basket_Size"] = dist["Basket_Size"].astype(str)
    fig = px.bar(dist, x="Basket_Size", y="Jumlah Transaksi", color="Basket_Size", color_discrete_sequence=PALETTE,
                 text="Jumlah Transaksi", title=f"Distribusi Basket_Size (rata-rata {bs.mean():.2f})")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False)
    show_big(fig, 400)

    st.subheader("4c · Format Transaksi untuk Apriori (transaksi × produk, 1/0)")
    bb = basket_matrix(clean)
    top_cols = bb.sum().nlargest(8).index
    prev = bb[top_cols].astype(int).head(10).reset_index()
    sparsity = 1 - bb.values.mean()
    st.markdown(f"Ukuran matriks: **{fmt_int(bb.shape[0])} transaksi × {fmt_int(bb.shape[1])} produk**; "
                f"kepadatan nilai 1 hanya **{pct1((1-sparsity)*100)}** (matriks jarang). Pratinjau 10 transaksi × 8 produk terlaris:")
    show_table(prev, formats={c: (lambda v: str(int(v))) for c in top_cols})

    st.subheader("4d · Matriks Fitur untuk K-Means")
    X = rfm[["Recency", "Frequency", "Monetary"]]
    st.markdown(f"Matriks fitur berukuran **{fmt_int(X.shape[0])} pelanggan × {X.shape[1]} fitur** (Recency, Frequency, Monetary).")
    show_table(X.head(8), formats={"Monetary": rp})

    st.subheader("4e · Standardisasi Fitur (StandardScaler)")
    Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    sc = pd.DataFrame({"Fitur": X.columns, "Rata-rata Awal": X.mean().values, "Std Awal": X.std().values,
                       "Min Awal": X.min().values, "Max Awal": X.max().values,
                       "Rata-rata Setelah": Xs.mean().round(6).values, "Std Setelah": Xs.std().round(3).values})
    show_table(sc)
    st.markdown("**Alasan:** K-Means memakai jarak Euclidean. Monetary (jutaan rupiah) jauh lebih besar skalanya daripada Recency (hari) "
                "dan Frequency (kali), sehingga tanpa standardisasi klaster hanya ditentukan oleh Monetary. StandardScaler "
                "(rata-rata 0, simpangan baku 1) membuat ketiga fitur berkontribusi setara. Dipilih dibanding Min-Max karena lebih tahan terhadap outlier ekstrem.")

    st.subheader("4f · Strategi untuk Dataset Berukuran Besar")
    mem_raw = raw.memory_usage(deep=True).sum() / 1e6
    mem_clean = clean.memory_usage(deep=True).sum() / 1e6
    strat = pd.DataFrame({
        "Strategi": ["Agregasi", "Pemrosesan bertahap (chunking)", "Sampling", "Optimasi tipe data & cache"],
        "Penjelasan": [
            "Transaksi diagregasi ke level pelanggan (RFM) sebelum K-Means, dan ke matriks transaksi × produk sebelum Apriori",
            "File besar dibaca per potongan (pd.read_excel/read_csv dengan chunksize) lalu dibersihkan per chunk dan digabung",
            "Untuk jutaan baris, ambil sampel acak/stratified (mis. 10–20%) untuk uji parameter; model final dijalankan pada data penuh",
            f"Kolom teks diubah ke 'category', angka ke int32/float32, hasil di-cache (st.cache_data). Memori saat ini: raw {mem_raw:.1f} MB → clean {mem_clean:.1f} MB",
        ],
        "Digunakan di sistem ini": ["✅ Ya", "➖ Tidak perlu (data ± belasan ribu baris)", "➖ Silhouette memakai sampel 5.000 bila pelanggan > 5.000", "✅ Ya (cache)"],
    })
    show_table(strat)

    st.subheader("4g · Dokumentasi Jumlah Data")
    kept = len(clean)
    removed = len(raw) - len(clean)
    changed = int(log.loc[log["Tahap"].isin(["1", "2", "9", "10"]), "Baris Terdampak"].sum())
    doc = pd.DataFrame({"Kategori": ["Dipertahankan", "Dihapus", "Diubah / diimputasi (total tindakan)", "Kolom baru ditambahkan"],
                        "Jumlah": [kept, removed, changed, 2],
                        "Keterangan": [f"{pct1(kept/len(raw)*100)} dari Data_Raw", f"{pct1(removed/len(raw)*100)} dari Data_Raw",
                                       "Normalisasi teks, imputasi, hitung ulang Line_Total, isi 'Tidak Diketahui'", "Basket_Size & Outlier_Flag"]})
    show_table(doc, bars={"Jumlah": "#7597de"})

    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ Data_Clean (CSV)", df_to_csv_bytes(clean), "data_clean.csv", "text/csv", use_container_width=True)
    d2.download_button("⬇ Data_Clean (Excel)", df_to_xlsx_bytes(clean), "data_clean.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    d3.download_button("⬇ Dokumentasi Preprocessing (CSV)", df_to_csv_bytes(log), "dokumentasi_preprocessing.csv", "text/csv", use_container_width=True)

# =============================================================================
# PAGE: APRIORI  (Tahap 3 — Soal 5 & 6)
# =============================================================================
def shelf_table(rules, n=6):
    seen, rows = set(), []
    for _, r in rules.iterrows():
        key = frozenset([r["antecedents"], r["consequents"]])
        if key in seen: continue
        seen.add(key)
        zona = "Rak sejajar / bersebelahan" if r["lift"] >= 2 else "Etalase yang sama (jarak dekat)"
        rows.append({"Produk A": r["antecedents"], "Produk B": r["consequents"], "Penempatan": zona,
                     "Alasan (hasil analisis)": f"Lift {r['lift']:.2f}, confidence {r['confidence']*100:.1f}%, support {r['support']*100:.2f}%"})
        if len(rows) >= n: break
    return pd.DataFrame(rows)

def render_apriori():
    page_header("🔗 Apriori / Market Basket Analysis", "Tahap 3 · Soal 5 (analisis pola) dan Soal 6 (interpretasi pola pembelian)")
    if st.session_state.df_clean is None:
        need_clean(); return
    df = st.session_state.df_clean

    # ---- 5a ----
    st.subheader("5a · Matriks Transaksi × Produk")
    bb = basket_matrix(df)
    top_cols = bb.sum().nlargest(8).index
    st.markdown(f"Matriks **{fmt_int(bb.shape[0])} transaksi × {fmt_int(bb.shape[1])} produk** (1 = produk dibeli pada transaksi itu).")
    show_table(bb[top_cols].astype(int).head(8).reset_index(), formats={c: (lambda v: str(int(v))) for c in top_cols})

    # ---- 5b ----
    st.subheader("5b · Menentukan Minimum Support & Minimum Confidence")
    bs = df.drop_duplicates("Transaction_ID")["Basket_Size"]
    multi = (bs >= 2).mean() * 100
    st.markdown(f"""
    <div class="alert-info"><b>Dasar pemilihan parameter (dari karakteristik dataset):</b><br>
    • Rata-rata basket size hanya <b>{bs.mean():.2f}</b> produk dan hanya <b>{multi:.1f}%</b> transaksi berisi ≥ 2 produk, serta ada
      <b>{fmt_int(bb.shape[1])}</b> produk berbeda → peluang sebuah pasangan produk muncul bersama kecil. Support tinggi (mis. 20%) tidak akan menemukan
      pola apa pun, sehingga support awal dipilih <b>rendah (0,010 = 1%)</b>. Pada ±{fmt_int(bb.shape[0])} transaksi, 1% ≈ {fmt_int(bb.shape[0]*0.01)} transaksi, cukup untuk dianggap pola, bukan kebetulan.<br>
    • <b>Tujuan bisnis</b>: rekomendasi bundling harus cukup meyakinkan → confidence minimal <b>0,30</b> (≥ 30% pembeli A juga membeli B), lalu
      aturan diurutkan dengan <b>lift</b> agar yang dipilih benar-benar bukan sekadar karena produk B populer.<br>
    • Slider di bawah dapat diubah bebas; hasil dihitung ulang saat tombol dijalankan.</div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    min_support = col1.slider("Minimum Support", 0.001, 0.500, 0.010, 0.001, format="%.3f",
                              help="Persentase transaksi yang memuat itemset.")
    min_confidence = col2.slider("Minimum Confidence", 0.05, 0.95, 0.30, 0.05, format="%.2f",
                                 help="Peluang B dibeli jika A dibeli.")

    if st.button("▶ Jalankan Apriori", type="primary"):
        with st.spinner("Menjalankan algoritma Apriori..."):
            freq, rules, n_trans = run_apriori_cached(df, min_support, min_confidence)
        if freq is None:
            st.warning("Tidak ada frequent itemset. Turunkan min support.")
        elif rules is None or len(rules) == 0:
            st.warning("Tidak ada association rule. Turunkan min confidence atau min support.")
        else:
            st.session_state.apriori_result = {"freq": freq, "rules": rules, "n_trans": n_trans,
                                               "min_support": min_support, "min_confidence": min_confidence}
            st.success(f"✅ Ditemukan {len(freq)} frequent itemsets dan {len(rules)} rules.")

    res = st.session_state.apriori_result
    if res is None:
        return
    freq, rules, n_trans = res["freq"], res["rules"], res["n_trans"]
    st.caption(f"Parameter aktif: Min Support = {res['min_support']:.3f} | Min Confidence = {res['min_confidence']:.2f}")
    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi_card("k1", "Frequent Itemsets", fmt_int(len(freq)), "memenuhi min support"), unsafe_allow_html=True)
    k2.markdown(kpi_card("k3", "Association Rules", fmt_int(len(rules)), "memenuhi min confidence"), unsafe_allow_html=True)
    k3.markdown(kpi_card("k5", "Lift Tertinggi", f"{rules['lift'].max():.2f}", "kekuatan asosiasi"), unsafe_allow_html=True)

    # ---- 5c ----
    st.markdown("---")
    st.subheader("5c · Frequent Itemsets (Top 20)")
    only_multi = st.checkbox("Tampilkan hanya itemset ≥ 2 produk", value=True)
    fv = freq.copy()
    if only_multi:
        fv = fv[fv["length"] >= 2]
    fv["itemsets"] = fv["itemsets"].apply(lambda x: ", ".join(sorted(x)))
    fv = fv.sort_values("support", ascending=False).head(20)[["itemsets", "support", "length"]]
    fv.columns = ["Itemset (produk)", "Support", "Jumlah Item"]
    if len(fv):
        show_table(fv, formats={"Support": lambda v: f"{v*100:.2f}%".replace(".", ",")}, bars={"Support": "#7597de"}, max_height=520)
    else:
        st.info("Tidak ada itemset ≥ 2 produk pada support ini. Turunkan min support.")

    # ---- 5d,e,f ----
    st.markdown("---")
    st.subheader("5d–5f · Aturan Asosiasi & Top 10 (support, confidence, lift)")
    st.caption("Kriteria Top 10: hanya aturan dengan **lift > 1** (asosiasi positif), diurutkan dari **lift tertinggi**, lalu confidence tertinggi.")
    good = rules[rules["lift"] > 1]
    top10 = (good if len(good) else rules).head(10).copy()
    tv = top10[["antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"]].copy()
    tv.columns = ["Jika membeli (Antecedents)", "Maka membeli (Consequents)", "Support", "Confidence", "Lift", "Leverage", "Conviction"]
    show_table(tv, formats={"Support": lambda v: f"{v*100:.2f}%".replace(".", ","),
                            "Confidence": lambda v: f"{v*100:.1f}%".replace(".", ","),
                            "Lift": lambda v: f"{v:.3f}", "Leverage": lambda v: f"{v:.4f}",
                            "Conviction": lambda v: "∞" if np.isinf(v) else f"{v:.3f}"},
               bars={"Support": "#7597de", "Confidence": "#c33764", "Lift": "#8b5cf6"})
    st.download_button("⬇ Unduh semua rules (CSV)", df_to_csv_bytes(rules.drop(columns=["antecedent support", "consequent support"], errors="ignore")),
                       "association_rules.csv", "text/csv")

    # ---- 5g ----
    st.markdown("---")
    st.subheader("5g · Visualisasi Hasil Apriori")
    fig_sc = px.scatter(rules, x="support", y="confidence", size="lift", color="lift",
                        hover_data={"antecedents": True, "consequents": True, "lift": ":.2f"},
                        color_continuous_scale="Plasma", size_max=30,
                        labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"})
    fig_sc.update_traces(marker=dict(line=dict(width=1.5, color="white")))
    fig_sc.update_layout(title="Sebaran Aturan: Support vs Confidence (warna & ukuran = Lift)", height=500)
    show(fig_sc)
    hm = lift_heatmap(rules)
    if hm is not None:
        show(hm, 520)
    fig_net = plot_rules_network(good if len(good) else rules, top_n=15)
    if fig_net is not None:
        st.caption("Arah panah = 'jika membeli A, maka membeli B'. Ketebalan panah sebanding dengan Lift.")
        show(fig_net)

    # ---- SOAL 6 ----
    st.markdown("---")
    st.header("🧠 Soal 6 · Interpretasi Pola Pembelian")
    st.subheader("6a · Kombinasi Produk Paling Menarik bagi Bisnis")
    cand = good[good["support"] >= good["support"].median()] if len(good) else rules
    best = (cand if len(cand) else rules).sort_values(["lift", "confidence"], ascending=False).iloc[0]
    insight(f"Kombinasi paling menarik: <b>{best['antecedents']}</b> → <b>{best['consequents']}</b> "
            f"(lift {best['lift']:.2f}, confidence {best['confidence']*100:.1f}%, support {best['support']*100:.2f}%). "
            "Dipilih karena lift tinggi (hubungan kuat) <u>dan</u> support di atas median (cukup sering terjadi sehingga berdampak nyata, bukan kasus langka).")

    st.subheader("6b · Arti Support, Confidence, dan Lift pada Aturan Terpilih")
    opts = [f"{i+1}. {r['antecedents']} → {r['consequents']}" for i, (_, r) in enumerate(top10.iterrows())]
    sel = st.selectbox("Pilih aturan untuk dijelaskan", opts)
    row = top10.iloc[opts.index(sel)]
    st.markdown(f'<div class="reco-card">{explain_rule(row, n_trans)}</div>', unsafe_allow_html=True)

    st.subheader("6c · Rekomendasi Cross-Selling / Bundling (3 teratas)")
    for i, (_, r) in enumerate(top10.head(3).iterrows(), start=1):
        st.markdown(f"""
        <div class="reco-card">
            <h4>Rekomendasi #{i} — Bundling "{r['antecedents']}" + "{r['consequents']}"</h4>
            <p><b>Bukti data:</b> support {r['support']:.2%}, confidence {r['confidence']:.2%}, lift {r['lift']:.2f}
               {"(> 1: asosiasi positif)" if r['lift'] > 1 else ""}.</p>
            <p><b>Tindakan:</b> tawarkan paket <i>{r['antecedents']}</i> + <i>{r['consequents']}</i> dengan potongan kecil; kasir menawarkan
               <i>{r['consequents']}</i> saat <i>{r['antecedents']}</i> dibeli; tampilkan "sering dibeli bersama" pada nota/aplikasi.</p>
            <p><b>KPI:</b> basket size naik ≥ 10%, penjualan produk consequents naik ≥ 5% dalam 1 bulan.</p>
        </div>""", unsafe_allow_html=True)

    st.subheader("6d · Produk yang Layak Ditempatkan Berdekatan (rak / etalase)")
    sh = shelf_table(good if len(good) else rules)
    if len(sh): show_table(sh)

    st.subheader("6e · Alasan Rekomendasi (berdasarkan hasil analisis, bukan perkiraan)")
    st.markdown("""
    - **Support** membuktikan kombinasi itu benar-benar sering terjadi pada transaksi nyata, jadi penataan rak berdampak pada banyak pelanggan.
    - **Confidence** menunjukkan peluang pembeli A ikut membeli B, sehingga tawaran bundling punya peluang diterima.
    - **Lift > 1** membuktikan hubungan itu bukan sekadar karena produk B memang populer, melainkan A benar-benar mendorong pembelian B.
    - Karena itu rekomendasi disusun dari angka ketiga metrik tersebut, bukan intuisi petugas toko.
    """)

# =============================================================================
# PAGE: K-MEANS  (Tahap 4 — Soal 7 & 8)
# =============================================================================
def render_kmeans():
    page_header("🎯 K-Means / Segmentasi Pelanggan (RFM)", "Tahap 4 · Soal 7 (pemodelan) dan Soal 8 (interpretasi & strategi)")
    if st.session_state.df_clean is None:
        need_clean(); return
    df = st.session_state.df_clean
    rfm, snapshot = build_rfm(df)
    if len(rfm) < 5:
        st.warning("Jumlah pelanggan terlalu sedikit untuk K-Means."); return

    st.subheader("7a · Fitur untuk Segmentasi")
    st.markdown("""
    Fitur **RFM** dipilih karena langsung menggambarkan perilaku belanja dan nilai pelanggan:
    **Recency** (seberapa baru belanja → indikasi aktif/akan pergi), **Frequency** (seberapa sering → loyalitas),
    **Monetary** (total belanja → nilai ekonomi). Seluruh fitur distandardisasi (StandardScaler) sebelum K-Means.
    """)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("k1", "Jumlah Pelanggan", fmt_int(len(rfm)), f"snapshot {snapshot.date()}"), unsafe_allow_html=True)
    c2.markdown(kpi_card("k2", "Rata-rata Recency", f"{n1(rfm['Recency'].mean())} hari", "terakhir belanja"), unsafe_allow_html=True)
    c3.markdown(kpi_card("k3", "Rata-rata Frekuensi", f"{n1(rfm['Frequency'].mean())}×", "transaksi / pelanggan"), unsafe_allow_html=True)
    c4.markdown(kpi_card("k4", "Rata-rata Monetary", rp(rfm["Monetary"].mean()), "belanja / pelanggan"), unsafe_allow_html=True)
    prev = rfm.head(10).copy()
    prev.columns = ["ID Pelanggan", "Recency (hari)", "Frekuensi (transaksi)", "Monetary (Rp)"]
    show_table(prev, formats={"Monetary (Rp)": rp}, bars={"Recency (hari)": "#ffb347", "Frekuensi (transaksi)": "#7597de", "Monetary (Rp)": "#10b981"})

    st.markdown("---")
    st.subheader("7b–7c · Elbow Method & Silhouette Score")
    max_k_limit = min(20, len(rfm) - 1)
    col1, col2 = st.columns(2)
    k_value = col1.slider("Pilih Jumlah K (Klaster)", 2, max(2, max_k_limit), min(4, max_k_limit))
    max_k_elbow = col2.slider("Uji Elbow s.d. K", 3, max(3, max_k_limit), min(8, max_k_limit))
    with st.spinner(f"Menghitung Elbow & Silhouette untuk k = 2..{max_k_elbow}"):
        elb = elbow_cached(rfm, max_k_elbow)
    best_k = render_elbow(elb)
    st.markdown(f'<div class="alert-success">💡 Silhouette tertinggi pada <b>k = {best_k}</b> (skor {max(elb["sils"]):.3f}). '
                'Cari juga titik siku (penurunan inertia mulai melandai) pada grafik.</div>', unsafe_allow_html=True)

    seg = segment_customers(df, k_value)
    if seg is None:
        st.warning("Segmentasi gagal: pelanggan terlalu sedikit."); return
    out, profile, sil, _, k_used = seg
    st.session_state.kmeans_result = {"k": k_used, "silhouette": sil}

    st.subheader("7d · Jumlah Klaster yang Digunakan & Alasan")
    sil_at = elb["sils"][elb["ks"].index(k_used)] if k_used in elb["ks"] else sil
    st.markdown(f"""
    <div class="reco-card" style="border-top-color:#7597de;">
    <p>Jumlah klaster yang dipakai: <b>k = {k_used}</b> (silhouette = <b>{sil:.3f}</b>; k terbaik menurut silhouette pada rentang uji adalah k = {best_k}
    dengan skor {max(elb['sils']):.3f}).</p>
    <p><b>Alasan:</b> (1) Silhouette {'sudah tertinggi' if k_used == best_k else 'masih berdekatan dengan nilai tertinggi' if abs(sil_at - max(elb['sils'])) < 0.05 else 'lebih rendah dari k terbaik, namun dipertimbangkan bersama faktor lain'};
    (2) pada grafik Elbow, penurunan inertia setelah titik ini cenderung melandai; (3) jumlah segmen harus <b>dapat ditafsirkan dan ditindaklanjuti</b>
    manajemen (tiap segmen punya strategi berbeda) — terlalu banyak klaster sulit dikelola, terlalu sedikit menyembunyikan perbedaan perilaku.
    Anda dapat menggeser slider untuk membandingkan.</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader(f"7e · Hasil Pengelompokan (k = {k_used})")
    top_seg = profile.sort_values("Monetary_Total", ascending=False).iloc[0]
    r1, r2, r3, r4 = st.columns(4)
    r1.markdown(kpi_card("k1", "Jumlah Klaster", str(k_used), "hasil K-Means"), unsafe_allow_html=True)
    r2.markdown(kpi_card("k3", "Silhouette Score", f"{sil:.3f}", "kualitas pemisahan"), unsafe_allow_html=True)
    r3.markdown(kpi_card("k2", "Total Pelanggan", fmt_int(profile["Jumlah_Pelanggan"].sum()), "dikelompokkan"), unsafe_allow_html=True)
    r4.markdown(kpi_card("k5", "Kontributor Terbesar", top_seg["Label"], f"C{int(top_seg['Cluster'])} · {rp(top_seg['Monetary_Total'])}"), unsafe_allow_html=True)

    st.subheader("8a · Profil Setiap Klaster (RFM) + 8b · Label & Dasar Pelabelan")
    disp = profile.copy()
    disp["Pct_Pelanggan"] = disp["Jumlah_Pelanggan"] / disp["Jumlah_Pelanggan"].sum() * 100
    disp["Pct_Monetary"] = disp["Monetary_Total"] / disp["Monetary_Total"].sum() * 100
    disp = disp.sort_values("Monetary_Total", ascending=False)
    t1 = disp[["Cluster", "Label", "Jumlah_Pelanggan", "Pct_Pelanggan", "Recency_Mean", "Frequency_Mean",
               "Monetary_Mean", "Monetary_Median", "Monetary_Total", "Pct_Monetary"]].copy()
    t1.columns = ["Klaster", "Segmen", "Jumlah Pelanggan", "% Pelanggan", "Recency Rata-rata (hari)", "Frekuensi Rata-rata (×)",
                  "Monetary Rata-rata", "Monetary Median", "Monetary Total", "% Monetary"]
    show_table(t1, formats={"Klaster": lambda v: f"C{int(v)}", "Jumlah Pelanggan": fmt_int, "% Pelanggan": pct1,
                            "Recency Rata-rata (hari)": n1, "Frekuensi Rata-rata (×)": n1, "Monetary Rata-rata": rp,
                            "Monetary Median": rp, "Monetary Total": rp, "% Monetary": pct1},
               badge="Segmen", bars={"Jumlah Pelanggan": "#7597de", "% Pelanggan": "#7597de", "Monetary Rata-rata": "#10b981",
                                     "Monetary Total": "#10b981", "% Monetary": "#10b981"})
    basis = disp[["Cluster", "Label", "Dasar"]].copy()
    basis.columns = ["Klaster", "Segmen", "Dasar pelabelan (dibandingkan rata-rata seluruh pelanggan)"]
    show_table(basis, formats={"Klaster": lambda v: f"C{int(v)}"}, badge="Segmen")

    pv = profile.copy()
    pv["Nama"] = pv["Label"] + "<br>(C" + pv["Cluster"].astype(str) + ")"
    ca, cb = st.columns(2)
    with ca:
        fig = px.bar(pv, x="Nama", y="Jumlah_Pelanggan", color="Nama", text="Jumlah_Pelanggan", color_discrete_sequence=PALETTE,
                     title="Jumlah Pelanggan per Klaster", labels={"Jumlah_Pelanggan": "Jumlah Pelanggan", "Nama": ""})
        fig.update_layout(showlegend=False); fig.update_traces(textposition="outside")
        show(fig, 440)
    with cb:
        fig = px.bar(pv, x="Nama", y="Monetary_Total", color="Nama", color_discrete_sequence=PALETTE,
                     title="Total Monetary per Klaster (Rp)", labels={"Monetary_Total": "Total Monetary (Rp)", "Nama": ""})
        fig.update_layout(showlegend=False)
        show(fig, 440)

    st.subheader("7f · Visualisasi Klaster (Scatter Plot)")
    op = out.copy()
    op["Nama"] = op["Label"] + " (C" + op["Cluster"].astype(str) + ")"
    sa, sb = st.columns(2)
    with sa:
        fig = px.scatter(op, x="Recency", y="Monetary", color="Nama", size="Frequency", hover_data=["Customer_ID", "Frequency"],
                         color_discrete_sequence=PALETTE, title="Recency vs Monetary", labels={"Recency": "Recency (hari)", "Monetary": "Monetary (Rp)"})
        show(fig, 480)
    with sb:
        fig = px.scatter(op, x="Frequency", y="Monetary", color="Nama", hover_data=["Customer_ID", "Recency"],
                         color_discrete_sequence=PALETTE, title="Frequency vs Monetary", labels={"Frequency": "Frekuensi (transaksi)", "Monetary": "Monetary (Rp)"})
        show(fig, 480)

    st.subheader("8c · Karakteristik Utama Tiap Segmen (berdasarkan hasil perhitungan)")
    for _, r in disp.iterrows():
        color = SEGMENT_COLORS.get(r["Label"], "#c33764")
        st.markdown(f"""
        <div class="reco-card" style="border-top-color:{color};">
        <h4>{r['Label']} (C{int(r['Cluster'])}) — {fmt_int(r['Jumlah_Pelanggan'])} pelanggan ({pct1(r['Pct_Pelanggan'])})</h4>
        <p>Rata-rata belanja terakhir <b>{n1(r['Recency_Mean'])} hari</b> lalu, <b>{n1(r['Frequency_Mean'])}×</b> transaksi,
        belanja rata-rata <b>{rp(r['Monetary_Mean'])}</b> (median {rp(r['Monetary_Median'])}); menyumbang <b>{pct1(r['Pct_Monetary'])}</b> dari seluruh nilai belanja.</p>
        </div>""", unsafe_allow_html=True)

    st.subheader("🏅 Top 10 Pelanggan Bernilai Tertinggi")
    top_c = out.sort_values("Monetary", ascending=False).head(10)[["Customer_ID", "Label", "Cluster", "Recency", "Frequency", "Monetary"]].reset_index(drop=True)
    top_c.insert(0, "Peringkat", range(1, len(top_c) + 1))
    top_c.columns = ["Peringkat", "ID Pelanggan", "Segmen", "Klaster", "Recency (hari)", "Frekuensi (transaksi)", "Monetary (Rp)"]
    show_table(top_c, formats={"Peringkat": lambda v: f"#{int(v)}", "Klaster": lambda v: f"C{int(v)}", "Monetary (Rp)": rp},
               badge="Segmen", bars={"Monetary (Rp)": "#10b981", "Frekuensi (transaksi)": "#7597de"})
    st.download_button("⬇ Unduh Hasil Segmentasi Pelanggan (CSV)", df_to_csv_bytes(out), "hasil_segmentasi_kmeans.csv", "text/csv")

    st.subheader("8d · Strategi Pemasaran per Segmen")
    for _, row in disp.iterrows():
        strat = STRATEGIES.get(row["Label"], {"aktif": "-", "kpi": "-"})
        color = SEGMENT_COLORS.get(row["Label"], "#c33764")
        st.markdown(f"""
        <div class="reco-card" style="border-top-color:{color};">
            <h4>🎯 {row['Label']} (C{int(row['Cluster'])}) — {fmt_int(row['Jumlah_Pelanggan'])} pelanggan</h4>
            <p><b>Strategi:</b> {strat['aktif']}</p>
            <p><b>KPI:</b> {strat['kpi']}</p>
        </div>""", unsafe_allow_html=True)

    st.subheader("8e · Segmen yang Menjadi Prioritas Perusahaan")
    for no, lab, c, why in priority_segments(profile):
        color = SEGMENT_COLORS.get(lab, "#c33764")
        st.markdown(f"""
        <div class="reco-card" style="border-top-color:{color};">
            <h4>{no}: {lab} (C{c})</h4><p><b>Alasan:</b> {why}</p>
        </div>""", unsafe_allow_html=True)

# =============================================================================
# PAGE: DASHBOARD BI  (Tahap 5 — Soal 9)
# =============================================================================
def render_dashboard():
    page_header("📊 Dashboard BI", "Tahap 5 · Soal 9 — KPI, analisis penjualan, Apriori, K-Means, interaktivitas & integrasi")
    if st.session_state.df_raw is None:
        need_data(); return
    df = st.session_state.df_raw
    dv = build_eda_data(df)

    # ---- ringkasan data awal ----
    with st.expander("🗂️ Ringkasan Data Awal (Data_Raw dari Excel)", expanded=False):
        total_rows, n_cols = len(df), df.shape[1]
        n_dup = int(df.duplicated().sum()); miss_cells = int(df.isna().sum().sum()); miss_rows = int(df.isna().any(axis=1).sum())
        a1, a2, a3, a4 = st.columns(4)
        a1.markdown(kpi_card("k2", "Total Data Awal", f"{fmt_int(total_rows)} data", f"{n_cols} kolom"), unsafe_allow_html=True)
        a2.markdown(kpi_card("k6", "Data Duplikat", f"{fmt_int(n_dup)} data", f"{pct1(n_dup/max(total_rows,1)*100)}"), unsafe_allow_html=True)
        a3.markdown(kpi_card("k4", "Missing Value", f"{fmt_int(miss_cells)} sel", f"{pct1(miss_cells/max(total_rows*n_cols,1)*100)} sel"), unsafe_allow_html=True)
        a4.markdown(kpi_card("k5", "Data Ada Missing", f"{fmt_int(miss_rows)} data", f"{pct1(miss_rows/max(total_rows,1)*100)}"), unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=420)

    # ---- segmentasi global (untuk filter segmen & integrasi) ----
    seg = segment_customers(dv, current_k())
    if seg is not None:
        seg_out, seg_profile, seg_sil, _, k_dash = seg
        dv = dv.merge(seg_out[["Customer_ID", "Label"]].rename(columns={"Label": "Segmen"}), on="Customer_ID", how="left")
        dv["Segmen"] = dv["Segmen"].fillna("Tidak tersegmentasi")
    else:
        seg_out = seg_profile = None
        dv["Segmen"] = "Tidak tersegmentasi"

    # ---- interaktivitas (Soal 9e) ----
    st.subheader("🎛️ Filter Interaktif")
    f1, f2 = st.columns(2)
    with f1:
        date_range = st.date_input("Filter periode transaksi",
                                   [dv["Transaction_Date"].min().date(), dv["Transaction_Date"].max().date()])
    with f2:
        cat_filter = st.multiselect("Filter kategori", sorted(dv["Category"].dropna().unique()), default=[])
    f3, f4 = st.columns(2)
    with f3:
        prod_filter = st.multiselect("Filter produk", sorted(dv["Product_Name"].dropna().unique()), default=[])
    with f4:
        seg_filter = st.multiselect("Filter segmen pelanggan", sorted(dv["Segmen"].unique()), default=[],
                                    help="Segmen berasal dari K-Means (k mengikuti menu K-Means, default 4).")
    dff = dv.copy()
    if len(date_range) == 2:
        dff = dff[(dff["Transaction_Date"].dt.date >= date_range[0]) & (dff["Transaction_Date"].dt.date <= date_range[1])]
    if cat_filter: dff = dff[dff["Category"].isin(cat_filter)]
    if prod_filter: dff = dff[dff["Product_Name"].isin(prod_filter)]
    if seg_filter: dff = dff[dff["Segmen"].isin(seg_filter)]
    if dff.empty:
        st.warning("⚠️ Tidak ada data untuk filter yang dipilih."); return
    st.caption(f"Data terfilter: {fmt_int(len(dff))} baris dari {fmt_int(len(dv))} baris Data_Clean (otomatis dibersihkan).")

    # ---- KPI (Soal 9a) ----
    st.subheader("💰 Key Performance Indicators")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("k1", "Total Penjualan", rp(dff["Line_Total"].sum()), "pendapatan (terfilter)"), unsafe_allow_html=True)
    c2.markdown(kpi_card("k2", "Jumlah Transaksi", fmt_int(dff["Transaction_ID"].nunique()), "Transaction_ID unik"), unsafe_allow_html=True)
    c3.markdown(kpi_card("k3", "Pelanggan Unik", fmt_int(dff["Customer_ID"].nunique()), "pelanggan aktif pada periode ini"), unsafe_allow_html=True)
    c4.markdown(kpi_card("k4", "Rata-rata Basket Size", f"{dff.drop_duplicates('Transaction_ID')['Basket_Size'].mean():.2f}", "produk per transaksi"), unsafe_allow_html=True)

    t_sales, t_apr, t_km, t_int = st.tabs(["📈 Analisis Penjualan", "🔗 Analisis Apriori", "🎯 Analisis K-Means", "🧩 Integrasi Apriori × K-Means (Bonus)"])

    # ---- 9b Penjualan ----
    with t_sales:
        dfm = dff.copy()
        dfm["Bulan"] = dfm["Transaction_Date"].dt.to_period("M").dt.to_timestamp()
        monthly = dfm.groupby("Bulan")["Line_Total"].sum().reset_index()
        st.markdown("#### 📅 Tren Penjualan Bulanan")
        fig = px.line(monthly, x="Bulan", y="Line_Total", markers=True, labels={"Line_Total": "Total Penjualan (Rp)"},
                      color_discrete_sequence=["#c33764"])
        fig.update_traces(mode="lines+markers+text", text=[fmt_jt(v) for v in monthly["Line_Total"]], textposition="top center",
                          textfont=dict(size=15, color="#7c3aed"), line=dict(width=4),
                          marker=dict(size=11, color="#c33764", line=dict(width=2, color="white")),
                          hovertemplate="%{x|%B %Y}<br>Rp %{y:,.0f}<extra></extra>")
        fig.update_xaxes(dtick="M1", tickformat="%b %Y", tickangle=-30)
        ymin, ymax = monthly["Line_Total"].min(), monthly["Line_Total"].max()
        pad = max((ymax - ymin) * 0.2, ymax * 0.03)
        fig.update_yaxes(range=[max(ymin - pad, 0), ymax + pad])
        show_big(fig, 460)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🏆 10 Produk Penjualan Tertinggi")
            top_p = dff.groupby("Product_Name")["Line_Total"].sum().nlargest(10).reset_index()
            fig = px.bar(top_p, x="Line_Total", y="Product_Name", orientation="h", color="Product_Name", color_discrete_sequence=PALETTE,
                         text=[fmt_jt(v) for v in top_p["Line_Total"]], labels={"Line_Total": "Penjualan (Rp)", "Product_Name": ""})
            fig.update_traces(textposition="outside", textfont=dict(size=16), cliponaxis=False)
            fig.update_layout(showlegend=False, yaxis=dict(autorange="reversed"))
            fig.update_xaxes(range=[0, top_p["Line_Total"].max() * 1.32])
            show_big(fig, 560)
        with col2:
            st.markdown("#### 📦 Distribusi Penjualan per Kategori")
            cat_data = dff.groupby("Category")["Line_Total"].sum().reset_index().sort_values("Line_Total", ascending=False)
            fig = px.bar(cat_data, x="Category", y="Line_Total", color="Category", color_discrete_sequence=PALETTE,
                         text=[fmt_jt(v) for v in cat_data["Line_Total"]], labels={"Line_Total": "Penjualan (Rp)", "Category": "Kategori"})
            fig.update_traces(textposition="outside", textfont=dict(size=16), cliponaxis=False)
            fig.update_layout(showlegend=False)
            fig.update_xaxes(tickangle=-35)
            fig.update_yaxes(range=[0, cat_data["Line_Total"].max() * 1.18])
            show_big(fig, 560)
        best = monthly.loc[monthly["Line_Total"].idxmax()]
        insight(f"<b>Informasi utama untuk manajemen:</b> penjualan puncak pada <b>{best['Bulan']:%B %Y}</b> ({rp(best['Line_Total'])}); "
                f"produk terlaris <b>{top_p.iloc[0]['Product_Name']}</b>; kategori terbesar <b>{cat_data.iloc[0]['Category']}</b> "
                f"({pct1(cat_data.iloc[0]['Line_Total']/cat_data['Line_Total'].sum()*100)} penjualan). Fokuskan promosi & stok pada bulan, produk, dan kategori ini.")

    # ---- 9c Apriori ----
    with t_apr:
        pa, pb = st.columns(2)
        ms = pa.slider("Min Support (dashboard)", 0.001, 0.100, 0.010, 0.001, format="%.3f", key="dash_ms")
        mc = pb.slider("Min Confidence (dashboard)", 0.05, 0.95, 0.30, 0.05, format="%.2f", key="dash_mc")
        freq, rules, n_trans = run_apriori_cached(dff, ms, mc)
        if rules is None or len(rules) == 0:
            st.info("Tidak ada aturan pada parameter/filter ini. Turunkan support/confidence atau longgarkan filter.")
        else:
            good = rules[rules["lift"] > 1]
            top = (good if len(good) else rules).head(10).copy()
            k1, k2, k3 = st.columns(3)
            k1.markdown(kpi_card("k1", "Aturan Asosiasi", fmt_int(len(rules)), "memenuhi parameter"), unsafe_allow_html=True)
            k2.markdown(kpi_card("k3", "Lift Tertinggi", f"{rules['lift'].max():.2f}", "aturan utama"), unsafe_allow_html=True)
            k3.markdown(kpi_card("k5", "Confidence Tertinggi", pct1(rules["confidence"].max() * 100), ""), unsafe_allow_html=True)
            st.markdown("#### Aturan Asosiasi Utama (support, confidence, lift)")
            tv = top[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
            tv.columns = ["Jika membeli", "Maka membeli", "Support", "Confidence", "Lift"]
            show_table(tv, formats={"Support": lambda v: f"{v*100:.2f}%".replace(".", ","), "Confidence": lambda v: f"{v*100:.1f}%".replace(".", ","),
                                    "Lift": lambda v: f"{v:.3f}"}, bars={"Lift": "#8b5cf6", "Confidence": "#c33764"})
            top["Kombinasi"] = top["antecedents"] + "  →  " + top["consequents"]
            fig = px.bar(top, x="lift", y="Kombinasi", orientation="h", color="lift", color_continuous_scale="Magma",
                         text=top["lift"].round(2), title="Visualisasi Kombinasi Produk (berdasarkan Lift)", labels={"lift": "Lift"})
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
            show_big(fig, 520)
            fig_sc = px.scatter(rules, x="support", y="confidence", size="lift", color="lift", color_continuous_scale="Plasma", size_max=28,
                                hover_data={"antecedents": True, "consequents": True}, title="Support vs Confidence (warna = Lift)")
            show(fig_sc, 460)
            t = top.iloc[0]
            insight(f"<b>Informasi utama:</b> aturan terkuat <b>{t['antecedents']} → {t['consequents']}</b> (lift {t['lift']:.2f}, confidence "
                    f"{t['confidence']*100:.1f}%). Jadikan dasar bundling dan penempatan rak berdekatan.")

    # ---- 9d K-Means ----
    with t_km:
        if seg_profile is None:
            st.info("Pelanggan terlalu sedikit untuk segmentasi.")
        else:
            prof = seg_profile if not seg_filter else seg_profile[seg_profile["Label"].isin(seg_filter)]
            outv = seg_out if not seg_filter else seg_out[seg_out["Label"].isin(seg_filter)]
            if prof.empty:
                st.info("Segmen terpilih tidak ada pada klaster saat ini.")
            else:
                st.caption(f"Segmentasi K-Means dengan k = {k_dash} (silhouette {seg_sil:.3f}). Ubah k di menu K-Means.")
                pv = prof.copy()
                pv["Nama"] = pv["Label"] + "<br>(C" + pv["Cluster"].astype(str) + ")"
                ka, kb = st.columns(2)
                with ka:
                    fig = px.bar(pv, x="Nama", y="Jumlah_Pelanggan", color="Nama", text="Jumlah_Pelanggan", color_discrete_sequence=PALETTE,
                                 title="Jumlah Pelanggan dalam Setiap Klaster", labels={"Jumlah_Pelanggan": "Pelanggan", "Nama": ""})
                    fig.update_layout(showlegend=False); fig.update_traces(textposition="outside")
                    show_big(fig, 440)
                with kb:
                    fig = px.pie(pv, names="Label", values="Monetary_Total", hole=0.45, color="Label", color_discrete_map=SEGMENT_COLORS,
                                 title="Kontribusi Belanja per Segmen")
                    fig.update_traces(textinfo="percent+label", textfont=dict(size=15))
                    show_big(fig, 440)
                st.markdown("#### Karakteristik Masing-masing Klaster")
                tp = prof.sort_values("Monetary_Total", ascending=False)[["Cluster", "Label", "Jumlah_Pelanggan", "Recency_Mean", "Frequency_Mean", "Monetary_Mean", "Monetary_Total"]].copy()
                tp.columns = ["Klaster", "Segmen", "Pelanggan", "Recency (hari)", "Frekuensi (×)", "Monetary Rata-rata", "Monetary Total"]
                show_table(tp, formats={"Klaster": lambda v: f"C{int(v)}", "Pelanggan": fmt_int, "Recency (hari)": n1, "Frekuensi (×)": n1,
                                        "Monetary Rata-rata": rp, "Monetary Total": rp}, badge="Segmen",
                           bars={"Pelanggan": "#7597de", "Monetary Total": "#10b981"})
                st.markdown("#### Visualisasi Segmentasi Pelanggan")
                op = outv.copy(); op["Nama"] = op["Label"] + " (C" + op["Cluster"].astype(str) + ")"
                fig = px.scatter(op, x="Recency", y="Monetary", color="Nama", size="Frequency", hover_data=["Customer_ID", "Frequency"],
                                 color_discrete_sequence=PALETTE, labels={"Recency": "Recency (hari)", "Monetary": "Monetary (Rp)"},
                                 title="Segmentasi Pelanggan: Recency vs Monetary")
                show_big(fig, 500)
                p1 = priority_segments(seg_profile)[0]
                insight(f"<b>Informasi utama:</b> {p1[0]} = <b>{p1[1]}</b> — {p1[3]}")

    # ---- Bonus integrasi ----
    with t_int:
        if seg_profile is None:
            st.info("Segmentasi belum tersedia.")
        else:
            labels_avail = sorted(dff["Segmen"].unique())
            pick = st.selectbox("Pilih segmen pelanggan", labels_avail)
            sub = dff[dff["Segmen"] == pick]
            st.caption(f"{fmt_int(sub['Customer_ID'].nunique())} pelanggan · {fmt_int(sub['Transaction_ID'].nunique())} transaksi pada segmen ini (sesuai filter di atas).")
            ia, ib = st.columns(2)
            with ia:
                tp = sub.groupby("Product_Name")["Line_Total"].sum().nlargest(8).reset_index()
                fig = px.bar(tp, x="Line_Total", y="Product_Name", orientation="h", color="Product_Name", color_discrete_sequence=PALETTE,
                             text=[fmt_jt(v) for v in tp["Line_Total"]], title=f"Produk Teratas — {pick}", labels={"Line_Total": "Penjualan (Rp)", "Product_Name": ""})
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_layout(showlegend=False, yaxis=dict(autorange="reversed"))
                fig.update_xaxes(range=[0, tp["Line_Total"].max() * 1.3])
                show_big(fig, 480)
            with ib:
                pairs = top_pairs_cached(sub, 8)
                if pairs:
                    pdf = pd.DataFrame([{"Kombinasi": f"{a} + {b}", "Frekuensi": c} for (a, b), c in pairs])
                    fig = px.bar(pdf, x="Frekuensi", y="Kombinasi", orientation="h", color="Frekuensi", color_continuous_scale="Magma",
                                 text="Frekuensi", title=f"Kombinasi Produk Relevan — {pick}")
                    fig.update_traces(textposition="outside", cliponaxis=False)
                    fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
                    show_big(fig, 480)
                    insight(f"<b>Insight:</b> segmen <b>{pick}</b> paling sering membeli bersama <b>{pairs[0][0][0]}</b> + <b>{pairs[0][0][1]}</b> "
                            f"({fmt_int(pairs[0][1])} transaksi). Tawarkan bundling ini khusus untuk segmen tersebut.")
                else:
                    st.info("Tidak ada transaksi multi-produk pada segmen ini.")

    st.markdown('<div class="alert-info">📸 <b>Dokumentasi:</b> ambil tangkapan layar tiap tab di atas dan setiap kotak “Informasi utama” sebagai bukti luaran dashboard.</div>',
                unsafe_allow_html=True)

# =============================================================================
# PAGE: REKOMENDASI  (Tahap 6 — Soal 10)
# =============================================================================
def get_apriori_for_reco(df):
    res = st.session_state.apriori_result
    if res is not None:
        return res["rules"], res["n_trans"]
    freq, rules, n, ms, mc = default_apriori(df)
    return rules, n

def render_recommend():
    page_header("💡 Rekomendasi Keputusan Bisnis", "Tahap 6 · Soal 10 — Rekomendasi preskriptif berbasis hasil analisis")
    if st.session_state.df_clean is None:
        need_clean(); return
    df = st.session_state.df_clean
    rules, n_trans = get_apriori_for_reco(df)
    seg = segment_customers(df, current_k())
    total_sales = df["Line_Total"].sum()
    avg_basket = df.drop_duplicates("Transaction_ID")["Basket_Size"].mean()

    st.markdown(f"""<div class="alert-info"><b>Ringkasan Data_Clean:</b> {fmt_int(len(df))} baris · {fmt_int(df['Transaction_ID'].nunique())} transaksi ·
    {fmt_int(df['Customer_ID'].nunique())} pelanggan · penjualan {rp(total_sales)} · basket size rata-rata {avg_basket:.2f}.</div>""", unsafe_allow_html=True)

    good = None
    if rules is not None and len(rules):
        good = rules[rules["lift"] > 1]
        good = good if len(good) else rules

    # ---- 10a ----
    st.subheader("a. Strategi Penempatan Produk (berdasarkan Apriori)")
    if good is not None:
        sh = shelf_table(good, 6)
        show_table(sh)
        top3 = good.head(3)
        for i, (_, r) in enumerate(top3.iterrows(), 1):
            st.markdown(f"- **Bundling #{i}:** {r['antecedents']} + {r['consequents']} (lift {r['lift']:.2f}, confidence {r['confidence']*100:.1f}%)")
    else:
        st.info("Belum ada aturan asosiasi. Jalankan menu Apriori atau turunkan min support.")

    # ---- 10b ----
    st.subheader("b. Strategi Retensi Pelanggan (berdasarkan K-Means)")
    if seg is not None:
        out, profile, sil, _, k_used = seg
        tot_m = profile["Monetary_Total"].sum()
        rows = []
        for _, r in profile.sort_values("Monetary_Total", ascending=False).iterrows():
            rows.append({"Segmen": r["Label"], "Klaster": f"C{int(r['Cluster'])}", "Pelanggan": int(r["Jumlah_Pelanggan"]),
                         "% Belanja": r["Monetary_Total"] / tot_m * 100,
                         "Tindakan": STRATEGIES[r["Label"]]["aktif"], "KPI": STRATEGIES[r["Label"]]["kpi"]})
        show_table(pd.DataFrame(rows), formats={"Pelanggan": fmt_int, "% Belanja": pct1}, badge="Segmen", bars={"% Belanja": "#10b981"})
        st.caption(f"Segmentasi memakai k = {k_used} (ubah di menu K-Means).")
    else:
        st.info("Pelanggan terlalu sedikit untuk segmentasi.")

    # ---- 10c ----
    st.subheader("c. Prioritas Tindakan (urgensi, potensi manfaat, biaya, kemudahan)")
    st.caption("Skor 1–5 (5 = terbaik: paling mendesak / paling besar manfaat / paling murah / paling mudah). Skor adalah penilaian analis berdasarkan hasil analisis.")
    pr = pd.DataFrame({
        "Rekomendasi": ["Win-back pelanggan Berisiko Tidak Kembali", "Program loyalty untuk Loyal / High-Value", "Bundling & penempatan rak (Apriori)", "Konversi Non-Member → Member & promo bulan puncak"],
        "Urgensi": [5, 4, 3, 3], "Potensi Manfaat": [4, 5, 4, 3], "Biaya (5 = murah)": [3, 3, 5, 4], "Kemudahan": [4, 3, 5, 4],
    })
    pr["Total Skor"] = pr[["Urgensi", "Potensi Manfaat", "Biaya (5 = murah)", "Kemudahan"]].sum(axis=1)
    pr = pr.sort_values("Total Skor", ascending=False).reset_index(drop=True)
    pr.insert(0, "Peringkat", range(1, len(pr) + 1))
    show_table(pr, formats={"Peringkat": lambda v: f"#{int(v)}"}, bars={"Total Skor": "#8b5cf6"})

    # ---- 10d ----
    st.subheader("d. Estimasi Dampak Bisnis (ESTIMASI berbasis asumsi)")
    st.caption("Geser asumsi di bawah; angka dihitung dari data riil sebagai baseline × asumsi. Ini bukan hasil aktual.")
    a1, a2, a3 = st.columns(3)
    up_bundle = a1.slider("Asumsi kenaikan penjualan produk bundling (%)", 1, 20, 5)
    react = a2.slider("Asumsi reaktivasi pelanggan berisiko (%)", 1, 30, 10)
    ret = a3.slider("Asumsi tambahan belanja segmen Loyal (%)", 1, 20, 5)
    base_b = 0
    if good is not None:
        prods = set()
        for _, r in good.head(3).iterrows():
            prods.update(r["antecedents"].split(", ")); prods.update(r["consequents"].split(", "))
        base_b = df.loc[df["Product_Name"].isin(prods), "Line_Total"].sum()
    base_r = base_l = 0
    if seg is not None:
        base_r = profile.loc[profile["Label"] == "Berisiko Tidak Kembali", "Monetary_Total"].sum()
        base_l = profile.loc[profile["Label"] == "Loyal / High-Value", "Monetary_Total"].sum()
    imp = pd.DataFrame({
        "Rekomendasi": ["Bundling 3 aturan teratas", "Win-back pelanggan berisiko", "Loyalty segmen Loyal"],
        "Baseline dari data (Rp)": [base_b, base_r, base_l],
        "Asumsi": [f"naik {up_bundle}%", f"{react}% pelanggan kembali", f"belanja naik {ret}%"],
        "Estimasi Dampak (Rp)": [base_b * up_bundle / 100, base_r * react / 100, base_l * ret / 100],
        "Indikator Keberhasilan": ["Penjualan produk bundling & basket size", "Reactivation rate", "Retention rate & AOV"],
        "Cara Mengukur": ["Bandingkan 30 hari sebelum vs sesudah", "Pelanggan berisiko yang belanja lagi dalam 60 hari", "A/B test member loyalty vs kontrol"],
    })
    show_table(imp, formats={"Baseline dari data (Rp)": rp, "Estimasi Dampak (Rp)": rp}, bars={"Estimasi Dampak (Rp)": "#10b981"})
    st.markdown(f'<div class="alert-warn">Total estimasi (jika semua asumsi tercapai): <b>{rp(imp["Estimasi Dampak (Rp)"].sum())}</b> — '
                f'≈ {pct1(imp["Estimasi Dampak (Rp)"].sum()/max(total_sales,1)*100)} dari total penjualan periode data. Ini <b>estimasi</b>, bukan dampak aktual.</div>',
                unsafe_allow_html=True)

    # ---- 10e ----
    st.subheader("e. Rencana Implementasi")
    plan = pd.DataFrame({
        "Jangka": ["Pendek (0–30 hari)", "Pendek (0–30 hari)", "Pendek (0–30 hari)", "Menengah (1–3 bulan)", "Menengah (3–12 bulan)"],
        "Tindakan": ["Susun paket bundling dari 3 aturan teratas & ubah penempatan rak", "Kirim voucher win-back ke segmen Berisiko",
                     "Pelatihan kasir untuk menawarkan produk pasangan", "Luncurkan program loyalty bertingkat; personalisasi promo via WhatsApp",
                     "Otomatisasi dashboard BI & pembaruan segmentasi bulanan; perbaiki validasi input data"],
        "Penanggung jawab": ["Manajer toko", "Tim pemasaran", "Supervisor kasir", "Tim pemasaran", "Tim IT / analis data"],
        "Output": ["Rak & paket siap", "Daftar & kampanye terkirim", "SOP penawaran", "Program berjalan", "Dashboard rutin & data lebih bersih"],
    })
    show_table(plan)

    # ---- 10f ----
    st.subheader("f. Evaluasi — KPI Keberhasilan")
    kpi = pd.DataFrame({
        "KPI": ["Basket size", "Penjualan produk bundling", "Retention rate Loyal", "Reactivation rate Berisiko", "Repeat purchase rate pelanggan baru", "Proporsi Member"],
        "Baseline (data saat ini)": [f"{avg_basket:.2f} produk/transaksi", rp(base_b), "—  ukur awal bulan", "0% (belum ada kampanye)", "—  ukur awal bulan", "lihat EDA"],
        "Target": ["≥ +10%", "≥ +5% / bulan", "≥ 85%", "≥ 10%", "≥ 20% dalam 30 hari", "naik bertahap"],
        "Frekuensi Evaluasi": ["Bulanan", "Bulanan", "Triwulanan", "Bulanan", "Bulanan", "Triwulanan"],
    })
    show_table(kpi)

    st.markdown("""
    <div class="reco-card" style="border-top-color:#ffb347;">
        <h4>⚠️ Hasil Data vs Estimasi vs Dampak Aktual</h4>
        <ul>
            <li><b>Hasil dari data (fakta):</b> penjualan, jumlah transaksi, aturan Apriori (support/confidence/lift), profil klaster RFM.</li>
            <li><b>Estimasi berdasarkan asumsi:</b> angka kenaikan %, reaktivasi, retention pada tabel 10d — target hipotesis.</li>
            <li><b>Dampak aktual:</b> baru bisa dinyatakan <u>setelah implementasi</u> dan pengukuran (A/B test atau before-after).</li>
        </ul>
    </div>""", unsafe_allow_html=True)

# =============================================================================
# PAGE: LAPORAN & LUARAN
# =============================================================================
def build_report_md():
    raw, clean = st.session_state.df_raw, st.session_state.df_clean
    rules, n_trans = get_apriori_for_reco(clean)
    seg = segment_customers(clean, current_k())
    L = [f"# Laporan Analisis — Integrasi Data Mining & Business Intelligence",
         f"**Nama:** {IDENTITAS['Nama']}  |  **NPM:** {IDENTITAS['NPM']}  |  **Mata Kuliah:** {IDENTITAS['Mata Kuliah']}", "",
         "## 1. Business Understanding",
         "Perusahaan ritel memiliki data transaksi besar yang baru dipakai untuk laporan rutin. Analisis ini mencari pola produk yang dibeli bersama, "
         "segmentasi pelanggan, dan rekomendasi tindakan, setelah data dibersihkan.", "",
         "## 2. EDA",
         f"- Data raw: {fmt_int(len(raw))} baris × {raw.shape[1]} kolom, {raw['Transaction_Date'].min():%d-%m-%Y} s/d {raw['Transaction_Date'].max():%d-%m-%Y}",
         f"- Duplikat: {fmt_int(raw.duplicated().sum())}; sel missing: {fmt_int(raw.isna().sum().sum())}", "",
         "## 3. Preprocessing",
         f"- Data_Clean: {fmt_int(len(clean))} baris (dibuang {fmt_int(len(raw)-len(clean))})",
         f"- Total penjualan: {rp(clean['Line_Total'].sum())}; transaksi: {fmt_int(clean['Transaction_ID'].nunique())}; pelanggan: {fmt_int(clean['Customer_ID'].nunique())}"]
    for r in st.session_state.prep_log:
        L.append(f"  - Tahap {r['Tahap']}: {r['Tindakan']} — {r['Alasan / Justifikasi']} (terdampak {fmt_int(r['Baris Terdampak'])}, dibuang {fmt_int(r['Baris Dibuang'])})")
    L += ["", "## 4. Apriori"]
    if rules is not None and len(rules):
        g = rules[rules["lift"] > 1]; g = (g if len(g) else rules).head(5)
        for _, r in g.iterrows():
            L.append(f"- {r['antecedents']} → {r['consequents']}: support {r['support']*100:.2f}%, confidence {r['confidence']*100:.1f}%, lift {r['lift']:.2f}")
    else:
        L.append("- Tidak ada aturan pada parameter default.")
    L += ["", "## 5. K-Means"]
    if seg is not None:
        _, prof, sil, _, k = seg
        L.append(f"- k = {k}, silhouette = {sil:.3f}")
        for _, r in prof.sort_values("Monetary_Total", ascending=False).iterrows():
            L.append(f"- {r['Label']} (C{int(r['Cluster'])}): {fmt_int(r['Jumlah_Pelanggan'])} pelanggan, R={n1(r['Recency_Mean'])} hari, F={n1(r['Frequency_Mean'])}×, M={rp(r['Monetary_Mean'])}")
        L += ["", "## 6. Rekomendasi"]
        for no, lab, c, why in priority_segments(prof):
            L.append(f"- {no}: {lab} — {why}")
    L += ["- Terapkan bundling & penempatan rak berdasarkan aturan Apriori teratas.",
          "- Jalankan win-back dan loyalty sesuai segmen.",
          "- Catatan: angka kenaikan pada rekomendasi adalah ESTIMASI berbasis asumsi; dampak aktual diukur setelah implementasi."]
    return "\n".join(L)

def render_laporan():
    page_header("📑 Laporan & Luaran", "Bagian E — Luaran yang dikumpulkan: laporan, dataset, kode, dashboard, dokumentasi, presentasi")
    st.subheader("✅ Checklist Luaran vs Sistem")
    chk = pd.DataFrame({
        "No": [1, 2, 3, 4, 5, 6],
        "Luaran": ["Laporan analisis", "Dataset hasil preprocessing + dokumentasi", "Notebook / kode program", "Dashboard BI", "Dokumentasi visualisasi", "Presentasi hasil"],
        "Tersedia di Sistem": ["Menu ini → unduh laporan (.md)", "Menu Preprocessing → unduh Data_Clean & dokumentasi", "File app.py (seluruh proses EDA–K-Means)",
                               "Menu Dashboard BI", "Daftar di bawah + tangkapan layar", "Ringkasan temuan di bawah"],
        "Status": ["✅" if st.session_state.df_clean is not None else "⏳ jalankan Preprocessing",
                   "✅" if st.session_state.df_clean is not None else "⏳ jalankan Preprocessing", "✅", "✅", "✅", "✅" if st.session_state.df_clean is not None else "⏳"],
    })
    show_table(chk)

    st.subheader("🖼️ Dokumentasi Visualisasi")
    viz = pd.DataFrame({
        "Menu": ["EDA", "EDA", "EDA", "Preprocessing", "Apriori", "Apriori", "Apriori", "K-Means", "K-Means", "K-Means", "Dashboard", "Dashboard"],
        "Visualisasi": ["Boxplot Quantity & Unit_Price", "Top 10 pasangan produk", "Tren penjualan bulanan & kontribusi kategori", "Distribusi Basket_Size & sebelum–sesudah",
                        "Scatter support–confidence", "Heatmap lift", "Network graph aturan", "Elbow & Silhouette", "Scatter klaster", "Jumlah pelanggan & monetary per klaster",
                        "KPI + tren + top produk + kategori", "Tab Apriori, K-Means, Integrasi"],
        "Tujuan": ["Mendeteksi outlier", "Menjawab Q1", "Menjawab Q3", "Bukti kualitas data & fitur", "Melihat sebaran aturan", "Hubungan antar produk", "Arah hubungan produk",
                   "Memilih k", "Melihat pemisahan segmen", "Ukuran & nilai segmen", "Informasi eksekutif", "Insight terintegrasi"],
    })
    show_table(viz)

    if st.session_state.df_clean is None:
        st.warning("⚠️ Jalankan Preprocessing agar laporan dan unduhan tersedia."); return
    clean = st.session_state.df_clean
    st.subheader("⬇ Unduhan")
    md = build_report_md()
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ Laporan Analisis (.md)", md.encode("utf-8"), "laporan_analisis.md", "text/markdown", use_container_width=True)
    d2.download_button("⬇ Data_Clean (Excel)", df_to_xlsx_bytes(clean), "data_clean.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    d3.download_button("⬇ Dokumentasi Preprocessing (CSV)", df_to_csv_bytes(pd.DataFrame(st.session_state.prep_log)), "dokumentasi_preprocessing.csv", "text/csv", use_container_width=True)
    with st.expander("👀 Pratinjau Laporan Analisis", expanded=False):
        st.markdown(md)

    st.subheader("🎤 Ringkasan Temuan & Rekomendasi untuk Presentasi")
    rules, _ = get_apriori_for_reco(clean)
    seg = segment_customers(clean, current_k())
    pts = [f"Data raw {fmt_int(len(st.session_state.df_raw))} baris dibersihkan menjadi {fmt_int(len(clean))} baris siap analisis."]
    if rules is not None and len(rules):
        g = rules[rules["lift"] > 1]; r0 = (g if len(g) else rules).iloc[0]
        pts.append(f"Pola terkuat: {r0['antecedents']} → {r0['consequents']} (lift {r0['lift']:.2f}) → dasar bundling & penataan rak.")
    if seg is not None:
        p1 = priority_segments(seg[1])[0]
        pts.append(f"Segmen prioritas: {p1[1]} — {p1[3]}")
    pts.append("Rekomendasi: bundling, loyalty, win-back; dampak masih ESTIMASI sampai diukur setelah implementasi.")
    for p in pts:
        insight(p)

# =============================================================================
# MAIN
# =============================================================================
def main():
    if st.session_state.stage == "cover":
        render_fixed_logo(cover=True)
        inject_stage_css("cover")
        render_cover()
        return
    render_fixed_logo(cover=False)
    if st.session_state.stage == "login" or not st.session_state.logged_in:
        inject_stage_css("login")
        render_login()
        return

    page = render_sidebar()
    routes = {P_UPLOAD: render_upload, P_BU: render_business, P_EDA: render_eda, P_PREP: render_preprocessing,
              P_APR: render_apriori, P_KM: render_kmeans, P_DASH: render_dashboard, P_REC: render_recommend, P_LAP: render_laporan}
    routes[page]()

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; padding:26px; color:#64748b; font-size:14px;
                background: linear-gradient(135deg,#fdf2f8,#ede9fe);
                border-radius:16px; margin-top:18px; border: 1px solid #fbcfe8;">
        <b style="color:#7c3aed; font-size:16px;">BI Analytics System</b> —
        {JENIS_UJIAN} Integrasi Data Mining &amp; Business Intelligence<br>
        <span style="color:#334155; font-size:15px; font-weight:600;">
            {IDENTITAS['Nama']} · NPM {IDENTITAS['NPM']} · {IDENTITAS['Mata Kuliah']}
        </span><br>
        Built with ❤️ using <b>Streamlit</b> · <b>pandas</b> · <b>mlxtend</b> · <b>scikit-learn</b> · <b>Plotly</b><br>
        <span style="color:#94a3b8;">© 2026 · Demo Login: admin / admin123</span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()