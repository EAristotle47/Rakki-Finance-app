import os
import io
import json
import re
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from datetime import datetime

# -----------------------------
# Basic page setup
# -----------------------------
st.set_page_config(
    page_title="Rakki Finance App",
    page_icon="💰",
    layout="wide",
)

HIDE_DEFAULT_STYLES = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .topbar {position: sticky; top: 0; z-index: 999; background: white; border-bottom: 1px solid #eee;}
        .topbar-inner {max-width: 1200px; margin: 0 auto; padding: 0.8rem 1rem; display: flex; gap: 1rem; align-items: center;}
        .brand {font-weight: 700; font-size: 1.1rem;}
        .brand-emoji {margin-right: .3rem}
        .pills {display:flex; gap: .5rem; flex-wrap: wrap}
        .pill {background:#f5f7fb; border:1px solid #e9edf5; padding: .25rem .6rem; border-radius: 999px; font-size: .85rem}
        .footerbar {text-align:center; color:#6b7280; font-size:.85rem; margin-top:2rem; padding:2rem 0; border-top:1px solid #eee}
        .muted {color:#6b7280}
        .metric-grid {display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 1rem}
        @media (max-width: 900px){ .metric-grid{grid-template-columns: repeat(2, minmax(0,1fr));} }
    </style>
"""
st.markdown(HIDE_DEFAULT_STYLES, unsafe_allow_html=True)

st.markdown(
    """
    <div class="topbar">
      <div class="topbar-inner">
        <div class="brand"><span class="brand-emoji">💰</span>Rakki Finance</div>
        <div class="pills">
          <span class="pill">Upload CSV / Excel / Google Sheets</span>
          <span class="pill">Auto-categorize</span>
          <span class="pill">Shareable</span>
          <span class="pill">Download Reports</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Persistent category storage
# -----------------------------
CATEGORIES_FILE = "categories.json"
DEFAULT_CATEGORIES = {"Uncategorized": []}

if "categories" not in st.session_state:
    if os.path.exists(CATEGORIES_FILE):
        try:
            with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
                st.session_state.categories = json.load(f)
        except Exception:
            st.session_state.categories = DEFAULT_CATEGORIES.copy()
    else:
        st.session_state.categories = DEFAULT_CATEGORIES.copy()

def save_categories_to_disk():
    with open(CATEGORIES_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.categories, f, ensure_ascii=False, indent=2)

# -----------------------------
# Utility functions
# -----------------------------
def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, regex=True)].copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def _infer_and_clean_amount(series: pd.Series) -> pd.Series:
    def parse(x):
        if pd.isna(x): return np.nan
        if isinstance(x, (int, float, np.number)): return float(x)
        s = str(x).strip()
        if s == "": return np.nan
        neg = False
        if s.startswith("(") and s.endswith(")"):
            neg = True
            s = s[1:-1]
        s = re.sub(r"[^0-9.\-]", "", s.replace(",", ""))
        try:
            val = float(s)
            return -val if neg else val
        except Exception:
            return np.nan
    return series.apply(parse)

def _try_parse_date(series: pd.Series) -> pd.Series:
    for fmt in ("%d %b %Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return pd.to_datetime(series, format=fmt, errors="raise")
        except Exception:
            pass
    return pd.to_datetime(series, errors="coerce")

def load_from_uploaded(file) -> pd.DataFrame | None:
    try:
        name = file.name.lower()
        if name.endswith(".csv"): df = pd.read_csv(file, index_col=False)
        elif name.endswith((".xlsx", ".xls", ".ods")): df = pd.read_excel(file)
        else: st.error("Unsupported file type."); return None
        return _clean_columns(df)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def load_from_google_sheets(url: str) -> pd.DataFrame | None:
    try:
        m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
        if not m: st.error("Invalid Google Sheets link."); return None
        sheet_id = m.group(1)
        gid_match = re.search(r"[?&]gid=(\d+)", url)
        gid = gid_match.group(1) if gid_match else "0"
        export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        resp = requests.get(export_url, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        return _clean_columns(df)
    except Exception as e:
        st.error(f"Couldn't fetch Google Sheet: {e}")
        return None

def categorize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    if "Category" not in df.columns: df["Category"] = "Uncategorized"
    for category, keywords in st.session_state.categories.items():
        if category == "Uncategorized" or not keywords: continue
        lowered = [str(k).lower().strip() for k in keywords if str(k).strip()]
        if "Details" in df.columns:
            mask = df["Details"].astype(str).str.lower().str.strip().isin(lowered)
            df.loc[mask, "Category"] = category
    return df

def add_keyword_to_category(category: str, keyword: str) -> bool:
    keyword = str(keyword or "").strip()
    if not keyword: return False
    if category not in st.session_state.categories: st.session_state.categories[category] = []
    if keyword not in st.session_state.categories[category]:
        st.session_state.categories[category].append(keyword)
        save_categories_to_disk()
        return True
    return False

# -----------------------------
# Sidebar: Data input & options
# -----------------------------
with st.sidebar:
    st.subheader("Data Input")
    uploaded = st.file_uploader("Upload transactions (CSV / Excel)", type=["csv", "xlsx", "xls", "ods"])
    gs_url = st.text_input("Or paste a public Google Sheets link")
    cols = st.columns([1,1])
    with cols[0]: use_sample = st.button("Load sample data")
    with cols[1]: reset_state = st.button("Reset categories")

    st.markdown("---")
    st.subheader("Filters")
    start_date = st.date_input("Start date", value=None)
    end_date = st.date_input("End date", value=None)
    text_query = st.text_input("Search 'Details'")

    st.markdown("---")
    st.subheader("Export")
    exp_fmt = st.selectbox("Export format", ["CSV", "Excel"], index=0)

if reset_state:
    st.session_state.categories = DEFAULT_CATEGORIES.copy()
    save_categories_to_disk()
    st.success("Categories reset.")

# -----------------------------
# Load and preprocess data
# -----------------------------
raw_df = None
if uploaded: raw_df = load_from_uploaded(uploaded)
elif gs_url.strip(): raw_df = load_from_google_sheets(gs_url.strip())
elif use_sample:
    raw_df = pd.DataFrame({"Date": ["12 Jan 2025","13 Jan 2025"], "Details": ["Starbucks","Payroll"], "Debit/Credit": ["Debit","Credit"], "Amount": ["5.50","2500.00"]})

if raw_df is not None:
    df = raw_df.copy()
    if "Amount" in df.columns: df["Amount"] = _infer_and_clean_amount(df["Amount"])
    if "Date" in df.columns: df["Date"] = _try_parse_date(df["Date"]).dt.tz_localize(None)
    if "Debit/Credit" in df.columns:
        dc = df["Debit/Credit"].astype(str).str.lower().str.strip()
        dc = dc.replace({"credits":"credit","cr":"credit","dr":"debit"})
        df["Debit/Credit"] = dc.str.title()
    df = categorize_transactions(df)

    if start_date: df = df[df["Date"] >= pd.to_datetime(start_date)]
    if end_date: df = df[df["Date"] <= pd.to_datetime(end_date)]
    if text_query.strip() and "Details" in df.columns:
       df = df[df["Details"].astype(str).str.contains(text_query, case=False, na=False)]
 
