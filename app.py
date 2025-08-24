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
# Basic page setup (website-y)
# -----------------------------
st.set_page_config(
    page_title="Rakki Finance App",
    page_icon="💰",
    layout="wide",
)

HIDE_DEFAULT_STYLES = """
    <style>
        /* hide streamlit default hamburger & footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        /* website-like top bar */
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
# Data loading utilities
# -----------------------------

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # strip spaces and drop Unnamed columns
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, regex=True)].copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def _infer_and_clean_amount(series: pd.Series) -> pd.Series:
    # Handles strings with commas, parentheses for negatives, currency symbols
    def parse(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return np.nan
        # Handle parentheses negative: (123.45)
        neg = False
        if s.startswith("(") and s.endswith(")"):
            neg = True
            s = s[1:-1]
        # Remove currency symbols and commas
        s = re.sub(r"[^0-9.\-]", "", s.replace(",", ""))
        try:
            val = float(s)
            return -val if neg else val
        except Exception:
            return np.nan
    return series.apply(parse)


def _try_parse_date(series: pd.Series) -> pd.Series:
    # Try common bank formats (e.g., 12 Jan 2025, 2025-01-12, 12/01/2025)
    for fmt in ("%d %b %Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return pd.to_datetime(series, format=fmt, errors="raise")
        except Exception:
            pass
    # Fallback to pandas inference
    return pd.to_datetime(series, errors="coerce")


def load_from_uploaded(file) -> pd.DataFrame | None:
    """Load CSV/Excel from Streamlit uploader."""
    try:
        name = file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(file, index_col=False)
        elif name.endswith((".xlsx", ".xls", ".ods")):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file type. Please upload CSV, XLSX, XLS, or ODS.")
            return None
        return _clean_columns(df)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def load_from_google_sheets(url: str) -> pd.DataFrame | None:
    """Load from a public Google Sheets URL by converting to CSV export."""
    try:
        m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
        if not m:
            st.error("That doesn't look like a valid Google Sheets link.")
            return None
        sheet_id = m.group(1)
        # default to first sheet (gid=0). If user included gid param, preserve it
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


# -----------------------------
# Categorization
# -----------------------------

def categorize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    if "Category" not in df.columns:
        df["Category"] = "Uncategorized"
    for category, keywords in st.session_state.categories.items():
        if category == "Uncategorized" or not keywords:
            continue
        lowered = [str(k).lower().strip() for k in keywords if str(k).strip()]
        if not lowered:
            continue
        # Mark category where Details matches any keyword exactly (case-insensitive)
        if "Details" in df.columns:
            mask = df["Details"].astype(str).str.lower().str.strip().isin(lowered)
            df.loc[mask, "Category"] = category
    return df


def add_keyword_to_category(category: str, keyword: str) -> bool:
    keyword = str(keyword or "").strip()
    if not keyword:
        return False
    if category not in st.session_state.categories:
        st.session_state.categories[category] = []
    if keyword not in st.session_state.categories[category]:
        st.session_state.categories[category].append(keyword)
        save_categories_to_disk()
        return True
    return False


# -----------------------------
# Sidebar — Data input & Options
# -----------------------------
with st.sidebar:
    st.subheader("Data Input")
    uploaded = st.file_uploader("Upload transactions (CSV / Excel)", type=["csv", "xlsx", "xls", "ods"]) 
    gs_url = st.text_input("Or paste a public Google Sheets link")
    cols = st.columns([1,1])
    with cols[0]:
        use_sample = st.button("Load sample data")
    with cols[1]:
        reset_state = st.button("Reset categories")

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
# Load data
# -----------------------------
raw_df = None
if uploaded is not None:
    raw_df = load_from_uploaded(uploaded)
elif gs_url.strip():
    raw_df = load_from_google_sheets(gs_url.strip())
elif use_sample:
    # Simple sample for demo/testing
    raw_df = pd.DataFrame({
        "Date": ["12 Jan 2025", "13 Jan 2025", "15 Jan 2025", "15 Jan 2025"],
        "Details": ["Starbucks", "Grab Ride", "Payroll", "Shopee"],
        "Debit/Credit": ["Debit", "Debit", "Credit", "Debit"],
        "Amount": ["5.50", "12.00", "2500.00", "45.20"],
    })

if raw_df is not None:
    df = raw_df.copy()
    # Clean & normalize essential columns if present
    if "Amount" in df.columns:
        df["Amount"] = _infer_and_clean_amount(df["Amount"]) 
    if "Date" in df.columns:
        df["Date"] = _try_parse_date(df["Date"]).dt.tz_localize(None)
    # Standardize Debit/Credit values
    if "Debit/Credit" in df.columns:
        dc = df["Debit/Credit"].astype(str).str.lower().str.strip()
        dc = dc.replace({"credits": "credit", "cr": "credit", "dr": "debit"})
        df["Debit/Credit"] = dc.str.title()

    # Categorize
    df = categorize_transactions(df)

    # Apply filters
    if start_date:
        df = df[df["Date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["Date"] <= pd.to_datetime(end_date)]
    if text_query.strip():
        if "Details" in df.columns:
            df = df[df["Details"].astype(str).str.contains(text_query, case=False, na=False)]

    # Ensure minimal required columns
    required = ["Date", "Details", "Debit/Credit", "Amount", "Category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Your data is missing required columns: {', '.join(missing)}")
    else:
        # -----------------------------
        # Header & key metrics
        # -----------------------------
        st.markdown("### Overview")
        total_expense = float(df.loc[df["Debit/Credit"] == "Debit", "Amount"].sum())
        total_payments = float(df.loc[df["Debit/Credit"] == "Credit", "Amount"].sum())
        balance = total_payments - total_expense
        colm = st.container()
        with colm:
            st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Expenses", f"{total_expense:,.2f} SGD")
            c2.metric("Total Payments", f"{total_payments:,.2f} SGD")
            c3.metric("Balance", f"{balance:,.2f} SGD")
            c4.metric("Transactions", f"{len(df):,}")
            st.markdown('</div>', unsafe_allow_html=True)

        # -----------------------------
        # Tabs
        # -----------------------------
        tab1, tab2, tab3 = st.tabs(["Expenses", "Payments", "Categories"])

        # Expenses tab
        with tab1:
            st.subheader("Your Expenses")
            debits_df = df[df["Debit/Credit"] == "Debit"].copy()
            st.session_state["debits_df"] = debits_df.copy()

            edited = st.data_editor(
                debits_df[["Date", "Details", "Amount", "Category"]],
                column_config={
                    "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                    "Amount": st.column_config.NumberColumn("Amount", format="%.2f SGD"),
                    "Category": st.column_config.SelectboxColumn(
                        "Category", options=list(st.session_state.categories.keys())
                    ),
                },
                hide_index=True,
                use_container_width=True,
                key="category_editor",
            )
            if st.button("Apply Changes", type="primary"):
                for idx, row in edited.iterrows():
                    new_cat = row["Category"]
                    old_cat = debits_df.at[idx, "Category"]
                    if new_cat != old_cat:
                        debits_df.at[idx, "Category"] = new_cat
                        add_keyword_to_category(new_cat, row["Details"])  # learn keyword
                st.session_state["debits_df"] = debits_df
                st.success("Changes applied & keywords learned.")

            # Summary + charts
            st.markdown("#### Expense Summary")
            category_totals = (
                (st.session_state["debits_df"][["Category", "Amount"]]
                 .groupby("Category")["Amount"].sum().reset_index())
                .sort_values("Amount", ascending=False)
            )
            st.dataframe(
                category_totals,
                column_config={"Amount": st.column_config.NumberColumn("Amount", format="%.2f SGD")},
                hide_index=True,
                use_container_width=True,
            )
            if not category_totals.empty:
                fig = px.pie(category_totals, values="Amount", names="Category", title="Expenses by Category")
                st.plotly_chart(fig, use_container_width=True)
                bar = px.bar(category_totals, x="Category", y="Amount", title="Top Categories", text_auto='.2s')
                st.plotly_chart(bar, use_container_width=True)

        # Payments tab
        with tab2:
            st.subheader("Payments (Credits)")
            credits_df = df[df["Debit/Credit"] == "Credit"].copy()
            total_pay = credits_df["Amount"].sum()
            st.metric("Total Payments", f"{total_pay:,.2f} SGD")
            st.dataframe(
                credits_df[["Date", "Details", "Amount", "Category"]].sort_values("Date"),
                column_config={
                    "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                    "Amount": st.column_config.NumberColumn("Amount", format="%.2f SGD"),
                },
                hide_index=True,
                use_container_width=True,
            )

        # Categories tab
        with tab3:
            st.subheader("Manage Categories & Keywords")
            left, right = st.columns([1, 1])
            with left:
                new_category = st.text_input("New category name")
                if st.button("Add Category"):
                    if new_category and new_category not in st.session_state.categories:
                        st.session_state.categories[new_category] = []
                        save_categories_to_disk()
                        st.success(f"Added '{new_category}'.")
                    else:
                        st.warning("Enter a unique category name.")
                st.markdown("##### Categories")
                cat_df = pd.DataFrame([
                    {"Category": cat, "Keywords": ", ".join(words)}
                    for cat, words in st.session_state.categories.items()
                ])
                st.dataframe(cat_df, hide_index=True, use_container_width=True)

            with right:
                pick = st.selectbox("Choose category to add a keyword", list(st.session_state.categories.keys()))
                kw = st.text_input("Keyword (must match 'Details' exactly to auto-categorize)")
                if st.button("Add Keyword"):
                    if add_keyword_to_category(pick, kw):
                        st.success(f"Added keyword to '{pick}'.")
                    else:
                        st.warning("Nothing added (empty or duplicate).")
                st.markdown("##### Import/Export")
                # Export categories JSON
                cat_bytes = io.BytesIO(json.dumps(st.session_state.categories, ensure_ascii=False, indent=2).encode("utf-8"))
                st.download_button("Download categories.json", data=cat_bytes, file_name="categories.json", mime="application/json")
                # Import categories JSON
                cat_up = st.file_uploader("Upload categories.json", type=["json"], key="catjson")
                if cat_up is not None:
                    try:
                        incoming = json.load(cat_up)
                        if isinstance(incoming, dict):
                            st.session_state.categories = incoming
                            save_categories_to_disk()
                            st.success("Categories imported.")
                        else:
                            st.error("Invalid format. Expected an object of {category: [keywords]}.")
                    except Exception as e:
                        st.error(f"Failed to import: {e}")

        # -----------------------------
        # Export cleaned data
        # -----------------------------
        st.markdown("### Download Cleaned Data")
        cleaned = df.copy()
        # reorder
        cleaned = cleaned[["Date", "Details", "Debit/Credit", "Amount", "Category"]]
        cleaned["Date"] = cleaned["Date"].dt.strftime("%d %b %Y")

        if exp_fmt == "CSV":
            b = io.BytesIO()
            cleaned.to_csv(b, index=False).seek(0)
            st.download_button("Download CSV", data=b, file_name="transactions_cleaned.csv", mime="text/csv")
        else:
            b = io.BytesIO()
            with pd.ExcelWriter(b, engine="xlsxwriter") as writer:
                cleaned.to_excel(writer, index=False, sheet_name="Transactions")
            b.seek(0)
            st.download_button("Download Excel", data=b, file_name="transactions_cleaned.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.markdown(
        """
        ### Welcome to Rakki Finance
        Upload a CSV/Excel file or paste a public Google Sheets link in the sidebar to begin. 
        You can also try sample data. The app cleans data, removes *Unnamed* columns, auto-detects dates & amounts, and lets you manage categories.
        """
    )

st.markdown(
    """
    <div class="footerbar">Built with ❤️ using Streamlit. Share this app by deploying to Streamlit Community Cloud and sending your app link.</div>
    """,
    unsafe_allow_html=True,
)
