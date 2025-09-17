"""
FSL Fixed Income Portfolio — Streamlit App (Improved UI)
------------------------------------------------------

Now presented in a dashboard-style interface with clear cards, charts, and metrics
instead of a raw spreadsheet-style table.

Run:
  streamlit run fsl_bond_portfolio_streamlit_app.py
"""
from __future__ import annotations

import os
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
# Optional Plotly (legacy). Safe import so missing package won't crash.
try:
    import plotly.express as px  # noqa: F401
except Exception:
    px = None
import plotly.express as px

import asyncio
import nest_asyncio
nest_asyncio.apply()
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
from ib_insync import IB, Bond, util
util.patchAsyncio()
import asyncio

# Ensure event loop exists before ib_insync/eventkit import issues
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

util.patchAsyncio()

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="FSL Bond Portfolio", page_icon="📊", layout="wide")

# ----------------------------
# IB CONNECTION
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_ib() -> IB:
    util.patchAsyncio()
    return IB()

ib = get_ib()

# Sidebar
st.sidebar.header("IBKR Connection")
ib_host = st.sidebar.text_input("Host", os.getenv("IB_HOST", "127.0.0.1"))
ib_port = st.sidebar.number_input("Port", value=int(os.getenv("IB_PORT", "7497")))
ib_client = st.sidebar.number_input("Client ID", value=int(os.getenv("IB_CLIENT_ID", "1101")))
connect_btn = st.sidebar.button("Connect")

if connect_btn:
    try:
        ib.connect(ib_host, ib_port, clientId=ib_client, readonly=True, timeout=5)
        st.sidebar.success("Connected to IBKR")
    except Exception as e:
        st.sidebar.error(f"Could not connect: {e}")

# ----------------------------
# MAIN APP
# ----------------------------
st.title("📈 Firstline Securities Ltd — Bond Portfolio")
st.caption("A clearer dashboard view of your fixed income portfolio with IBKR live data.")

# Upload
uploaded = st.file_uploader("Upload portfolio file (Excel/CSV)", type=["xlsx", "xls", "csv"])
if uploaded is not None:
    if uploaded.name.lower().endswith("csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
else:
    st.info("Upload a file to get started.")
    df = pd.DataFrame(columns=["Issuer","Security","ISIN","Units","Purchase Px","Close Bid","Coupon","Maturity Date"])

# Simple portfolio calculations
if not df.empty:
    df["Invested"] = df["Units"] * df["Purchase Px"]
    df["Market Value"] = df["Units"] * df["Close Bid"]
    df["P/L $"] = df["Market Value"] - df["Invested"]
    df["P/L %"] = (df["P/L $"] / df["Invested"]) * 100

    # Top-level KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Invested", f"${df['Invested'].sum():,.2f}")
    with c2:
        st.metric("Market Value", f"${df['Market Value'].sum():,.2f}")
    with c3:
        st.metric("Total P/L", f"${df['P/L $'].sum():,.2f}", f"{df['P/L %'].mean():.2f}%")
    with c4:
        st.metric("# Holdings", len(df))

    st.divider()

    # Portfolio composition
    st.subheader("Portfolio Breakdown")
    if "Issuer" in df.columns:
        fig = px.pie(df, names="Issuer", values="Market Value", title="Market Value by Issuer")
        st.plotly_chart(fig, use_container_width=True)

    # Table view
    st.subheader("Holdings")
    show_cols = ["Issuer","Security","ISIN","Units","Purchase Px","Close Bid","Invested","Market Value","P/L $","P/L %"]
    # === Nicer UI ===

def safe_weighted_avg(values, weights, default=np.nan):
    v = pd.to_numeric(values, errors='coerce')
    w = pd.to_numeric(weights, errors='coerce')
    if v.empty or w.empty:
        return default
    w = w.fillna(0)
    total_w = float(w.sum())
    if total_w <= 0:
        return default
    v = v.fillna(0)
    try:
        return float(np.average(v, weights=w))
    except ZeroDivisionError:
        return default

import altair as alt

# Header KPIs
# Safe numeric helpers for KPIs
def _num_series(df, col):
    s = df[col] if col in df.columns else pd.Series(dtype=float)
    return pd.to_numeric(s, errors='coerce')

def fmt_money(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"${x:,.2f}"

def fmt_pct(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}%"

mv_s = _num_series(out_df, 'Market_Value')
pl_s = _num_series(out_df, 'Dollar_PL')
ytm_s = _num_series(out_df, 'BidYTM')

colK1, colK2, colK3, colK4 = st.columns(4)
colK1.metric("Total Market Value", fmt_money(float(mv_s.sum()) if not mv_s.empty else float('nan')))
colK2.metric("Total $ P/L", fmt_money(float(pl_s.sum()) if not pl_s.empty else float('nan')))
colK3.metric("Avg Bid YTM", fmt_pct(float(ytm_s.mean()) if not ytm_s.empty else float('nan')))
wd = safe_weighted_avg(out_df['Duration'] if 'Duration' in out_df.columns else pd.Series(dtype=float), mv_s)
colK4.metric("Weighted Duration", "—" if pd.isna(wd) else f"{wd:.2f}y")

# Filters
st.subheader("Filters")
fc1, fc2, fc3 = st.columns(3)
all_issuers = sorted([x for x in out_df['Issuer'].dropna().unique()])
all_accounts = sorted([x for x in out_df['Account'].dropna().unique()])
all_allocs = sorted([x for x in out_df['Allocation'].dropna().unique()])
sel_issuer = fc1.multiselect("Issuer", all_issuers)
sel_account = fc2.multiselect("Account", all_accounts)
sel_alloc = fc3.multiselect("Allocation", all_allocs)

mask = pd.Series(True, index=out_df.index)
if sel_issuer:
    mask &= out_df['Issuer'].isin(sel_issuer)
if sel_account:
    mask &= out_df['Account'].isin(sel_account)
if sel_alloc:
    mask &= out_df['Allocation'].isin(sel_alloc)
flt = out_df[mask].copy()

# Helper to make selection labels consistently
def build_options_map(df: pd.DataFrame):
    labels = (df['Issuer'].fillna('') + ' — ' + df['Security'].fillna('') + ' (' + df['ISIN'].fillna('') + ')')
    options = labels.tolist()
    idx_map = {opt: i for i, opt in zip(df.index.tolist(), options)}
    # Correction: build mapping from label->row index
    idx_map = {label: idx for label, idx in zip(options, df.index.tolist())}
    return options, idx_map

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Holdings", "Risk", "Cashflows"])

with tab1:
    st.caption("Portfolio overview & breakdowns")

    # Allocation by account
    alloc_df = flt.groupby('Account', dropna=False)['Market_Value'].sum().reset_index()
    if not alloc_df.empty:
        alloc_df = alloc_df.sort_values('Market_Value', ascending=False)
        chart1 = (
            alt.Chart(alloc_df)
            .mark_bar()
            .encode(
                x=alt.X('Market_Value:Q', title='Market Value ($)'),
                y=alt.Y('Account:N', sort='-x'),
                tooltip=['Account','Market_Value']
            )
            .properties(height=300)
        )
        st.altair_chart(chart1, use_container_width=True)
    else:
        st.info("No holdings to plot by account.")

    # Maturity ladder by year
    mat = flt.dropna(subset=['Maturity_Date']).copy()
    if not mat.empty:
        mat['Year'] = pd.to_datetime(mat['Maturity_Date']).dt.year
        ladder = mat.groupby('Year')['Market_Value'].sum().reset_index()
        chart2 = (
            alt.Chart(ladder)
            .mark_bar()
            .encode(
                x=alt.X('Year:O'),
                y=alt.Y('Market_Value:Q', title='Market Value ($)'),
                tooltip=['Year','Market_Value']
            )
            .properties(height=300)
        )
        st.altair_chart(chart2, use_container_width=True)
    else:
        st.info("No maturity data available.")

    # YTM histogram
    ytms = flt['BidYTM'].dropna()
    if not ytms.empty:
        hist = pd.DataFrame({'BidYTM': ytms})
        chart3 = (
            alt.Chart(hist)
            .mark_bar()
            .encode(
                x=alt.X('BidYTM:Q', bin=alt.Bin(maxbins=30), title='Bid YTM (%)'),
                y=alt.Y('count():Q', title='Count')
            )
            .properties(height=300)
        )
        st.altair_chart(chart3, use_container_width=True)
    else:
        st.info("No Bid YTM data to plot.")

with tab2:
    st.caption("Clean holdings table with key columns and quick selection.")
    nice_cols = ['Issuer','Security','ISIN','Account','Allocation','Units','Purchase_Px','Close_Bid','Market_Value','Dollar_PL','Percent_PL','PurchYTM','BidYTM','Duration','DV01','Maturity_Date']
    view = flt[nice_cols].copy()
    view.rename(columns={
        'Purchase_Px':'Purchase Px','Close_Bid':'Bid Px','Market_Value':'Market Value',
        'Dollar_PL':'$ P/L','Percent_PL':'% P/L','PurchYTM':'Purch YTM','BidYTM':'Bid YTM',
        'Maturity_Date':'Maturity','DV01':'DV01 (per 100)'
    }, inplace=True)
    st.dataframe(view, hide_index=True, use_container_width=True)

    st.markdown("### Bond details")
    options, idx_map = build_options_map(flt)
    choice = st.selectbox("Choose a bond", options) if options else None
    if choice:
        r = flt.loc[idx_map[choice]].to_dict()
        c1, c2, c3 = st.columns(3)
        c1.metric("Bid Px", f"{(r.get('Close_Bid') or float('nan')):,.2f}")
        c2.metric("Bid YTM", f"{(r.get('BidYTM') or float('nan')):,.2f}%")
        c3.metric("Duration", f"{(r.get('Duration') or float('nan')):,.2f}y")
        st.write({k: r.get(k) for k in ['Issuer','Security','ISIN','Account','Allocation','Units','Coupon','Maturity_Date','Days_to_Maturity']})

with tab3:
    st.caption("Risk lenses: DV01 and Duration by holding.")
    risk = flt[['Issuer','Security','ISIN','Market_Value','Duration','DV01']].dropna(subset=['DV01']).copy()
    if not risk.empty:
        risk['AbsDV01'] = risk['DV01'] * risk['Market_Value'].fillna(0) / 100.0
        st.dataframe(risk.sort_values('AbsDV01', ascending=False), hide_index=True, use_container_width=True)
        rchart = (
            alt.Chart(risk)
            .mark_bar()
            .encode(
                x=alt.X('AbsDV01:Q', title='Portfolio DV01 ($ per 1bp)'),
                y=alt.Y('Security:N', sort='-x'),
                tooltip=['ISIN','AbsDV01']
            )
            .properties(height=400)
        )
        st.altair_chart(rchart, use_container_width=True)
    else:
        st.info("No risk data available.")

with tab4:
    st.caption("Projected fixed-coupon cashflows for a selected bond.")
    options2, idx_map2 = build_options_map(flt)
    pick = st.selectbox("Bond", options2, key='cf_pick') if options2 else None
    if pick:
        row = flt.loc[idx_map2[pick]]
        cp = row.get('Coupon')
        coupon_dec = (cp/100.0) if (pd.notna(cp) and cp > 1) else (cp if pd.notna(cp) else 0.0)
        if pd.notna(row.get('Maturity_Date')):
            cf = cashflow_schedule(row['Maturity_Date'], coupon_dec, 2, 100, date.today())
            cf_df = pd.DataFrame(cf)
            st.dataframe(cf_df, hide_index=True, use_container_width=True)
            # Optional bar of cashflows
            cfc = (
                alt.Chart(cf_df)
                .mark_bar()
                .encode(
                    x=alt.X('date:T', title='Payment Date'),
                    y=alt.Y('amount:Q', title='Amount')
                )
                .properties(height=300)
            )
            st.altair_chart(cfc, use_container_width=True)
        else:
            st.info("Missing maturity date for cashflow projection.")

# --- Portfolio totals & download ---
with st.expander("Totals & Averages", expanded=True):
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Market Value", fmt_money(float(mv_s.sum()) if not mv_s.empty else float('nan')))
        invested_s = _num_series(out_df, 'Invested')
        st.metric("Total Invested", fmt_money(float(invested_s.sum()) if not invested_s.empty else float('nan')))
    with cols[1]:
        st.metric("Total $ P/L", fmt_money(float(pl_s.sum()) if not pl_s.empty else float('nan')))
        pctpl_s = _num_series(out_df, 'Percent_PL')
        st.metric("Avg % P/L", fmt_pct(float(pctpl_s.mean()) if not pctpl_s.empty else float('nan')))
    with cols[2]:
        purch_s = _num_series(out_df, 'PurchYTM')
        st.metric("Avg Purch YTM", fmt_pct(float(purch_s.mean()) if not purch_s.empty else float('nan')))
        st.metric("Avg Bid YTM", fmt_pct(float(ytm_s.mean()) if not ytm_s.empty else float('nan')))
    with cols[3]:
        dv01_s = _num_series(out_df, 'DV01')
        st.metric("Portfolio DV01 (per 100)", fmt_money(float(dv01_s.sum()) if not dv01_s.empty else float('nan')))
        wd2 = safe_weighted_avg(out_df['Duration'] if 'Duration' in out_df.columns else pd.Series(dtype=float), mv_s)
        st.metric("Weighted Duration", "—" if pd.isna(wd2) else f"{wd2:.2f}y")

csv = styled.to_csv(index=False).encode("utf-8")
st.download_button("Download table as CSV", data=csv, file_name=f"fsl_bond_portfolio_{date.today()}.csv")

st.caption(
    "Notes: YTM/Duration/Convexity are approximations for fixed-rate bullet bonds using simple day count and semi-annual coupons. "
    "For floating-rate/odd coupons, consider enhancing with full analytics (e.g., QuantLib). Margin figures are estimates—configure "
    "init/maintenance % and borrow rate in the sidebar or wire in IB 'what-if' calculations per bond."
)
