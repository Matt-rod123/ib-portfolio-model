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
import plotly.express as px

from ib_insync import IB, Bond, util
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

import altair as alt

# Header KPIs
colK1, colK2, colK3, colK4 = st.columns(4)
with colK1:
    st.metric("Total Market Value", f"${out_df['Market_Value'].sum(skipna=True):,.2f}")
with colK2:
    st.metric("Total $ P/L", f"${out_df['Dollar_PL'].sum(skipna=True):,.2f}")
with colK3:
    st.metric("Avg Bid YTM", f"{out_df['BidYTM'].mean(skipna=True):.2f}%")
with colK4:
    st.metric("Weighted Duration", f"{np.average(out_df['Duration'].fillna(0), weights=out_df['Market_Value'].fillna(0) + 1e-9):.2f}y")

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

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Holdings", "Risk", "Cashflows"])

with tab1:
    st.caption("Portfolio overview & breakdowns")
    # Allocation by account (bar)
    alloc_df = flt.groupby('Account', dropna=False)['Market_Value'].sum().reset_index().sort_values('Market_Value', ascending=False)
    chart1 = alt.Chart(alloc_df).mark_bar().encode(x=alt.X('Market_Value:Q', title='Market Value ($)'), y=alt.Y('Account:N', sort='-x'), tooltip=['Account','Market_Value']).properties(height=300)
    st.altair_chart(chart1, use_container_width=True)

    # Maturity ladder by year
    mat = flt.dropna(subset=['Maturity_Date']).copy()
    if not mat.empty:
        mat['Year'] = pd.to_datetime(mat['Maturity_Date']).dt.year
        ladder = mat.groupby('Year')['Market_Value'].sum().reset_index()
        chart2 = alt.Chart(ladder).mark_bar().encode(x=alt.X('Year:O'), y=alt.Y('Market_Value:Q', title='Market Value ($)'), tooltip=['Year','Market_Value']).properties(height=300)
        st.altair_chart(chart2, use_container_width=True)

    # YTM histogram
    ytms = flt['BidYTM'].dropna()
    if not ytms.empty:
        hist = pd.DataFrame({'BidYTM': ytms})
        chart3 = alt.Chart(hist).mark_bar().encode(alt.X('BidYTM:Q', bin=alt.Bin(maxbins=30), title='Bid YTM (%)'), y='count()').properties(height=300)
        st.altair_chart(chart3, use_container_width=True)

with tab2:
    st.caption("Clean holdings table with key columns and quick selection.")
    nice_cols = ['Issuer','Security','ISIN','Account','Allocation','Units','Purchase_Px','Close_Bid','Market_Value','Dollar_PL','Percent_PL','PurchYTM','BidYTM','Duration','DV01','Maturity_Date']
    view = flt[nice_cols].copy()
    # Pretty labels
    labels = {
        'Purchase_Px':'Purchase Px','Close_Bid':'Bid Px','Market_Value':'Market Value',
        'Dollar_PL':'$ P/L','Percent_PL':'% P/L','PurchYTM':'Purch YTM','BidYTM':'Bid YTM',
        'Maturity_Date':'Maturity','DV01':'DV01 (per 100)'
    }
    view.rename(columns=labels, inplace=True)
    st.dataframe(view, hide_index=True, use_container_width=True)

    # Select a bond for details
    st.markdown("### Bond details")
    options = (flt['Issuer'].fillna('') + ' — ' + flt['Security'].fillna('') + ' (' + flt['ISIN'].fillna('') + ')').tolist()
    idx_map = {opt:i for i,opt in enumerate(options)}
    choice = st.selectbox("Choose a bond", options) if options else None
    if choice:
        r = flt.iloc[idx_map[choice]].to_dict()
        c1, c2, c3 = st.columns(3)
        c1.metric("Bid Px", f"{r.get('Close_Bid', float('nan')):,.2f}")
        c2.metric("Bid YTM", f"{r.get('BidYTM', float('nan')):,.2f}%")
        c3.metric("Duration", f"{r.get('Duration', float('nan')):,.2f}y")
        st.write({k: r.get(k) for k in ['Issuer','Security','ISIN','Account','Allocation','Units','Coupon','Maturity_Date','Days_to_Maturity']})

with tab3:
    st.caption("Risk lenses: DV01 and duration by holding.")
    risk = flt[['Issuer','Security','ISIN','Market_Value','Duration','DV01']].dropna(subset=['DV01']).copy()
    if not risk.empty:
        risk['AbsDV01'] = risk['DV01'] * risk['Market_Value'].fillna(0) / 100.0
        st.dataframe(risk.sort_values('AbsDV01', ascending=False), hide_index=True, use_container_width=True)
        rchart = alt.Chart(risk).mark_bar().encode(x=alt.X('AbsDV01:Q', title='Portfolio DV01 ($ per 1bp)'), y=alt.Y('Security:N', sort='-x'), tooltip=['ISIN','AbsDV01']).properties(height=400)
        st.altair_chart(rchart, use_container_width=True)
    else:
        st.info("No risk data available.")

with tab4:
    st.caption("Projected fixed-coupon cashflows for a selected bond.")
    options2 = (flt['Issuer'].fillna('') + ' — ' + flt['Security'].fillna('') + ' (' + flt['ISIN'].fillna('') + ')').tolist()
    pick = st.selectbox("Bond", options2, key='cf_pick') if options2 else None
    if pick:
        r = flt.iloc[idx_map[pick]]
        coupon_dec = (r['Coupon']/100.0) if (pd.notna(r['Coupon']) and r['Coupon']>1) else r['Coupon']
        cf = cashflow_schedule(r['Maturity_Date'], coupon_dec if pd.notna(coupon_dec) else 0.0, 2, 100, date.today()) if pd.notna(r['Maturity_Date']) else []
        if cf:
            cf_df = pd.DataFrame(cf)
            st.dataframe(cf_df, hide_index=True, use_container_width=True)
        else:
            st.info("Missing coupon or maturity to build cashflows.")

    # Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download portfolio CSV", data=csv, file_name=f"portfolio_{date.today()}.csv")

else:
    st.warning("No data loaded yet.")

st.caption("Note: Metrics are simplified. For advanced yield/duration/convexity analytics, extend with QuantLib.")
