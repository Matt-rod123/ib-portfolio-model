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
# MAIN APP (IB-first, upload hidden)
# ----------------------------

st.title("📈 Firstline Securities Ltd — Bond Portfolio")
st.caption("Primary data source: IBKR via TWS/Gateway. CSV/XLSX import is hidden under Advanced.")

# --- Advanced (hidden) upload -------------------------------------------------
with st.expander("Advanced: Import CSV/XLSX (optional)", expanded=False):
    up = st.file_uploader("Upload portfolio file", type=["xlsx","xls","csv"], label_visibility="collapsed")
    df_upload = None
    if up is not None:
        try:
            df_upload = pd.read_csv(up) if up.name.lower().endswith("csv") else pd.read_excel(up)
            st.success("File loaded.")
        except Exception as e:
            st.error(f"Could not read file: {e}")

# --- IB -> Positions -> DataFrame --------------------------------------------

def _to_date(yyyymmdd: str):
    try:
        if not yyyymmdd or not isinstance(yyyymmdd, str):
            return pd.NaT
        y, m, d = int(yyyymmdd[0:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8])
        return date(y, m, d)
    except Exception:
        return pd.NaT

@st.cache_data(show_spinner=False, ttl=15)
def ib_positions_df() -> pd.DataFrame:
    rows = []
    try:
        # Ensure fresh snapshot of positions
        ib.reqPositions()
        ib.sleep(0.2)
        positions = ib.positions()
    except Exception:
        positions = []
    for p in positions:
        con = p.contract
        # Only bonds
        secType = getattr(con, 'secType', '')
        if str(secType).upper() != 'BOND' and not isinstance(con, Bond):
            continue
        qty = float(getattr(p, 'position', 0.0) or 0.0)
        avg_cost = float(getattr(p, 'avgCost', float('nan')) or float('nan'))
        # Try to get a bid/last/close snapshot
        bid_px = np.nan
        try:
            tkr = ib.reqMktData(con, genericTickList="", snapshot=True)
            ib.sleep(0.35)
            cand = [tkr.bid, tkr.last, tkr.close]
            cand = [c for c in cand if c is not None and not (isinstance(c, float) and np.isnan(c))]
            if cand:
                bid_px = float(cand[0])
                if bid_px < 2:  # IB sometimes returns in 1.xx rather than 100.xx
                    bid_px *= 100.0
        except Exception:
            pass
        # Identify fields
        issuer = getattr(con, 'symbol', None) or getattr(con, 'longName', None) or ''
        isin = getattr(con, 'isin', None) or getattr(con, 'secId', None) or getattr(con, 'cusip', None) or ''
        coupon = getattr(con, 'coupon', None)
        maturity = _to_date(getattr(con, 'maturity', ''))
        account = getattr(p, 'account', '')
        # Compute
        invested = qty * avg_cost if (qty and not np.isnan(avg_cost)) else np.nan
        mval = qty * bid_px if (qty and not np.isnan(bid_px)) else np.nan
        dpl = (mval - invested) if (not np.isnan(mval) and not np.isnan(invested)) else np.nan
        ppl = (dpl / invested * 100.0) if (not np.isnan(dpl) and invested and invested != 0) else np.nan
        rows.append({
            'Issuer': issuer,
            'Security': getattr(con, 'localSymbol', '') or getattr(con, 'description', '') or '',
            'ISIN': isin,
            'Account': account,
            'Allocation': '',
            'Units': qty,
            'Purchase_Px': avg_cost,
            'Close_Bid': bid_px,
            'Invested': invested,
            'Market_Value': mval,
            'Dollar_PL': dpl,
            'Percent_PL': ppl,
            'Coupon': coupon,
            'Maturity_Date': maturity,
            # Placeholders for analytics not computed here
            'Duration': np.nan,
            'DV01': np.nan,
            'BidYTM': np.nan,
            'PurchYTM': np.nan,
        })
    df = pd.DataFrame(rows)
    return df

# Decide source (IB preferred)
if ib.isConnected():
    df_main = ib_positions_df()
    if df_main.empty:
        st.warning("Connected to IBKR but no bond positions were returned.")
elif 'df_upload' in locals() and df_upload is not None:
    df_main = df_upload.copy()
    st.info("Using uploaded file (IB not connected).")
else:
    df_main = pd.DataFrame()
    st.info("Connect to IBKR to load positions, or use Advanced import.")

# Guarantee columns
needed = ['Issuer','Security','ISIN','Account','Allocation','Units','Purchase_Px','Close_Bid','Invested','Market_Value','Dollar_PL','Percent_PL','Coupon','Maturity_Date','Duration','DV01','BidYTM','PurchYTM']
for c in needed:
    if c not in df_main.columns:
        df_main[c] = np.nan

# --- KPIs ---------------------------------------------------------------------

def _num_series(df, col):
    s = df[col] if col in df.columns else pd.Series(dtype=float)
    return pd.to_numeric(s, errors='coerce')

def fmt_money(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"${x:,.2f}"

def fmt_pct(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}%"

mv_s = _num_series(df_main, 'Market_Value')
pl_s = _num_series(df_main, 'Dollar_PL')
colK1, colK2, colK3, colK4 = st.columns(4)
colK1.metric("Total Invested", fmt_money(float(_num_series(df_main,'Invested').sum()) if not df_main.empty else float('nan')))
colK2.metric("Market Value", fmt_money(float(mv_s.sum()) if not df_main.empty else float('nan')))
colK3.metric("Total P/L", fmt_money(float(pl_s.sum()) if not df_main.empty else float('nan')))
colK4.metric("# Holdings", int(df_main.shape[0]))

st.divider()

# --- Filters ------------------------------------------------------------------

def col_values(df: pd.DataFrame, col: str):
    if col in df.columns:
        vals = pd.Series(df[col]).dropna().astype(str)
        vals = [v for v in vals.tolist() if v.strip() != ""]
        return sorted(vals)
    return []

fc1, fc2, fc3 = st.columns(3)
sel_issuer = fc1.multiselect("Issuer", col_values(df_main,'Issuer'))
sel_account = fc2.multiselect("Account", col_values(df_main,'Account'))
sel_alloc = fc3.multiselect("Allocation", col_values(df_main,'Allocation'))

mask = pd.Series(True, index=df_main.index)
if sel_issuer and 'Issuer' in df_main.columns:
    mask &= df_main['Issuer'].astype(str).isin(sel_issuer)
if sel_account and 'Account' in df_main.columns:
    mask &= df_main['Account'].astype(str).isin(sel_account)
if sel_alloc and 'Allocation' in df_main.columns:
    mask &= df_main['Allocation'].astype(str).isin(sel_alloc)
flt = df_main[mask].copy()

# --- Tabs ---------------------------------------------------------------------
import altair as alt

tab1, tab2, tab3 = st.tabs(["Overview", "Holdings", "Risk"])

with tab1:
    st.caption("Portfolio overview & breakdowns")
    # Allocation by account
    if 'Account' in flt.columns and 'Market_Value' in flt.columns and not flt.empty:
        alloc_df = flt.groupby('Account', dropna=False)['Market_Value'].sum().reset_index().sort_values('Market_Value', ascending=False)
        if not alloc_df.empty:
            chart1 = alt.Chart(alloc_df).mark_bar().encode(
                x=alt.X('Market_Value:Q', title='Market Value ($)'),
                y=alt.Y('Account:N', sort='-x'),
                tooltip=['Account','Market_Value']
            ).properties(height=300)
            st.altair_chart(chart1, use_container_width=True)
        else:
            st.info("No holdings to plot by account.")
    else:
        st.info("No account data available.")

    # Maturity ladder by year
    if 'Maturity_Date' in flt.columns and not flt['Maturity_Date'].dropna().empty:
        mat = flt.dropna(subset=['Maturity_Date']).copy()
        mat['Year'] = pd.to_datetime(mat['Maturity_Date']).dt.year
        ladder = mat.groupby('Year')['Market_Value'].sum().reset_index()
        chart2 = alt.Chart(ladder).mark_bar().encode(
            x=alt.X('Year:O'),
            y=alt.Y('Market_Value:Q', title='Market Value ($)'),
            tooltip=['Year','Market_Value']
        ).properties(height=300)
        st.altair_chart(chart2, use_container_width=True)
    else:
        st.info("No maturity data available.")

with tab2:
    st.caption("Holdings (live from IBKR when connected)")
    nice_cols = ['Issuer','Security','ISIN','Account','Allocation','Units','Purchase_Px','Close_Bid','Invested','Market_Value','Dollar_PL','Percent_PL','Coupon','Maturity_Date']
    for c in nice_cols:
        if c not in flt.columns:
            flt[c] = np.nan
    view = flt[nice_cols].copy()
    view.rename(columns={
        'Purchase_Px':'Purchase Px','Close_Bid':'Bid Px','Market_Value':'Market Value',
        'Dollar_PL':'$ P/L','Percent_PL':'% P/L','Maturity_Date':'Maturity'
    }, inplace=True)
    st.dataframe(view, hide_index=True, use_container_width=True)

with tab3:
    st.caption("Risk lenses: DV01 and Duration (placeholders unless analytics added)")
    if 'DV01' in flt.columns and flt['DV01'].notna().any():
        risk = flt[['Issuer','Security','ISIN','Market_Value','Duration','DV01']].dropna(subset=['DV01']).copy()
        risk['AbsDV01'] = risk['DV01'] * risk['Market_Value'].fillna(0) / 100.0
        st.dataframe(risk.sort_values('AbsDV01', ascending=False), hide_index=True, use_container_width=True)
        rchart = alt.Chart(risk).mark_bar().encode(
            x=alt.X('AbsDV01:Q', title='Portfolio DV01 ($ per 1bp)'),
            y=alt.Y('Security:N', sort='-x'),
            tooltip=['ISIN','AbsDV01']
        ).properties(height=400)
        st.altair_chart(rchart, use_container_width=True)
    else:
        st.info("DV01/Duration not computed in this build.")

# Footer
st.caption("Data source priority: IBKR positions when connected; optional file import is hidden above. Prices are snapshots; some bonds may need manual qualification for quotes.")
