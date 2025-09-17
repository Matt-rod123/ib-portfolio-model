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
        dpl = (mval - invested) if (not np.isnan(m
