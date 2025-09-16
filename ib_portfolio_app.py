# ib_portfolio_app.py
# Streamlit app to view an IBKR portfolio via TWS/IB Gateway.
# Shows: Issuer, Security Type, Face Value, Units, Purchase Price, Close Bid,
# Market Value, P/L, %P/L, and a Margin summary.

import math
import time
from typing import List, Dict, Any

import streamlit as st
import pandas as pd

# You need: pip install ib-insync streamlit pandas
# TWS/IB Gateway must be running; API → Settings → "Enable ActiveX and Socket Clients" (socket API)
try:
    from ib_insync import IB
except Exception:
    st.error("ib_insync is required. Install with: pip install ib-insync")
    raise

st.set_page_config(page_title="IBKR Portfolio (TWS API)", layout="wide")
st.title("Firstline – Fixed Income Portfolio (IBKR)")
st.caption("Issuer, Security, Face Value, Units, Purchase Px, Close Bid, Market Value, P/L, %P/L, Margin")

# ---------------- Sidebar: Connection ----------------
st.sidebar.header("TWS Connection")
host = st.sidebar.text_input("Host", "127.0.0.1")
port = st.sidebar.number_input("Port (7497 paper, 7496 live)", value=7497, step=1)
client_id = st.sidebar.number_input("Client ID", value=1, step=1)

if "ib" not in st.session_state:
    st.session_state.ib = IB()
if "connected" not in st.session_state:
    st.session_state.connected = False

colA, colB = st.sidebar.columns(2)
if colA.button("Connect", type="primary", disabled=st.session_state.connected):
    try:
        st.session_state.ib.connect(host, int(port), int(client_id), timeout=5)
        st.session_state.connected = st.session_state.ib.isConnected()
        if st.session_state.connected:
            st.success("Connected to TWS/Gateway ✅")
        else:
            st.error("Failed to connect.")
    except Exception as e:
        st.session_state.connected = False
        st.error(f"Connection error: {e}")

if colB.button("Disconnect", disabled=not st.session_state.connected):
    try:
        st.session_state.ib.disconnect()
    finally:
        st.session_state.connected = False
        st.warning("Disconnected.")

# ---------------- Helpers ----------------
def get_account_id(ib: IB) -> str:
    """Pick an account ID from positions or accountSummary."""
    try:
        positions = ib.positions()
        if positions:
            return positions[0].account
    except Exception:
        pass
    try:
        smry = ib.accountSummary()
        accts = sorted({v.account for v in smry})
        return accts[0] if accts else ""
    except Exception:
        return ""

def fetch_portfolio_table(ib: IB) -> pd.DataFrame:
    """
    Build a DataFrame with one row per position:
    Issuer, Security, Face Value, Units, Purchase Px, Close Bid, Market Value, P/L, % P/L.
    """
    positions = ib.reqPositions()
    rows: List[Dict[str, Any]] = []
    contracts = []
    issuer_cache: Dict[int, str] = {}

    # Qualify contracts & pull issuer names
    for pos in positions:
        c = pos.contract
        contracts.append(c)
        try:
            ib.qualifyContracts(c)
        except Exception:
            pass
        issuer = None
        try:
            cds = ib.reqContractDetails(c)
            if cds:
                issuer = cds[0].longName or cds[0].issuerId or None
        except Exception:
            pass
        issuer_cache[getattr(c, "conId", 0) or id(c)] = issuer or getattr(c, "symbol", "") or ""

    # Snapshot market data for Bid (requires permissions)
    tickers = []
    for c in contracts:
        try:
            t = ib.reqMktData(c, "", snapshot=True)
            tickers.append(t)
        except Exception:
            tickers.append(None)

    ib.sleep(1.5)  # wait briefly for snapshots

    # Build table rows
    for pos, t in zip(positions, tickers):
        c = pos.contract
        units = pos.position or 0.0
        avg_cost = pos.avgCost or 0.0
        sec_type = getattr(c, "secType", "")
        con_id = getattr(c, "conId", 0)
        issuer = issuer_cache.get(con_id or id(c), getattr(c, "symbol", ""))

        # Bid with fallbacks
        bid = None
        if t is not None:
            bid_val = getattr(t, "bid", float("nan"))
            bid = bid_val if not math.isnan(bid_val) and bid_val != 0 else None
            if bid is None:
                last_val = getattr(t, "last", float("nan"))
                close_val = getattr(t, "close", float("nan"))
                bid = (last_val if not math.isnan(last_val) else None) or \
                      (close_val if not math.isnan(close_val) else None)

        # Market value: prefer ib.portfolio() valuation, else approximate
        mkt_val = None
        try:
            pf_items = st.session_state.ib.portfolio()
            mkt_val = next((it.marketValue for it in pf_items
                            if getattr(it.contract, "conId", -1) == con_id), None)
        except Exception:
            pass
        if mkt_val is None and bid is not None and units:
            mkt_val = bid * units

        pl = None
        if mkt_val is not None and avg_cost:
            pl = mkt_val - (avg_cost * units)

        pl_pct = None
        if pl is not None and units and avg_cost:
            pl_pct = pl / (abs(units) * avg_cost)

        face_value = abs(units) if sec_type == "BOND" else ""

        rows.append({
            "Issuer": issuer,
            "Security": sec_type,
            "Face Value": face_value,
            "Units": units,
            "Purchase Px": avg_cost,
            "Close Bid": bid,
            "Market Value": mkt_val,
            "P/L": pl,
            "% P/L": pl_pct
        })

    df = pd.DataFrame(rows, columns=[
        "Issuer", "Security", "Face Value", "Units", "Purchase Px",
        "Close Bid", "Market Value", "P/L", "% P/L"
    ])
    return df

# ---------------- Main UI ----------------
if st.session_state.connected:
    acct = get_account_id(st.session_state.ib)
    st.sidebar.write(f"**Account:** {acct or '(unknown)'}")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        refresh = st.button("Refresh Portfolio", type="primary")
    with c2:
        auto = st.checkbox("Auto-refresh every 10s", value=False)
    with c3:
        show_margin = st.checkbox("Show Margin Summary", value=True)

    if refresh or auto:
        df = fetch_portfolio_table(st.session_state.ib)
        if not df.empty:
            fmt = {
                "Face Value": "{:,.0f}",
                "Units": "{:,.0f}",
                "Purchase Px": "{:,.2f}",
                "Close Bid": "{:,.2f}",
                "Market Value": "{:,.2f}",
                "P/L": "{:,.2f}",
                "% P/L": "{:.2%}",
            }
            st.dataframe(df.style.format(fmt), use_container_width=True)
        else:
            st.info("No positions returned.")

        if show_margin:
            try:
                smry = st.session_state.ib.accountSummary()
                tags = {v.tag: v.value for v in smry}
                init = float(tags.get("InitMarginReq", "0") or "0")
                maint = float(tags.get("MaintMarginReq", "0") or "0")
                netliq = float(tags.get("NetLiquidation", "0") or "0")
                colA, colB, colC, colD = st.columns(4)
                colA.metric("Net Liquidation", f"{netliq:,.2f}")
                colB.metric("Init Margin Req ($)", f"{init:,.2f}")
                colC.metric("Maint Margin Req ($)", f"{maint:,.2f}")
                pct = (init / netliq) if netliq else 0.0
                colD.metric("Init Margin %", f"{pct:.2%}")
            except Exception as e:
                st.warning(f"Margin summary not available: {e}")

        if auto:
            time.sleep(10)
            st.experimental_rerun()
else:
    st.info("Connect to TWS/Gateway (left sidebar) to load data.")
