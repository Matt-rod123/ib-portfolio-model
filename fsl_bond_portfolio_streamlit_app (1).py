"""
Firstline Securities Ltd — Bond Portfolio (Streamlit + IBKR TWS API)

How to run locally
------------------
1) Python 3.10+
2) `pip install -r requirements.txt`
3) Start IB Gateway or Trader Workstation with API enabled.
4) `streamlit run fsl_bond_portfolio_streamlit_app.py`

Minimal requirements.txt (create this file next to the app):
-----------------------------------------------------------
streamlit
pandas
numpy
ib-insync
openpyxl
python-dateutil

Security note: set IB connection params from environment variables when deploying
(IB_HOST, IB_PORT, IB_CLIENT_ID). On Streamlit Cloud or your own server, run an
IB Gateway alongside the app (same private network) so the socket is reachable.
"""
from __future__ import annotations
import asyncio

# Create and set a loop before importing ib_insync/eventkit
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

from ib_insync import IB, Bond, util
util.patchAsyncio()

import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import streamlit as st

# ----------------------------
# UI CONFIG
# ----------------------------
st.set_page_config(
    page_title="FSL Fixed Income Portfolio",
    page_icon="📈",
    layout="wide",
)

# ----------------------------
# HELPERS — FINANCE MATH
# ----------------------------

def yearfrac(d1: date, d2: date, basis: str = "30/360") -> float:
    """Approximate year fraction between two dates.
    Supports 'act/365' and '30/360' conventions (simple versions).
    """
    if basis.lower() in {"act/365", "actual/365"}:
        return (d2 - d1).days / 365.0
    # 30/360 US (simplified)
    d1_day = min(d1.day, 30)
    d2_day = 30 if (d1.day == 31 and d2.day == 31) else min(d2.day, 30)
    months = (d2.year - d1.year) * 12 + (d2.month - d1.month)
    days = d2_day - d1_day
    return (months * 30 + days) / 360.0


def cashflow_schedule(maturity: date, coupon: float, freq: int, face: float, settle: date) -> List[Dict[str, Any]]:
    """Generate fixed coupon cashflows from next coupon after settle to maturity.
    coupon in decimal (e.g., 0.065 for 6.5%). face nominal per *unit*.
    """
    if freq not in [1, 2, 4]:
        freq = 2
    # Assume standard semi-annual timeline ending at maturity
    # Roll back coupons until strictly greater than settle
    cf = []
    pay = maturity
    step = {1: relativedelta(years=1), 2: relativedelta(months=6), 4: relativedelta(months=3)}[freq]
    while pay > settle:
        cf.append({"date": pay, "amount": face * coupon / freq})
        pay = pay - step
    cf = list(reversed(cf))
    if cf and cf[-1]["date"] == maturity:
        cf[-1]["amount"] += face  # add principal at maturity
    else:
        # if no coupon dates (very short), just principal at maturity
        cf.append({"date": maturity, "amount": face * (1 + coupon / freq)})
    return cf


def price_from_ytm(ytm: float, maturity: date, coupon: float, freq: int, face: float, settle: date) -> float:
    """Clean price per 100 face given YTM (annual, in decimal)."""
    cfs = cashflow_schedule(maturity, coupon, freq, face, settle)
    pv = 0.0
    for c in cfs:
        t = yearfrac(settle, c["date"], "act/365")
        pv += c["amount"] / ((1 + ytm / freq) ** (freq * t))
    return pv / (face / 100.0)


def ytm_from_price(clean_price: float, maturity: date, coupon: float, freq: int, face: float, settle: date) -> float:
    """Solve for YTM (decimal, annual) from clean price per 100 face using Newton's method."""
    if clean_price <= 0:
        return np.nan
    # Initial guess ~ coupon/price
    y = max(0.0001, coupon / max(0.01, clean_price / 100.0))
    for _ in range(40):
        p = price_from_ytm(y, maturity, coupon, freq, face, settle)
        # numeric derivative
        dy = 1e-6
        p2 = price_from_ytm(y + dy, maturity, coupon, freq, face, settle)
        dpdy = (p2 - p) / dy
        if abs(dpdy) < 1e-8:
            break
        y_new = y - (p - clean_price) / dpdy
        if abs(y_new - y) < 1e-10:
            return max(y_new, 0.0)
        y = max(y_new, 0.0)
    return max(y, 0.0)


def macaulay_duration_convexity(ytm: float, maturity: date, coupon: float, freq: int, face: float, settle: date) -> tuple[float, float]:
    """Return (ModifiedDuration, Convexity) in years. DV01 per 100 face = ModDur * Price / 100."""
    cfs = cashflow_schedule(maturity, coupon, freq, face, settle)
    md = 0.0
    cx = 0.0
    pv = 0.0
    for c in cfs:
        t = yearfrac(settle, c["date"], "act/365")
        disc = (1 + ytm / freq) ** (freq * t)
        pv_c = c["amount"] / disc
        pv += pv_c
        tau = t
        md += tau * pv_c
        cx += tau * (tau + 1 / freq) * pv_c  # approximation
    if pv == 0:
        return (np.nan, np.nan)
    macaulay = md / pv
    modified = macaulay / (1 + ytm / freq)
    convexity = cx / (pv * (1 + ytm / freq) ** 2)
    return modified, convexity


# ----------------------------
# DATA CLASSES
# ----------------------------
@dataclass
class BondRow:
    RIC: Optional[str] = None
    Issuer: Optional[str] = None
    Security: Optional[str] = None
    ISIN: Optional[str] = None
    Account: Optional[str] = None
    Allocation: Optional[str] = None
    Face_Value: float = 0.0
    Units: float = 0.0
    Purchase_Px: float = np.nan  # clean px / 100
    Invested: float = np.nan
    Close_Bid: float = np.nan

    # P&L block
    Market_Value: float = np.nan
    Dollar_PL: float = np.nan
    Percent_PL: float = np.nan

    # Yield/coupon block
    Mrgn_Cost: float = np.nan
    PurchYTM: float = np.nan
    PYTM_Sprd: float = np.nan
    BidYTM: float = np.nan
    Current_Yld: float = np.nan
    Curr_Sprd: float = np.nan
    Coupon: float = np.nan
    Cpn_Sprd: float = np.nan
    Init_Margin_Pct: float = 0.25  # default 25%
    I_Cash_Equity: float = np.nan

    # Margin details
    I_Margined: float = np.nan
    Marg_Int: float = np.nan
    Mnt_Margin: float = np.nan

    # Risk
    Duration: float = np.nan
    Convexity: float = np.nan
    DV01: float = np.nan

    # Dates
    Maturity_Date: Optional[date] = None
    Days_to_Maturity: Optional[int] = None
    Mat_Year: Optional[int] = None
    Cpn_Date_1: Optional[str] = None
    Cpn_Date_2: Optional[str] = None


# ----------------------------
# IB CONNECTION
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_ib() -> IB:
    # Double patch for safety in some hosting envs
    util.patchAsyncio()
    ib = IB()
    return ib


def ensure_connected(ib: IB, host: str, port: int, client_id: int) -> bool:
    if ib.isConnected():
        return True
    try:
        ib.connect(host, port, clientId=client_id, readonly=True, timeout=5)
        return True
    except Exception as e:
        st.warning(f"Could not connect to IBKR TWS/Gateway: {e}")
        return False


# ----------------------------
# MARKET DATA FETCH
# ----------------------------
@st.cache_data(show_spinner=False, ttl=30)
def fetch_bid_and_contract(ib: IB, isin: Optional[str]) -> tuple[Optional[float], Optional[Bond]]:
    if not isin:
        return (None, None)
    contract = Bond(secIdType="ISIN", secId=isin)
    try:
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            return (None, None)
        c = qualified[0]
        ticker = ib.reqMktData(c, genericTickList="", snapshot=True)
        ib.sleep(0.5)
        bid = ticker.bid if not math.isnan(ticker.bid or float('nan')) else None
        # fallback to last / close
        if bid is None:
            last = None if math.isnan(ticker.last or float('nan')) else ticker.last
            close = None if math.isnan(ticker.close or float('nan')) else ticker.close
            bid = last or close
        return (bid, c)
    except Exception:
        return (None, None)


# ----------------------------
# PORTFOLIO PIPELINE
# ----------------------------
REQUIRED_COLUMNS = [
    "RIC", "Issuer", "Security", "ISIN", "Account", "Allocation",
    "Face Value", "Units", "Purchase Px", "Invested", "Close Bid",
    # Optional analytics inputs
    "Coupon", "Mat. Year", "Maturity Date", "Cpn Date 1", "Cpn Date 2", "Init Margin %",
]

COLUMN_RENAMES = {
    "Face Value": "Face_Value",
    "Purchase Px": "Purchase_Px",
    "Close Bid": "Close_Bid",
    "Init Margin %": "Init_Margin_Pct",
    "Maturity Date": "Maturity_Date",
    "Cpn Date 1": "Cpn_Date_1",
    "Cpn Date 2": "Cpn_Date_2",
}


def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Standardize column names
    for old, new in COLUMN_RENAMES.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)
    # Ensure expected columns exist
    for c in [c for c in BondRow.__dataclass_fields__.keys() if c not in df.columns]:
        df[c] = np.nan
    # Coerce numeric
    num_cols = [
        "Face_Value", "Units", "Purchase_Px", "Invested", "Close_Bid", "Coupon",
        "Init_Margin_Pct",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Dates
    if "Maturity_Date" in df.columns:
        df["Maturity_Date"] = pd.to_datetime(df["Maturity_Date"], errors="coerce").dt.date
    if "Mat_Year" in df.columns:
        df["Mat_Year"] = pd.to_numeric(df["Mat_Year"], errors="coerce").astype("Int64")
    return df


def compute_row_metrics(row: pd.Series, ib: Optional[IB], live_prices: bool, maint_margin_pct: float, borrow_rate: float) -> Dict[str, Any]:
    data = BondRow(**{k: row.get(k) for k in BondRow.__dataclass_fields__.keys()})

    # Pull live bid if requested (fallback to spreadsheet value)
    bid_px = float(row.get("Close_Bid")) if not np.isnan(row.get("Close_Bid", np.nan)) else None
    contract = None
    if live_prices and ib is not None and row.get("ISIN"):
        live_bid, contract = fetch_bid_and_contract(ib, row.get("ISIN"))
        if live_bid is not None:
            bid_px = live_bid * 100 if live_bid < 2 else live_bid  # IB may return in % or raw
    data.Close_Bid = bid_px

    # Face per unit assumed = 100 unless explicitly provided via Face_Value/Units
    face_per_unit = 100.0
    units = float(row.get("Units") or 0.0)
    face_val = float(row.get("Face_Value") or (units * face_per_unit))

    # Invested default = units * purchase_px
    purch_px = float(row.get("Purchase_Px") or np.nan)
    invested = row.get("Invested")
    if pd.isna(invested) and not np.isnan(purch_px):
        invested = units * purch_px

    data.Face_Value = face_val
    data.Units = units
    data.Purchase_Px = purch_px
    data.Invested = invested

    # Market value
    if bid_px is not None and not np.isnan(bid_px):
        mv = units * bid_px
        data.Market_Value = mv
        if invested is not None and not np.isnan(invested):
            data.Dollar_PL = mv - invested
            data.Percent_PL = (data.Dollar_PL / invested) * 100 if invested else np.nan

    # Coupon/Yield block
    coupon = float(row.get("Coupon") or (getattr(contract, "coupon", None) or np.nan))
    if not np.isnan(coupon):
        coupon_dec = coupon / 100.0 if coupon > 1.0 else coupon
    else:
        coupon_dec = np.nan

    maturity = row.get("Maturity_Date")
    if pd.isna(maturity) or maturity is None:
        # Try Mat_Year + Cpn Date 2 (assume last coupon in year)
        mat_year = row.get("Mat_Year")
        if pd.notna(mat_year):
            maturity = date(int(mat_year), 12, 31)
    settle = date.today()

    freq = 2  # default semi-annual
    if not np.isnan(coupon_dec) and not pd.isna(maturity):
        # Compute BidYTM from bid price per 100
        if bid_px is not None and not np.isnan(bid_px):
            data.BidYTM = ytm_from_price(bid_px, maturity, coupon_dec, freq, 100, settle) * 100
        # PurchYTM from purchase price
        if not np.isnan(purch_px):
            data.PurchYTM = ytm_from_price(purch_px, maturity, coupon_dec, freq, 100, settle) * 100
        if not np.isnan(data.PurchYTM) and not np.isnan(data.BidYTM):
            data.PYTM_Sprd = data.BidYTM - data.PurchYTM
        # Current yield ≈ coupon / price
        if bid_px is not None and not np.isnan(bid_px):
            data.Current_Yld = (coupon_dec * 100) / (bid_px / 100)
        # Spreads relative to coupon
        data.Cpn_Sprd = (data.BidYTM - (coupon_dec * 100)) if not np.isnan(data.BidYTM) else np.nan
        data.Curr_Sprd = (data.Current_Yld - (coupon_dec * 100)) if not np.isnan(data.Current_Yld) else np.nan
        # Risk metrics
        if not np.isnan(data.BidYTM):
            mdur, conv = macaulay_duration_convexity(data.BidYTM / 100.0, maturity, coupon_dec, freq, 100, settle)
            data.Duration = mdur
            data.Convexity = conv
            if bid_px is not None and not np.isnan(bid_px):
                data.DV01 = mdur * (bid_px) / 100.0

    data.Coupon = coupon if not np.isnan(coupon) else np.nan

    # Days to maturity
    if maturity:
        data.Maturity_Date = maturity
        data.Days_to_Maturity = (maturity - settle).days
        data.Mat_Year = maturity.year

    # Margin block (approx, configurable)
    init_pct = float(row.get("Init_Margin_Pct") or 0.25)
    data.Init_Margin_Pct = init_pct * 100 if init_pct < 1 else init_pct
    init_pct_dec = init_pct if init_pct < 1 else init_pct / 100.0
    maint_pct_dec = maint_margin_pct

    if not np.isnan(data.Market_Value):
        data.I_Cash_Equity = data.Market_Value * init_pct_dec
        data.I_Margined = data.Market_Value * init_pct_dec
        data.Mnt_Margin = data.Market_Value * maint_pct_dec
        # Simple daily interest estimate on margined amount
        data.Marg_Int = data.I_Margined * (borrow_rate / 100.0) / 360.0

    # Return dict for DataFrame assembly
    out = asdict(data)
    return out


# ----------------------------
# SIDEBAR & INPUTS
# ----------------------------
st.sidebar.header("IBKR Connection")
ib_host = st.sidebar.text_input("Host", os.getenv("IB_HOST", "127.0.0.1"))
ib_port = st.sidebar.number_input("Port", value=int(os.getenv("IB_PORT", "7497")))
ib_client = st.sidebar.number_input("Client ID", value=int(os.getenv("IB_CLIENT_ID", "1101")))
connect_btn = st.sidebar.button("Connect to IBKR")

st.sidebar.header("Analytics Settings")
live_prices = st.sidebar.toggle("Use live IBKR bid prices", value=True)
maint_margin_pct = st.sidebar.slider("Maintenance Margin % (estimate)", 0.05, 0.5, 0.25, 0.01)
borrow_rate = st.sidebar.number_input("Borrow Rate % p.a. (estimate)", value=8.0, step=0.25)

ib = get_ib()
if connect_btn:
    ensure_connected(ib, ib_host, ib_port, ib_client)

if ib.isConnected():
    st.sidebar.success("Connected to IBKR")
else:
    st.sidebar.info("Not connected. The app can still run from spreadsheet values.")

# ----------------------------
# MAIN: INPUT TABLE
# ----------------------------
st.title("FIRSTLINE SECURITIES LTD — Fixed Income Portfolio")
st.caption("Live model reflecting your Excel template. Upload the sheet and (optionally) pull live prices from IBKR.")

uploaded = st.file_uploader("Upload Excel/CSV based on template", type=["xlsx", "xls", "csv"])

if uploaded is not None:
    if uploaded.name.lower().endswith("csv"):
        raw_df = pd.read_csv(uploaded)
    else:
        raw_df = pd.read_excel(uploaded, engine="openpyxl")
else:
    st.info("No file uploaded yet. You can still see an empty table below.")
    raw_df = pd.DataFrame(columns=REQUIRED_COLUMNS)

# Normalize columns
in_df = normalize_input(raw_df)

# Compute metrics row by row
rows = []
for _, r in in_df.iterrows():
    rows.append(
        compute_row_metrics(r, ib if ib.isConnected() else None, live_prices, maint_margin_pct, borrow_rate)
    )

out_df = pd.DataFrame(rows)

# Presentation — order columns to mirror your screenshots
ORDERED_COLUMNS = [
    # Block 1
    "RIC", "Issuer", "Security", "ISIN", "Account", "Allocation",
    "Face_Value", "Units", "Purchase_Px", "Invested", "Close_Bid",
    # Block 2
    "Market_Value", "Dollar_PL", "Percent_PL", "Mrgn_Cost", "PurchYTM", "PYTM_Sprd",
    "BidYTM", "Current_Yld", "Curr_Sprd", "Coupon", "Cpn_Sprd", "Init_Margin_Pct", "I_Cash_Equity",
    # Block 3
    "I_Margined", "Marg_Int", "Mnt_Margin", "Duration", "Convexity", "DV01",
    "Maturity_Date", "Days_to_Maturity", "Mat_Year", "Cpn_Date_1", "Cpn_Date_2",
]

# Ensure columns exist
for c in ORDERED_COLUMNS:
    if c not in out_df.columns:
        out_df[c] = np.nan

# Formatting helpers
money_cols = [
    "Face_Value", "Units", "Purchase_Px", "Invested", "Close_Bid", "Market_Value",
    "Dollar_PL", "I_Cash_Equity", "I_Margined", "Marg_Int", "Mnt_Margin", "DV01",
]
percent_cols = ["Percent_PL", "PurchYTM", "PYTM_Sprd", "BidYTM", "Current_Yld", "Curr_Sprd", "Coupon", "Cpn_Sprd", "Init_Margin_Pct"]
int_cols = ["Days_to_Maturity", "Mat_Year"]

styled = out_df[ORDERED_COLUMNS].copy()

# Apply formats
for c in money_cols:
    if c in styled.columns:
        styled[c] = styled[c].map(lambda x: np.nan if pd.isna(x) else round(float(x), 2))
for c in percent_cols:
    if c in styled.columns:
        styled[c] = styled[c].map(lambda x: np.nan if pd.isna(x) else round(float(x), 2))
for c in int_cols:
    if c in styled.columns:
        styled[c] = styled[c].astype("Int64")

# Display
st.dataframe(
    styled,
    use_container_width=True,
    height=650,
)

# Portfolio totals (bottom bar like in the screenshots)
with st.expander("Totals & Averages", expanded=True):
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Market Value", f"${out_df['Market_Value'].sum(skipna=True):,.2f}")
        st.metric("Total Invested", f"${out_df['Invested'].sum(skipna=True):,.2f}")
    with cols[1]:
        st.metric("Total $ P/L", f"${out_df['Dollar_PL'].sum(skipna=True):,.2f}")
        st.metric("Avg % P/L", f"{out_df['Percent_PL'].mean(skipna=True):.2f}%")
    with cols[2]:
        st.metric("Avg Purch YTM", f"{out_df['PurchYTM'].mean(skipna=True):.2f}%")
        st.metric("Avg Bid YTM", f"{out_df['BidYTM'].mean(skipna=True):.2f}%")
    with cols[3]:
        st.metric("Portfolio DV01 (per 100)", f"${out_df['DV01'].sum(skipna=True):,.2f}")
        st.metric("Weighted Duration", f"{np.average(out_df['Duration'].fillna(0), weights=out_df['Market_Value'].fillna(0) + 1e-9):.2f}y")

# Download
csv = styled.to_csv(index=False).encode("utf-8")
st.download_button("Download table as CSV", data=csv, file_name=f"fsl_bond_portfolio_{date.today()}.csv")

st.caption(
    "Notes: YTM/Duration/Convexity are approximations for fixed-rate bullet bonds using simple day count and semi-annual coupons. "
    "For floating-rate/odd coupons, consider enhancing with full analytics (e.g., QuantLib). Margin figures are estimates—configure "
    "init/maintenance % and borrow rate in the sidebar or wire in IB 'what-if' calculations per bond."
)
