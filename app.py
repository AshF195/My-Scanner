# breakout_scanner.py
# Streamlit Breakout Continuation Scanner (Customizable Filters)
# ---------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
from datetime import datetime
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import MACD

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Breakout Continuation Scanner",
    layout="wide"
)

st.title("Breakout Continuation Scanner")

# ---------------------------------------------------
# FILTER DEFAULTS & SESSION STATE
# ---------------------------------------------------

# Define our default threshold values
FILTER_DEFAULTS = {
    "min_rvol": 1.0,
    "min_1m_trend": 0.0,
    "min_6m_trend": 0.0,
    "max_ath_dist": -35.0,
    "max_rsi": 85.0,
    "max_bbpos": 1.1,
    "max_bw": 95.0,
    "max_accel": 4.0
}

# Initialize session state for each filter if they don't exist
for key, val in FILTER_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------
# GITHUB SETTINGS
# ---------------------------------------------------

GITHUB_USER   = "AshF195"
GITHUB_REPO   = "My-Scanner"
GITHUB_BRANCH = "main"

GITHUB_API = (
    f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}"
    f"/git/trees/{GITHUB_BRANCH}?recursive=1"
)

GITHUB_RAW = (
    f"https://raw.githubusercontent.com/{GITHUB_USER}"
    f"/{GITHUB_REPO}/{GITHUB_BRANCH}"
)

# ---------------------------------------------------
# RAG COLOUR SCHEME
# ---------------------------------------------------

RAG_GREEN = "background-color: rgba(0, 200, 80,  0.28)"
RAG_AMBER = "background-color: rgba(255, 170, 0,  0.28)"
RAG_RED   = "background-color: rgba(220, 50,  50, 0.28)"
RAG_NONE  = ""

# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------

@st.cache_data(ttl=300)
def get_github_csv_files():
    try:
        resp = requests.get(GITHUB_API, timeout=10)
        resp.raise_for_status()
        tree = resp.json().get("tree", [])
        csv_files = [
            item["path"]
            for item in tree
            if item["path"].lower().endswith(".csv")
            and item["type"] == "blob"
        ]
        return sorted(csv_files)
    except Exception as e:
        st.error(f"Could not fetch file list from GitHub: {e}")
        return []

@st.cache_data(ttl=300)
def load_csv_from_github(path):
    url = f"{GITHUB_RAW}/{path}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))

def get_earnings(stock):
    """Returns (display_str, days_until_int)."""
    # Method 1: stock.earnings_dates
    try:
        ed = stock.earnings_dates
        if ed is not None and not ed.empty:
            tz = ed.index.tz
            now_tz = pd.Timestamp.now(tz=tz) if tz else pd.Timestamp.now()
            future = ed[ed.index > now_tz]
            if not future.empty:
                next_date = future.index.min().date()
                days = (next_date - datetime.now().date()).days
                return f"{next_date} ({days}d)", days
    except Exception: pass

    # Method 2: stock.calendar as dict
    try:
        cal = stock.calendar
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
            if dates:
                next_date = pd.to_datetime(dates[0]).date()
                days = (next_date - datetime.now().date()).days
                return f"{next_date} ({days}d)", days
    except Exception: pass

    return "N/A", 999

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Scanner Settings")

all_csv_files = get_github_csv_files()

if not all_csv_files:
    st.error("No CSV files found.")
    st.stop()

selected_markets = st.sidebar.multiselect(
    "Markets",
    options=all_csv_files,
    default=all_csv_files[:1] if all_csv_files else [],
    format_func=lambda x: x.split("/")[-1].replace(".csv", "").upper()
)

max_stocks = st.sidebar.slider("Max Stocks Per Market", 10, 500, 100)
strict_mode = st.sidebar.checkbox("Strict Mode", value=True)
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# --- STRICT FILTER CUSTOMIZATION SECTION ---
with st.sidebar.expander("🛠️ Amend Strict Thresholds"):
    st.session_state.min_rvol = st.number_input("Min RVOL", value=st.session_state.min_rvol, step=0.1)
    st.session_state.max_rsi = st.number_input("Max RSI", value=st.session_state.max_rsi, step=1.0)
    st.session_state.min_1m_trend = st.number_input("Min 1M Trend %", value=st.session_state.min_1m_trend, step=1.0)
    st.session_state.min_6m_trend = st.number_input("Min 6M Trend %", value=st.session_state.min_6m_trend, step=1.0)
    st.session_state.max_ath_dist = st.number_input("Max ATH Distance %", value=st.session_state.max_ath_dist, step=1.0)
    st.session_state.max_bbpos = st.number_input("Max BBPos (if MACD neg)", value=st.session_state.max_bbpos, step=0.1)
    st.session_state.max_bw = st.number_input("Max BW % (if MACD neg)", value=st.session_state.max_bw, step=1.0)
    st.session_state.max_accel = st.number_input("Max Accel (if MACD neg)", value=st.session_state.max_accel, step=0.5)
    
    if st.button("Restore Defaults"):
        for key, val in FILTER_DEFAULTS.items():
            st.session_state[key] = val
        st.rerun()

# ---------------------------------------------------
# LOAD TICKERS
# ---------------------------------------------------

all_tickers = []
for market_path in selected_markets:
    try:
        df_market = load_csv_from_github(market_path)
        if "Ticker" in df_market.columns:
            tickers = df_market["Ticker"].dropna().tolist()[:max_stocks]
            all_tickers.extend(tickers)
    except Exception: continue

unique_tickers = list(dict.fromkeys(all_tickers))
st.sidebar.markdown(f"**{len(unique_tickers)}** tickers loaded.")

# ---------------------------------------------------
# METRIC FUNCTIONS
# ---------------------------------------------------

def calculate_metrics(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period="1y", auto_adjust=True)
        if len(hist) < 150: return None

        close, volume = hist["Close"], hist["Volume"]
        current_price = close.iloc[-1]

        # Indicators
        bb = BollingerBands(close=close, window=20, window_dev=2)
        upper, lower, middle = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg()
        bbpos = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
        bw_series = ((upper - lower) / middle) * 100
        bw_percentile = bw_series.rank(pct=True).iloc[-1] * 100
        rsi = RSIIndicator(close=close, window=14).rsi().iloc[-1]
        
        # RVOL
        rvol = volume.iloc[-2] / volume.iloc[-22:-2].mean()

        # MACD
        histo = MACD(close).macd_diff()
        macd_delta = histo.iloc[-1] - histo.iloc[-4]

        # Accel
        ret3 = close.pct_change(3)
        accel_z = (ret3.iloc[-1] - ret3.tail(126).mean()) / ret3.tail(126).std()

        # Trends
        t1d = ((current_price / close.iloc[-2]) - 1) * 100
        t1w = ((current_price / close.iloc[-5]) - 1) * 100
        t1m = ((current_price / close.iloc[-21]) - 1) * 100
        t6m = ((current_price / close.iloc[-126]) - 1) * 100
        ath_dist = ((current_price / close.max()) - 1) * 100

        # Earnings
        earn_disp, earn_days = get_earnings(stock)

        # Apply Strict Filters (Earnings Filter Removed)
        if strict_mode:
            reason = None
            if macd_delta < 0:                       reason = f"MACD delta neg ({macd_delta:.3f})"
            elif rvol < st.session_state.min_rvol:   reason = f"RVOL < {st.session_state.min_rvol}"
            elif t1m < st.session_state.min_1m_trend: reason = f"1M Trend < {st.session_state.min_1m_trend}%"
            elif t6m < st.session_state.min_6m_trend: reason = f"6M Trend < {st.session_state.min_6m_trend}%"
            elif ath_dist < st.session_state.max_ath_dist: reason = f"ATH Dist < {st.session_state.max_ath_dist}%"
            elif rsi > st.session_state.max_rsi:     reason = f"RSI > {st.session_state.max_rsi}"
            elif bbpos > st.session_state.max_bbpos and macd_delta < 0: reason = "Ext BBPos + Neg MACD"
            elif bw_percentile > st.session_state.max_bw and macd_delta < 0: reason = "BW Climax + Neg MACD"
            elif accel_z > st.session_state.max_accel and macd_delta < 0: reason = "Accel Explode + Neg MACD"

            if reason:
                return {"_debug": True, "Ticker": ticker, "Filtered By": reason} if debug_mode else None

        # Helper Labels
        def lbl(v, t):
            if t=="bb": return "EXT" if v>1.1 else "PUSH" if v>0.8 else "WEAK" if v<0.2 else "MID"
            if t=="bw": return "CLIMAX" if v>90 else "EXP" if v>70 else "SQUEEZE" if v<20 else "NORMAL"
            if t=="ac": return "EXPLODE" if v>4 else "SURGE" if v>2 else "FAST" if v>1 else "PUSH" if v>0 else "SLOW"
            if t=="rs": return "EUPH" if v>80 else "HOT" if v>70 else "STR" if v>55 else "MID"
            if t=="rv": return "EXT" if v>5 else "HIGH" if v>3 else "ACT" if v>1.5 else "LOW"
            if t=="ma": return "UP STR" if v>0.3 else "UP BUILD" if v>0.05 else "FLAT" if v>-0.05 else "DN WEAK" if v>-0.2 else "DN FAIL"

        return {
            "Ticker": ticker, "BBPos": f"{bbpos:.2f} {lbl(bbpos, 'bb')}", "BW%": f"{bw_percentile:.0f} {lbl(bw_percentile, 'bw')}",
            "Accel": f"{accel_z:.2f} {lbl(accel_z, 'ac')}", "RSI": f"{rsi:.0f} {lbl(rsi, 'rs')}", "RVOL": f"{rvol:.2f} {lbl(rvol, 'rv')}",
            "MACD": lbl(macd_delta, 'ma'), "Earn": earn_disp, "1D": f"{t1d:.1f}%", "1W": f"{t1w:.1f}%", "1M": f"{t1m:.1f}%", 
            "6M": f"{t6m:.1f}%", "ATH%": f"{ath_dist:.1f}%",
            "_bbpos": bbpos, "_bw": bw_percentile, "_accel": accel_z, "_rsi": rsi, "_rvol": rvol, "_macd": macd_delta,
            "_trend_1d": t1d, "_trend_1w": t1w, "_trend_1m": t1m, "_trend_6m": t6m, "_ath": ath_dist, "_earnings_days": float(earn_days)
        }
    except Exception: return None

# ---------------------------------------------------
# STYLING & RESULTS
# ---------------------------------------------------

def rag_color(col, v):
    try: v = float(v)
    except: return RAG_NONE
    if col == "BBPos": return RAG_GREEN if 0.6<=v<=1.0 else RAG_AMBER if (0.4<=v<0.6 or 1.0<v<=1.1) else RAG_RED
    if col == "BW%": return RAG_RED if v>90 else RAG_AMBER if v>80 else RAG_GREEN if v>=25 else RAG_AMBER
    if col == "Accel": return RAG_GREEN if 1.0<=v<=4.0 else RAG_AMBER if 0.0<=v<1.0 else RAG_RED
    if col == "RSI": return RAG_GREEN if 55<=v<=70 else RAG_AMBER if 70<v<=80 else RAG_RED
    if col == "RVOL": return RAG_GREEN if v>1.5 else RAG_AMBER if v>=1.0 else RAG_RED
    if col == "MACD": return RAG_GREEN if v>0.05 else RAG_AMBER if v>=-0.05 else RAG_RED
    if col in ("1D", "1W", "1M", "6M"): return RAG_GREEN if v>0 else RAG_AMBER if v>-5 else RAG_RED
    if col == "ATH%": return RAG_GREEN if v>=-10 else RAG_AMBER if v>=-25 else RAG_RED
    if col == "Earn": return RAG_NONE if v>=999 else RAG_GREEN if v>14 else RAG_AMBER if v>3 else RAG_RED
    return RAG_NONE

RAG_COL_MAP = { "BBPos": "_bbpos", "BW%": "_bw", "Accel": "_accel", "RSI": "_rsi", "RVOL": "_rvol", "MACD": "_macd", "1D": "_trend_1d", "1W": "_trend_1w", "1M": "_trend_1m", "6M": "_trend_6m", "ATH%": "_ath", "Earn": "_earnings_days" }

def apply_rag_styles(df_full):
    df_disp = df_full.drop(columns=list(RAG_COL_MAP.values()))
    style_df = pd.DataFrame("", index=df_disp.index, columns=df_disp.columns)
    for disp, num in RAG_COL_MAP.items():
        if disp in df_disp.columns:
            style_df[disp] = df_full[num].apply(lambda v, c=disp: rag_color(c, v))
    return df_disp.style.apply(lambda col: style_df[col.name], axis=0)

# ---------------------------------------------------
# EXECUTION
# ---------------------------------------------------

if st.button(f"Run Scanner ({len(unique_tickers)} tickers)"):
    results, debug_results = [], []
    progress = st.progress(0)
    for i, t in enumerate(unique_tickers):
        data = calculate_metrics(t)
        if data:
            if data.get("_debug"): debug_results.append(data)
            else: results.append(data)
        progress.progress((i + 1) / len(unique_tickers))

    if debug_mode and debug_results:
        st.subheader("Debug: Filtered Stocks")
        debug_df = pd.DataFrame(debug_results).drop(columns=["_debug"])
        st.dataframe(debug_df, use_container_width=True, hide_index=True)

    if not results:
        st.warning("No stocks passed the filters.")
    else:
        df = pd.DataFrame(results).sort_values(by=["_bbpos", "_accel", "_rvol"], ascending=False).reset_index(drop=True)
        st.subheader("Scanner Results")
        st.dataframe(apply_rag_styles(df), use_container_width=True, height=700)
        st.success(f"{len(df)} stocks found.")
