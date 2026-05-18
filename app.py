# breakout_scanner.py
# Streamlit Breakout Continuation Scanner
# ---------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
import os
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
# PERSISTENCE & PORTFOLIO LOGIC (NEW)
# ---------------------------------------------------
PORTFOLIO_FILE = "portfolio_db.csv"

def save_portfolio(p_dict):
    if p_dict:
        df = pd.DataFrame.from_dict(p_dict, orient="index")

        # The ticker is already used as the dictionary key,
        # so remove any duplicate Ticker column before saving.
        if "Ticker" in df.columns:
            df = df.drop(columns=["Ticker"])

        df.index.name = "Ticker"
        df.reset_index().to_csv(PORTFOLIO_FILE, index=False)

    elif os.path.exists(PORTFOLIO_FILE):
        os.remove(PORTFOLIO_FILE)

# ---------------------------------------------------
# SESSION STATE & DEFAULT SETTINGS
# ---------------------------------------------------

FILTER_DEFAULTS = {
    "st_rvol": 1.5,
    "st_1m_trend": 0.0,
    "st_6m_trend": 0.0,
    "st_ath_dist": -20.0,
    "st_max_rsi": 80.0,
    "st_max_bbpos": 1.1,
    "st_max_bw": 95.0,
    "st_max_accel": 5.0
}

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
# FETCH & LOAD DATA
# ---------------------------------------------------

@st.cache_data(ttl=300)
def get_github_csv_files():
    try:
        resp = requests.get(GITHUB_API, timeout=10)
        resp.raise_for_status()
        tree = resp.json().get("tree", [])
        return sorted([i["path"] for i in tree if i["path"].lower().endswith(".csv") and i["type"] == "blob"])
    except Exception as e:
        st.error(f"GitHub Error: {e}")
        return []

@st.cache_data(ttl=300)
def load_csv_from_github(path):
    url = f"{GITHUB_RAW}/{path}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))

# ---------------------------------------------------
# SIDEBAR & NAVIGATION
# ---------------------------------------------------

st.sidebar.title("🚀 Navigation")
app_mode = st.sidebar.radio("View:", ["Scanner", "My Portfolio"])

st.sidebar.divider()
st.sidebar.header("Scanner Settings")

all_csv_files = get_github_csv_files()
if not all_csv_files:
    st.stop()

selected_markets = st.sidebar.multiselect(
    "Markets",
    options=all_csv_files,
    default=all_csv_files[:1] if all_csv_files else [],
    format_func=lambda x: x.split("/")[-1].replace(".csv", "").upper()
)

max_stocks = st.sidebar.slider("Max Stocks To Scan (per market)", 10, 500, 100)
strict_mode = st.sidebar.checkbox("Strict Mode", value=True)
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

with st.sidebar.expander("🛠️ Strict Filter Settings"):
    st.session_state.st_rvol = st.number_input("Min RVOL", value=st.session_state.st_rvol, step=0.1)
    st.session_state.st_max_rsi = st.number_input("Max RSI", value=st.session_state.st_max_rsi, step=1.0)
    st.session_state.st_1m_trend = st.number_input("Min 1M Trend %", value=st.session_state.st_1m_trend, step=1.0)
    st.session_state.st_6m_trend = st.number_input("Min 6M Trend %", value=st.session_state.st_6m_trend, step=1.0)
    st.session_state.st_ath_dist = st.number_input("Max ATH Distance %", value=st.session_state.st_ath_dist, step=1.0)
    st.session_state.st_max_bbpos = st.number_input("Max BBPos (if MACD neg)", value=st.session_state.st_max_bbpos, step=0.1)
    
    if st.button("Restore Defaults"):
        for key, val in FILTER_DEFAULTS.items():
            st.session_state[key] = val
        st.rerun()

# ---------------------------------------------------
# CORE LOGIC FUNCTIONS
# ---------------------------------------------------

def get_earnings(stock):
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
    except: pass
    try:
        cal = stock.calendar
        if isinstance(cal, dict) and "Earnings Date" in cal:
            next_date = pd.to_datetime(cal["Earnings Date"][0]).date()
            days = (next_date - datetime.now().date()).days
            return f"{next_date} ({days}d)", days
    except: pass
    return "N/A", 999

def calculate_metrics(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period="1y", auto_adjust=True)
        if len(hist) < 150: return None

        close, volume = hist["Close"], hist["Volume"]
        current_price = close.iloc[-1]

        bb = BollingerBands(close=close, window=20, window_dev=2)
        upper, lower, middle = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg()
        bbpos = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
        bw_pct = (((upper - lower) / middle) * 100).rank(pct=True).iloc[-1] * 100
        rsi = RSIIndicator(close=close, window=14).rsi().iloc[-1]
        rvol = volume.iloc[-2] / volume.iloc[-22:-2].mean()
        
        histo = MACD(close).macd_diff()
        macd_delta = histo.iloc[-1] - histo.iloc[-4]

        ret3 = close.pct_change(3)
        accel_z = (ret3.iloc[-1] - ret3.tail(126).mean()) / ret3.tail(126).std()

        t1d = ((current_price / close.iloc[-2]) - 1) * 100
        t1w = ((current_price / close.iloc[-5]) - 1) * 100
        t1m = ((current_price / close.iloc[-21]) - 1) * 100
        t6m = ((current_price / close.iloc[-126]) - 1) * 100
        ath_dist = ((current_price / close.max()) - 1) * 100

        earn_disp, earn_days = get_earnings(stock)

        if strict_mode:
            reason = None
            if macd_delta < 0: reason = f"MACD neg ({macd_delta:.3f})"
            elif rvol < st.session_state.st_rvol: reason = f"RVOL < {st.session_state.st_rvol}"
            elif t1m < st.session_state.st_1m_trend: reason = f"1M Trend < {st.session_state.st_1m_trend}%"
            elif t6m < st.session_state.st_6m_trend: reason = f"6M Trend < {st.session_state.st_6m_trend}%"
            elif ath_dist < st.session_state.st_ath_dist: reason = f"ATH Dist < {st.session_state.st_ath_dist}%"
            elif rsi > st.session_state.st_max_rsi: reason = f"RSI > {st.session_state.st_max_rsi}"
            elif bbpos > st.session_state.st_max_bbpos and macd_delta < 0: reason = "Ext BBPos + Neg MACD"
            if reason:
                if debug_mode: return {"_debug": True, "Ticker": ticker, "Filtered By": reason}
                return None

        def bb_lbl(x): return "EXT" if x > 1.1 else "PUSH" if x > 0.8 else "WEAK" if x < 0.2 else "MID"
        def bw_lbl(x): return "CLIMAX" if x > 90 else "EXP" if x > 70 else "SQUEEZE" if x < 20 else "NORMAL"
        def ac_lbl(x): return "EXPLODE" if x > 4 else "SURGE" if x > 2 else "FAST" if x > 1 else "PUSH" if x > 0 else "SLOW"
        def rs_lbl(x): return "EUPH" if x > 80 else "HOT" if x > 70 else "STR" if x > 55 else "MID"
        def rv_lbl(x): return "EXT" if x > 5 else "HIGH" if x > 3 else "ACT" if x > 1.5 else "LOW"
        def ma_lbl(x): return "UP STR" if x > 0.3 else "UP BUILD" if x > 0.05 else "FLAT" if x > -0.05 else "DN WEAK" if x > -0.2 else "DN FAIL"

        return {
            "Ticker": ticker,
            "Price": current_price,
            "BBPos": f"{bbpos:.2f} {bb_lbl(bbpos)}",
            "BW%": f"{bw_pct:.0f} {bw_lbl(bw_pct)}",
            "Accel": f"{accel_z:.2f} {ac_lbl(accel_z)}",
            "RSI": f"{rsi:.0f} {rs_lbl(rsi)}",
            "RVOL": f"{rvol:.2f} {rv_lbl(rvol)}",
            "MACD": ma_lbl(macd_delta),
            "Earn": earn_disp,
            "1D": f"{t1d:.1f}%", "1W": f"{t1w:.1f}%", "1M": f"{t1m:.1f}%", "6M": f"{t6m:.1f}%", "ATH%": f"{ath_dist:.1f}%",
            "_bbpos": bbpos, "_bw": bw_pct, "_accel": accel_z, "_rsi": rsi, "_rvol": rvol, "_macd": macd_delta,
            "_trend_1d": t1d, "_trend_1w": t1w, "_trend_1m": t1m, "_trend_6m": t6m, "_ath": ath_dist, "_earnings_days": float(earn_days)
        }
    except: return None

# ---------------------------------------------------
# RAG COLOUR LOGIC
# ---------------------------------------------------

def rag_color(col, val):
    try: 
        if isinstance(val, str) and "%" in val: v = float(val.replace("%",""))
        else: v = float(val)
    except: 
        if "🟢" in str(val): return RAG_GREEN
        if "🟡" in str(val): return RAG_AMBER
        if "🔴" in str(val): return RAG_RED
        return RAG_NONE
    if col == "BBPos": return RAG_GREEN if 0.6 <= v <= 1.0 else RAG_AMBER if (0.4 <= v < 0.6 or 1.0 < v <= 1.1) else RAG_RED
    if col == "BW%": return RAG_RED if v > 90 else RAG_AMBER if v > 80 else RAG_GREEN if v >= 25 else RAG_AMBER
    if col == "Accel": return RAG_GREEN if 1.0 <= v <= 4.0 else RAG_AMBER if 0.0 <= v < 1.0 else RAG_RED
    if col == "RSI": return RAG_GREEN if 55 <= v <= 80 else RAG_AMBER if 80 < v <= 85 else RAG_RED
    if col == "RVOL": return RAG_GREEN if v > 1.5 else RAG_AMBER if v >= 1.0 else RAG_RED
    if col == "MACD": return RAG_GREEN if v > 0.05 else RAG_AMBER if v >= -0.05 else RAG_RED
    if col in ("1D", "1W", "1M", "6M", "Gain/Loss"): return RAG_GREEN if v > 0 else RAG_AMBER if v > -5 else RAG_RED
    if col == "ATH%": return RAG_GREEN if v >= -15 else RAG_AMBER if v >= -25 else RAG_RED
    if col == "Earn": return RAG_NONE if v >= 999 else RAG_GREEN if v > 14 else RAG_AMBER if v > 3 else RAG_RED
    return RAG_NONE

RAG_MAP = {"BBPos": "_bbpos", "BW%": "_bw", "Accel": "_accel", "RSI": "_rsi", "RVOL": "_rvol", "MACD": "_macd", "1D": "_trend_1d", "1W": "_trend_1w", "1M": "_trend_1m", "6M": "_trend_6m", "ATH%": "_ath", "Earn": "_earnings_days"}

def apply_rag_styles(df_full):
    # Filter columns to only those that exist in df_full
    existing_map = {k: v for k, v in RAG_MAP.items() if k in df_full.columns and v in df_full.columns}
    # These extra columns are for Portfolio mode
    if "Flag" in df_full.columns: existing_map["Flag"] = "Flag"
    if "Gain/Loss" in df_full.columns: existing_map["Gain/Loss"] = "Gain/Loss"
    
    df_disp = df_full.drop(columns=[v for v in existing_map.values() if v in df_full.columns and v != "Flag" and v != "Gain/Loss"])
    style_df = pd.DataFrame("", index=df_disp.index, columns=df_disp.columns)
    
    for disp, num in existing_map.items():
        if disp in df_disp.columns:
            style_df[disp] = df_full[num].apply(lambda v, c=disp: rag_color(c, v))
    return df_disp.style.apply(lambda col: style_df[col.name], axis=0)

# ---------------------------------------------------
# APP MODES
# ---------------------------------------------------

if app_mode == "Scanner":
    all_tickers = []

    for market_path in selected_markets:
        try:
            df_market = load_csv_from_github(market_path)
            if "Ticker" in df_market.columns:
                all_tickers.extend(df_market["Ticker"].dropna().tolist()[:max_stocks])
        except:
            pass

    unique_tickers = list(dict.fromkeys(all_tickers))

    if "last_results" not in st.session_state:
        st.session_state.last_results = None

    if st.button(f"Run Scanner ({len(unique_tickers)} tickers)"):
        results, debug_results = [], []

        progress = st.progress(0)

        for i, ticker in enumerate(unique_tickers):
            data = calculate_metrics(ticker)

            if data:
                if data.get("_debug"):
                    debug_results.append(data)
                else:
                    results.append(data)

            progress.progress((i + 1) / len(unique_tickers))

        if debug_mode and debug_results:
            st.subheader("Debug: Filtered Stocks")
            st.dataframe(
                pd.DataFrame(debug_results).drop(columns=["_debug"]),
                use_container_width=True,
                hide_index=True
            )

        if not results:
            st.session_state.last_results = None
            st.warning("No stocks passed filters.")
        else:
            df = (
                pd.DataFrame(results)
                .sort_values(by=["_bbpos", "_accel", "_rvol"], ascending=False)
                .reset_index(drop=True)
            )

            st.session_state.last_results = df

    if st.session_state.last_results is not None:
        df = st.session_state.last_results

        st.subheader("Scanner Results")

        st.dataframe(
            apply_rag_styles(df.drop(columns=["Price"])),
            use_container_width=True,
            height=500
        )

        st.divider()
        st.subheader("📥 Save to Portfolio")

        with st.form("save_to_portfolio_form"):
            to_add = st.multiselect(
                "Select Tickers to Baseline:",
                options=df["Ticker"].tolist()
            )

            submitted = st.form_submit_button("Confirm Save to portfolio_db.csv")

            if submitted:
                for t in to_add:
                    snap = df[df["Ticker"] == t].to_dict("records")[0]
                    snap["Baseline_Date"] = datetime.now().strftime("%Y-%m-%d")
                    st.session_state.portfolio[t] = snap

                save_portfolio(st.session_state.portfolio)

                st.success(f"Saved {len(to_add)} tickers.")
