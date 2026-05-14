# breakout_scanner.py
# Streamlit Breakout Continuation Scanner
# ---------------------------------------
# Requirements:
# pip install streamlit yfinance pandas numpy ta requests

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
# GITHUB SETTINGS — update these two lines
# ---------------------------------------------------

GITHUB_USER = "AshF195"       # <-- change this
GITHUB_REPO = "My-Scanner"      # <-- change this
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
# FETCH CSV LIST FROM GITHUB
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

# ---------------------------------------------------
# LOAD CSV FROM GITHUB
# ---------------------------------------------------

@st.cache_data(ttl=300)
def load_csv_from_github(path):
    url = f"{GITHUB_RAW}/{path}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Scanner Settings")

all_csv_files = get_github_csv_files()

if not all_csv_files:
    st.error(
        "No CSV files found. Check your GITHUB_USER / GITHUB_REPO values at the top of the script."
    )
    st.stop()

selected_markets = st.sidebar.multiselect(
    "Markets (select one or more)",
    options=all_csv_files,
    default=all_csv_files[:1] if all_csv_files else [],
    format_func=lambda x: x.split("/")[-1].replace(".csv", "").upper()
)

if not selected_markets:
    st.info("Select at least one market from the sidebar to continue.")
    st.stop()

max_stocks = st.sidebar.slider(
    "Maximum Stocks To Scan (per market)",
    10,
    500,
    100
)

strict_mode = st.sidebar.checkbox(
    "Strict Mode",
    value=True
)

# ---------------------------------------------------
# LOAD TICKERS FROM ALL SELECTED MARKETS
# ---------------------------------------------------

all_tickers = []

for market_path in selected_markets:
    try:
        df_market = load_csv_from_github(market_path)
        if "Ticker" not in df_market.columns:
            st.warning(
                f"Skipping {market_path} — no 'Ticker' column found."
            )
            continue
        tickers = df_market["Ticker"].dropna().tolist()[:max_stocks]
        all_tickers.extend(tickers)
    except Exception as e:
        st.warning(f"Could not load {market_path}: {e}")

# Deduplicate while preserving order
seen = set()
unique_tickers = []
for t in all_tickers:
    if t not in seen:
        seen.add(t)
        unique_tickers.append(t)

st.sidebar.markdown(
    f"**{len(unique_tickers)}** unique tickers loaded across "
    f"{len(selected_markets)} market(s)"
)

# ---------------------------------------------------
# METRIC FUNCTIONS
# ---------------------------------------------------

def calculate_metrics(ticker):

    try:

        stock = yf.Ticker(ticker)

        hist = stock.history(period="1y", auto_adjust=True)

        if len(hist) < 150:
            return None

        close = hist["Close"]
        volume = hist["Volume"]

        current_price = close.iloc[-1]

        # ---------------------------------------------------
        # BOLLINGER BANDS
        # ---------------------------------------------------

        bb = BollingerBands(close=close, window=20, window_dev=2)

        upper = bb.bollinger_hband()
        lower = bb.bollinger_lband()
        middle = bb.bollinger_mavg()

        bbpos = (
            (current_price - lower.iloc[-1]) /
            (upper.iloc[-1] - lower.iloc[-1])
        )

        bandwidth = (
            (upper.iloc[-1] - lower.iloc[-1]) /
            middle.iloc[-1]
        ) * 100

        # Bandwidth Percentile
        bandwidth_series = (
            (upper - lower) / middle
        ) * 100

        bw_percentile = (
            bandwidth_series.rank(pct=True).iloc[-1]
        ) * 100

        # ---------------------------------------------------
        # RSI
        # ---------------------------------------------------

        rsi = RSIIndicator(close=close, window=14).rsi().iloc[-1]

        # ---------------------------------------------------
        # RVOL
        # ---------------------------------------------------

        avg_volume = volume.tail(20).mean()

        rvol = volume.iloc[-1] / avg_volume

        # ---------------------------------------------------
        # MACD HISTOGRAM SLOPE
        # ---------------------------------------------------

        macd = MACD(close)

        histo = macd.macd_diff()

        macd_delta = histo.iloc[-1] - histo.iloc[-4]

        # ---------------------------------------------------
        # ACCELERATION Z-SCORE
        # ---------------------------------------------------

        returns_3d = close.pct_change(3)

        current_3d = returns_3d.iloc[-1]

        mean_3d = returns_3d.tail(126).mean()
        std_3d = returns_3d.tail(126).std()

        accel_z = (current_3d - mean_3d) / std_3d

        # ---------------------------------------------------
        # TREND RETURNS
        # ---------------------------------------------------

        def calc_return(days):
            return (
                (close.iloc[-1] / close.iloc[-days]) - 1
            ) * 100

        trend_1d = calc_return(2)
        trend_1w = calc_return(5)
        trend_1m = calc_return(21)
        trend_6m = calc_return(126)

        # ---------------------------------------------------
        # ATH DISTANCE
        # ---------------------------------------------------

        ath = close.max()

        ath_distance = (
            (current_price / ath) - 1
        ) * 100

        # ---------------------------------------------------
        # EARNINGS DATE
        # ---------------------------------------------------

        earnings_display = "N/A"
        earnings_days = 999

        try:
            cal = stock.calendar

            if cal is not None and not cal.empty:

                earnings_date = pd.to_datetime(
                    cal.iloc[0][0]
                ).date()

                earnings_days = (
                    earnings_date - datetime.now().date()
                ).days

                earnings_display = f"{earnings_date} ({earnings_days}d)"

        except:
            pass

        # ---------------------------------------------------
        # STRICT FILTERS
        # ---------------------------------------------------

        if strict_mode:

            remove = False

            if macd_delta < 0:
                remove = True

            if rvol < 1:
                remove = True

            if trend_1m < 0:
                remove = True

            if trend_6m < 0:
                remove = True

            if ath_distance < -35:
                remove = True

            if rsi > 85:
                remove = True

            if bbpos > 1.1 and macd_delta < 0:
                remove = True

            if bw_percentile > 95 and macd_delta < 0:
                remove = True

            if accel_z > 4 and macd_delta < 0:
                remove = True

            if earnings_days <= 1:
                remove = True

            if remove:
                return None

        # ---------------------------------------------------
        # LABELS
        # ---------------------------------------------------

        def bbpos_label(x):
            if x > 1.1:
                return "EXT"
            elif x > 0.8:
                return "PUSH"
            elif x < 0.2:
                return "WEAK"
            return "MID"

        def bw_label(x):
            if x > 90:
                return "CLIMAX"
            elif x > 70:
                return "EXP"
            elif x < 20:
                return "SQUEEZE"
            return "NORMAL"

        def accel_label(x):
            if x > 4:
                return "EXPLODE"
            elif x > 2:
                return "SURGE"
            elif x > 1:
                return "FAST"
            elif x > 0:
                return "PUSH"
            return "SLOW"

        def rsi_label(x):
            if x > 80:
                return "EUPH"
            elif x > 70:
                return "HOT"
            elif x > 55:
                return "STR"
            return "MID"

        def rvol_label(x):
            if x > 5:
                return "EXT"
            elif x > 3:
                return "HIGH"
            elif x > 1.5:
                return "ACT"
            return "LOW"

        def macd_label(x):
            if x > 0.3:
                return "↑ STR"
            elif x > 0.05:
                return "↑ BUILD"
            elif x > -0.05:
                return "→ FLAT"
            elif x > -0.2:
                return "↓ WEAK"
            return "↓ FAIL"

        return {
            "Ticker": ticker,

            "BBPos":
                f"{bbpos:.2f} {bbpos_label(bbpos)}",

            "BW%":
                f"{bw_percentile:.0f} {bw_label(bw_percentile)}",

            "Accel":
                f"{accel_z:.2f} {accel_label(accel_z)}",

            "RSI":
                f"{rsi:.0f} {rsi_label(rsi)}",

            "RVOL":
                f"{rvol:.2f} {rvol_label(rvol)}",

            "MACDΔ":
                macd_label(macd_delta),

            "Earn":
                earnings_display,

            "1D":
                f"{trend_1d:.1f}%",

            "1W":
                f"{trend_1w:.1f}%",

            "1M":
                f"{trend_1m:.1f}%",

            "6M":
                f"{trend_6m:.1f}%",

            "ATH%":
                f"{ath_distance:.1f}%",

            # Hidden sort fields
            "_bbpos": bbpos,
            "_accel": accel_z,
            "_rvol": rvol,
            "_macd": macd_delta,
            "_bw": bw_percentile
        }

    except:
        return None

# ---------------------------------------------------
# RUN SCAN
# ---------------------------------------------------

scan_button = st.button(
    f"Run Scanner ({len(unique_tickers)} tickers)"
)

if scan_button:

    results = []

    progress = st.progress(0)

    total = len(unique_tickers)

    for i, ticker in enumerate(unique_tickers):

        data = calculate_metrics(ticker)

        if data:
            results.append(data)

        progress.progress((i + 1) / total)

    if len(results) == 0:
        st.warning("No stocks matched filters.")
        st.stop()

    df = pd.DataFrame(results)

    # ---------------------------------------------------
    # SORTING
    # ---------------------------------------------------

    df = df.sort_values(
        by=["_bbpos", "_accel", "_rvol"],
        ascending=False
    )

    # Drop hidden columns
    hidden_cols = [
        "_bbpos",
        "_accel",
        "_rvol",
        "_macd",
        "_bw"
    ]

    df_display = df.drop(columns=hidden_cols)

    # ---------------------------------------------------
    # DISPLAY
    # ---------------------------------------------------

    st.subheader("Scanner Results")

    st.dataframe(
        df_display,
        use_container_width=True,
        height=700
    )

    st.success(f"{len(df_display)} stocks matched filters.")
