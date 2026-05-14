# breakout_scanner.py
# ---------------------------------------
# Breakout Continuation Scanner (Unified Universe + Market Filter)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import glob
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
# LOAD ALL MARKETS
# ---------------------------------------------------

DATA_FOLDER = "markets"

csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

all_data = []

for file in csv_files:
    df = pd.read_csv(file)

    if "Ticker" not in df.columns:
        continue

    market_name = os.path.splitext(os.path.basename(file))[0]

    df["Market"] = market_name

    all_data.append(df)

market_df = pd.concat(all_data, ignore_index=True)

# ---------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------

st.sidebar.header("Scanner Settings")

selected_markets = st.sidebar.multiselect(
    "Select Markets",
    sorted(market_df["Market"].unique()),
    default=sorted(market_df["Market"].unique())
)

max_stocks = st.sidebar.slider(
    "Max Stocks To Scan",
    10,
    500,
    100
)

strict_mode = st.sidebar.checkbox(
    "Strict Mode (recommended)",
    value=True
)

run_scan = st.sidebar.button("Run Scanner")

# ---------------------------------------------------
# FILTER BY MARKET
# ---------------------------------------------------

filtered_df = market_df[
    market_df["Market"].isin(selected_markets)
]

tickers = filtered_df["Ticker"].dropna().unique().tolist()[:max_stocks]

# ---------------------------------------------------
# METRICS FUNCTION
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

        bbpos = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])

        bandwidth_series = ((upper - lower) / middle) * 100
        bw_percentile = bandwidth_series.rank(pct=True).iloc[-1] * 100

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
        # MACD HISTOGRAM + SLOPE
        # ---------------------------------------------------

        macd = MACD(close)
        histo = macd.macd_diff()

        macd_delta = histo.iloc[-1] - histo.iloc[-4]

        # ---------------------------------------------------
        # ACCELERATION (Z SCORE)
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
            return ((close.iloc[-1] / close.iloc[-days]) - 1) * 100

        trend_1d = calc_return(2)
        trend_1w = calc_return(5)
        trend_1m = calc_return(21)
        trend_6m = calc_return(126)

        # ---------------------------------------------------
        # ATH DISTANCE
        # ---------------------------------------------------

        ath = close.max()
        ath_distance = ((current_price / ath) - 1) * 100

        # ---------------------------------------------------
        # EARNINGS
        # ---------------------------------------------------

        earnings_display = "N/A"
        earnings_days = 999

        try:
            cal = stock.calendar

            if cal is not None and not cal.empty:

                earnings_date = pd.to_datetime(cal.iloc[0][0]).date()

                earnings_days = (earnings_date - datetime.now().date()).days

                earnings_display = f"{earnings_date} ({earnings_days}d)"

        except:
            pass

        # ---------------------------------------------------
        # STRICT FILTERS
        # ---------------------------------------------------

        if strict_mode:

            if (
                macd_delta < 0 or
                rvol < 1 or
                trend_1m < 0 or
                trend_6m < 0 or
                ath_distance < -35 or
                rsi > 85 or
                (bbpos > 1.1 and macd_delta < 0) or
                (bw_percentile > 95 and macd_delta < 0) or
                (accel_z > 4 and macd_delta < 0) or
                earnings_days <= 1
            ):
                return None

        # ---------------------------------------------------
        # LABELS
        # ---------------------------------------------------

        def label_bb(x):
            return "EXT" if x > 1.1 else "PUSH" if x > 0.8 else "MID" if x > 0.2 else "WEAK"

        def label_bw(x):
            return "CLIMAX" if x > 90 else "EXP" if x > 70 else "SQUEEZE" if x < 20 else "NORM"

        def label_acc(x):
            return "EXPLODE" if x > 4 else "SURGE" if x > 2 else "FAST" if x > 1 else "PUSH" if x > 0 else "SLOW"

        def label_rsi(x):
            return "EUPH" if x > 80 else "HOT" if x > 70 else "STR" if x > 55 else "MID"

        def label_rvol(x):
            return "EXT" if x > 5 else "HIGH" if x > 3 else "ACT" if x > 1.5 else "LOW"

        def label_macd(x):
            return "↑ STR" if x > 0.3 else "↑ BUILD" if x > 0.05 else "→ FLAT" if x > -0.05 else "↓ WEAK" if x > -0.2 else "↓ FAIL"

        return {
            "Ticker": ticker,
            "Market": filtered_df.loc[filtered_df["Ticker"] == ticker, "Market"].values[0],

            "BBPos": f"{bbpos:.2f} {label_bb(bbpos)}",
            "BW%": f"{bw_percentile:.0f} {label_bw(bw_percentile)}",
            "Accel": f"{accel_z:.2f} {label_acc(accel_z)}",
            "RSI": f"{rsi:.0f} {label_rsi(rsi)}",
            "RVOL": f"{rvol:.2f} {label_rvol(rvol)}",
            "MACDΔ": label_macd(macd_delta),

            "Earn": earnings_display,

            "1D": f"{trend_1d:.1f}%",
            "1W": f"{trend_1w:.1f}%",
            "1M": f"{trend_1m:.1f}%",
            "6M": f"{trend_6m:.1f}%",

            "ATH%": f"{ath_distance:.1f}%"
        }

# ---------------------------------------------------
# RUN SCANNER
# ---------------------------------------------------

if run_scan:

    results = []
    progress = st.progress(0)

    for i, ticker in enumerate(tickers):

        data = calculate_metrics(ticker)

        if data:
            results.append(data)

        progress.progress((i + 1) / len(tickers))

    if not results:
        st.warning("No stocks passed filters.")
        st.stop()

    df = pd.DataFrame(results)

    df = df.sort_values(
        by=["BBPos", "Accel", "RVOL"],
        ascending=False
    )

    st.subheader("Breakout Candidates")

    st.dataframe(df, use_container_width=True, height=700)

    st.success(f"{len(df)} stocks found.")
