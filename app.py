# breakout_scanner.py
# Streamlit Breakout Continuation Scanner
# ---------------------------------------
# Requirements:
# pip install streamlit yfinance pandas numpy ta

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import MACD
import os
import glob

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Breakout Continuation Scanner",
    layout="wide"
)

st.title("Breakout Continuation Scanner")

# ---------------------------------------------------
# SETTINGS
# ---------------------------------------------------

DATA_FOLDER = "markets"

# ---------------------------------------------------
# LOAD MARKET FILES DYNAMICALLY
# ---------------------------------------------------

csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
market_files = [os.path.basename(f) for f in csv_files]

if not market_files:
    st.error("No CSV files found in /markets folder")
    st.stop()

st.sidebar.header("Scanner Settings")

selected_market = st.sidebar.selectbox(
    "Market",
    market_files
)

max_stocks = st.sidebar.slider(
    "Maximum Stocks To Scan",
    10,
    500,
    100
)

strict_mode = st.sidebar.checkbox(
    "Strict Mode",
    value=True
)

run_scan = st.sidebar.button("Run Scanner")

# ---------------------------------------------------
# LOAD SELECTED CSV
# ---------------------------------------------------

csv_path = os.path.join(DATA_FOLDER, selected_market)

market_df = pd.read_csv(csv_path)

if "Ticker" not in market_df.columns:
    st.error("CSV must contain a 'Ticker' column")
    st.stop()

tickers = market_df["Ticker"].dropna().tolist()[:max_stocks]

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

        bbpos = (
            (current_price - lower.iloc[-1]) /
            (upper.iloc[-1] - lower.iloc[-1])
        )

        bandwidth = (
            (upper.iloc[-1] - lower.iloc[-1]) /
            middle.iloc[-1]
        ) * 100

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

            "BBPos": f"{bbpos:.2f} {bbpos_label(bbpos)}",
            "BW%": f"{bw_percentile:.0f} {bw_label(bw_percentile)}",
            "Accel": f"{accel_z:.2f} {accel_label(accel_z)}",
            "RSI": f"{rsi:.0f} {rsi_label(rsi)}",
            "RVOL": f"{rvol:.2f} {rvol_label(rvol)}",
            "MACDΔ": macd_label(macd_delta),

            "Earn": earnings_display,

            "1D": f"{trend_1d:.1f}%",
            "1W": f"{trend_1w:.1f}%",
            "1M": f"{trend_1m:.1f}%",
            "6M": f"{trend_6m:.1f}%",

            "ATH%": f"{ath_distance:.1f}%"
        }

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None

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
