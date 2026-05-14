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
# Semi-transparent so they work in both light & dark mode
# ---------------------------------------------------

RAG_GREEN = "background-color: rgba(0, 200, 80,  0.28)"
RAG_AMBER = "background-color: rgba(255, 170, 0,  0.28)"
RAG_RED   = "background-color: rgba(220, 50,  50, 0.28)"
RAG_NONE  = ""

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
        "No CSV files found. "
        "Check GITHUB_USER / GITHUB_REPO at the top of the script."
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
    10, 500, 100
)

strict_mode = st.sidebar.checkbox(
    "Strict Mode",
    value=True
)

debug_mode = st.sidebar.checkbox(
    "Debug Mode (show why stocks are filtered)",
    value=False
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
# EARNINGS DATE LOOKUP
# Three-method cascade for maximum yfinance compatibility
# ---------------------------------------------------

def get_earnings(stock):
    """
    Returns (display_str, days_until_int).
    Tries earnings_dates -> calendar dict -> calendar DataFrame.
    """
    # Method 1: stock.earnings_dates (most reliable in recent yfinance)
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
    except Exception:
        pass

    # Method 2: stock.calendar as dict (yfinance >= 0.2.x)
    try:
        cal = stock.calendar
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
            if dates:
                next_date = pd.to_datetime(dates[0]).date()
                days = (next_date - datetime.now().date()).days
                return f"{next_date} ({days}d)", days
    except Exception:
        pass

    # Method 3: stock.calendar as DataFrame (older yfinance)
    try:
        cal = stock.calendar
        if cal is not None and hasattr(cal, "empty") and not cal.empty:
            next_date = pd.to_datetime(cal.iloc[0][0]).date()
            days = (next_date - datetime.now().date()).days
            return f"{next_date} ({days}d)", days
    except Exception:
        pass

    return "N/A", 999

# ---------------------------------------------------
# METRIC FUNCTIONS
# ---------------------------------------------------

def calculate_metrics(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period="1y", auto_adjust=True)

        if len(hist) < 150:
            return None

        close  = hist["Close"]
        volume = hist["Volume"]

        current_price = close.iloc[-1]

        # Bollinger Bands
        bb     = BollingerBands(close=close, window=20, window_dev=2)
        upper  = bb.bollinger_hband()
        lower  = bb.bollinger_lband()
        middle = bb.bollinger_mavg()

        bbpos = (
            (current_price - lower.iloc[-1]) /
            (upper.iloc[-1] - lower.iloc[-1])
        )

        bandwidth_series = ((upper - lower) / middle) * 100
        bw_percentile    = bandwidth_series.rank(pct=True).iloc[-1] * 100

        # RSI
        rsi = RSIIndicator(close=close, window=14).rsi().iloc[-1]

        # RVOL
        # Use iloc[-2] (last COMPLETE trading day) to avoid partial-day
        # volume distortion when the scanner runs during market hours.
        # Today's bar accumulates volume throughout the session, so
        # scanning at 9am gives ~1/8th of a full day → RVOL looks tiny.
        complete_vol = volume.iloc[-2]
        avg_volume   = volume.iloc[-22:-2].mean()   # 20-day avg, same window
        rvol         = complete_vol / avg_volume

        # MACD Histogram Slope
        macd_obj   = MACD(close)
        histo      = macd_obj.macd_diff()
        macd_delta = histo.iloc[-1] - histo.iloc[-4]

        # Acceleration Z-Score
        returns_3d = close.pct_change(3)
        current_3d = returns_3d.iloc[-1]
        mean_3d    = returns_3d.tail(126).mean()
        std_3d     = returns_3d.tail(126).std()
        accel_z    = (current_3d - mean_3d) / std_3d

        # Trend Returns
        def calc_return(days):
            return ((close.iloc[-1] / close.iloc[-days]) - 1) * 100

        trend_1d = calc_return(2)
        trend_1w = calc_return(5)
        trend_1m = calc_return(21)
        trend_6m = calc_return(126)

        # ATH Distance
        ath          = close.max()
        ath_distance = ((current_price / ath) - 1) * 100

        # Earnings
        earnings_display, earnings_days = get_earnings(stock)

        # Strict Filters
        if strict_mode:
            filter_reason = None
            if macd_delta < 0:
                filter_reason = f"MACD delta negative ({macd_delta:.3f})"
            elif rvol < 1:
                filter_reason = f"RVOL too low ({rvol:.2f})"
            elif trend_1m < 0:
                filter_reason = f"1M trend negative ({trend_1m:.1f}%)"
            elif trend_6m < 0:
                filter_reason = f"6M trend negative ({trend_6m:.1f}%)"
            elif ath_distance < -35:
                filter_reason = f"Too far from ATH ({ath_distance:.1f}%)"
            elif rsi > 85:
                filter_reason = f"RSI too high ({rsi:.0f})"
            elif bbpos > 1.1 and macd_delta < 0:
                filter_reason = f"BBPos extended + MACD neg"
            elif bw_percentile > 95 and macd_delta < 0:
                filter_reason = f"BW climax + MACD neg"
            elif accel_z > 4 and macd_delta < 0:
                filter_reason = f"Accel exploding + MACD neg"
            elif earnings_days <= 1:
                filter_reason = f"Earnings too close ({earnings_days}d)"

            if filter_reason:
                if debug_mode:
                    return {"_debug": True, "Ticker": ticker, "Filtered By": filter_reason}
                return None

        # Labels
        def bbpos_label(x):
            if x > 1.1: return "EXT"
            if x > 0.8: return "PUSH"
            if x < 0.2: return "WEAK"
            return "MID"

        def bw_label(x):
            if x > 90: return "CLIMAX"
            if x > 70: return "EXP"
            if x < 20: return "SQUEEZE"
            return "NORMAL"

        def accel_label(x):
            if x > 4:  return "EXPLODE"
            if x > 2:  return "SURGE"
            if x > 1:  return "FAST"
            if x > 0:  return "PUSH"
            return "SLOW"

        def rsi_label(x):
            if x > 80: return "EUPH"
            if x > 70: return "HOT"
            if x > 55: return "STR"
            return "MID"

        def rvol_label(x):
            if x > 5:  return "EXT"
            if x > 3:  return "HIGH"
            if x > 1.5: return "ACT"
            return "LOW"

        def macd_label(x):
            if x >  0.3:  return "UP STR"
            if x >  0.05: return "UP BUILD"
            if x > -0.05: return "FLAT"
            if x > -0.2:  return "DN WEAK"
            return "DN FAIL"

        return {
            # Display columns
            "Ticker": ticker,
            "BBPos":  f"{bbpos:.2f} {bbpos_label(bbpos)}",
            "BW%":    f"{bw_percentile:.0f} {bw_label(bw_percentile)}",
            "Accel":  f"{accel_z:.2f} {accel_label(accel_z)}",
            "RSI":    f"{rsi:.0f} {rsi_label(rsi)}",
            "RVOL":   f"{rvol:.2f} {rvol_label(rvol)}",
            "MACD":   macd_label(macd_delta),
            "Earn":   earnings_display,
            "1D":     f"{trend_1d:.1f}%",
            "1W":     f"{trend_1w:.1f}%",
            "1M":     f"{trend_1m:.1f}%",
            "6M":     f"{trend_6m:.1f}%",
            "ATH%":   f"{ath_distance:.1f}%",
            # Hidden numeric columns (used for RAG colouring)
            "_bbpos":         bbpos,
            "_bw":            bw_percentile,
            "_accel":         accel_z,
            "_rsi":           rsi,
            "_rvol":          rvol,
            "_macd":          macd_delta,
            "_trend_1d":      trend_1d,
            "_trend_1w":      trend_1w,
            "_trend_1m":      trend_1m,
            "_trend_6m":      trend_6m,
            "_ath":           ath_distance,
            "_earnings_days": float(earnings_days),
        }

    except Exception:
        return None

# ---------------------------------------------------
# RAG COLOUR LOGIC
# ---------------------------------------------------

def rag_color(col_name, val):
    """Returns a CSS string for a display column given its numeric value."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return RAG_NONE

    if col_name == "BBPos":
        if 0.6 <= v <= 1.0:   return RAG_GREEN  # sweet-spot upper band
        if 0.4 <= v < 0.6:    return RAG_AMBER  # mid-band
        if 1.0 < v <= 1.1:    return RAG_AMBER  # mildly extended
        return RAG_RED                            # very weak or very extended

    if col_name == "BW%":
        if v > 90:             return RAG_RED    # climax / overextended
        if v > 80:             return RAG_AMBER  # getting stretched
        if v >= 25:            return RAG_GREEN  # healthy expansion
        return RAG_AMBER                          # tight squeeze

    if col_name == "Accel":
        if 1.0 <= v <= 4.0:   return RAG_GREEN  # surging nicely
        if 0.0 <= v < 1.0:    return RAG_AMBER  # mild push
        return RAG_RED                            # decelerating or over-extended

    if col_name == "RSI":
        if 55 <= v <= 70:      return RAG_GREEN  # strong trend
        if 70 < v <= 80:       return RAG_AMBER  # hot
        return RAG_RED                            # euphoric or too weak

    if col_name == "RVOL":
        if v > 1.5:            return RAG_GREEN  # active
        if v >= 1.0:           return RAG_AMBER  # average
        return RAG_RED                            # low

    if col_name == "MACD":
        if v > 0.05:           return RAG_GREEN  # building momentum
        if v >= -0.05:         return RAG_AMBER  # flat
        return RAG_RED                            # rolling over

    if col_name in ("1D", "1W", "1M", "6M"):
        if v > 0:              return RAG_GREEN
        if v > -5:             return RAG_AMBER
        return RAG_RED

    if col_name == "ATH%":
        if v >= -10:           return RAG_GREEN  # near ATH
        if v >= -25:           return RAG_AMBER
        return RAG_RED

    if col_name == "Earn":
        if v >= 999:           return RAG_NONE   # N/A — no earnings data
        if v > 14:             return RAG_GREEN  # plenty of runway
        if v > 3:              return RAG_AMBER  # getting close
        return RAG_RED                            # imminent

    return RAG_NONE


# Map each display column to its hidden numeric column
RAG_COL_MAP = {
    "BBPos": "_bbpos",
    "BW%":   "_bw",
    "Accel": "_accel",
    "RSI":   "_rsi",
    "RVOL":  "_rvol",
    "MACD":  "_macd",
    "1D":    "_trend_1d",
    "1W":    "_trend_1w",
    "1M":    "_trend_1m",
    "6M":    "_trend_6m",
    "ATH%":  "_ath",
    "Earn":  "_earnings_days",
}

HIDDEN_COLS = list(RAG_COL_MAP.values())


def apply_rag_styles(df_full):
    """
    Accepts df_full (with hidden numeric cols included).
    Returns a pandas Styler on df_display with RAG colours applied.
    """
    df_display = df_full.drop(columns=HIDDEN_COLS)

    # Build a same-shape DataFrame of CSS strings
    style_df = pd.DataFrame(
        "", index=df_display.index, columns=df_display.columns
    )

    for disp_col, num_col in RAG_COL_MAP.items():
        if disp_col in df_display.columns and num_col in df_full.columns:
            style_df[disp_col] = df_full[num_col].apply(
                lambda v, c=disp_col: rag_color(c, v)
            )

    return df_display.style.apply(
        lambda col: style_df[col.name], axis=0
    )

# ---------------------------------------------------
# RUN SCAN
# ---------------------------------------------------

scan_button = st.button(
    f"Run Scanner  ({len(unique_tickers)} tickers)"
)

if scan_button:

    results       = []
    debug_results = []
    progress      = st.progress(0)
    total         = len(unique_tickers)

    for i, ticker in enumerate(unique_tickers):
        data = calculate_metrics(ticker)
        if data:
            if data.get("_debug"):
                debug_results.append(data)
            else:
                results.append(data)
        progress.progress((i + 1) / total)

    # ── Debug panel ──────────────────────────────────────────────
    if debug_mode and debug_results:
        st.subheader(f"Debug: {len(debug_results)} stocks filtered by strict mode")
        debug_df = pd.DataFrame(debug_results).drop(columns=["_debug"])
        # Tally filter reasons
        tally = debug_df["Filtered By"].value_counts().reset_index()
        tally.columns = ["Filter Reason", "Count"]
        col_d1, col_d2 = st.columns([1, 2])
        with col_d1:
            st.markdown("**Filter reason breakdown**")
            st.dataframe(tally, use_container_width=True, hide_index=True)
        with col_d2:
            st.markdown("**All filtered tickers**")
            st.dataframe(debug_df, use_container_width=True, height=300, hide_index=True)
        st.divider()

    if not results:
        if debug_mode:
            st.warning(
                f"No stocks passed filters. See debug table above — "
                f"{len(debug_results)} stocks were filtered out."
            )
        else:
            st.warning(
                "No stocks matched filters. "
                "Try disabling Strict Mode, or enable Debug Mode to see why stocks are being dropped."
            )
        st.stop()

    df = pd.DataFrame(results)

    # Sort by breakout strength
    df = df.sort_values(
        by=["_bbpos", "_accel", "_rvol"],
        ascending=False
    ).reset_index(drop=True)

    # Apply RAG styling
    st.subheader("Scanner Results")

    styled = apply_rag_styles(df)

    st.dataframe(
        styled,
        use_container_width=True,
        height=700
    )

    st.success(
        f"{len(df)} stocks matched filters across "
        f"{len(selected_markets)} market(s)."
    )

    # RAG colour key
    with st.expander("RAG Colour Key"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("🟢 **GREEN — strong / ideal**")
            st.markdown(
                "BBPos 0.6–1.0 · BW% 25–80 · Accel 1–4 · "
                "RSI 55–70 · RVOL >1.5 · MACD >0.05 · "
                "Trends +ve · ATH% >-10% · Earn >14d"
            )
        with col2:
            st.markdown("🟡 **AMBER — borderline / caution**")
            st.markdown(
                "BBPos mid or mildly ext · BW% squeeze or >80 · "
                "Accel 0–1 · RSI 70–80 · RVOL 1–1.5 · "
                "MACD flat · Trends -5–0% · ATH% -10–25% · Earn 4–14d"
            )
        with col3:
            st.markdown("🔴 **RED — weak / risky**")
            st.markdown(
                "BBPos <0.4 or >1.1 · BW% >90 climax · "
                "Accel <0 · RSI >80 or <55 · RVOL <1 · "
                "MACD <-0.05 · Trends <-5% · ATH% <-25% · Earn ≤3d"
            )
