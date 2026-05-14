# breakout_scanner.py
# Streamlit Breakout Scanner & Persistent Portfolio Tracker
# ---------------------------------------------------------
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
# CONFIG & PERSISTENCE
# ---------------------------------------------------
st.set_page_config(page_title="Breakout Scanner & Portfolio", layout="wide")

PORTFOLIO_FILE = "portfolio_db.csv"

def load_portfolio_from_disk():
    if os.path.exists(PORTFOLIO_FILE):
        return pd.read_csv(PORTFOLIO_FILE).set_index("Ticker").to_dict('index')
    return {}

def save_portfolio_to_disk(portfolio_dict):
    if portfolio_dict:
        df = pd.DataFrame.from_dict(portfolio_dict, orient='index')
        df.index.name = "Ticker"
        df.reset_index().to_csv(PORTFOLIO_FILE, index=False)
    else:
        if os.path.exists(PORTFOLIO_FILE):
            os.remove(PORTFOLIO_FILE)

# Initialize Session State
if "portfolio" not in st.session_state:
    st.session_state.portfolio = load_portfolio_from_disk()

FILTER_DEFAULTS = {
    "st_rvol": 1.5, "st_1m_trend": 0.0, "st_6m_trend": 0.0,
    "st_ath_dist": -20.0, "st_max_rsi": 80.0, "st_max_bbpos": 1.1
}
for key, val in FILTER_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------
# GITHUB DATA FETCHING
# ---------------------------------------------------
GITHUB_USER, GITHUB_REPO, GITHUB_BRANCH = "AshF195", "My-Scanner", "main"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/git/trees/{GITHUB_BRANCH}?recursive=1"
GITHUB_RAW = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}"

@st.cache_data(ttl=300)
def get_github_csv_files():
    try:
        resp = requests.get(GITHUB_API, timeout=10)
        tree = resp.json().get("tree", [])
        return sorted([i["path"] for i in tree if i["path"].lower().endswith(".csv")])
    except: return []

@st.cache_data(ttl=300)
def load_csv_from_github(path):
    return pd.read_csv(io.StringIO(requests.get(f"{GITHUB_RAW}/{path}").text))

# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------
st.sidebar.title("🚀 Navigation")
app_mode = st.sidebar.radio("Go to:", ["Breakout Scanner", "My Portfolio"])

st.sidebar.divider()
st.sidebar.header("Scanner Settings")
all_csv_files = get_github_csv_files()
selected_markets = st.sidebar.multiselect("Markets", options=all_csv_files, default=all_csv_files[:1] if all_csv_files else [])
max_stocks = st.sidebar.slider("Max Stocks Per Market", 10, 500, 100)
strict_mode = st.sidebar.checkbox("Strict Mode", value=True)

with st.sidebar.expander("🛠️ Strict Filter Settings"):
    st.session_state.st_rvol = st.number_input("Min RVOL", value=st.session_state.st_rvol, step=0.1)
    st.session_state.st_max_rsi = st.number_input("Max RSI", value=st.session_state.st_max_rsi, step=1.0)
    st.session_state.st_1m_trend = st.number_input("Min 1M Trend %", value=st.session_state.st_1m_trend, step=1.0)
    st.session_state.st_6m_trend = st.number_input("Min 6M Trend %", value=st.session_state.st_6m_trend, step=1.0)
    st.session_state.st_ath_dist = st.number_input("Max ATH Distance %", value=st.session_state.st_ath_dist, step=1.0)
    
    if st.button("Restore Defaults"):
        for k, v in FILTER_DEFAULTS.items(): st.session_state[k] = v
        st.rerun()

# ---------------------------------------------------
# INDICATOR LOGIC
# ---------------------------------------------------
def get_metrics(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y", auto_adjust=True)
        if len(hist) < 100: return None
        
        close = hist["Close"]
        curr_p = close.iloc[-1]
        
        bb = BollingerBands(close=close)
        bbp = (curr_p - bb.bollinger_lband().iloc[-1]) / (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1])
        rsi = RSIIndicator(close=close).rsi().iloc[-1]
        
        histo = MACD(close).macd_diff()
        m_delta = histo.iloc[-1] - histo.iloc[-4]
        rvol = hist["Volume"].iloc[-2] / hist["Volume"].iloc[-22:-2].mean()
        
        # Returns for table
        t1m = ((curr_p / close.iloc[-21]) - 1) * 100
        t6m = ((curr_p / close.iloc[-126]) - 1) * 100
        ath_dist = ((curr_p / close.max()) - 1) * 100

        return {
            "Ticker": ticker, "Price": curr_p, "RSI": rsi, "RVOL": rvol, 
            "MACD_Delta": m_delta, "BBPos": bbp, "1M": t1m, "6M": t6m, "ATH%": ath_dist,
            "Date_Added": datetime.now().strftime("%Y-%m-%d")
        }
    except: return None

def get_portfolio_status(baseline, current):
    # Logic for status flags
    price_change = ((current['Price'] / baseline['Price']) - 1) * 100
    
    # 🔴 NEGATIVE: Price drops > 3% OR MACD goes significantly negative
    if price_change < -3.0 or current['MACD_Delta'] < -0.05:
        return "🔴 NEGATIVE (Exit/Weak)", price_change
    
    # 🟢 POSITIVE: Price up > 2% AND MACD is still positive
    elif price_change > 2.0 and current['MACD_Delta'] > 0:
        return "🟢 POSITIVE (Trend Up)", price_change
    
    # 🟡 NEUTRAL: Price is range-bound
    else:
        return "🟡 NEUTRAL (Steady)", price_change

# ---------------------------------------------------
# SCANNER MODE
# ---------------------------------------------------
if app_mode == "Breakout Scanner":
    st.header("Breakout Continuation Scanner")
    
    tickers_to_scan = []
    for path in selected_markets:
        try:
            df = load_csv_from_github(path)
            if "Ticker" in df.columns: tickers_to_scan.extend(df["Ticker"].dropna().tolist()[:max_stocks])
        except: continue
    
    unique_scan_list = list(dict.fromkeys(tickers_to_scan))

    if st.button(f"Run Scanner ({len(unique_scan_list)} tickers)"):
        results = []
        prog = st.progress(0)
        for i, t in enumerate(unique_scan_list):
            m = get_metrics(t)
            if m:
                # Apply Strict Filters
                passed = True
                if strict_mode:
                    if m['MACD_Delta'] < 0 or m['RVOL'] < st.session_state.st_rvol or m['RSI'] > st.session_state.st_max_rsi or m['1M'] < st.session_state.st_1m_trend or m['ATH%'] < st.session_state.st_ath_dist:
                        passed = False
                
                if passed: results.append(m)
            prog.progress((i+1)/len(unique_scan_list))
        
        if results:
            df_res = pd.DataFrame(results)
            st.session_state.last_results = df_res
            st.dataframe(df_res.drop(columns=["Date_Added"]), use_container_width=True)
            
            st.divider()
            st.subheader("📥 Add to Portfolio")
            st.info("Saving a stock creates a baseline of its current Price, MACD, and RSI.")
            to_add = st.multiselect("Select Tickers:", options=df_res["Ticker"].tolist())
            
            if st.button("Save Selected to portfolio_db.csv"):
                for t in to_add:
                    snap = df_res[df_res["Ticker"] == t].to_dict('records')[0]
                    st.session_state.portfolio[t] = snap
                
                save_portfolio_to_disk(st.session_state.portfolio)
                st.success(f"Saved {len(to_add)} tickers to permanent storage.")
        else:
            st.warning("No stocks matched your filters.")

# ---------------------------------------------------
# PORTFOLIO MODE
# ---------------------------------------------------
elif app_mode == "My Portfolio":
    st.header("My Portfolio Tracker")
    
    if not st.session_state.portfolio:
        st.info("No stocks saved. Use the Scanner to add stocks to your portfolio.")
    else:
        st.write(f"Comparing current market data against baselines saved in `{PORTFOLIO_FILE}`")
        
        port_results = []
        prog_p = st.progress(0)
        tickers = list(st.session_state.portfolio.keys())
        
        for i, ticker in enumerate(tickers):
            baseline = st.session_state.portfolio[ticker]
            current = get_metrics(ticker)
            if current:
                status, change = get_portfolio_status(baseline, current)
                port_results.append({
                    "Ticker": ticker,
                    "Flag": status,
                    "Gain/Loss %": f"{change:.2f}%",
                    "Price (Entry)": f"${baseline['Price']:.2f}",
                    "Price (Now)": f"${current['Price']:.2f}",
                    "MACD (Entry)": f"{baseline['MACD_Delta']:.3f}",
                    "MACD (Now)": f"{current['MACD_Delta']:.3f}",
                    "RSI (Now)": f"{current['RSI']:.0f}",
                    "Date Saved": baseline['Date_Added']
                })
            prog_p.progress((i+1)/len(tickers))
        
        df_p = pd.DataFrame(port_results)
        
        # Display table with some basic styling
        def color_status(val):
            if "🔴" in val: return "color: #ff4b4b"
            if "🟢" in val: return "color: #00cc50"
            return ""

        st.dataframe(df_p.style.applymap(color_status, subset=['Flag']), use_container_width=True, hide_index=True)
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            ticker_to_rem = st.selectbox("Remove a stock:", options=[""] + tickers)
            if st.button("Remove Ticker"):
                if ticker_to_rem in st.session_state.portfolio:
                    del st.session_state.portfolio[ticker_to_rem]
                    save_portfolio_to_disk(st.session_state.portfolio)
                    st.rerun()
        
        with col2:
            if st.button("Clear All Data"):
                st.session_state.portfolio = {}
                save_portfolio_to_disk({})
                st.rerun()
