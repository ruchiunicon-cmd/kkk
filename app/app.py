"""
NSE Options & Indices Dashboard (free)  -  Streamlit app

Data sources (sab free):
  - NSE option chain (OI, OI change, PCR, max pain, support/resistance, IV, LTP)
  - NSE FII/DII cash data + participant-wise F&O OI (FII/DII/Pro/Client)
  - Yahoo Finance (yfinance): index prices, VIX, global cues (thoda delayed)
  - Apna trade journal (CSV)

Run:  pip install -r requirements.txt   ->   streamlit run app.py
Disclaimer: sirf analysis tool hai, trading advice nahi.
"""
import io
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:  # pragma: no cover
    IST = None

st.set_page_config(page_title="NSE Options Dashboard", layout="wide", page_icon="📈")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
FNO = {
    "NIFTY":      {"yf": "^NSEI",                 "step": 50},
    "BANKNIFTY":  {"yf": "^NSEBANK",              "step": 100},
    "FINNIFTY":   {"yf": "NIFTY_FIN_SERVICE.NS",  "step": 50},
    "MIDCPNIFTY": {"yf": "NIFTY_MID_SELECT.NS",   "step": 25},
}
VIX_TICKER = "^INDIAVIX"

# Yahoo ke tickers best-effort hain. Jo na mile wo skip ho jata hai.
INDICES = {
    "Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK", "Sensex": "^BSESN",
    "Nifty Next 50": "^NSMIDCP", "Nifty Midcap 50": "^NSEMDCP50",
    "Nifty Midcap 100": "NIFTY_MIDCAP_100.NS", "Nifty Smallcap 100": "^CNXSC",
    "Nifty 100": "^CNX100", "Nifty 500": "^CRSLDX",
    "Nifty Fin Services": "NIFTY_FIN_SERVICE.NS", "Nifty Midcap Select": "NIFTY_MID_SELECT.NS",
    "Nifty IT": "^CNXIT", "Nifty Auto": "^CNXAUTO", "Nifty Pharma": "^CNXPHARMA",
    "Nifty FMCG": "^CNXFMCG", "Nifty Metal": "^CNXMETAL", "Nifty Realty": "^CNXREALTY",
    "Nifty Energy": "^CNXENERGY", "Nifty Infra": "^CNXINFRA", "Nifty PSU Bank": "^CNXPSUBANK",
    "Nifty Media": "^CNXMEDIA", "Nifty Pvt Bank": "NIFTY_PVT_BANK.NS",
    "Nifty Healthcare": "NIFTY_HEALTHCARE.NS", "Nifty Oil & Gas": "NIFTY_OIL_AND_GAS.NS",
    "Nifty Cons Durables": "NIFTY_CONSR_DURBL.NS",
}
GLOBAL = {
    "S&P 500": "^GSPC", "Nasdaq": "^IXIC", "Dow Jones": "^DJI", "Nikkei 225": "^N225",
    "Hang Seng": "^HSI", "Crude WTI": "CL=F", "Brent": "BZ=F", "Gold": "GC=F",
    "USD/INR": "INR=X", "US 10Y Yield": "^TNX", "Dollar Index": "DX-Y.NYB", "India VIX": "^INDIAVIX",
}

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain",
}


def now_ist():
    return datetime.now(IST) if IST else datetime.now()


# ----------------------------------------------------------------------------
# INDICATORS
# ----------------------------------------------------------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df, n=10):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def supertrend(df, n=10, m=3.0):
    """returns (line, direction)  direction: +1 uptrend, -1 downtrend"""
    a = atr(df, n).values
    hl2 = ((df["High"] + df["Low"]) / 2).values
    c = df["Close"].values
    ub = hl2 + m * a
    lb = hl2 - m * a
    fub, flb = ub.copy(), lb.copy()
    dirn = np.ones(len(df))
    st_line = np.full(len(df), np.nan)
    for i in range(1, len(df)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if c[i] > fub[i - 1]:
            dirn[i] = 1
        elif c[i] < flb[i - 1]:
            dirn[i] = -1
        else:
            dirn[i] = dirn[i - 1]
        st_line[i] = flb[i] if dirn[i] == 1 else fub[i]
    return pd.Series(st_line, index=df.index), pd.Series(dirn, index=df.index)


# ----------------------------------------------------------------------------
# DATA: YAHOO
# ----------------------------------------------------------------------------
def _fix_index(df):
    if df is None or df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    if IST:
        df.index = df.index.tz_convert(IST)
    return df


@st.cache_data(ttl=120, show_spinner=False)
def get_hist(ticker, interval, period):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        df = df.dropna(subset=["Close"])
        return _fix_index(df)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def get_quotes(tickers: dict):
    """last, change% for a dict name->ticker"""
    def one(item):
        name, tk = item
        d = get_hist(tk, "1d", "10d")
        if d is None or len(d) < 2:
            return name, None
        return name, (float(d["Close"].iloc[-1]), float(d["Close"].iloc[-1] / d["Close"].iloc[-2] * 100 - 100))
    with ThreadPoolExecutor(8) as ex:
        res = dict(ex.map(one, tickers.items()))
    return res


def analyze_tech(ticker, interval, ema_f, ema_s, rsi_len, st_n, st_m):
    h = get_hist(ticker, interval, "5d")
    d = get_hist(ticker, "1d", "1mo")
    if h is None or h.empty or d is None or len(d) < 2:
        return None
    h = h.copy()
    h["ema_f"] = ema(h["Close"], ema_f)
    h["ema_s"] = ema(h["Close"], ema_s)
    h["rsi"] = rsi(h["Close"], rsi_len)
    h["st"], h["st_dir"] = supertrend(h, st_n, st_m)
    last = h.iloc[-1]
    today = h.index[-1].date()
    prev_days = d[d.index.date < today]
    if prev_days.empty:
        return None
    prev = prev_days.iloc[-1]
    day_h = h[h.index.date == today]
    orange = day_h.between_time("09:15", "09:29")
    or_ready = len(orange) >= 3 and (len(day_h) >= 3)
    out = {
        "close": float(last["Close"]), "ema_f": float(last["ema_f"]), "ema_s": float(last["ema_s"]),
        "rsi": float(last["rsi"]), "st": float(last["st"]) if not np.isnan(last["st"]) else np.nan,
        "st_dir": int(last["st_dir"]), "prev_close": float(prev["Close"]),
        "pdh": float(prev["High"]), "pdl": float(prev["Low"]),
        "day_chg": float(last["Close"] / prev["Close"] * 100 - 100),
        "gap": float(day_h["Open"].iloc[0] / prev["Close"] * 100 - 100) if len(day_h) else np.nan,
        "or_high": float(orange["High"].max()) if or_ready else np.nan,
        "or_low": float(orange["Low"].min()) if or_ready else np.nan,
        "or_ready": bool(or_ready),
        "last_bar": h.index[-1], "is_today": today == now_ist().date(),
        "df": h,
    }
    return out


# ----------------------------------------------------------------------------
# DATA: NSE
# ----------------------------------------------------------------------------
@st.cache_resource(ttl=600)
def nse_session():
    s = requests.Session()
    s.headers.update(NSE_HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=10)
        s.get("https://www.nseindia.com/option-chain", timeout=10)
    except Exception:
        pass
    return s


@st.cache_data(ttl=60, show_spinner=False)
def fetch_chain_json(symbol):
    s = nse_session()
    # 1) purana endpoint
    try:
        r = s.get(f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}", timeout=15)
        if r.status_code == 200 and r.text.strip().startswith("{"):
            j = r.json()
            if j.get("records", {}).get("data"):
                return j
    except Exception:
        pass
    # 2) naya (v3) endpoint
    r = s.get(f"https://www.nseindia.com/api/option-chain-contract-info?symbol={symbol}", timeout=15)
    r.raise_for_status()
    exps = r.json().get("expiryDates", [])
    if not exps:
        raise RuntimeError("NSE se expiry list nahi mili")
    r2 = s.get(f"https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol={symbol}&expiry={exps[0]}", timeout=15)
    r2.raise_for_status()
    j = r2.json()
    j.setdefault("records", {}).setdefault("expiryDates", exps)
    return j


def parse_chain_json(j):
    rec = j["records"]
    spot = rec.get("underlyingValue")
    rows = []
    for d in rec["data"]:
        ce, pe = d.get("CE", {}) or {}, d.get("PE", {}) or {}
        rows.append(dict(
            expiry=d.get("expiryDate") or d.get("expiryDates"), strike=d["strikePrice"],
            ce_oi=ce.get("openInterest", 0), ce_chg=ce.get("changeinOpenInterest", 0),
            ce_vol=ce.get("totalTradedVolume", 0), ce_iv=ce.get("impliedVolatility", np.nan), ce_ltp=ce.get("lastPrice", np.nan),
            pe_oi=pe.get("openInterest", 0), pe_chg=pe.get("changeinOpenInterest", 0),
            pe_vol=pe.get("totalTradedVolume", 0), pe_iv=pe.get("impliedVolatility", np.nan), pe_ltp=pe.get("lastPrice", np.nan),
        ))
    df = pd.DataFrame(rows)
    if spot is None and not df.empty:
        spot = float(df.loc[(df["ce_ltp"] - df["pe_ltp"]).abs().idxmin(), "strike"])
    return df.sort_values(["expiry", "strike"]), spot


def parse_chain_csv(file, spot_hint=None):
    """NSE website se download kiya hua option chain CSV."""
    raw = pd.read_csv(file, header=1, thousands=",")
    raw = raw.replace("-", np.nan)

    def num(col):
        return pd.to_numeric(raw.iloc[:, col].astype(str).str.replace(",", ""), errors="coerce")
    df = pd.DataFrame({
        "expiry": "CSV", "strike": num(11),
        "ce_oi": num(1), "ce_chg": num(2), "ce_vol": num(3), "ce_iv": num(4), "ce_ltp": num(5),
        "pe_oi": num(21), "pe_chg": num(20), "pe_vol": num(19), "pe_iv": num(18), "pe_ltp": num(17),
    }).dropna(subset=["strike"])
    df = df.fillna({"ce_oi": 0, "ce_chg": 0, "ce_vol": 0, "pe_oi": 0, "pe_chg": 0, "pe_vol": 0})
    spot = spot_hint or float(df.loc[(df["ce_ltp"] - df["pe_ltp"]).abs().idxmin(), "strike"])
    return df.sort_values("strike"), spot


def chain_metrics(df, spot):
    df = df.copy()
    for c in ["ce_oi", "pe_oi", "ce_chg", "pe_chg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    tot_ce, tot_pe = df["ce_oi"].sum(), df["pe_oi"].sum()
    pcr = tot_pe / tot_ce if tot_ce > 0 else np.nan
    K, ceoi, peoi = df["strike"].values, df["ce_oi"].values, df["pe_oi"].values
    pain = [(ceoi * np.maximum(k - K, 0)).sum() + (peoi * np.maximum(K - k, 0)).sum() for k in K]
    max_pain = float(K[int(np.argmin(pain))]) if len(K) else np.nan
    atm_row = df.iloc[(df["strike"] - spot).abs().argmin()]
    above, below = df[df["strike"] >= spot], df[df["strike"] <= spot]
    resist = above.nlargest(3, "ce_oi")["strike"].tolist()
    support = below.nlargest(3, "pe_oi")["strike"].tolist()
    net_chg = df["pe_chg"].sum() - df["ce_chg"].sum()
    base = max(tot_ce + tot_pe, 1)
    return dict(
        pcr=pcr, max_pain=max_pain, atm=float(atm_row["strike"]), atm_ce=atm_row["ce_ltp"], atm_pe=atm_row["pe_ltp"],
        atm_iv=np.nanmean([atm_row["ce_iv"], atm_row["pe_iv"]]), straddle=atm_row["ce_ltp"] + atm_row["pe_ltp"],
        resist=resist, support=support, ce_chg=df["ce_chg"].sum(), pe_chg=df["pe_chg"].sum(),
        net_chg=net_chg, net_chg_pct=net_chg / base * 100, tot_ce=tot_ce, tot_pe=tot_pe,
    )


@st.cache_data(ttl=600, show_spinner=False)
def fetch_fii_dii():
    s = nse_session()
    r = s.get("https://www.nseindia.com/api/fiidiiTradeReact", timeout=15)
    r.raise_for_status()
    return pd.DataFrame(r.json())


@st.cache_data(ttl=900, show_spinner=False)
def fetch_participant_oi():
    """Pichhle 10 din mein jo latest file mile (weekend/holiday skip)."""
    s = nse_session()
    for back in range(0, 10):
        d = (now_ist() - timedelta(days=back))
        if d.weekday() >= 5:
            continue
        ds = d.strftime("%d%m%Y")
        url = f"https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{ds}.csv"
        try:
            r = s.get(url, timeout=15)
            if r.status_code == 200 and len(r.text) > 200:
                df = pd.read_csv(io.StringIO(r.text), skiprows=1)
                df.columns = [c.strip() for c in df.columns]
                return d.strftime("%d-%b-%Y"), df
        except Exception:
            continue
    return None, None


# ----------------------------------------------------------------------------
# SCORING
# ----------------------------------------------------------------------------
def score_index(t, cm, vix_chg, p):
    """returns total, list of (label, value, note)"""
    items = []

    def add(label, val, note):
        items.append((label, val, note))

    if t:
        c = t["close"]
        tr = 1.0 if (c > t["ema_f"] > t["ema_s"]) else -1.0 if (c < t["ema_f"] < t["ema_s"]) else 0.0
        add("Trend (EMA)", tr, f"EMA{p['ema_f']}/{p['ema_s']}")
        add("Supertrend", 1.0 if t["st_dir"] == 1 else -1.0, "Up" if t["st_dir"] == 1 else "Down")
        add("RSI", 1.0 if t["rsi"] >= 55 else -1.0 if t["rsi"] <= 45 else 0.0, f"{t['rsi']:.0f}")
        if t["or_ready"]:
            v = 1.0 if c > t["or_high"] else -1.0 if c < t["or_low"] else 0.0
            add("Opening Range (9:15-9:30)", v, "Upar" if v > 0 else "Neeche" if v < 0 else "Andar")
        else:
            add("Opening Range (9:15-9:30)", 0.0, "Abhi ban raha / data nahi")
        v = 1.0 if c > t["pdh"] else -1.0 if c < t["pdl"] else 0.0
        add("Prev Day High/Low", v, "PDH ke upar" if v > 0 else "PDL ke neeche" if v < 0 else "Range ke andar")
    if cm:
        pcr = cm["pcr"]
        v = 0.0 if np.isnan(pcr) else 1.0 if pcr >= p["pcr_bull"] else -1.0 if pcr <= p["pcr_bear"] else 0.0
        add("PCR (OI)", v, "n/a" if np.isnan(pcr) else f"{pcr:.2f}")
        v = 0.5 if cm["net_chg_pct"] > 2 else -0.5 if cm["net_chg_pct"] < -2 else 0.0
        add("OI change (Put vs Call)", v, f"{cm['net_chg_pct']:+.1f}% of total OI")
    if vix_chg is not None and not np.isnan(vix_chg):
        v = 0.5 if vix_chg <= -p["vix_thr"] else -0.5 if vix_chg >= p["vix_thr"] else 0.0
        add("India VIX", v, f"{vix_chg:+.1f}%")
    return sum(i[1] for i in items), items


def suggestion(total, cm, min_score):
    if total >= min_score:
        ltp = f" @ {cm['atm_ce']:.1f}" if cm and not np.isnan(cm["atm_ce"]) else ""
        atm = f"{cm['atm']:.0f} " if cm else ""
        return "BULLISH - UPAR", f"BUY {atm}CE{ltp}", "green"
    if total <= -min_score:
        ltp = f" @ {cm['atm_pe']:.1f}" if cm and not np.isnan(cm["atm_pe"]) else ""
        atm = f"{cm['atm']:.0f} " if cm else ""
        return "BEARISH - NEECHE", f"BUY {atm}PE{ltp}", "red"
    return "NEUTRAL / SIDEWAYS", "WAIT - No trade", "gray"


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
interval = st.sidebar.selectbox("Technical timeframe", ["5m", "15m", "1m"], index=0,
                                help="Subah 9:30 ke liye 5m. Candle close hone ke baad dekho.")
p = dict(
    ema_f=st.sidebar.number_input("EMA fast", 2, 100, 20),
    ema_s=st.sidebar.number_input("EMA slow", 3, 300, 50),
    rsi_len=st.sidebar.number_input("RSI length", 2, 50, 14),
    st_n=st.sidebar.number_input("Supertrend ATR len", 1, 50, 10),
    st_m=st.sidebar.number_input("Supertrend multiplier", 0.5, 10.0, 3.0, 0.5),
    pcr_bull=st.sidebar.number_input("PCR >= x => Bullish", 0.5, 3.0, 1.2, 0.05),
    pcr_bear=st.sidebar.number_input("PCR <= x => Bearish", 0.1, 2.0, 0.8, 0.05),
    vix_thr=st.sidebar.number_input("VIX change % threshold", 0.5, 20.0, 3.0, 0.5),
)
min_score = st.sidebar.number_input("Min |score| for signal", 1.0, 7.0, 2.5, 0.5,
                                    help="Max score 7. Zyada = kam lekin strict signals.")
refresh = st.sidebar.number_input("Auto-refresh (seconds, 0 = off)", 0, 600, 0, 30)
if refresh > 0:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=int(refresh * 1000), key="auto")
    except Exception:
        st.sidebar.info("Auto-refresh ke liye: pip install streamlit-autorefresh")
if st.sidebar.button("🔄 Cache clear / Refresh now"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.caption("Sirf analysis tool hai, trading advice nahi. Options mein poori premium doob sakti hai.")

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.title("📈 NSE Options & Indices Dashboard")
n = now_ist()
mins = n.hour * 60 + n.minute
if n.weekday() >= 5:
    status = "Weekend - market band (data pichhle trading din ka)"
elif mins < 9 * 60 + 15:
    status = "Pre-open. Global cues aur option chain dekhne ka time."
elif mins < 9 * 60 + 30:
    status = "Opening range ban raha hai (9:15-9:30). 9:30 candle close ka wait karo."
elif mins < 9 * 60 + 35:
    status = "9:30 candle close ho gayi - pehla decision point."
elif mins <= 15 * 60 + 30:
    status = "Market open."
else:
    status = "Market close (data aaj ka final)."
st.caption(f"IST {n.strftime('%d-%b-%Y %H:%M:%S')}  |  {status}  |  Yahoo data ~15 min delayed ho sakta hai.")

tabs = st.tabs(["🎯 Signals", "🧾 Option Chain", "🏦 FII / DII", "🏆 Indices Strength", "🌍 Global Cues", "📓 Journal", "ℹ️ Help"])

# ----------------------------------------------------------------------------
# TAB 1: SIGNALS
# ----------------------------------------------------------------------------
with tabs[0]:
    vix = get_hist(VIX_TICKER, "1d", "10d")
    vix_chg = float(vix["Close"].iloc[-1] / vix["Close"].iloc[-2] * 100 - 100) if vix is not None and len(vix) >= 2 else None
    vix_last = float(vix["Close"].iloc[-1]) if vix is not None and len(vix) else None

    summary, details = [], {}
    prog = st.progress(0.0, text="Data la raha hoon...")
    for i, (name, cfg) in enumerate(FNO.items()):
        t = analyze_tech(cfg["yf"], interval, p["ema_f"], p["ema_s"], p["rsi_len"], p["st_n"], p["st_m"])
        cm, chain_err = None, None
        try:
            j = fetch_chain_json(name)
            cdf, spot = parse_chain_json(j)
            first_exp = cdf["expiry"].iloc[0]
            cdf = cdf[cdf["expiry"] == first_exp]
            spot = spot or (t["close"] if t else None)
            cm = chain_metrics(cdf, spot) if spot else None
            if cm:
                cm["expiry"] = first_exp
        except Exception as e:
            chain_err = str(e)[:120]
        total, items = score_index(t, cm, vix_chg, p)
        bias, trade, color = suggestion(total, cm, min_score)
        details[name] = (t, cm, items, bias, trade, color, chain_err)
        summary.append({
            "Index": name,
            "Spot": round(t["close"], 2) if t else None,
            "Day %": round(t["day_chg"], 2) if t else None,
            "Gap %": round(t["gap"], 2) if t and not np.isnan(t["gap"]) else None,
            "RSI": round(t["rsi"]) if t else None,
            "PCR": round(cm["pcr"], 2) if cm else None,
            "Max Pain": cm["max_pain"] if cm else None,
            "Support": ", ".join(f"{x:.0f}" for x in cm["support"][:2]) if cm else None,
            "Resistance": ", ".join(f"{x:.0f}" for x in cm["resist"][:2]) if cm else None,
            "Score": round(total, 1), "Bias": bias, "Trade idea": trade,
        })
        prog.progress((i + 1) / len(FNO))
    prog.empty()

    avg = np.mean([r["Score"] for r in summary])
    mk = "BULLISH" if avg >= min_score * 0.6 else "BEARISH" if avg <= -min_score * 0.6 else "MIXED / SIDEWAYS"
    c1, c2, c3 = st.columns(3)
    c1.metric("Market sentiment", mk, f"avg score {avg:.1f}")
    c2.metric("India VIX", f"{vix_last:.2f}" if vix_last else "-", f"{vix_chg:+.2f}%" if vix_chg is not None else None, delta_color="inverse")
    c3.metric("Timeframe", interval)

    sdf = pd.DataFrame(summary)

    def color_bias(v):
        return "background-color:#1f7a3a;color:white" if "BULL" in str(v) else "background-color:#a32020;color:white" if "BEAR" in str(v) else ""
    st.dataframe(sdf.style.map(color_bias, subset=["Bias"]), use_container_width=True, hide_index=True)

    for name, (t, cm, items, bias, trade, color, chain_err) in details.items():
        with st.expander(f"{name}  |  {bias}  |  {trade}", expanded=False):
            if chain_err:
                st.warning(f"Option chain nahi mila ({chain_err}). 'Option Chain' tab mein CSV upload karke PCR/OI use kar sakte ho.")
            colA, colB = st.columns([1, 1])
            with colA:
                st.markdown("**Score breakdown**")
                st.dataframe(pd.DataFrame(items, columns=["Factor", "Score", "Detail"]), hide_index=True, use_container_width=True)
            with colB:
                if t:
                    st.markdown("**Levels**")
                    st.write(f"PDH / PDL: **{t['pdh']:.1f} / {t['pdl']:.1f}**")
                    if t["or_ready"]:
                        st.write(f"Opening Range: **{t['or_low']:.1f} - {t['or_high']:.1f}**")
                    st.write(f"Supertrend level: **{t['st']:.1f}**")
                    if not t["is_today"]:
                        st.info("Last candle aaj ki nahi hai (market band ya data delay).")
                if cm:
                    st.write(f"Expiry: **{cm['expiry']}** | ATM: **{cm['atm']:.0f}** | ATM IV: **{cm['atm_iv']:.1f}%** | Straddle: **{cm['straddle']:.1f}**")
                    st.write(f"Support (max Put OI): **{', '.join(f'{x:.0f}' for x in cm['support'])}**")
                    st.write(f"Resistance (max Call OI): **{', '.join(f'{x:.0f}' for x in cm['resist'])}**")
            if t is not None:
                dfp = t["df"].tail(120)
                fig = go.Figure(go.Candlestick(x=dfp.index, open=dfp["Open"], high=dfp["High"], low=dfp["Low"], close=dfp["Close"], name=name))
                fig.add_trace(go.Scatter(x=dfp.index, y=dfp["ema_f"], name=f"EMA{p['ema_f']}", line=dict(width=1)))
                fig.add_trace(go.Scatter(x=dfp.index, y=dfp["ema_s"], name=f"EMA{p['ema_s']}", line=dict(width=1)))
                fig.update_layout(height=380, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 2: OPTION CHAIN
# ----------------------------------------------------------------------------
with tabs[1]:
    colx, coly = st.columns([1, 2])
    sym = colx.selectbox("Index", list(FNO.keys()))
    up = coly.file_uploader("NSE se download kiya option chain CSV (agar live fetch fail ho)", type=["csv"])
    cdf, spot, err = None, None, None
    try:
        if up is not None:
            cdf, spot = parse_chain_csv(up)
        else:
            cdf, spot = parse_chain_json(fetch_chain_json(sym))
    except Exception as e:
        err = str(e)
    if err or cdf is None or cdf.empty:
        st.error(f"Option chain load nahi hua: {err}. NSE kabhi kabhi cloud/VPN IP block karta hai. "
                 "nseindia.com/option-chain par 'Download (.csv)' karke yahan upload karo.")
    else:
        exps = list(dict.fromkeys(cdf["expiry"].tolist()))
        exp = st.selectbox("Expiry", exps)
        cdf = cdf[cdf["expiry"] == exp]
        cm = chain_metrics(cdf, spot)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Spot", f"{spot:,.1f}")
        m2.metric("PCR (OI)", f"{cm['pcr']:.2f}")
        m3.metric("Max Pain", f"{cm['max_pain']:.0f}")
        m4.metric("ATM straddle", f"{cm['straddle']:.1f}")
        m5.metric("ATM IV", f"{cm['atm_iv']:.1f}%")
        st.write(f"Support (max Put OI): **{', '.join(f'{x:.0f}' for x in cm['support'])}**  |  "
                 f"Resistance (max Call OI): **{', '.join(f'{x:.0f}' for x in cm['resist'])}**")
        width = st.slider("ATM ke aas-paas kitne strikes dikhane hain", 5, 40, 15)
        idx = (cdf["strike"] - spot).abs().values.argmin()
        view = cdf.iloc[max(0, idx - width): idx + width + 1]
        fig = go.Figure()
        fig.add_bar(x=view["strike"], y=view["ce_oi"], name="Call OI", marker_color="#d9534f")
        fig.add_bar(x=view["strike"], y=view["pe_oi"], name="Put OI", marker_color="#2e9b5b")
        fig.add_vline(x=spot, line_dash="dash", annotation_text="Spot")
        fig.update_layout(barmode="group", height=380, title="Open Interest", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        fig2 = go.Figure()
        fig2.add_bar(x=view["strike"], y=view["ce_chg"], name="Call OI chg", marker_color="#d9534f")
        fig2.add_bar(x=view["strike"], y=view["pe_chg"], name="Put OI chg", marker_color="#2e9b5b")
        fig2.add_vline(x=spot, line_dash="dash")
        fig2.update_layout(barmode="group", height=320, title="Change in OI (aaj)", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(view[["strike", "ce_oi", "ce_chg", "ce_iv", "ce_ltp", "pe_ltp", "pe_iv", "pe_chg", "pe_oi"]],
                     use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
# TAB 3: FII / DII
# ----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("FII / DII - Cash market (NSE)")
    try:
        fd = fetch_fii_dii()
        st.dataframe(fd, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"FII/DII cash data nahi mila: {str(e)[:120]}")
    st.subheader("Participant-wise F&O OI (FII / DII / Pro / Client)")
    dstr, po = fetch_participant_oi()
    if po is None:
        st.warning("Participant OI file nahi mili (NSE block ya holiday). nseindia.com > Market Data > F&O > Participant wise OI se dekh lo.")
    else:
        st.caption(f"Latest file: {dstr}")
        st.dataframe(po, use_container_width=True, hide_index=True)
        try:
            fii = po[po.iloc[:, 0].astype(str).str.upper().str.contains("FII")].iloc[0]
            lng = float(fii["Future Index Long"])
            sht = float(fii["Future Index Short"])
            a, b, c = st.columns(3)
            a.metric("FII Index Fut Long", f"{lng:,.0f}")
            b.metric("FII Index Fut Short", f"{sht:,.0f}")
            c.metric("FII Long %", f"{lng / (lng + sht) * 100:.1f}%")
        except Exception:
            pass

# ----------------------------------------------------------------------------
# TAB 4: INDICES STRENGTH
# ----------------------------------------------------------------------------
with tabs[3]:
    extra = st.text_input("Apne Yahoo tickers jodo (comma separated, jaise ^CNXIT, NIFTY_PVT_BANK.NS)", "")
    inds = dict(INDICES)
    for tk in [x.strip() for x in extra.split(",") if x.strip()]:
        inds[tk] = tk

    @st.cache_data(ttl=600, show_spinner="Indices ka data la raha hoon...")
    def strength(inds_items):
        def one(item):
            name, tk = item
            d = get_hist(tk, "1d", "6mo")
            if d is None or len(d) < 30:
                return None
            c = d["Close"]
            e20, e50 = ema(c, 20).iloc[-1], ema(c, 50).iloc[-1]
            r = rsi(c, 14).iloc[-1]
            tr = 1 if (c.iloc[-1] > e20 > e50) else -1 if (c.iloc[-1] < e20 < e50) else 0
            sc = tr + (1 if r >= 55 else -1 if r <= 45 else 0) + (1 if c.iloc[-1] > c.iloc[-6] else -1)
            return {"Index": name, "Last": round(float(c.iloc[-1]), 2), "Day %": round(float(c.iloc[-1] / c.iloc[-2] * 100 - 100), 2),
                    "5D %": round(float(c.iloc[-1] / c.iloc[-6] * 100 - 100), 2), "1M %": round(float(c.iloc[-1] / c.iloc[-22] * 100 - 100), 2),
                    "RSI": round(float(r)), "Trend": "UP" if tr > 0 else "DOWN" if tr < 0 else "MIXED", "Score": sc}
        with ThreadPoolExecutor(8) as ex:
            rows = [r for r in ex.map(one, inds_items) if r]
        return pd.DataFrame(rows)

    sdf2 = strength(tuple(inds.items()))
    if sdf2.empty:
        st.warning("Data nahi aaya. Internet/Yahoo check karo.")
    else:
        sdf2 = sdf2.sort_values(["Score", "Day %"], ascending=False)
        nb, nr = int((sdf2["Score"] >= 2).sum()), int((sdf2["Score"] <= -2).sum())
        st.metric("Breadth", f"Bullish {nb} | Bearish {nr} | Neutral {len(sdf2) - nb - nr}", f"{len(sdf2)} indices mile")

        def cbias(v):
            return "background-color:#1f7a3a;color:white" if v >= 2 else "background-color:#a32020;color:white" if v <= -2 else ""
        st.dataframe(sdf2.style.map(cbias, subset=["Score"]), use_container_width=True, hide_index=True)
        st.caption("Ye daily-data based strength hai (trend + RSI + 5-din momentum). Jo ticker Yahoo par nahi mila wo skip hua.")

# ----------------------------------------------------------------------------
# TAB 5: GLOBAL CUES
# ----------------------------------------------------------------------------
with tabs[4]:
    q = get_quotes(GLOBAL)
    rows = [{"Market": k, "Last": round(v[0], 2), "Change %": round(v[1], 2)} for k, v in q.items() if v]
    if rows:
        gdf = pd.DataFrame(rows)
        st.dataframe(gdf.style.map(lambda v: "color:#2e9b5b" if isinstance(v, float) and v > 0 else "color:#d9534f" if isinstance(v, float) and v < 0 else "", subset=["Change %"]),
                     use_container_width=True, hide_index=True)
    st.info("GIFT Nifty Yahoo par nahi milta. NSE IX / Google par 'GIFT Nifty' dekh lo (pre-market ke liye important).")

# ----------------------------------------------------------------------------
# TAB 6: JOURNAL
# ----------------------------------------------------------------------------
with tabs[5]:
    JF = "journal.csv"
    cols = ["Date", "Index", "Signal", "Entry", "SL", "Target", "Exit", "Qty", "PnL", "Reason / Notes"]
    jdf = pd.read_csv(JF) if os.path.exists(JF) else pd.DataFrame(columns=cols)
    jdf = st.data_editor(jdf, num_rows="dynamic", use_container_width=True, key="journal")
    c1, c2 = st.columns(2)
    if c1.button("💾 Save journal"):
        jdf.to_csv(JF, index=False)
        st.success("Saved (journal.csv)")
    c2.download_button("⬇️ Download CSV", jdf.to_csv(index=False), "journal.csv", "text/csv")
    pnl = pd.to_numeric(jdf["PnL"], errors="coerce").dropna()
    if len(pnl):
        w = (pnl > 0).sum()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trades", len(pnl))
        m2.metric("Win rate", f"{w / len(pnl) * 100:.0f}%")
        m3.metric("Total PnL", f"{pnl.sum():,.0f}")
        m4.metric("Avg win / Avg loss", f"{pnl[pnl > 0].mean() if w else 0:,.0f} / {pnl[pnl < 0].mean() if (pnl < 0).any() else 0:,.0f}")
    st.caption("Streamlit Cloud par file temporary hoti hai - hamesha Download CSV karke rakho.")

# ----------------------------------------------------------------------------
# TAB 7: HELP
# ----------------------------------------------------------------------------
with tabs[6]:
    st.markdown("""
**Score kaise banta hai (max 7):** Trend (EMA) ±1, Supertrend ±1, RSI ±1, Opening Range ±1, Prev-day High/Low ±1,
PCR ±1, OI change ±0.5, VIX ±0.5. Score `Min |score|` se zyada ho to CE/PE idea aata hai, warna WAIT.

**Subah ka use:** 9:00-9:15 option chain + global cues. 9:30 candle close ke baad Signals tab. 9:45 par confirm.

**Limits:**
- Ye thumb-rule score hai, backtest nahi hua. Accuracy ka koi number nahi hai.
- NSE ka data unofficial web endpoints se aata hai; kabhi bhi band/badal sakta hai ya cloud IP block ho sakta hai. Tab CSV upload use karo.
- Yahoo data delayed ho sakta hai aur kuch index tickers na mile.
- Index options ka premium, theta aur IV risk alag hai - direction sahi ho tab bhi loss ho sakta hai.
- Sirf analysis tool hai, trading advice nahi.
""")
