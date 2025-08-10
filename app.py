# app.py - Streamlit RSI Dashboard (Binance, Cloud-friendly + 100 symbols + live price)
import math
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------
# Mirrors & HTTP session
# ------------------------------
BINANCE_BASES = [
    "https://data-api.binance.vision",  # پیشنهاد اصلی (کمتر دچار 451)
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; RSI-Dashboard/1.0; +https://streamlit.io)"
})
TIMEOUT = 15
MAX_WORKERS = 12
RETRIES_PER_SYMBOL = 3

# ------------------------------
# Endpoints
# ------------------------------
def url(path: str) -> List[str]:
    return [f"{base}{path}" for base in BINANCE_BASES]

KLINES_URLS = url("/api/v3/klines")
TICKER_PRICE_URLS = url("/api/v3/ticker/price")
TICKER_24H_URLS = url("/api/v3/ticker/24hr")

# ------------------------------
# RSI (Wilder)
# ------------------------------
def compute_rsi_wilder(closes: List[float], period: int = 14) -> Optional[float]:
    if closes is None or len(closes) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        change = closes[i] - closes[i-1]
        if change > 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i-1]
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = math.inf if avg_loss == 0 else (avg_gain / avg_loss)
        rsi = 100 - 100 / (1 + rs)
    return float(rsi)

def rsi_series(closes: List[float], period: int = 14, take: int = 60) -> List[float]:
    if closes is None or len(closes) < period + 1:
        return []
    series = []
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        change = closes[i] - closes[i-1]
        if change > 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period
    rs = math.inf if avg_loss == 0 else (avg_gain / avg_loss)
    rsi = 100 - 100 / (1 + rs)
    series.append(rsi)
    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i-1]
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = math.inf if avg_loss == 0 else (avg_gain / avg_loss)
        rsi = 100 - 100 / (1 + rs)
        series.append(rsi)
    return [round(x, 2) for x in series[-take:]]

# ------------------------------
# Binance fetchers (with mirrors & retries)
# ------------------------------
def _get_json(urls: List[str], params: Dict[str, Any]) -> Any:
    last_err = None
    shuffled = urls[:]
    random.shuffle(shuffled)
    for _ in range(RETRIES_PER_SYMBOL):
        for u in shuffled:
            try:
                r = SESSION.get(u, params=params, timeout=TIMEOUT)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                time.sleep(0.3 + 0.3 * random.random())
        random.shuffle(shuffled)
    raise last_err

def fetch_klines(symbol: str, interval: str = "1m", limit: int = 500) -> Dict[str, Any]:
    data = _get_json(KLINES_URLS, {"symbol": symbol, "interval": interval, "limit": limit})
    closes = [float(k[4]) for k in data]
    times = [int(k[0]) for k in data]
    return {"symbol": symbol, "closes": closes, "times": times}

def fetch_price(symbol: str) -> Optional[float]:
    try:
        data = _get_json(TICKER_PRICE_URLS, {"symbol": symbol})
        # پاسخ می‌تواند dict یا list باشد (برای همه سمبل‌ها). ما dict تک‌سمبلی می‌گیریم.
        if isinstance(data, dict) and "price" in data:
            return float(data["price"])
        return None
    except Exception:
        return None

def fetch_top100_usdt_symbols() -> List[str]:
    """
    از /ticker/24hr همه سمبل‌ها را می‌گیریم، فقط USDT و بدون توکن‌های اهرمی (UP/DOWN) را نگه می‌داریم،
    بعد بر اساس quoteVolume مرتب می‌کنیم و 100 تای اول را برمی‌داریم.
    """
    data = _get_json(TICKER_24H_URLS, {})
    rows = []
    for item in data:
        sym = item.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        # حذف توکن‌های اهرمی/خاص
        if any(bad in sym for bad in ["UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT", "PERP"]):
            continue
        try:
            qv = float(item.get("quoteVolume", "0"))  # بر حسب USDT
        except Exception:
            qv = 0.0
        rows.append((sym, qv))
    rows.sort(key=lambda x: x[1], reverse=True)
    symbols = [s for s, _ in rows[:100]]
    # اگر به هر دلیلی کمتر شد، fallback به چند سمبل اصلی
    if len(symbols) < 20:
        fallback = ["BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","SOLUSDT","ADAUSDT","DOGEUSDT","TRXUSDT","TONUSDT","DOTUSDT"]
        for f in fallback:
            if f not in symbols:
                symbols.append(f)
    return symbols

# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title="RSI Dashboard", page_icon="📈", layout="wide")
st.title("📈 داشبورد لحظه‌ای RSI و قیمت برای ۱۰۰ کوین برتر (Binance)")
st.caption("انتخاب ۱۰۰ کوین برتر به‌صورت داینامیک (quoteVolume 24h). اشباع خرید سبز، اشباع فروش قرمز. علامت ✅ اگر RSI دقیقاً 25 یا 75 شود.")

with st.sidebar:
    st.subheader("تنظیمات")
    interval = st.selectbox("تایم‌فریم", ["1m", "5m", "15m"], index=0)
    period = st.number_input("دوره RSI", min_value=2, max_value=100, value=14, step=1)
    limit = st.number_input("تعداد کندل‌ها (≤1000)", min_value=200, max_value=1000, value=500, step=50,
                            help="هر چه بیشتر باشد RSI دقیق‌تر اما کندتر می‌شود.")
    auto_every = st.number_input("رفرش خودکار (ثانیه)", min_value=0, max_value=600, value=45, step=5,
                                 help="برای رعایت ریت‌لیمیت بهتر است ≥ 45s باشد.")
    search = st.text_input("جستجوی نماد", "")
    st.markdown("---")
    st.write("🔁 برای به‌روزرسانی لیست ۱۰۰تایی:")
    refresh_symbols = st.button("بارگیری ۱۰۰ کوین برتر")

# لیست داینامیک 100 تایی (کش می‌کنیم تا هر 5 دقیقه یک‌بار تازه شود)
@st.cache_data(ttl=300, show_spinner=True)
def get_top100_symbols_cached() -> List[str]:
    return fetch_top100_usdt_symbols()

if refresh_symbols or "symbols_cached" not in st.session_state:
    st.session_state["symbols_cached"] = get_top100_symbols_cached()

symbols = st.session_state["symbols_cached"]

# جستجو
if search:
    symbols = [s for s in symbols if search.lower() in s.lower()]

# Auto-refresh
if auto_every and auto_every > 0:
    try:
        st.autorefresh(interval=int(auto_every) * 1000, key="rsi_autorefresh")
    except Exception:
        pass

# دسته‌ای کدنویسی برای گرفتن کِلاین‌ها
@st.cache_data(ttl=30, show_spinner=False)
def load_batch(symbols: List[str], interval: str, limit: int) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    workers = min(len(symbols), MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_klines, sym, interval, limit): sym for sym in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                results[sym] = fut.result()
            except Exception as e:
                results[sym] = {"symbol": sym, "closes": None, "times": None, "error": str(e)}
    return results

with st.spinner("در حال دریافت داده‌ها از بایننس..."):
    raw_map = load_batch(symbols, interval, limit)

# قیمت لحظه‌ای برای همه نمادها (به‌صورت موازی، اما کنترل‌شده)
def load_prices(symbols: List[str]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    workers = min(len(symbols), 16)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_price, sym): sym for sym in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            out[sym] = fut.result()
    return out

with st.spinner("در حال دریافت قیمت لحظه‌ای..."):
    price_map = load_prices(symbols)

# ------------------------------
# Helpers
# ------------------------------
def mark_if_threshold(value: Optional[float]) -> str:
    """اگر RSI دقیقاً 25.00 یا 75.00 شود، ✅ بزن (گرد به دو رقم اعشار)."""
    if value is None:
        return ""
    val2 = round(float(value), 2)
    return " ✅" if val2 in (25.00, 75.00) else ""

def status_label(rsi: Optional[float]) -> str:
    if rsi is None:
        return "—"
    if rsi >= 70:
        return "🟢 اشباع خرید"
    if rsi <= 30:
        return "🔴 اشباع فروش"
    return "میانی"

def line_color_for(rsi: Optional[float]) -> Optional[str]:
    if rsi is None:
        return None
    if rsi >= 70:
        return "green"
    if rsi <= 30:
        return "red"
    return None  # default

# ------------------------------
# Build table
# ------------------------------
rows = []
now_iso = datetime.now(timezone.utc).isoformat()
latest_rsi_map: Dict[str, Optional[float]] = {}
last_price_map: Dict[str, Optional[float]] = {}

for sym in symbols:
    payload = raw_map.get(sym, {})
    closes = payload.get("closes")
    price = price_map.get(sym)
    last_price_map[sym] = price

    if closes:
        rsi_now = compute_rsi_wilder(closes, period)
        latest_rsi_map[sym] = rsi_now
        series = rsi_series(closes, period, take=60)

        rsi_display = f"{round(rsi_now, 2)}{mark_if_threshold(rsi_now)}" if rsi_now is not None else None
        price_display = f"{price:,.6f}" if isinstance(price, (int, float)) else "—"

        rows.append({
            "Symbol": sym,
            "Price": price_display,
            "RSI": rsi_display,
            "Status": status_label(rsi_now),
            "Updated(UTC)": now_iso,
            "Series": series
        })
    else:
        latest_rsi_map[sym] = None
        rows.append({
            "Symbol": sym,
            "Price": f"{price:,.6f}" if isinstance(price, (int, float)) else "—",
            "RSI": None,
            "Status": "—",
            "Updated(UTC)": now_iso,
            "Series": [],
            "Error": payload.get("error", "")
        })

df = pd.DataFrame(rows)

# نمایش جدول (ستون Series مخفی)
st.dataframe(df.drop(columns=["Series"]), use_container_width=True, hide_index=True)

# ------------------------------
# Detail chart
# ------------------------------
pick_symbol = st.selectbox("نمایش نمودار برای نماد", df["Symbol"])
series = df[df["Symbol"] == pick_symbol]["Series"].iloc[0]
last_rsi = latest_rsi_map.get(pick_symbol)

st.subheader(f"روند RSI برای {pick_symbol}")
if series:
    fig, ax = plt.subplots()
    lc = line_color_for(last_rsi)
    if lc:
        ax.plot(range(len(series)), series, color=lc)
    else:
        ax.plot(range(len(series)), series)

    ax.axhline(70, linestyle="--", color="green")  # اشباع خرید (سبز)
    ax.axhline(30, linestyle="--", color="red")    # اشباع فروش (قرمز)
    ax.set_ylim(0, 100)
    ax.set_ylabel("RSI(14)")
    ax.set_xlabel("آخرین نقاط")
    st.pyplot(fig)
else:
    st.info("داده‌ای برای نمایش موجود نیست.")
