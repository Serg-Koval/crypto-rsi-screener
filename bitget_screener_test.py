
import time
import requests
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://api.bitget.com"
PRODUCT_TYPE = "usdt-futures"

RSI_PERIOD = 14
RSI_4H_ALERT = 80
RSI_4H_EXTREME = 85
RSI_1H_ALERT = 82

MIN_PRICE_CHANGE_24H = 4
MIN_VOLUME_USD_24H = 5_000_000

PRE_FILTER_TOP_N = 40
FINAL_TOP_N = 30

CANDLE_LIMIT_1H = 500
CANDLE_LIMIT_4H = 500
REQUEST_DELAY_SECONDS = 0.15

WATCH_SYMBOLS = ["VELVETUSDT"]


def create_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "bitget-rsi-screener-test/1.0"})
    return session


SESSION = create_session()


def safe_get_json(endpoint, params=None):
    if params is None:
        params = {}

    url = BASE_URL + endpoint
    response = SESSION.get(url, params=params, timeout=20)

    if response.status_code != 200:
        print("HTTP error:", response.status_code)
        print(response.text[:1000])
        raise Exception(f"HTTP error: {response.status_code}")

    data = response.json()

    if str(data.get("code")) != "00000":
        print("Bitget API error response:")
        print(data)
        raise Exception(f"Bitget API error: {data.get('msg')}")

    return data


def get_contracts():
    data = safe_get_json(
        "/api/v2/mix/market/contracts",
        {"productType": PRODUCT_TYPE},
    )
    return data["data"]


def get_tickers():
    data = safe_get_json(
        "/api/v2/mix/market/tickers",
        {"productType": PRODUCT_TYPE},
    )
    return data["data"]


def get_candles(symbol, granularity, limit):
    data = safe_get_json(
        "/api/v2/mix/market/candles",
        {
            "productType": PRODUCT_TYPE,
            "symbol": symbol,
            "granularity": granularity,
            "limit": str(limit),
        },
    )
    return data["data"]


def normalize_change(value):
    if value is None or pd.isna(value):
        return np.nan

    value = float(value)

    # Bitget often returns change24h as a ratio, e.g. 0.12 = 12%
    if abs(value) <= 2:
        return value * 100

    return value


def contracts_to_df(contracts):
    df = pd.DataFrame(contracts)

    if df.empty:
        return df

    if "symbol" not in df.columns:
        raise Exception("No symbol column in contracts response.")

    df["symbol"] = df["symbol"].astype(str)
    return df.reset_index(drop=True)


def tickers_to_df(tickers):
    df = pd.DataFrame(tickers)

    if df.empty:
        return df

    if "symbol" not in df.columns:
        raise Exception("No symbol column in tickers response.")

    numeric_cols = [
        "lastPr",
        "last",
        "open24h",
        "high24h",
        "low24h",
        "change24h",
        "priceChangePercent",
        "baseVolume",
        "quoteVolume",
        "usdtVolume",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "lastPr" in df.columns:
        df["price"] = df["lastPr"]
    elif "last" in df.columns:
        df["price"] = df["last"]
    else:
        raise Exception("No price column found.")

    if "change24h" in df.columns:
        df["price_change_24h_percent"] = df["change24h"].apply(normalize_change)
    elif "priceChangePercent" in df.columns:
        df["price_change_24h_percent"] = df["priceChangePercent"].apply(normalize_change)
    elif "open24h" in df.columns:
        df["price_change_24h_percent"] = ((df["price"] - df["open24h"]) / df["open24h"]) * 100
    else:
        df["price_change_24h_percent"] = np.nan

    if "usdtVolume" in df.columns:
        df["volume_usd_24h_est"] = df["usdtVolume"]
    elif "quoteVolume" in df.columns:
        df["volume_usd_24h_est"] = df["quoteVolume"]
    elif "baseVolume" in df.columns:
        df["volume_usd_24h_est"] = df["baseVolume"] * df["price"]
    else:
        df["volume_usd_24h_est"] = np.nan

    return df


def candles_to_df(rows):
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if df.shape[1] < 6:
        raise Exception(f"Unexpected candles format. Columns: {df.shape[1]}")

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "base_volume",
        "quote_volume",
    ]

    df = df.iloc[:, : min(df.shape[1], len(columns))]
    df.columns = columns[: df.shape[1]]

    for col in ["open", "high", "low", "close", "base_volume", "quote_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def wilder_rma(series, period):
    values = series.astype(float).to_numpy()
    rma = np.full(len(values), np.nan)

    valid = np.where(~np.isnan(values))[0]

    if len(valid) < period:
        return pd.Series(rma, index=series.index)

    seed = valid[:period]
    seed_end = seed[-1]
    seed_value = np.nanmean(values[seed])

    rma[seed_end] = seed_value
    previous = seed_value

    for i in range(seed_end + 1, len(values)):
        current = values[i]

        if np.isnan(current):
            rma[i] = previous
            continue

        current_rma = ((previous * (period - 1)) + current) / period
        rma[i] = current_rma
        previous = current_rma

    return pd.Series(rma, index=series.index)


def calculate_rsi(df, period=RSI_PERIOD):
    df = df.copy()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    gain.iloc[0] = np.nan
    loss.iloc[0] = np.nan

    avg_gain = wilder_rma(gain, period)
    avg_loss = wilder_rma(loss, period)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50)

    df["rsi"] = rsi
    return df


def get_last_closed_candle(df):
    # Bitget does not provide OKX-style confirm field.
    # We use previous candle as safer closed candle.
    if len(df) >= 2:
        return df.iloc[-2]
    return df.iloc[-1]


def closed_24h_quote_volume(df_1h):
    if df_1h is None or df_1h.empty or "quote_volume" not in df_1h.columns:
        return None

    closed = df_1h.iloc[:-1].copy()

    if len(closed) < 24:
        return None

    return float(closed.tail(24)["quote_volume"].sum())


def volume_change_24h(df_1h):
    if df_1h is None or df_1h.empty or "quote_volume" not in df_1h.columns:
        return None

    closed = df_1h.iloc[:-1].copy()

    if len(closed) < 48:
        return None

    prev_24 = closed.iloc[-48:-24]["quote_volume"].sum()
    last_24 = closed.iloc[-24:]["quote_volume"].sum()

    if prev_24 == 0:
        return None

    return float(((last_24 - prev_24) / prev_24) * 100)


def format_large_number(value):
    if value is None or pd.isna(value):
        return "N/A"

    value = float(value)

    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"

    return f"{value:.2f}"


def classify_signal(rsi_1h, rsi_4h, volume_24h, price_change_24h):
    volume_ok = volume_24h is not None and volume_24h >= MIN_VOLUME_USD_24H
    price_ok = price_change_24h >= MIN_PRICE_CHANGE_24H

    if not volume_ok or not price_ok:
        return "NO_SIGNAL", "Filters not passed"

    rsi_4h_alert = rsi_4h > RSI_4H_ALERT
    rsi_4h_extreme = rsi_4h > RSI_4H_EXTREME
    rsi_1h_alert = rsi_1h > RSI_1H_ALERT

    if rsi_4h_alert and rsi_1h_alert:
        return "COMBINED_OVERHEAT", f"RSI 1H > {RSI_1H_ALERT} and RSI 4H > {RSI_4H_ALERT}"

    if rsi_4h_extreme:
        return "EXTREME_4H", f"RSI 4H > {RSI_4H_EXTREME}"

    if rsi_4h_alert:
        return "STRONG_4H", f"RSI 4H > {RSI_4H_ALERT}"

    if rsi_1h_alert:
        return "EXTREME_1H", f"RSI 1H > {RSI_1H_ALERT}"

    return "NO_SIGNAL", "RSI filters not passed"


def build_market_universe():
    contracts = contracts_to_df(get_contracts())
    tickers = tickers_to_df(get_tickers())

    df = contracts.merge(
        tickers,
        on="symbol",
        how="inner",
        suffixes=("_contract", "_ticker"),
    )

    df = df.dropna(
        subset=[
            "price",
            "price_change_24h_percent",
            "volume_usd_24h_est",
        ]
    ).copy()

    return df


def prefilter_candidates(df_market):
    df = df_market.copy()

    df = df[
        (df["price_change_24h_percent"] >= MIN_PRICE_CHANGE_24H)
        & (df["volume_usd_24h_est"] >= MIN_VOLUME_USD_24H)
    ].copy()

    df = df.sort_values(
        by=["price_change_24h_percent", "volume_usd_24h_est"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return df.head(PRE_FILTER_TOP_N)


def analyze_candidate(symbol, row):
    raw_1h = get_candles(symbol, "1H", CANDLE_LIMIT_1H)
    df_1h = calculate_rsi(candles_to_df(raw_1h), RSI_PERIOD)

    raw_4h = get_candles(symbol, "4H", CANDLE_LIMIT_4H)
    df_4h = calculate_rsi(candles_to_df(raw_4h), RSI_PERIOD)

    last_1h = get_last_closed_candle(df_1h)
    last_4h = get_last_closed_candle(df_4h)

    rsi_1h = float(last_1h["rsi"])
    rsi_4h = float(last_4h["rsi"])
    price = float(last_1h["close"])
    price_change = float(row["price_change_24h_percent"])

    volume_24h = closed_24h_quote_volume(df_1h)
    volume_change = volume_change_24h(df_1h)

    signal_level, reason = classify_signal(
        rsi_1h=rsi_1h,
        rsi_4h=rsi_4h,
        volume_24h=volume_24h,
        price_change_24h=price_change,
    )

    return {
        "exchange": "Bitget",
        "symbol": f"{symbol}.P",
        "raw_symbol": symbol,
        "price": price,
        "rsi_1h": rsi_1h,
        "rsi_4h": rsi_4h,
        "price_change_24h_percent": price_change,
        "volume_usd_24h_exact": volume_24h,
        "volume_change_24h_percent": volume_change,
        "signal_level": signal_level,
        "reason": reason,
    }


def run_screener():
    print("=" * 120)
    print("BITGET RSI SCREENER TEST")
    print("=" * 120)

    df_market = build_market_universe()
    print("Total Bitget USDT futures universe:", len(df_market))

    for symbol in WATCH_SYMBOLS:
        found = symbol in set(df_market["symbol"].astype(str))
        print(f"Watch symbol {symbol}: {'FOUND' if found else 'NOT FOUND'}")

    df_candidates = prefilter_candidates(df_market)
    print("Prefiltered candidates:", len(df_candidates))

    print("\nTop prefiltered candidates:")
    preview = df_candidates[[
        "symbol",
        "price",
        "price_change_24h_percent",
        "volume_usd_24h_est",
    ]].head(15).copy()

    preview["price"] = preview["price"].round(8)
    preview["price_change_24h_percent"] = preview["price_change_24h_percent"].round(2)
    preview["volume_usd_24h_est"] = preview["volume_usd_24h_est"].apply(format_large_number)

    print(preview.to_string(index=False))

    results = []

    for index, row in df_candidates.iterrows():
        symbol = row["symbol"]
        print(f"[{index + 1}/{len(df_candidates)}] Analyzing {symbol}...")

        try:
            results.append(analyze_candidate(symbol, row))
        except Exception as e:
            print(f"Error while analyzing {symbol}: {e}")

        time.sleep(REQUEST_DELAY_SECONDS)

    df_results = pd.DataFrame(results)

    if df_results.empty:
        print("No analysis results.")
        return

    rank = {
        "COMBINED_OVERHEAT": 4,
        "EXTREME_4H": 3,
        "STRONG_4H": 2,
        "EXTREME_1H": 1,
        "NO_SIGNAL": 0,
    }

    df_results["signal_rank"] = df_results["signal_level"].map(rank)

    df_results = df_results.sort_values(
        by=[
            "signal_rank",
            "rsi_4h",
            "rsi_1h",
            "price_change_24h_percent",
            "volume_usd_24h_exact",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    df_active = df_results[df_results["signal_level"] != "NO_SIGNAL"].copy()

    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print("Total universe:", len(df_market))
    print("Prefiltered candidates:", len(df_candidates))
    print("Active signals:", len(df_active))

    if df_active.empty:
        print("No active Bitget signals.")
        return

    df_out = df_active.copy()
    df_out["price"] = df_out["price"].map(lambda x: f"{float(x):.2f}")
    df_out["rsi_1h"] = df_out["rsi_1h"].round(2)
    df_out["rsi_4h"] = df_out["rsi_4h"].round(2)
    df_out["chg_24h_%"] = df_out["price_change_24h_percent"].round(2)
    df_out["vol_24h"] = df_out["volume_usd_24h_exact"].apply(format_large_number)
    df_out["vol_chg_24h_%"] = df_out["volume_change_24h_percent"].round(2)

    cols = [
        "exchange",
        "signal_level",
        "symbol",
        "price",
        "rsi_1h",
        "rsi_4h",
        "chg_24h_%",
        "vol_24h",
        "vol_chg_24h_%",
        "reason",
    ]

    print("\n" + "=" * 120)
    print("ACTIVE SIGNALS ONLY")
    print("=" * 120)
    print(df_out[cols].head(FINAL_TOP_N).to_string(index=False))


def main():
    run_screener()


if __name__ == "__main__":
    main()
