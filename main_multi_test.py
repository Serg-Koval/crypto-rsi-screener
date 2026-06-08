import os
import time
import html
import requests
import pandas as pd
import numpy as np

from datetime import datetime
from zoneinfo import ZoneInfo
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# CONFIG
# ============================================================

KYIV_TZ = ZoneInfo("Europe/Kyiv")

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

SEND_MESSAGE_IF_NO_SIGNALS = True

OKX_BASE_URL = "https://www.okx.com"
OKX_INST_TYPE = "SWAP"
OKX_SETTLE_CCY = "USDT"

BITGET_BASE_URL = "https://api.bitget.com"
BITGET_PRODUCT_TYPE = "usdt-futures"


# ============================================================
# HTTP SESSION
# ============================================================

def create_session():
    session = requests.Session()

    retry_strategy = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": "multi-exchange-rsi-screener-test/1.0"
    })

    return session


SESSION = create_session()


# ============================================================
# COMMON HELPERS
# ============================================================

def safe_get_json(base_url, endpoint, params=None, provider_name="provider"):
    if params is None:
        params = {}

    url = base_url + endpoint
    response = SESSION.get(url, params=params, timeout=20)

    if response.status_code != 200:
        print(f"{provider_name} HTTP error:", response.status_code)
        print(response.text[:1000])
        raise Exception(f"{provider_name} HTTP error: {response.status_code}")

    try:
        data = response.json()
    except Exception as e:
        print(f"{provider_name} JSON parse error.")
        print(response.text[:1000])
        raise e

    return data


def calculate_wilder_rma(series, period):
    values = series.astype(float).to_numpy()
    rma = np.full(len(values), np.nan)

    valid_indexes = np.where(~np.isnan(values))[0]

    if len(valid_indexes) < period:
        return pd.Series(rma, index=series.index)

    seed_indexes = valid_indexes[:period]
    seed_end_index = seed_indexes[-1]

    seed_value = np.nanmean(values[seed_indexes])
    rma[seed_end_index] = seed_value

    previous_value = seed_value

    for i in range(seed_end_index + 1, len(values)):
        current_value = values[i]

        if np.isnan(current_value):
            rma[i] = previous_value
            continue

        current_rma = ((previous_value * (period - 1)) + current_value) / period
        rma[i] = current_rma
        previous_value = current_rma

    return pd.Series(rma, index=series.index)


def calculate_rsi(df, period=RSI_PERIOD):
    df = df.copy()

    delta = df["close"].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    gain.iloc[0] = np.nan
    loss.iloc[0] = np.nan

    avg_gain = calculate_wilder_rma(gain, period)
    avg_loss = calculate_wilder_rma(loss, period)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50)

    df["rsi"] = rsi

    return df


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


def format_price_2(value):
    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):.2f}"


def format_percent_2(value, show_plus=True):
    if value is None or pd.isna(value):
        return "N/A"

    value = float(value)

    if show_plus:
        return f"{value:+.2f}"

    return f"{value:.2f}"


def parse_float_from_value(value):
    if value is None:
        return None

    try:
        text = str(value).replace("%", "").replace("+", "").strip()

        if text.upper() == "N/A":
            return None

        return float(text)

    except Exception:
        return None


def classify_signal(rsi_1h, rsi_4h, exact_volume_24h, price_change_24h):
    volume_ok = (
        exact_volume_24h is not None and
        exact_volume_24h >= MIN_VOLUME_USD_24H
    )

    price_change_ok = price_change_24h >= MIN_PRICE_CHANGE_24H

    if not volume_ok or not price_change_ok:
        return {
            "signal_level": "NO_SIGNAL",
            "reason": "Filters not passed"
        }

    rsi_4h_condition = rsi_4h > RSI_4H_ALERT
    rsi_4h_extreme_condition = rsi_4h > RSI_4H_EXTREME
    rsi_1h_condition = rsi_1h > RSI_1H_ALERT
    combined_condition = rsi_4h_condition and rsi_1h_condition

    if combined_condition:
        return {
            "signal_level": "COMBINED_OVERHEAT",
            "reason": f"RSI 1H > {RSI_1H_ALERT} and RSI 4H > {RSI_4H_ALERT}"
        }

    if rsi_4h_extreme_condition:
        return {
            "signal_level": "EXTREME_4H",
            "reason": f"RSI 4H > {RSI_4H_EXTREME}"
        }

    if rsi_4h_condition:
        return {
            "signal_level": "STRONG_4H",
            "reason": f"RSI 4H > {RSI_4H_ALERT}"
        }

    if rsi_1h_condition:
        return {
            "signal_level": "EXTREME_1H",
            "reason": f"RSI 1H > {RSI_1H_ALERT}"
        }

    return {
        "signal_level": "NO_SIGNAL",
        "reason": "RSI filters not passed"
    }


def get_signal_rank(signal_level):
    ranks = {
        "COMBINED_OVERHEAT": 4,
        "EXTREME_4H": 3,
        "STRONG_4H": 2,
        "EXTREME_1H": 1,
        "NO_SIGNAL": 0,
    }

    return ranks.get(signal_level, 0)


# ============================================================
# OKX PROVIDER
# ============================================================

def okx_get_all_instruments():
    data = safe_get_json(
        base_url=OKX_BASE_URL,
        endpoint="/api/v5/public/instruments",
        params={"instType": OKX_INST_TYPE},
        provider_name="OKX",
    )

    if data.get("code") != "0":
        raise Exception(f"OKX API error: {data.get('msg')}")

    return data["data"]


def okx_get_all_tickers():
    data = safe_get_json(
        base_url=OKX_BASE_URL,
        endpoint="/api/v5/market/tickers",
        params={"instType": OKX_INST_TYPE},
        provider_name="OKX",
    )

    if data.get("code") != "0":
        raise Exception(f"OKX API error: {data.get('msg')}")

    return data["data"]


def okx_get_candles(inst_id, bar, limit):
    data = safe_get_json(
        base_url=OKX_BASE_URL,
        endpoint="/api/v5/market/candles",
        params={
            "instId": inst_id,
            "bar": bar,
            "limit": limit,
        },
        provider_name="OKX",
    )

    if data.get("code") != "0":
        raise Exception(f"OKX API error: {data.get('msg')}")

    return data


def okx_instruments_to_dataframe(instruments):
    df = pd.DataFrame(instruments)

    df = df[
        (df["instType"] == OKX_INST_TYPE)
        & (df["settleCcy"] == OKX_SETTLE_CCY)
        & (df["state"] == "live")
        & (df["instId"].str.endswith("-USDT-SWAP"))
    ].copy()

    return df.reset_index(drop=True)


def okx_tickers_to_dataframe(tickers):
    df = pd.DataFrame(tickers)

    numeric_columns = [
        "last",
        "open24h",
        "high24h",
        "low24h",
        "vol24h",
        "volCcy24h",
        "volCcyQuote24h",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "ts" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")

    return df


def okx_candles_to_dataframe(raw_candles):
    rows = raw_candles["data"]

    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "volume_currency",
            "quote_volume",
            "confirm",
        ],
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "volume_currency",
        "quote_volume",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")
    df["confirm"] = df["confirm"].astype(int)

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def okx_get_last_closed_candle(df):
    closed_df = df[df["confirm"] == 1].copy()

    if closed_df.empty:
        raise Exception("OKX: no closed candles found.")

    return closed_df.iloc[-1]


def okx_closed_24h_quote_volume(df_1h):
    closed_df = df_1h[df_1h["confirm"] == 1].copy()

    if len(closed_df) < 24:
        return None

    return float(closed_df.tail(24)["quote_volume"].sum())


def okx_volume_change_24h(df_1h):
    closed_df = df_1h[df_1h["confirm"] == 1].copy()

    if len(closed_df) < 48:
        return None

    prev_24 = closed_df.iloc[-48:-24]
    last_24 = closed_df.iloc[-24:]

    prev_volume = prev_24["quote_volume"].sum()
    last_volume = last_24["quote_volume"].sum()

    if prev_volume == 0:
        return None

    return float(((last_volume - prev_volume) / prev_volume) * 100)


def okx_make_report_symbol(inst_id):
    return inst_id.replace("-SWAP", "").replace("-", "") + ".P"


def okx_build_market_universe():
    instruments = okx_get_all_instruments()
    df_instruments = okx_instruments_to_dataframe(instruments)

    tickers = okx_get_all_tickers()
    df_tickers = okx_tickers_to_dataframe(tickers)

    df = df_instruments.merge(
        df_tickers,
        on="instId",
        how="inner",
        suffixes=("_instrument", "_ticker"),
    )

    df["price_change_24h_percent"] = (
        (df["last"] - df["open24h"]) / df["open24h"]
    ) * 100

    if "volCcyQuote24h" in df.columns and df["volCcyQuote24h"].notna().any():
        df["volume_usd_24h_est"] = df["volCcyQuote24h"]
    else:
        df["volume_usd_24h_est"] = df["volCcy24h"] * df["last"]

    df = df.dropna(
        subset=[
            "last",
            "open24h",
            "price_change_24h_percent",
            "volume_usd_24h_est",
        ]
    ).copy()

    return df


def okx_prefilter_candidates(df_market):
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


def okx_analyze_candidate(inst_id, ticker_row):
    raw_1h = okx_get_candles(inst_id=inst_id, bar="1H", limit=CANDLE_LIMIT_1H)
    df_1h = okx_candles_to_dataframe(raw_1h)
    df_1h = calculate_rsi(df_1h, period=RSI_PERIOD)

    raw_4h = okx_get_candles(inst_id=inst_id, bar="4H", limit=CANDLE_LIMIT_4H)
    df_4h = okx_candles_to_dataframe(raw_4h)
    df_4h = calculate_rsi(df_4h, period=RSI_PERIOD)

    last_1h = okx_get_last_closed_candle(df_1h)
    last_4h = okx_get_last_closed_candle(df_4h)

    exact_volume_24h = okx_closed_24h_quote_volume(df_1h)
    volume_change_24h = okx_volume_change_24h(df_1h)

    rsi_1h = float(last_1h["rsi"])
    rsi_4h = float(last_4h["rsi"])
    price = float(last_1h["close"])
    price_change_24h = float(ticker_row["price_change_24h_percent"])

    classification = classify_signal(
        rsi_1h=rsi_1h,
        rsi_4h=rsi_4h,
        exact_volume_24h=exact_volume_24h,
        price_change_24h=price_change_24h,
    )

    return {
        "exchange": "OKX",
        "provider": "OKX",
        "raw_symbol": inst_id,
        "symbol": okx_make_report_symbol(inst_id),
        "price": price,
        "rsi_1h": rsi_1h,
        "rsi_4h": rsi_4h,
        "volume_usd_24h_exact": exact_volume_24h,
        "volume_change_24h_percent": volume_change_24h,
        "price_change_24h_percent": price_change_24h,
        "signal_level": classification["signal_level"],
        "reason": classification["reason"],
    }


def run_okx_screener():
    print("\n" + "=" * 120)
    print("RUNNING OKX PROVIDER")
    print("=" * 120)

    df_market = okx_build_market_universe()
    total_universe_count = len(df_market)

    df_candidates = okx_prefilter_candidates(df_market)
    prefiltered_count = len(df_candidates)

    print("OKX total universe:", total_universe_count)
    print("OKX prefiltered:", prefiltered_count)

    results = []

    for index, row in df_candidates.iterrows():
        inst_id = row["instId"]

        try:
            result = okx_analyze_candidate(inst_id, row)
            results.append(result)

        except Exception as e:
            print(f"OKX error while analyzing {inst_id}: {e}")

        time.sleep(REQUEST_DELAY_SECONDS)

    df_results = pd.DataFrame(results)

    if df_results.empty:
        return df_results, total_universe_count, prefiltered_count, 0

    df_results["signal_rank"] = df_results["signal_level"].apply(get_signal_rank)

    active_count = len(df_results[df_results["signal_level"] != "NO_SIGNAL"])

    return df_results, total_universe_count, prefiltered_count, active_count


# ============================================================
# BITGET PROVIDER
# ============================================================

def bitget_success(data):
    return str(data.get("code")) == "00000"


def bitget_get_contracts():
    data = safe_get_json(
        base_url=BITGET_BASE_URL,
        endpoint="/api/v2/mix/market/contracts",
        params={"productType": BITGET_PRODUCT_TYPE},
        provider_name="Bitget",
    )

    if not bitget_success(data):
        raise Exception(f"Bitget API error: {data.get('msg')}")

    return data["data"]


def bitget_get_tickers():
    data = safe_get_json(
        base_url=BITGET_BASE_URL,
        endpoint="/api/v2/mix/market/tickers",
        params={"productType": BITGET_PRODUCT_TYPE},
        provider_name="Bitget",
    )

    if not bitget_success(data):
        raise Exception(f"Bitget API error: {data.get('msg')}")

    return data["data"]


def bitget_get_candles(symbol, granularity, limit):
    data = safe_get_json(
        base_url=BITGET_BASE_URL,
        endpoint="/api/v2/mix/market/candles",
        params={
            "productType": BITGET_PRODUCT_TYPE,
            "symbol": symbol,
            "granularity": granularity,
            "limit": str(limit),
        },
        provider_name="Bitget",
    )

    if not bitget_success(data):
        raise Exception(f"Bitget API error: {data.get('msg')}")

    return data["data"]


def bitget_contracts_to_dataframe(contracts):
    df = pd.DataFrame(contracts)

    if df.empty:
        return df

    if "symbol" not in df.columns:
        raise Exception("Bitget contracts response has no symbol column.")

    df["symbol"] = df["symbol"].astype(str)

    return df.reset_index(drop=True)


def bitget_normalize_change(value):
    if value is None or pd.isna(value):
        return np.nan

    value = float(value)

    # Bitget can return change as ratio, e.g. 0.12 = 12%.
    if abs(value) <= 2:
        return value * 100

    return value


def bitget_tickers_to_dataframe(tickers):
    df = pd.DataFrame(tickers)

    if df.empty:
        return df

    if "symbol" not in df.columns:
        raise Exception("Bitget tickers response has no symbol column.")

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
        raise Exception("Bitget: no price column found.")

    if "change24h" in df.columns:
        df["price_change_24h_percent"] = df["change24h"].apply(bitget_normalize_change)
    elif "priceChangePercent" in df.columns:
        df["price_change_24h_percent"] = df["priceChangePercent"].apply(bitget_normalize_change)
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


def bitget_candles_to_dataframe(rows):
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if df.shape[1] < 6:
        raise Exception(f"Bitget: unexpected candle format. Columns: {df.shape[1]}")

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


def bitget_get_last_closed_candle(df):
    # Bitget does not provide OKX-style confirm field.
    # Use previous candle as safer closed candle.
    if df is None or df.empty:
        raise Exception("Bitget: empty candles dataframe.")

    if len(df) >= 2:
        return df.iloc[-2]

    return df.iloc[-1]


def bitget_closed_24h_quote_volume(df_1h):
    if df_1h is None or df_1h.empty or "quote_volume" not in df_1h.columns:
        return None

    closed_df = df_1h.iloc[:-1].copy()

    if len(closed_df) < 24:
        return None

    return float(closed_df.tail(24)["quote_volume"].sum())


def bitget_volume_change_24h(df_1h):
    if df_1h is None or df_1h.empty or "quote_volume" not in df_1h.columns:
        return None

    closed_df = df_1h.iloc[:-1].copy()

    if len(closed_df) < 48:
        return None

    prev_24 = closed_df.iloc[-48:-24]["quote_volume"].sum()
    last_24 = closed_df.iloc[-24:]["quote_volume"].sum()

    if prev_24 == 0:
        return None

    return float(((last_24 - prev_24) / prev_24) * 100)


def bitget_build_market_universe():
    contracts = bitget_get_contracts()
    tickers = bitget_get_tickers()

    df_contracts = bitget_contracts_to_dataframe(contracts)
    df_tickers = bitget_tickers_to_dataframe(tickers)

    df = df_contracts.merge(
        df_tickers,
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


def bitget_prefilter_candidates(df_market):
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


def bitget_analyze_candidate(symbol, ticker_row):
    raw_1h = bitget_get_candles(symbol=symbol, granularity="1H", limit=CANDLE_LIMIT_1H)
    df_1h = bitget_candles_to_dataframe(raw_1h)
    df_1h = calculate_rsi(df_1h, period=RSI_PERIOD)

    raw_4h = bitget_get_candles(symbol=symbol, granularity="4H", limit=CANDLE_LIMIT_4H)
    df_4h = bitget_candles_to_dataframe(raw_4h)
    df_4h = calculate_rsi(df_4h, period=RSI_PERIOD)

    last_1h = bitget_get_last_closed_candle(df_1h)
    last_4h = bitget_get_last_closed_candle(df_4h)

    exact_volume_24h = bitget_closed_24h_quote_volume(df_1h)
    volume_change_24h = bitget_volume_change_24h(df_1h)

    rsi_1h = float(last_1h["rsi"])
    rsi_4h = float(last_4h["rsi"])
    price = float(last_1h["close"])
    price_change_24h = float(ticker_row["price_change_24h_percent"])

    classification = classify_signal(
        rsi_1h=rsi_1h,
        rsi_4h=rsi_4h,
        exact_volume_24h=exact_volume_24h,
        price_change_24h=price_change_24h,
    )

    return {
        "exchange": "Bitget",
        "provider": "Bitget",
        "raw_symbol": symbol,
        "symbol": f"{symbol}.P",
        "price": price,
        "rsi_1h": rsi_1h,
        "rsi_4h": rsi_4h,
        "volume_usd_24h_exact": exact_volume_24h,
        "volume_change_24h_percent": volume_change_24h,
        "price_change_24h_percent": price_change_24h,
        "signal_level": classification["signal_level"],
        "reason": classification["reason"],
    }


def run_bitget_screener():
    print("\n" + "=" * 120)
    print("RUNNING BITGET PROVIDER")
    print("=" * 120)

    df_market = bitget_build_market_universe()
    total_universe_count = len(df_market)

    df_candidates = bitget_prefilter_candidates(df_market)
    prefiltered_count = len(df_candidates)

    print("Bitget total universe:", total_universe_count)
    print("Bitget prefiltered:", prefiltered_count)

    results = []

    for index, row in df_candidates.iterrows():
        symbol = row["symbol"]

        try:
            result = bitget_analyze_candidate(symbol, row)
            results.append(result)

        except Exception as e:
            print(f"Bitget error while analyzing {symbol}: {e}")

        time.sleep(REQUEST_DELAY_SECONDS)

    df_results = pd.DataFrame(results)

    if df_results.empty:
        return df_results, total_universe_count, prefiltered_count, 0

    df_results["signal_rank"] = df_results["signal_level"].apply(get_signal_rank)

    active_count = len(df_results[df_results["signal_level"] != "NO_SIGNAL"])

    return df_results, total_universe_count, prefiltered_count, active_count


# ============================================================
# TELEGRAM
# ============================================================

def get_telegram_credentials():
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token:
        raise Exception("TELEGRAM_BOT_TOKEN is missing.")

    if not chat_id:
        raise Exception("TELEGRAM_CHAT_ID is missing.")

    return bot_token, chat_id


def telegram_signal_label(signal_level):
    labels = {
        "COMBINED_OVERHEAT": "🔴🔴 COMBINED OVERHEAT",
        "EXTREME_4H": "🔴 EXTREME 4H",
        "STRONG_4H": "🟠 STRONG 4H",
        "EXTREME_1H": "🟡 EXTREME 1H",
        "NO_SIGNAL": "⚪ NO SIGNAL",
    }

    return labels.get(signal_level, signal_level)


def telegram_signal_badges(row):
    badges = []

    chg_24h = parse_float_from_value(row.get("chg_24h_%"))
    vol_chg = parse_float_from_value(row.get("vol_chg_24h_%"))

    if chg_24h is not None and chg_24h >= 20:
        badges.append("🚀 Strong pump")

    if vol_chg is not None and vol_chg >= 100:
        badges.append("🔥 Volume spike")

    if vol_chg is not None and vol_chg < 0:
        badges.append("⚠️ Volume fading")

    return badges


def split_long_message(text, max_length=3900):
    if len(text) <= max_length:
        return [text]

    parts = []
    current_part = ""

    for line in text.split("\n"):
        if len(current_part) + len(line) + 1 > max_length:
            parts.append(current_part)
            current_part = line
        else:
            current_part += "\n" + line if current_part else line

    if current_part:
        parts.append(current_part)

    return parts


def send_telegram_message(text):
    bot_token, chat_id = get_telegram_credentials()

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    response = SESSION.post(url, json=payload, timeout=20)

    if response.status_code != 200:
        print("Telegram error response:")
        print(response.text)
        raise Exception(f"Telegram sendMessage error: {response.status_code}")

    print("Telegram message sent.")


def send_telegram_message_safe(text):
    for index, part in enumerate(split_long_message(text)):
        if index > 0:
            part = f"Part {index + 1}\n\n" + part

        send_telegram_message(part)
        time.sleep(1)


def prepare_active_output_table(df_active):
    if df_active is None or df_active.empty:
        return pd.DataFrame()

    df_out = df_active.copy()

    df_out["price"] = df_out["price"].apply(format_price_2)
    df_out["rsi_1h"] = df_out["rsi_1h"].round(2)
    df_out["rsi_4h"] = df_out["rsi_4h"].round(2)
    df_out["chg_24h_%"] = df_out["price_change_24h_percent"].apply(format_percent_2)
    df_out["vol_24h"] = df_out["volume_usd_24h_exact"].apply(format_large_number)
    df_out["vol_chg_24h_%"] = df_out["volume_change_24h_percent"].apply(format_percent_2)

    columns = [
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

    return df_out[columns]


def format_multi_provider_telegram(
    df_active_output,
    okx_total,
    okx_prefiltered,
    okx_active,
    bitget_total,
    bitget_prefiltered,
    bitget_active,
):
    now_kyiv = datetime.now(KYIV_TZ).strftime("%Y-%m-%d %H:%M Kyiv")

    lines = []

    lines.append("🚨 <b>Multi-Exchange RSI Screener TEST</b>")
    lines.append(f"🕒 <code>{html.escape(now_kyiv)}</code>")
    lines.append("")
    lines.append(f"OKX: <code>{okx_active}</code> signals / <code>{okx_total}</code> universe")
    lines.append(f"Bitget: <code>{bitget_active}</code> signals / <code>{bitget_total}</code> universe")
    lines.append("")

    if df_active_output is None or df_active_output.empty:
        lines.append("✅ Активних сигналів немає.")
        return "\n".join(lines)

    for idx, row in df_active_output.iterrows():
        signal_label = telegram_signal_label(row["signal_level"])

        exchange = html.escape(str(row["exchange"]))
        symbol = html.escape(str(row["symbol"]))
        price = html.escape(str(row["price"]))
        rsi_1h = html.escape(str(row["rsi_1h"]))
        rsi_4h = html.escape(str(row["rsi_4h"]))
        chg_24h = html.escape(str(row["chg_24h_%"]))
        vol_24h = html.escape(str(row["vol_24h"]))
        vol_chg = html.escape(str(row["vol_chg_24h_%"]))
        reason = html.escape(str(row["reason"]))

        lines.append(f"{idx + 1}) {signal_label}")
        lines.append(f"📌 <b>{symbol}</b>")
        lines.append(f"🏦 Exchange: <b>{exchange}</b>")
        lines.append(f"💵 Price: <code>{price}</code>")
        lines.append(f"📊 RSI 1H / 4H: <code>{rsi_1h} / {rsi_4h}</code>")
        lines.append(f"📈 24h: <code>{chg_24h}%</code>")
        lines.append(f"💰 Vol: <code>{vol_24h}</code>")
        lines.append(f"🔥 Vol chg: <code>{vol_chg}%</code>")

        badges = telegram_signal_badges(row)

        if badges:
            lines.append("")
            for badge in badges:
                lines.append(badge)

        lines.append("")
        lines.append(f"Reason: <i>{reason}</i>")

        if idx != len(df_active_output) - 1:
            lines.append("")
            lines.append("────────────")
            lines.append("")

    return "\n".join(lines)


# ============================================================
# MULTI PROVIDER RUNNER
# ============================================================

def run_multi_provider_screener():
    print("\n" + "=" * 120)
    print("RUNNING MULTI-PROVIDER SCREENER TEST")
    print("=" * 120)

    okx_results, okx_total, okx_prefiltered, okx_active = run_okx_screener()
    bitget_results, bitget_total, bitget_prefiltered, bitget_active = run_bitget_screener()

    frames = []

    if okx_results is not None and not okx_results.empty:
        frames.append(okx_results)

    if bitget_results is not None and not bitget_results.empty:
        frames.append(bitget_results)

    if not frames:
        df_all = pd.DataFrame()
    else:
        df_all = pd.concat(frames, ignore_index=True)

    if df_all.empty:
        df_active = pd.DataFrame()
    else:
        df_all["signal_rank"] = df_all["signal_level"].apply(get_signal_rank)

        df_all = df_all.sort_values(
            by=[
                "signal_rank",
                "rsi_4h",
                "rsi_1h",
                "price_change_24h_percent",
                "volume_usd_24h_exact",
            ],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)

        df_active = df_all[df_all["signal_level"] != "NO_SIGNAL"].copy()

    print("\n" + "=" * 120)
    print("MULTI-PROVIDER SUMMARY")
    print("=" * 120)
    print("OKX total universe:", okx_total)
    print("OKX prefiltered:", okx_prefiltered)
    print("OKX active:", okx_active)
    print("Bitget total universe:", bitget_total)
    print("Bitget prefiltered:", bitget_prefiltered)
    print("Bitget active:", bitget_active)
    print("Total active signals:", len(df_active))

    df_active_output = prepare_active_output_table(df_active)

    if df_active_output.empty:
        print("No active signals.")
    else:
        print("\n" + "=" * 120)
        print("ACTIVE SIGNALS ONLY")
        print("=" * 120)
        print(df_active_output.head(FINAL_TOP_N).to_string(index=False))

    if SEND_MESSAGE_IF_NO_SIGNALS or not df_active_output.empty:
        message = format_multi_provider_telegram(
            df_active_output=df_active_output.head(FINAL_TOP_N),
            okx_total=okx_total,
            okx_prefiltered=okx_prefiltered,
            okx_active=okx_active,
            bitget_total=bitget_total,
            bitget_prefiltered=bitget_prefiltered,
            bitget_active=bitget_active,
        )

        send_telegram_message_safe(message)


def main():
    get_telegram_credentials()
    run_multi_provider_screener()


if __name__ == "__main__":
    main()
