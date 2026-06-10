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

# New pump-detection thresholds.
# Closed 4H RSI is intentionally not used in signal classification.
EARLY_PUMP_RSI_1H_LIVE = 82
EARLY_PUMP_PRICE_CHANGE_24H = 8

ACTIVE_OVERHEAT_RSI_1H_LIVE = 82
ACTIVE_OVERHEAT_RSI_4H_LIVE = 75
ACTIVE_OVERHEAT_PRICE_CHANGE_24H = 10
ACTIVE_OVERHEAT_VOLUME_CHANGE_24H = 50

EXTREME_PUMP_RSI_1H_LIVE = 85
EXTREME_PUMP_RSI_4H_LIVE = 80
EXTREME_PUMP_PRICE_CHANGE_24H = 20
EXTREME_PUMP_VOLUME_CHANGE_24H = 100

RSI_1H_CLOSED_CONFIRMATION = 80

MIN_PRICE_CHANGE_24H = EARLY_PUMP_PRICE_CHANGE_24H
MIN_VOLUME_USD_24H = 5_000_000

# Short-analysis settings based on 1H OHLCV only.
LIQUIDITY_SWEEP_LOOKBACK = 30
PREMIUM_ZONE_LOOKBACK = 72
PREMIUM_ZONE_LOOKBACK_1H = 72
PREMIUM_ZONE_LOOKBACK_4H = 42
LOCAL_HIGH_LOOKBACKS = {
    "24H": 24,
    "48H": 48,
    "7D": 168,
}
REJECTION_VOLUME_LOOKBACK = 20
REJECTION_UPPER_WICK_BODY_MULTIPLIER = 1.5
REJECTION_VOLUME_MULTIPLIER = 1.5

PRE_FILTER_TOP_N = 40
FINAL_TOP_N = 30
RSI_ONLY_TOP_N = 20

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
        "User-Agent": "multi-exchange-rsi-screener/1.1"
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


def make_factor(key, label, status, points=0, detail=""):
    """
    Short-factor object.

    status:
    - confirmed
    - not_confirmed
    - not_enough_data
    """

    return {
        "key": key,
        "label": label,
        "status": status,
        "points": int(points),
        "detail": str(detail or ""),
    }


def get_volume_series(df):
    """
    Return the best available volume column for relative volume checks.
    Quote volume is preferred because it is comparable across instruments.
    """

    for col in ["quote_volume", "volume", "base_volume", "volume_currency"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")

    return pd.Series([np.nan] * len(df), index=df.index)


def detect_liquidity_sweep(df_1h, lookback=LIQUIDITY_SWEEP_LOOKBACK):
    if df_1h is None or df_1h.empty or len(df_1h) < lookback + 1:
        return make_factor(
            key="liquidity_sweep",
            label="Liquidity sweep",
            status="not_enough_data",
            detail=f"requires {lookback + 1} 1H candles",
        )

    df = df_1h.copy().reset_index(drop=True)
    current = df.iloc[-1]
    previous = df.iloc[-(lookback + 1):-1]

    previous_high = float(previous["high"].max())
    current_high = float(current["high"])
    current_close = float(current["close"])

    confirmed = current_high > previous_high and current_close < previous_high

    if confirmed:
        return make_factor(
            key="liquidity_sweep",
            label="Liquidity sweep",
            status="confirmed",
            points=2,
            detail=f"swept {lookback}H high",
        )

    return make_factor(
        key="liquidity_sweep",
        label="Liquidity sweep",
        status="not_confirmed",
        detail=f"{lookback}H high not swept/reclaimed",
    )


def classify_premium_position(position):
    if position is None or pd.isna(position):
        return "N/A"

    position = float(position)

    if position >= 0.786:
        return "Extreme Premium"

    if position >= 0.618:
        return "Premium"

    if position <= 0.382:
        return "Discount"

    return "Neutral"


def calculate_premium_position(df, lookback):
    if df is None or df.empty or len(df) < lookback + 1:
        return {
            "status": "not_enough_data",
            "position": None,
            "label": "N/A",
            "detail": f"requires {lookback + 1} candles",
        }

    work_df = df.copy().reset_index(drop=True)
    current = work_df.iloc[-1]
    previous = work_df.iloc[-(lookback + 1):-1]

    range_low = float(previous["low"].min())
    range_high = float(previous["high"].max())
    current_price = float(current["close"])
    range_size = range_high - range_low

    if range_size <= 0:
        return {
            "status": "not_enough_data",
            "position": None,
            "label": "N/A",
            "detail": "invalid range",
        }

    position = (current_price - range_low) / range_size
    label = classify_premium_position(position)

    return {
        "status": "ok",
        "position": float(position),
        "label": label,
        "detail": f"{label} {position:.2f}",
    }


def detect_premium_zone(
    df_1h,
    df_4h=None,
    lookback_1h=PREMIUM_ZONE_LOOKBACK_1H,
    lookback_4h=PREMIUM_ZONE_LOOKBACK_4H,
):
    premium_1h = calculate_premium_position(df_1h, lookback_1h)
    premium_4h = calculate_premium_position(df_4h, lookback_4h) if df_4h is not None else {
        "status": "not_enough_data",
        "position": None,
        "label": "N/A",
        "detail": "4H candles unavailable",
    }

    labels = []
    points = 0
    confirmed = False
    not_enough_data = True

    for tf, item in [("1H", premium_1h), ("4H", premium_4h)]:
        status = item.get("status")
        label = item.get("label", "N/A")
        position = item.get("position")

        if status == "ok":
            not_enough_data = False

            if position is not None and not pd.isna(position):
                labels.append(f"{tf}: {label} / {float(position):.2f}")
            else:
                labels.append(f"{tf}: {label}")

            if label == "Extreme Premium":
                points += 2
                confirmed = True
            elif label == "Premium":
                points += 1
                confirmed = True
        else:
            labels.append(f"{tf}: N/A")

    # Premium should matter, but should not dominate the setup score by itself.
    points = min(points, 3)

    detail = " | ".join(labels)

    factor = make_factor(
        key="premium_zone",
        label="Premium zone",
        status="confirmed" if confirmed else ("not_enough_data" if not_enough_data else "not_confirmed"),
        points=points,
        detail=detail,
    )

    factor["premium_1h_label"] = premium_1h.get("label", "N/A")
    factor["premium_1h_position"] = premium_1h.get("position")
    factor["premium_4h_label"] = premium_4h.get("label", "N/A")
    factor["premium_4h_position"] = premium_4h.get("position")

    return factor


def detect_local_high_update(df_1h, lookbacks=LOCAL_HIGH_LOOKBACKS):
    if df_1h is None or df_1h.empty:
        return make_factor(
            key="local_high_update",
            label="Local high update",
            status="not_enough_data",
            detail="no 1H candles",
        )

    df = df_1h.copy().reset_index(drop=True)
    current = df.iloc[-1]
    current_high = float(current["high"])

    # Check the highest-confidence tiers first. Points are not cumulative.
    tiers = [
        ("7D", lookbacks["7D"], 3),
        ("48H", lookbacks["48H"], 2),
        ("24H", lookbacks["24H"], 1),
    ]

    checked_any = False

    for label, lookback, points in tiers:
        if len(df) < lookback + 1:
            continue

        checked_any = True
        previous = df.iloc[-(lookback + 1):-1]
        previous_high = float(previous["high"].max())

        if current_high > previous_high:
            return make_factor(
                key="local_high_update",
                label="Local high update",
                status="confirmed",
                points=points,
                detail=f"new {label} high",
            )

    if checked_any:
        return make_factor(
            key="local_high_update",
            label="Local high update",
            status="not_confirmed",
            detail="no 24H/48H/7D high update",
        )

    return make_factor(
        key="local_high_update",
        label="Local high update",
        status="not_enough_data",
        detail="requires at least 25 1H candles",
    )


def detect_rejection_candle(
    df_1h,
    volume_lookback=REJECTION_VOLUME_LOOKBACK,
    wick_body_multiplier=REJECTION_UPPER_WICK_BODY_MULTIPLIER,
    volume_multiplier=REJECTION_VOLUME_MULTIPLIER,
):
    if df_1h is None or df_1h.empty or len(df_1h) < volume_lookback + 1:
        return make_factor(
            key="rejection_candle",
            label="Rejection candle",
            status="not_enough_data",
            detail=f"requires {volume_lookback + 1} 1H candles",
        )

    df = df_1h.copy().reset_index(drop=True)
    current = df.iloc[-1]
    previous = df.iloc[-(volume_lookback + 1):-1]

    open_price = float(current["open"])
    high_price = float(current["high"])
    low_price = float(current["low"])
    close_price = float(current["close"])

    candle_range = high_price - low_price

    if candle_range <= 0:
        return make_factor(
            key="rejection_candle",
            label="Rejection candle",
            status="not_enough_data",
            detail="invalid candle range",
        )

    body = abs(close_price - open_price)
    body_reference = max(body, candle_range * 0.05)
    upper_wick = high_price - max(open_price, close_price)
    midpoint = low_price + candle_range * 0.5

    volume_series = get_volume_series(df)
    current_volume = float(volume_series.iloc[-1]) if not pd.isna(volume_series.iloc[-1]) else np.nan
    previous_avg_volume = float(volume_series.iloc[-(volume_lookback + 1):-1].mean())

    if pd.isna(current_volume) or pd.isna(previous_avg_volume) or previous_avg_volume <= 0:
        return make_factor(
            key="rejection_candle",
            label="Rejection candle",
            status="not_enough_data",
            detail="volume unavailable",
        )

    wick_ok = upper_wick >= body_reference * wick_body_multiplier
    weak_close = close_price < midpoint
    volume_ok = current_volume >= previous_avg_volume * volume_multiplier

    confirmed = wick_ok and weak_close and volume_ok

    if confirmed:
        return make_factor(
            key="rejection_candle",
            label="Rejection candle",
            status="confirmed",
            points=2,
            detail=f"upper wick + weak close + volume {current_volume / previous_avg_volume:.1f}x",
        )

    return make_factor(
        key="rejection_candle",
        label="Rejection candle",
        status="not_confirmed",
        detail="no wick/weak-close/volume confirmation",
    )


def analyze_short_factors(df_1h, df_4h=None):
    premium_factor = detect_premium_zone(df_1h, df_4h)

    factors = [
        detect_liquidity_sweep(df_1h),
        premium_factor,
        detect_local_high_update(df_1h),
        detect_rejection_candle(df_1h),
    ]

    score = sum(int(factor.get("points", 0)) for factor in factors)
    confirmed_count = sum(1 for factor in factors if factor.get("status") == "confirmed")
    total_count = len(factors)

    return {
        "factors": factors,
        "score": score,
        "confirmed_count": confirmed_count,
        "total_count": total_count,
        "premium_factor": premium_factor,
    }


def calculate_scores(
    rsi_1h_live,
    rsi_1h_closed,
    rsi_4h_live,
    exact_volume_24h,
    volume_change_24h,
    price_change_24h,
    short_setup_score,
):
    pump_score = 0

    if price_change_24h >= 40:
        pump_score = 3
    elif price_change_24h >= 20:
        pump_score = 2
    elif price_change_24h >= 10:
        pump_score = 1

    rsi_score = 0

    if rsi_1h_live >= 85:
        rsi_score += 2
    elif rsi_1h_live >= 82:
        rsi_score += 1

    if rsi_4h_live >= 80:
        rsi_score += 2
    elif rsi_4h_live >= 75:
        rsi_score += 1

    if rsi_1h_closed >= 80:
        rsi_score += 1

    volume_score = 0

    if exact_volume_24h is not None and not pd.isna(exact_volume_24h) and exact_volume_24h >= MIN_VOLUME_USD_24H:
        volume_score += 1

    vol_chg = None if volume_change_24h is None or pd.isna(volume_change_24h) else float(volume_change_24h)

    if vol_chg is not None:
        if vol_chg >= 300:
            volume_score += 3
        elif vol_chg >= 100:
            volume_score += 2
        elif vol_chg >= 50:
            volume_score += 1

    final_score = pump_score + rsi_score + volume_score + short_setup_score

    return {
        "pump_score": pump_score,
        "rsi_score": rsi_score,
        "volume_score": volume_score,
        "short_setup_score": int(short_setup_score),
        "final_score": int(final_score),
    }


def classify_watch_signal(scores):
    pump_score = int(scores.get("pump_score", 0))
    rsi_score = int(scores.get("rsi_score", 0))
    short_setup_score = int(scores.get("short_setup_score", 0))

    if pump_score >= 2 and rsi_score >= 4 and short_setup_score >= 5:
        return {
            "signal_level": "HIGH_PRIORITY_SHORT_WATCH",
            "reason": "Strong pump + live RSI overheat + multiple short factors",
        }

    if pump_score >= 2 and rsi_score >= 3 and short_setup_score >= 2:
        return {
            "signal_level": "SHORT_WATCH",
            "reason": "Pump + live RSI overheat + short factor confirmation",
        }

    if pump_score >= 2 and rsi_score >= 3:
        return {
            "signal_level": "OVERHEAT_WATCH",
            "reason": "Pump + live RSI overheat, short setup not confirmed yet",
        }

    if pump_score >= 1 and rsi_score >= 1:
        return {
            "signal_level": "PUMP_WATCH",
            "reason": "Pump detected, short setup not confirmed yet",
        }

    return {
        "signal_level": "NO_SIGNAL",
        "reason": "Watch filters not passed",
    }


def classify_signal(
    rsi_1h_live,
    rsi_1h_closed,
    rsi_4h_live,
    exact_volume_24h,
    volume_change_24h,
    price_change_24h,
    short_setup_score=0,
):
    """
    Watchlist signal model.

    Uses scores instead of one hard RSI rule:
    - pump_score
    - rsi_score
    - volume_score
    - short_setup_score
    - final_score

    Does not use closed 4H RSI.
    """

    scores = calculate_scores(
        rsi_1h_live=rsi_1h_live,
        rsi_1h_closed=rsi_1h_closed,
        rsi_4h_live=rsi_4h_live,
        exact_volume_24h=exact_volume_24h,
        volume_change_24h=volume_change_24h,
        price_change_24h=price_change_24h,
        short_setup_score=short_setup_score,
    )

    classification = classify_watch_signal(scores)

    classification.update(scores)

    return classification


def get_signal_rank(signal_level):
    ranks = {
        "HIGH_PRIORITY_SHORT_WATCH": 4,
        "SHORT_WATCH": 3,
        "OVERHEAT_WATCH": 2,
        "PUMP_WATCH": 1,
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

def okx_get_last_live_candle(df):
    """
    OKX returns the current candle with confirm = 0 when it is still open.
    For live RSI we intentionally use the latest candle regardless of confirm.
    """

    if df is None or df.empty:
        raise Exception("OKX: empty candles dataframe.")

    return df.iloc[-1]

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

    last_1h_live = okx_get_last_live_candle(df_1h)
    last_1h_closed = okx_get_last_closed_candle(df_1h)
    last_4h_live = okx_get_last_live_candle(df_4h)

    exact_volume_24h = okx_closed_24h_quote_volume(df_1h)
    volume_change_24h = okx_volume_change_24h(df_1h)

    rsi_1h_live = float(last_1h_live["rsi"])
    rsi_1h_closed = float(last_1h_closed["rsi"])
    rsi_4h_live = float(last_4h_live["rsi"])
    price = float(last_1h_live["close"])
    price_change_24h = float(ticker_row["price_change_24h_percent"])

    short_analysis = analyze_short_factors(df_1h, df_4h)

    classification = classify_signal(
        rsi_1h_live=rsi_1h_live,
        rsi_1h_closed=rsi_1h_closed,
        rsi_4h_live=rsi_4h_live,
        exact_volume_24h=exact_volume_24h,
        volume_change_24h=volume_change_24h,
        price_change_24h=price_change_24h,
        short_setup_score=short_analysis["score"],
    )

    return {
        "exchange": "OKX",
        "provider": "OKX",
        "raw_symbol": inst_id,
        "symbol": okx_make_report_symbol(inst_id),
        "price": price,
        "rsi_1h_live": rsi_1h_live,
        "rsi_1h_closed": rsi_1h_closed,
        "rsi_4h_live": rsi_4h_live,
        "volume_usd_24h_exact": exact_volume_24h,
        "volume_change_24h_percent": volume_change_24h,
        "price_change_24h_percent": price_change_24h,
        "signal_level": classification["signal_level"],
        "reason": classification["reason"],
        "pump_score": classification["pump_score"],
        "rsi_score": classification["rsi_score"],
        "volume_score": classification["volume_score"],
        "short_setup_score": classification["short_setup_score"],
        "final_score": classification["final_score"],
        "short_factors": short_analysis["factors"],
        "confirmed_short_factors_count": short_analysis["confirmed_count"],
        "total_short_factors_count": short_analysis["total_count"],
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

def bitget_get_last_live_candle(df):
    """
    Bitget does not provide a confirm field.
    For live RSI we use the latest candle row.
    """

    if df is None or df.empty:
        raise Exception("Bitget: empty candles dataframe.")

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

    last_1h_live = bitget_get_last_live_candle(df_1h)
    last_1h_closed = bitget_get_last_closed_candle(df_1h)
    last_4h_live = bitget_get_last_live_candle(df_4h)

    exact_volume_24h = bitget_closed_24h_quote_volume(df_1h)
    volume_change_24h = bitget_volume_change_24h(df_1h)

    rsi_1h_live = float(last_1h_live["rsi"])
    rsi_1h_closed = float(last_1h_closed["rsi"])
    rsi_4h_live = float(last_4h_live["rsi"])
    price = float(last_1h_live["close"])
    price_change_24h = float(ticker_row["price_change_24h_percent"])

    short_analysis = analyze_short_factors(df_1h, df_4h)

    classification = classify_signal(
        rsi_1h_live=rsi_1h_live,
        rsi_1h_closed=rsi_1h_closed,
        rsi_4h_live=rsi_4h_live,
        exact_volume_24h=exact_volume_24h,
        volume_change_24h=volume_change_24h,
        price_change_24h=price_change_24h,
        short_setup_score=short_analysis["score"],
    )

    return {
        "exchange": "Bitget",
        "provider": "Bitget",
        "raw_symbol": symbol,
        "symbol": f"{symbol}.P",
        "price": price,
        "rsi_1h_live": rsi_1h_live,
        "rsi_1h_closed": rsi_1h_closed,
        "rsi_4h_live": rsi_4h_live,
        "volume_usd_24h_exact": exact_volume_24h,
        "volume_change_24h_percent": volume_change_24h,
        "price_change_24h_percent": price_change_24h,
        "signal_level": classification["signal_level"],
        "reason": classification["reason"],
        "pump_score": classification["pump_score"],
        "rsi_score": classification["rsi_score"],
        "volume_score": classification["volume_score"],
        "short_setup_score": classification["short_setup_score"],
        "final_score": classification["final_score"],
        "short_factors": short_analysis["factors"],
        "confirmed_short_factors_count": short_analysis["confirmed_count"],
        "total_short_factors_count": short_analysis["total_count"],
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
        "HIGH_PRIORITY_SHORT_WATCH": "🔴🔴 HIGH PRIORITY SHORT WATCH",
        "SHORT_WATCH": "🔴 SHORT WATCH",
        "OVERHEAT_WATCH": "🟠 OVERHEAT WATCH",
        "PUMP_WATCH": "🟡 PUMP WATCH",
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
    df_out["rsi_1h_live"] = df_out["rsi_1h_live"].round(2)
    df_out["rsi_1h_closed"] = df_out["rsi_1h_closed"].round(2)
    df_out["rsi_4h_live"] = df_out["rsi_4h_live"].round(2)
    df_out["chg_24h_%"] = df_out["price_change_24h_percent"].apply(format_percent_2)
    df_out["vol_24h"] = df_out["volume_usd_24h_exact"].apply(format_large_number)
    df_out["vol_chg_24h_%"] = df_out["volume_change_24h_percent"].apply(format_percent_2)

    columns = [
        "exchange",
        "signal_level",
        "symbol",
        "price",
        "rsi_1h_live",
        "rsi_1h_closed",
        "rsi_4h_live",
        "chg_24h_%",
        "vol_24h",
        "vol_chg_24h_%",
        "pump_score",
        "rsi_score",
        "volume_score",
        "short_setup_score",
        "final_score",
        "confirmed_short_factors_count",
        "total_short_factors_count",
        "reason",
    ]

    return df_out[columns]

def exchange_sort_key(exchange):
    order = {
        "OKX": 1,
        "Bitget": 2,
    }

    return order.get(str(exchange), 99)


def prepare_grouped_active_signals(df_active, max_groups=FINAL_TOP_N):
    """
    Group active signals by symbol.

    If the same symbol exists on OKX and Bitget, Telegram shows it once.
    Display values are taken from OKX when available; otherwise from the first available exchange.
    Ranking uses live RSI fields, not closed 4H RSI.
    """

    if df_active is None or df_active.empty:
        return []

    df = df_active.copy()
    df["signal_rank"] = df["signal_level"].apply(get_signal_rank)
    df["sort_volume"] = df["volume_usd_24h_exact"].fillna(0)
    df["sort_vol_chg"] = df["volume_change_24h_percent"].fillna(-999999)

    groups = []

    for symbol, group in df.groupby("symbol", sort=False):
        group_sorted = group.sort_values(
            by=[
                "signal_rank",
                "final_score",
                "short_setup_score",
                "rsi_4h_live",
                "rsi_1h_live",
                "rsi_1h_closed",
                "price_change_24h_percent",
                "sort_volume",
            ],
            ascending=[False, False, False, False, False, False, False, False],
        ).reset_index(drop=True)

        top_row = group_sorted.iloc[0]

        exchanges = sorted(
            group_sorted["exchange"].dropna().astype(str).unique().tolist(),
            key=exchange_sort_key,
        )

        details = []

        group_by_exchange = group_sorted.copy()
        group_by_exchange["exchange_order"] = group_by_exchange["exchange"].apply(exchange_sort_key)
        group_by_exchange = group_by_exchange.sort_values(
            by=["exchange_order", "signal_rank"],
            ascending=[True, False],
        )

        for _, row in group_by_exchange.iterrows():
            details.append({
                "exchange": str(row["exchange"]),
                "signal_level": str(row["signal_level"]),
                "reason": str(row["reason"]),
                "price": format_price_2(row["price"]),
                "rsi_1h_live": f"{float(row['rsi_1h_live']):.2f}",
                "rsi_1h_closed": f"{float(row['rsi_1h_closed']):.2f}",
                "rsi_4h_live": f"{float(row['rsi_4h_live']):.2f}",
                "chg_24h_%": format_percent_2(row["price_change_24h_percent"]),
                "vol_24h": format_large_number(row["volume_usd_24h_exact"]),
                "vol_chg_24h_%": format_percent_2(row["volume_change_24h_percent"]),
                "pump_score": int(row.get("pump_score", 0)),
                "rsi_score": int(row.get("rsi_score", 0)),
                "volume_score": int(row.get("volume_score", 0)),
                "short_setup_score": int(row.get("short_setup_score", 0)),
                "final_score": int(row.get("final_score", 0)),
                "short_factors": row.get("short_factors", []),
                "confirmed_short_factors_count": int(row.get("confirmed_short_factors_count", 0)),
                "total_short_factors_count": int(row.get("total_short_factors_count", 0)),
            })

        unique_reasons = []

        for reason in group_sorted["reason"].astype(str).tolist():
            if reason not in unique_reasons:
                unique_reasons.append(reason)

        max_price_change = group_sorted["price_change_24h_percent"].max()
        max_volume_change = group_sorted["volume_change_24h_percent"].max()
        max_volume = group_sorted["volume_usd_24h_exact"].max()
        max_final_score = group_sorted["final_score"].max() if "final_score" in group_sorted.columns else 0
        max_short_setup_score = group_sorted["short_setup_score"].max() if "short_setup_score" in group_sorted.columns else 0

        display_detail = details[0] if details else {}
        display_signal_level = str(display_detail.get("signal_level", top_row["signal_level"]))
        display_signal_rank = get_signal_rank(display_signal_level)

        groups.append({
            "symbol": str(symbol),
            "signal_level": display_signal_level,
            "signal_rank": int(display_signal_rank),
            "reason": str(display_detail.get("reason", top_row["reason"])),
            "reasons": unique_reasons,
            "exchanges": exchanges,
            "exchanges_text": " + ".join(exchanges),
            "best_exchange": str(top_row["exchange"]),
            "best_rsi_1h_live": float(group_sorted["rsi_1h_live"].max()),
            "best_rsi_1h_closed": float(group_sorted["rsi_1h_closed"].max()),
            "best_rsi_4h_live": float(group_sorted["rsi_4h_live"].max()),
            "best_price_change_24h_percent": float(max_price_change),
            "best_volume_usd_24h_exact": None if pd.isna(max_volume) else float(max_volume),
            "best_volume_change_24h_percent": None if pd.isna(max_volume_change) else float(max_volume_change),
            "best_final_score": int(max_final_score),
            "best_short_setup_score": int(max_short_setup_score),
            "details": details,
            "display_detail": display_detail,
        })

    groups = sorted(
        groups,
        key=lambda item: (
            item["signal_rank"],
            item.get("best_final_score", 0),
            item.get("best_short_setup_score", 0),
            item["best_rsi_4h_live"],
            item["best_rsi_1h_live"],
            item["best_rsi_1h_closed"],
            item["best_price_change_24h_percent"],
            item["best_volume_usd_24h_exact"] or 0,
        ),
        reverse=True,
    )

    return groups[:max_groups]

def prepare_grouped_output_table(grouped_signals):
    if not grouped_signals:
        return pd.DataFrame()

    rows = []

    for item in grouped_signals:
        rows.append({
            "signal_level": item["signal_level"],
            "symbol": item["symbol"],
            "exchanges": item["exchanges_text"],
            "best_exchange": item["best_exchange"],
            "best_rsi_1h_live": round(item["best_rsi_1h_live"], 2),
            "best_rsi_1h_closed": round(item["best_rsi_1h_closed"], 2),
            "best_rsi_4h_live": round(item["best_rsi_4h_live"], 2),
            "best_chg_24h_%": format_percent_2(item["best_price_change_24h_percent"]),
            "best_vol_24h": format_large_number(item["best_volume_usd_24h_exact"]),
            "best_vol_chg_24h_%": format_percent_2(item["best_volume_change_24h_percent"]),
            "final_score": item.get("best_final_score", 0),
            "short_setup_score": item.get("best_short_setup_score", 0),
            "reason": item["reason"],
        })

    return pd.DataFrame(rows)

def grouped_signal_badges(group):
    """
    Badges are calculated from the same values that Telegram displays.

    Rule:
    - if the symbol exists on OKX, Telegram displays OKX values;
    - otherwise Telegram displays the first available exchange values.
    """

    badges = []

    detail = group.get("display_detail")

    if not detail:
        return badges

    chg_24h = parse_float_from_value(detail.get("chg_24h_%"))
    vol_chg = parse_float_from_value(detail.get("vol_chg_24h_%"))

    if chg_24h is not None and chg_24h >= 20:
        badges.append("🚀 Strong pump")

    if vol_chg is not None and vol_chg >= 100:
        badges.append("🔥 Volume spike")

    if vol_chg is not None and vol_chg < 0:
        badges.append("⚠️ Volume fading")

    return badges


def short_factor_line(factor):
    status = factor.get("status")
    label = str(factor.get("label", "Unknown factor"))
    detail = str(factor.get("detail", "")).strip()

    if factor.get("key") == "premium_zone":
        return format_premium_line_from_factors([factor])["premium_factor_line"]

    if status == "confirmed":
        icon = "✅"
    elif status == "not_enough_data":
        icon = "⚪"
    else:
        icon = "❌"

    if detail:
        return f"{icon} {label} — {detail}"

    return f"{icon} {label}"


def format_short_factors_for_telegram(factors):
    if not factors:
        return ["⚪ Short factors not available"]

    return [short_factor_line(factor) for factor in factors]

def find_factor(factors, key):
    if not factors:
        return None

    for factor in factors:
        if factor.get("key") == key:
            return factor

    return None


def format_premium_line_from_factors(factors):
    factor = find_factor(factors, "premium_zone")

    if not factor:
        return {
            "premium_1h": "N/A",
            "premium_4h": "N/A",
            "premium_factor_line": "⚪ Premium zone — data unavailable",
        }

    label_1h = str(factor.get("premium_1h_label", "N/A"))
    label_4h = str(factor.get("premium_4h_label", "N/A"))
    position_1h = factor.get("premium_1h_position")
    position_4h = factor.get("premium_4h_position")

    if position_1h is not None and not pd.isna(position_1h):
        premium_1h = f"{label_1h} / {float(position_1h):.2f}"
    else:
        premium_1h = label_1h

    if position_4h is not None and not pd.isna(position_4h):
        premium_4h = f"{label_4h} / {float(position_4h):.2f}"
    else:
        premium_4h = label_4h

    if factor.get("status") == "confirmed":
        icon = "✅"
    elif factor.get("status") == "not_enough_data":
        icon = "⚪"
    else:
        icon = "❌"

    return {
        "premium_1h": premium_1h,
        "premium_4h": premium_4h,
        "premium_factor_line": f"{icon} Premium zone — 1H: {label_1h} | 4H: {label_4h}",
    }


def quality_pump_label(score):
    score = int(score or 0)

    if score >= 3:
        return "Extreme"
    if score == 2:
        return "Strong"
    if score == 1:
        return "Mild"
    return "None"


def quality_heat_label(score):
    score = int(score or 0)

    if score >= 4:
        return "Strong"
    if score >= 2:
        return "Moderate"
    if score == 1:
        return "Mild"
    return "None"


def quality_volume_label(score):
    score = int(score or 0)

    if score >= 4:
        return "Strong spike"
    if score >= 3:
        return "Strong spike"
    if score == 2:
        return "Strong"
    if score == 1:
        return "Normal"
    return "Weak"


def quality_setup_label(score, confirmed_count, total_count):
    score = int(score or 0)
    confirmed_count = int(confirmed_count or 0)
    total_count = int(total_count or 0)

    if score >= 5:
        level = "Strong"
    elif score >= 3:
        level = "Moderate"
    elif score >= 1:
        level = "Weak"
    else:
        level = "None"

    return f"{level} — {confirmed_count}/{total_count} factors"


def quality_priority_label(signal_level):
    mapping = {
        "HIGH_PRIORITY_SHORT_WATCH": "Critical",
        "SHORT_WATCH": "High",
        "OVERHEAT_WATCH": "Medium",
        "PUMP_WATCH": "Low",
        "NO_SIGNAL": "None",
    }

    return mapping.get(str(signal_level), "None")


def build_quality_labels(detail):
    pump_score = int(detail.get("pump_score", 0))
    rsi_score = int(detail.get("rsi_score", 0))
    volume_score = int(detail.get("volume_score", 0))
    short_setup_score = int(detail.get("short_setup_score", 0))
    confirmed_count = int(detail.get("confirmed_short_factors_count", 0))
    total_count = int(detail.get("total_short_factors_count", 0))
    signal_level = str(detail.get("signal_level", "NO_SIGNAL"))

    return {
        "pump_quality": quality_pump_label(pump_score),
        "heat_quality": quality_heat_label(rsi_score),
        "volume_quality": quality_volume_label(volume_score),
        "setup_quality": quality_setup_label(short_setup_score, confirmed_count, total_count),
        "priority_quality": quality_priority_label(signal_level),
    }


def classify_rsi_only_signal(row):
    """
    Heat-only calibration classification.

    This block ignores short-factor confirmation.
    It answers: "Why is this asset on the radar by pump / RSI / volume context?"
    """

    rsi_1h_live = float(row.get("rsi_1h_live", np.nan))
    rsi_1h_closed = float(row.get("rsi_1h_closed", np.nan))
    rsi_4h_live = float(row.get("rsi_4h_live", np.nan))
    price_change_24h = float(row.get("price_change_24h_percent", np.nan))
    volume_24h = row.get("volume_usd_24h_exact", np.nan)
    volume_change_24h = row.get("volume_change_24h_percent", np.nan)
    main_signal_level = str(row.get("signal_level", "NO_SIGNAL"))

    volume_ok = (
        volume_24h is not None and
        not pd.isna(volume_24h) and
        float(volume_24h) >= MIN_VOLUME_USD_24H
    )

    if not volume_ok:
        return {
            "rsi_only_level": "NO_HEAT_SIGNAL",
            "rsi_only_rank": 0,
            "rsi_only_reason": "24h volume below minimum",
        }

    vol_chg = None
    if volume_change_24h is not None and not pd.isna(volume_change_24h):
        vol_chg = float(volume_change_24h)

    reason_parts = []

    if main_signal_level != "NO_SIGNAL":
        reason_parts.append(f"score-model level {main_signal_level}")

    if rsi_1h_live >= EARLY_PUMP_RSI_1H_LIVE:
        reason_parts.append(f"RSI 1H live >= {EARLY_PUMP_RSI_1H_LIVE}")

    if rsi_4h_live >= ACTIVE_OVERHEAT_RSI_4H_LIVE:
        reason_parts.append(f"RSI 4H live >= {ACTIVE_OVERHEAT_RSI_4H_LIVE}")

    if price_change_24h >= EXTREME_PUMP_PRICE_CHANGE_24H:
        reason_parts.append(f"24h change >= {EXTREME_PUMP_PRICE_CHANGE_24H}%")

    if vol_chg is not None and vol_chg >= EXTREME_PUMP_VOLUME_CHANGE_24H:
        reason_parts.append(f"volume change >= {EXTREME_PUMP_VOLUME_CHANGE_24H}%")

    has_extreme_heat = (
        rsi_1h_live >= EXTREME_PUMP_RSI_1H_LIVE and
        rsi_4h_live >= EXTREME_PUMP_RSI_4H_LIVE and
        price_change_24h >= EXTREME_PUMP_PRICE_CHANGE_24H and
        vol_chg is not None and
        vol_chg >= EXTREME_PUMP_VOLUME_CHANGE_24H
    )

    if has_extreme_heat:
        if rsi_1h_closed >= RSI_1H_CLOSED_CONFIRMATION:
            reason_parts.append(f"RSI 1H closed >= {RSI_1H_CLOSED_CONFIRMATION}")

        return {
            "rsi_only_level": "EXTREME_HEAT",
            "rsi_only_rank": 3,
            "rsi_only_reason": " + ".join(reason_parts),
        }

    has_active_heat = (
        rsi_4h_live >= ACTIVE_OVERHEAT_RSI_4H_LIVE or
        rsi_1h_live >= EARLY_PUMP_RSI_1H_LIVE
    ) and (
        price_change_24h >= EXTREME_PUMP_PRICE_CHANGE_24H or
        (vol_chg is not None and vol_chg >= EXTREME_PUMP_VOLUME_CHANGE_24H)
    )

    if has_active_heat:
        return {
            "rsi_only_level": "ACTIVE_HEAT",
            "rsi_only_rank": 2,
            "rsi_only_reason": " + ".join(reason_parts),
        }

    has_pump_context = (
        main_signal_level != "NO_SIGNAL" or
        rsi_1h_live >= EARLY_PUMP_RSI_1H_LIVE or
        rsi_4h_live >= ACTIVE_OVERHEAT_RSI_4H_LIVE or
        price_change_24h >= EXTREME_PUMP_PRICE_CHANGE_24H or
        (vol_chg is not None and vol_chg >= EXTREME_PUMP_VOLUME_CHANGE_24H)
    )

    if has_pump_context:
        return {
            "rsi_only_level": "PUMP_CONTEXT",
            "rsi_only_rank": 1,
            "rsi_only_reason": " + ".join(reason_parts) if reason_parts else "pump / heat context",
        }

    return {
        "rsi_only_level": "NO_HEAT_SIGNAL",
        "rsi_only_rank": 0,
        "rsi_only_reason": "heat filters not passed",
    }


def rsi_only_signal_label(signal_level):
    labels = {
        "EXTREME_HEAT": "🔴 EXTREME HEAT",
        "ACTIVE_HEAT": "🟠 ACTIVE HEAT",
        "PUMP_CONTEXT": "🟡 PUMP CONTEXT",
        "NO_HEAT_SIGNAL": "⚪ NO HEAT SIGNAL",
    }

    return labels.get(signal_level, signal_level)


def prepare_grouped_rsi_only_signals(df_all, max_groups=RSI_ONLY_TOP_N):
    """
    Prepare a second comparison block based on heat indicators only.

    Uses all analyzed prefiltered candidates, including those that did not pass
    the short-score watch classification.
    """

    if df_all is None or df_all.empty:
        return []

    df = df_all.copy()

    classifications = df.apply(lambda row: classify_rsi_only_signal(row), axis=1)
    df["rsi_only_level"] = classifications.apply(lambda item: item["rsi_only_level"])
    df["rsi_only_rank"] = classifications.apply(lambda item: item["rsi_only_rank"])
    df["rsi_only_reason"] = classifications.apply(lambda item: item["rsi_only_reason"])

    df = df[df["rsi_only_level"] != "NO_HEAT_SIGNAL"].copy()

    if df.empty:
        return []

    df["sort_volume"] = df["volume_usd_24h_exact"].fillna(0)
    df["sort_vol_chg"] = df["volume_change_24h_percent"].fillna(-999999)

    groups = []

    for symbol, group in df.groupby("symbol", sort=False):
        group_sorted = group.sort_values(
            by=[
                "rsi_only_rank",
                "rsi_4h_live",
                "rsi_1h_live",
                "rsi_1h_closed",
                "price_change_24h_percent",
                "sort_vol_chg",
                "sort_volume",
            ],
            ascending=[False, False, False, False, False, False, False],
        ).reset_index(drop=True)

        # Display OKX values when available; otherwise first available exchange.
        display_group = group_sorted.copy()
        display_group["exchange_order"] = display_group["exchange"].apply(exchange_sort_key)
        display_group = display_group.sort_values(
            by=["exchange_order", "rsi_only_rank"],
            ascending=[True, False],
        ).reset_index(drop=True)

        display_row = display_group.iloc[0]
        top_row = group_sorted.iloc[0]

        groups.append({
            "symbol": str(symbol),
            "rsi_only_level": str(display_row["rsi_only_level"]),
            "rsi_only_rank": int(display_row["rsi_only_rank"]),
            "rsi_only_reason": str(display_row["rsi_only_reason"]),
            "price": format_price_2(display_row["price"]),
            "rsi_1h_live": f"{float(display_row['rsi_1h_live']):.2f}",
            "rsi_1h_closed": f"{float(display_row['rsi_1h_closed']):.2f}",
            "rsi_4h_live": f"{float(display_row['rsi_4h_live']):.2f}",
            "chg_24h_%": format_percent_2(display_row["price_change_24h_percent"]),
            "vol_24h": format_large_number(display_row["volume_usd_24h_exact"]),
            "vol_chg_24h_%": format_percent_2(display_row["volume_change_24h_percent"]),
            "short_setup_score": int(display_row.get("short_setup_score", 0)),
            "confirmed_short_factors_count": int(display_row.get("confirmed_short_factors_count", 0)),
            "total_short_factors_count": int(display_row.get("total_short_factors_count", 0)),
            "main_signal_level": str(display_row.get("signal_level", "NO_SIGNAL")),
            "pump_score": int(display_row.get("pump_score", 0)),
            "rsi_score": int(display_row.get("rsi_score", 0)),
            "volume_score": int(display_row.get("volume_score", 0)),
            "short_setup_score": int(display_row.get("short_setup_score", 0)),
            "short_factors": display_row.get("short_factors", []),
            "best_rsi_only_rank": int(top_row["rsi_only_rank"]),
            "best_rsi_4h_live": float(group_sorted["rsi_4h_live"].max()),
            "best_rsi_1h_live": float(group_sorted["rsi_1h_live"].max()),
            "best_rsi_1h_closed": float(group_sorted["rsi_1h_closed"].max()),
            "best_price_change_24h_percent": float(group_sorted["price_change_24h_percent"].max()),
            "best_volume_usd_24h_exact": float(group_sorted["sort_volume"].max()),
        })

    groups = sorted(
        groups,
        key=lambda item: (
            item["best_rsi_only_rank"],
            item["best_rsi_4h_live"],
            item["best_rsi_1h_live"],
            item["best_rsi_1h_closed"],
            item["best_price_change_24h_percent"],
            item["best_volume_usd_24h_exact"],
        ),
        reverse=True,
    )

    return groups[:max_groups]


def format_multi_provider_telegram(
    grouped_signals,
    grouped_rsi_only_signals,
    okx_total,
    okx_prefiltered,
    okx_active,
    bitget_total,
    bitget_prefiltered,
    bitget_active,
):
    """
    Test-mode Telegram report with two blocks:
    1. short-score analysis;
    2. heat indicators only calibration.
    """

    now_kyiv = datetime.now(KYIV_TZ).strftime("%Y-%m-%d %H:%M Kyiv")

    lines = []

    lines.append("📊 <b>Market Heat Scanner</b>")
    lines.append(f"🕒 <code>{html.escape(now_kyiv)}</code>")
    lines.append("")

    short_count = len(grouped_signals) if grouped_signals else 0
    heat_count = len(grouped_rsi_only_signals) if grouped_rsi_only_signals else 0

    lines.append(f"Block 1 — Short score analysis: <b>{short_count}</b>")
    lines.append(f"Block 2 — Heat indicators only: <b>{heat_count}</b>")
    lines.append("")
    lines.append("━━━━━━━━━━━━")
    lines.append("<b>Block 1 — Short score analysis</b>")
    lines.append("━━━━━━━━━━━━")
    lines.append("")

    if not grouped_signals:
        lines.append("✅ No short-score watch signals.")
    else:
        for idx, group in enumerate(grouped_signals):
            signal_label = telegram_signal_label(group["signal_level"])
            symbol = html.escape(str(group["symbol"]))

            detail = group.get("display_detail") or {}

            price = html.escape(str(detail.get("price", "N/A")))
            rsi_1h_live = html.escape(str(detail.get("rsi_1h_live", "N/A")))
            rsi_1h_closed = html.escape(str(detail.get("rsi_1h_closed", "N/A")))
            rsi_4h_live = html.escape(str(detail.get("rsi_4h_live", "N/A")))
            chg_24h = html.escape(str(detail.get("chg_24h_%", "N/A")))
            vol_24h = html.escape(str(detail.get("vol_24h", "N/A")))
            vol_chg = html.escape(str(detail.get("vol_chg_24h_%", "N/A")))

            short_factors = detail.get("short_factors", []) or []
            premium_info = format_premium_line_from_factors(short_factors)
            quality = build_quality_labels(detail)

            confirmed_count = int(detail.get("confirmed_short_factors_count", 0))
            total_count = int(detail.get("total_short_factors_count", len(short_factors)))

            lines.append(f"{idx + 1}) {signal_label}")
            lines.append(f"📌 <b>{symbol}</b>")
            lines.append("")
            lines.append(f"💵 Price: <code>{price}</code>")
            lines.append(f"📊 RSI 1H live/closed: <code>{rsi_1h_live} / {rsi_1h_closed}</code>")
            lines.append(f"📊 RSI 4H live: <code>{rsi_4h_live}</code>")
            lines.append(f"📈 24h: <code>{chg_24h}%</code>")
            lines.append(f"💰 Vol: <code>{vol_24h}</code>")
            lines.append(f"🔥 Vol chg: <code>{vol_chg}%</code>")

            lines.append("")
            lines.append("<b>Premium / Discount:</b>")
            lines.append(f"1H: <code>{html.escape(premium_info['premium_1h'])}</code>")
            lines.append(f"4H: <code>{html.escape(premium_info['premium_4h'])}</code>")

            lines.append("")
            lines.append("<b>Signal quality:</b>")
            lines.append(f"Pump: <code>{html.escape(quality['pump_quality'])}</code>")
            lines.append(f"RSI heat: <code>{html.escape(quality['heat_quality'])}</code>")
            lines.append(f"Volume: <code>{html.escape(quality['volume_quality'])}</code>")
            lines.append(f"Short setup: <code>{html.escape(quality['setup_quality'])}</code>")
            lines.append(f"Priority: <code>{html.escape(quality['priority_quality'])}</code>")

            lines.append("")
            lines.append(f"<b>Short factors:</b> <code>{confirmed_count}/{total_count}</code>")

            for factor_line in format_short_factors_for_telegram(short_factors):
                lines.append(html.escape(factor_line))

            lines.append("")

            reason = html.escape(str(group.get("reason", detail.get("reason", "N/A"))))
            lines.append(f"Reason: <i>{reason}</i>")

            if idx != len(grouped_signals) - 1:
                lines.append("")
                lines.append("────────────")
                lines.append("")

    lines.append("")
    lines.append("━━━━━━━━━━━━")
    lines.append("<b>Block 2 — Heat indicators only</b>")
    lines.append("━━━━━━━━━━━━")
    lines.append("")
    lines.append("This block ignores short-factor confirmation and is shown for calibration.")
    lines.append("")

    if not grouped_rsi_only_signals:
        lines.append("✅ No heat-indicator signals.")
    else:
        for idx, group in enumerate(grouped_rsi_only_signals):
            signal_label = rsi_only_signal_label(group["rsi_only_level"])
            symbol = html.escape(str(group["symbol"]))
            price = html.escape(str(group.get("price", "N/A")))
            rsi_1h_live = html.escape(str(group.get("rsi_1h_live", "N/A")))
            rsi_1h_closed = html.escape(str(group.get("rsi_1h_closed", "N/A")))
            rsi_4h_live = html.escape(str(group.get("rsi_4h_live", "N/A")))
            chg_24h = html.escape(str(group.get("chg_24h_%", "N/A")))
            vol_24h = html.escape(str(group.get("vol_24h", "N/A")))
            vol_chg = html.escape(str(group.get("vol_chg_24h_%", "N/A")))
            confirmed_count = int(group.get("confirmed_short_factors_count", 0))
            total_count = int(group.get("total_short_factors_count", 0))
            main_signal_level = html.escape(str(group.get("main_signal_level", "NO_SIGNAL")))
            reason = html.escape(str(group.get("rsi_only_reason", "N/A")))

            quality = {
                "pump_quality": quality_pump_label(group.get("pump_score", 0)),
                "heat_quality": quality_heat_label(group.get("rsi_score", 0)),
                "volume_quality": quality_volume_label(group.get("volume_score", 0)),
                "setup_quality": quality_setup_label(group.get("short_setup_score", 0), confirmed_count, total_count),
                "priority_quality": quality_priority_label(group.get("main_signal_level", "NO_SIGNAL")),
            }

            lines.append(f"{idx + 1}) {signal_label}")
            lines.append(f"📌 <b>{symbol}</b>")
            lines.append(f"💵 Price: <code>{price}</code>")
            lines.append(f"📊 RSI 1H live/closed: <code>{rsi_1h_live} / {rsi_1h_closed}</code>")
            lines.append(f"📊 RSI 4H live: <code>{rsi_4h_live}</code>")
            lines.append(f"📈 24h: <code>{chg_24h}%</code> | Vol: <code>{vol_24h}</code>")
            lines.append(f"🔥 Vol chg: <code>{vol_chg}%</code>")
            lines.append(f"Heat: <code>{html.escape(quality['heat_quality'])}</code> | Pump: <code>{html.escape(quality['pump_quality'])}</code> | Volume: <code>{html.escape(quality['volume_quality'])}</code>")
            lines.append(f"Short setup: <code>{html.escape(quality['setup_quality'])}</code>")
            lines.append(f"Score-model level: <code>{main_signal_level}</code>")
            lines.append(f"Reason: <i>{reason}</i>")

            if idx != len(grouped_rsi_only_signals) - 1:
                lines.append("")
                lines.append("────────────")
                lines.append("")

    return "\n".join(lines)


# ============================================================
# MULTI PROVIDER RUNNER
# ============================================================

def run_multi_provider_screener():
    print("\n" + "=" * 120)
    print("RUNNING MULTI-PROVIDER SCREENER")
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
                "final_score",
                "short_setup_score",
                "rsi_4h_live",
                "rsi_1h_live",
                "rsi_1h_closed",
                "price_change_24h_percent",
                "volume_usd_24h_exact",
            ],
            ascending=[False, False, False, False, False, False, False, False],
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
    grouped_signals = prepare_grouped_active_signals(df_active, max_groups=FINAL_TOP_N)
    grouped_rsi_only_signals = prepare_grouped_rsi_only_signals(df_all, max_groups=RSI_ONLY_TOP_N)
    df_grouped_output = prepare_grouped_output_table(grouped_signals)

    if df_active_output.empty:
        print("No active signals.")
    else:
        print("\n" + "=" * 120)
        print("ACTIVE SIGNALS BY EXCHANGE")
        print("=" * 120)
        print(df_active_output.head(FINAL_TOP_N).to_string(index=False))

        print("\n" + "=" * 120)
        print("ACTIVE SIGNALS GROUPED BY SYMBOL")
        print("=" * 120)
        print(df_grouped_output.head(FINAL_TOP_N).to_string(index=False))

    if SEND_MESSAGE_IF_NO_SIGNALS or grouped_signals or grouped_rsi_only_signals:
        message = format_multi_provider_telegram(
            grouped_signals=grouped_signals,
            grouped_rsi_only_signals=grouped_rsi_only_signals,
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
