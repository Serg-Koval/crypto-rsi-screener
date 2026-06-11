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

SCRIPT_VERSION = "p0-sweep-v4-compact-20260610-r4"

RSI_PERIOD = 14

# Watch thresholds.
# Closed 4H RSI is intentionally not used in signal classification.
EARLY_PUMP_RSI_1H_LIVE = 82
EARLY_PUMP_PRICE_CHANGE_24H = 8

PUMP_WATCH_PRICE_CHANGE_24H = 10

OVERHEAT_WATCH_RSI_1H_LIVE = 82
OVERHEAT_WATCH_RSI_1H_CLOSED = 80
OVERHEAT_WATCH_PRICE_CHANGE_24H = 15
OVERHEAT_WATCH_MIN_VOLUME_USD_24H = 5_000_000

EXTREME_PUMP_RSI_1H_LIVE = 85
EXTREME_PUMP_RSI_4H_LIVE = 80
EXTREME_PUMP_PRICE_CHANGE_24H = 20
EXTREME_PUMP_VOLUME_CHANGE_24H = 100

RSI_1H_CLOSED_CONFIRMATION = OVERHEAT_WATCH_RSI_1H_CLOSED

MIN_PRICE_CHANGE_24H = EARLY_PUMP_PRICE_CHANGE_24H
MIN_VOLUME_USD_24H = OVERHEAT_WATCH_MIN_VOLUME_USD_24H

# Short-analysis settings based on 1H OHLCV only.
LIQUIDITY_SWEEP_LOOKBACKS = {
    "24H": 24,
    "12H": 12,
}
# Require a real sweep, not a one-tick wick above the previous high.
LIQUIDITY_SWEEP_MIN_BREAK_PCT = 0.001      # 0.10% above previous high
LIQUIDITY_SWEEP_MIN_CLOSE_BACK_PCT = 0.0005 # 0.05% close back below previous high
LIQUIDITY_SWEEP_LIVE_INVALIDATION_PCT = 0.0005 # current/live candle reclaimed level
LIQUIDITY_SWEEP_MIN_LEVEL_AGE_BARS = 5 # swept high must be at least 5 hours old
LIQUIDITY_SWEEP_MIN_REACTION_PCT = 0.015 # price must have reacted at least 1.5% from the level before sweep
LIQUIDITY_SWEEP_EQUAL_HIGH_TOLERANCE_PCT = 0.0025 # highs within 0.25% are treated as equal-high liquidity
LIQUIDITY_SWEEP_MIN_UPPER_WICK_RANGE_PCT = 0.30 # sweep candle should show rejection/failure character
LIQUIDITY_SWEEP_SWING_LEFT_BARS = 2
LIQUIDITY_SWEEP_SWING_RIGHT_BARS = 2
LIQUIDITY_SWEEP_LEVEL_LOOKBACK_1H = 240
LIQUIDITY_SWEEP_LEVEL_LOOKBACK_4H = 120
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
TELEGRAM_MAX_SIGNALS = 10
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

    value = float(value)
    abs_value = abs(value)

    if abs_value >= 100:
        return f"{value:.2f}"

    if abs_value >= 1:
        return f"{value:.4f}"

    if abs_value >= 0.01:
        return f"{value:.5f}"

    if abs_value >= 0.0001:
        return f"{value:.8f}"

    return f"{value:.10f}"


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


def safe_float(value, default=np.nan):
    try:
        if value is None or pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def detect_overheat_watch_context(
    rsi_1h_live,
    rsi_1h_closed,
    exact_volume_24h,
    price_change_24h,
):
    rsi_1h_live_value = safe_float(rsi_1h_live)
    rsi_1h_closed_value = safe_float(rsi_1h_closed)
    volume_24h_value = safe_float(exact_volume_24h)
    price_change_24h_value = safe_float(price_change_24h)

    checks = [
        (
            rsi_1h_live_value >= OVERHEAT_WATCH_RSI_1H_LIVE,
            f"RSI 1H live >= {OVERHEAT_WATCH_RSI_1H_LIVE}",
        ),
        (
            rsi_1h_closed_value >= OVERHEAT_WATCH_RSI_1H_CLOSED,
            f"RSI 1H closed >= {OVERHEAT_WATCH_RSI_1H_CLOSED}",
        ),
        (
            price_change_24h_value >= OVERHEAT_WATCH_PRICE_CHANGE_24H,
            f"24h change >= {OVERHEAT_WATCH_PRICE_CHANGE_24H}%",
        ),
        (
            volume_24h_value >= OVERHEAT_WATCH_MIN_VOLUME_USD_24H,
            f"24h volume >= {format_large_number(OVERHEAT_WATCH_MIN_VOLUME_USD_24H)}",
        ),
    ]

    passed = [label for ok, label in checks if ok]
    missing = [label for ok, label in checks if not ok]

    is_overheat = len(missing) == 0

    return {
        "is_overheat": is_overheat,
        "passed": passed,
        "missing": missing,
        "reason": " + ".join(passed) if is_overheat else "Missing " + " / ".join(missing),
    }


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


def get_last_closed_candle_for_analysis(df):
    """
    Return the last closed candle for factor analysis.

    OKX has explicit confirm flag:
    - confirm = 0 means live candle;
    - confirm = 1 means closed candle.

    Other providers may not expose confirm in the normalized dataframe.
    In that case, the safest generic assumption is:
    - last row = live/current candle;
    - previous row = last closed candle.
    """

    if df is None or df.empty:
        return None, None

    work_df = df.copy().reset_index(drop=True)

    if "confirm" in work_df.columns:
        closed_df = work_df[work_df["confirm"] == 1].copy()

        if closed_df.empty:
            return None, None

        closed_index = int(closed_df.index[-1])
        return work_df.iloc[closed_index], closed_index

    if len(work_df) < 2:
        return None, None

    closed_index = len(work_df) - 2
    return work_df.iloc[closed_index], closed_index


def get_live_candle_for_analysis(df):
    if df is None or df.empty:
        return None, None

    work_df = df.copy().reset_index(drop=True)
    live_index = len(work_df) - 1

    return work_df.iloc[live_index], live_index



def get_timeframe_hours(timeframe):
    if timeframe == "4H":
        return 4

    return 1


def get_candle_time(candle):
    if candle is None:
        return None

    try:
        if "timestamp" in candle.index and not pd.isna(candle["timestamp"]):
            return pd.to_datetime(candle["timestamp"])
    except Exception:
        return None

    return None



def calculate_level_age_hours(candidate_index, level_index, timeframe="1H", candidate_time=None, level_time=None):
    if candidate_time is not None and level_time is not None:
        try:
            delta_hours = (pd.to_datetime(candidate_time) - pd.to_datetime(level_time)).total_seconds() / 3600

            if delta_hours >= 0:
                return float(delta_hours)
        except Exception:
            pass

    if candidate_index is None or level_index is None:
        return None

    return float(max(0, int(candidate_index) - int(level_index)) * get_timeframe_hours(timeframe))


def calculate_reaction_pct_after_level(df, level_index, candidate_index, level_price):
    """
    Measure whether price actually moved away from the level before the sweep.

    A visible liquidity level should not be only a fresh continuation high. After the
    level is created, price should react lower by at least
    LIQUIDITY_SWEEP_MIN_REACTION_PCT before it can be swept later.
    """

    if df is None or df.empty:
        return 0.0

    if level_index is None or candidate_index is None or level_price is None:
        return 0.0

    if int(level_index) >= int(candidate_index) - 1:
        return 0.0

    segment = df.iloc[int(level_index) + 1:int(candidate_index)]

    if segment.empty or "low" not in segment.columns:
        return 0.0

    lows = pd.to_numeric(segment["low"], errors="coerce").dropna()

    if lows.empty:
        return 0.0

    min_low = float(lows.min())
    level_price = float(level_price)

    if level_price <= 0:
        return 0.0

    return max(0.0, (level_price - min_low) / level_price)


def make_liquidity_level(
    price,
    level_type,
    timeframe,
    source_index,
    candidate_index,
    touches=1,
    quality=1,
    reaction_pct=0.0,
    candidate_time=None,
    level_time=None,
):
    age_hours = calculate_level_age_hours(
        candidate_index=candidate_index,
        level_index=source_index,
        timeframe=timeframe,
        candidate_time=candidate_time,
        level_time=level_time,
    )

    return {
        "price": float(price),
        "type": str(level_type),
        "timeframe": str(timeframe),
        "source_index": int(source_index),
        "source_time": None if level_time is None else pd.to_datetime(level_time),
        "age_hours": 0.0 if age_hours is None else float(age_hours),
        "touches": int(touches),
        "quality": int(quality),
        "reaction_pct": float(reaction_pct or 0.0),
    }


def liquidity_level_is_valid(level):
    if not level:
        return False

    if float(level.get("age_hours", 0.0)) < float(LIQUIDITY_SWEEP_MIN_LEVEL_AGE_BARS):
        return False

    if float(level.get("reaction_pct", 0.0)) < float(LIQUIDITY_SWEEP_MIN_REACTION_PCT):
        return False

    if float(level.get("price", 0.0)) <= 0:
        return False

    return True


def collect_swing_high_levels(df, candidate_index, timeframe="1H", max_lookback=None):
    if df is None or df.empty or candidate_index is None:
        return []

    work_df = df.copy().reset_index(drop=True)

    if "high" not in work_df.columns or "low" not in work_df.columns:
        return []

    candidate_index = int(candidate_index)
    left = int(LIQUIDITY_SWEEP_SWING_LEFT_BARS)
    right = int(LIQUIDITY_SWEEP_SWING_RIGHT_BARS)

    if candidate_index <= left + right:
        return []

    if max_lookback is None:
        max_lookback = len(work_df)

    start_index = max(0, candidate_index - int(max_lookback))
    candidate_time = get_candle_time(work_df.iloc[candidate_index]) if candidate_index < len(work_df) else None

    highs = pd.to_numeric(work_df["high"], errors="coerce")
    levels = []

    # Right side must be fully formed before candidate candle.
    for i in range(start_index + left, candidate_index - right):
        high_value = highs.iloc[i]

        if pd.isna(high_value):
            continue

        left_window = highs.iloc[i - left:i]
        right_window = highs.iloc[i + 1:i + right + 1]

        if left_window.empty or right_window.empty:
            continue

        # Use a strict right-side condition so the level is a real pivot, not a still-forming impulse high.
        is_swing_high = (
            float(high_value) >= float(left_window.max()) and
            float(high_value) > float(right_window.max())
        )

        if not is_swing_high:
            continue

        reaction_pct = calculate_reaction_pct_after_level(
            work_df,
            level_index=i,
            candidate_index=candidate_index,
            level_price=float(high_value),
        )
        level_time = get_candle_time(work_df.iloc[i])
        quality = 4 if timeframe == "4H" else 2

        level = make_liquidity_level(
            price=float(high_value),
            level_type=f"{timeframe} swing high",
            timeframe=timeframe,
            source_index=i,
            candidate_index=candidate_index,
            touches=1,
            quality=quality,
            reaction_pct=reaction_pct,
            candidate_time=candidate_time,
            level_time=level_time,
        )

        if liquidity_level_is_valid(level):
            levels.append(level)

    return levels


def collect_equal_high_levels_from_swings(swing_levels, timeframe="1H"):
    if not swing_levels:
        return []

    tolerance = float(LIQUIDITY_SWEEP_EQUAL_HIGH_TOLERANCE_PCT)
    sorted_levels = sorted(swing_levels, key=lambda item: float(item.get("price", 0.0)))
    clusters = []

    for level in sorted_levels:
        price = float(level["price"])
        assigned = False

        for cluster in clusters:
            reference = float(cluster["reference_price"])

            if reference > 0 and abs(price - reference) / reference <= tolerance:
                cluster["levels"].append(level)
                cluster["reference_price"] = float(np.mean([item["price"] for item in cluster["levels"]]))
                assigned = True
                break

        if not assigned:
            clusters.append({
                "reference_price": price,
                "levels": [level],
            })

    equal_levels = []

    for cluster in clusters:
        members = cluster["levels"]

        if len(members) < 2:
            continue

        price = max(float(item["price"]) for item in members)
        oldest_member = min(members, key=lambda item: int(item.get("source_index", 0)))
        source_index = int(oldest_member["source_index"])
        source_time = oldest_member.get("source_time")
        age_hours = max(float(item.get("age_hours", 0.0)) for item in members)
        reaction_pct = max(float(item.get("reaction_pct", 0.0)) for item in members)
        quality = 5 if timeframe == "4H" else 4

        equal_level = {
            "price": float(price),
            "type": f"{timeframe} equal highs",
            "timeframe": str(timeframe),
            "source_index": int(source_index),
            "source_time": None if source_time is None else pd.to_datetime(source_time),
            "age_hours": float(age_hours),
            "touches": int(len(members)),
            "quality": int(quality),
            "reaction_pct": float(reaction_pct),
        }

        if liquidity_level_is_valid(equal_level):
            equal_levels.append(equal_level)

    return equal_levels


def collect_rolling_high_levels(df, candidate_index, lookbacks=LIQUIDITY_SWEEP_LOOKBACKS):
    if df is None or df.empty or candidate_index is None:
        return []

    work_df = df.copy().reset_index(drop=True)
    candidate_index = int(candidate_index)
    candidate_time = get_candle_time(work_df.iloc[candidate_index]) if candidate_index < len(work_df) else None
    levels = []

    for label, lookback in sorted(lookbacks.items(), key=lambda item: item[1], reverse=True):
        if candidate_index < int(lookback):
            continue

        previous = work_df.iloc[candidate_index - int(lookback):candidate_index]

        if previous.empty:
            continue

        highs = pd.to_numeric(previous["high"], errors="coerce").dropna()

        if highs.empty:
            continue

        level_index = int(highs.idxmax())
        level_price = float(highs.loc[level_index])
        reaction_pct = calculate_reaction_pct_after_level(
            work_df,
            level_index=level_index,
            candidate_index=candidate_index,
            level_price=level_price,
        )
        level_time = get_candle_time(work_df.iloc[level_index])

        level = make_liquidity_level(
            price=level_price,
            level_type=f"{label} high",
            timeframe="1H",
            source_index=level_index,
            candidate_index=candidate_index,
            touches=1,
            quality=1 if label == "12H" else 2,
            reaction_pct=reaction_pct,
            candidate_time=candidate_time,
            level_time=level_time,
        )

        if liquidity_level_is_valid(level):
            levels.append(level)

    return levels


def dedupe_liquidity_levels(levels):
    deduped = []

    for level in levels or []:
        price = float(level.get("price", 0.0))

        if price <= 0:
            continue

        duplicate_index = None

        for i, existing in enumerate(deduped):
            existing_price = float(existing.get("price", 0.0))

            if existing_price > 0 and abs(price - existing_price) / existing_price <= LIQUIDITY_SWEEP_EQUAL_HIGH_TOLERANCE_PCT / 2:
                duplicate_index = i
                break

        if duplicate_index is None:
            deduped.append(level)
        else:
            existing = deduped[duplicate_index]

            if int(level.get("quality", 0)) > int(existing.get("quality", 0)):
                deduped[duplicate_index] = level

    return deduped


def collect_liquidity_levels_for_timeframe(df, candidate_index, timeframe="1H", include_rolling=False):
    if df is None or df.empty or candidate_index is None:
        return []

    max_lookback = LIQUIDITY_SWEEP_LEVEL_LOOKBACK_4H if timeframe == "4H" else LIQUIDITY_SWEEP_LEVEL_LOOKBACK_1H

    swing_levels = collect_swing_high_levels(
        df,
        candidate_index=int(candidate_index),
        timeframe=timeframe,
        max_lookback=max_lookback,
    )
    equal_levels = collect_equal_high_levels_from_swings(swing_levels, timeframe=timeframe)
    rolling_levels = collect_rolling_high_levels(df, int(candidate_index), lookbacks=LIQUIDITY_SWEEP_LOOKBACKS) if include_rolling else []

    return dedupe_liquidity_levels(swing_levels + equal_levels + rolling_levels)


def get_candle_timestamp_value(candle):
    try:
        if candle is not None and "timestamp" in candle.index and not pd.isna(candle["timestamp"]):
            return pd.to_datetime(candle["timestamp"])
    except Exception:
        return None

    return None


def get_index_before_time(df, timestamp):
    if df is None or df.empty or timestamp is None or "timestamp" not in df.columns:
        return None

    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    valid = timestamps < pd.to_datetime(timestamp)

    if not valid.any():
        return None

    return int(valid[valid].index[-1]) + 1


def previous_high_info_before_index(df, candle_index, lookback):
    """
    Compatibility helper for older diagnostics. The sweep model now uses
    level-based liquidity detection instead of a single rolling high.
    """

    if df is None or df.empty:
        return None

    if candle_index is None:
        return None

    if candle_index < lookback:
        return None

    previous = df.iloc[candle_index - lookback:candle_index]

    if previous.empty:
        return None

    highs = pd.to_numeric(previous["high"], errors="coerce")
    valid_highs = highs.dropna()

    if valid_highs.empty:
        return None

    high_index = int(valid_highs.idxmax())
    previous_high = float(valid_highs.loc[high_index])
    level_age_bars = int(candle_index - high_index)

    return {
        "high": previous_high,
        "index": high_index,
        "age_bars": level_age_bars,
    }


def previous_high_before_index(df, candle_index, lookback):
    high_info = previous_high_info_before_index(df, candle_index, lookback)

    if high_info is None:
        return None

    return float(high_info["high"])


def evaluate_sweep_against_previous_high(
    candle,
    previous_high,
    min_break_pct=LIQUIDITY_SWEEP_MIN_BREAK_PCT,
    min_close_back_pct=LIQUIDITY_SWEEP_MIN_CLOSE_BACK_PCT,
):
    if candle is None or previous_high is None:
        return False

    current_high = float(candle["high"])
    current_close = float(candle["close"])

    required_break_level = float(previous_high) * (1 + float(min_break_pct))
    required_close_back_level = float(previous_high) * (1 - float(min_close_back_pct))

    return current_high >= required_break_level and current_close <= required_close_back_level


def is_sweep_invalidated_by_live_candle(live_candle, swept_high):
    if live_candle is None or swept_high is None:
        return False

    live_close = float(live_candle["close"])
    reclaim_level = float(swept_high) * (1 + LIQUIDITY_SWEEP_LIVE_INVALIDATION_PCT)

    return live_close >= reclaim_level


def is_confirmation_timeframe_allowed(level, confirm_tf):
    level_tf = str((level or {}).get("timeframe", "1H"))
    return get_timeframe_hours(confirm_tf) >= get_timeframe_hours(level_tf)


def candle_has_sweep_failure_character(candle):
    """
    A buy-side sweep should show at least minimal failed-continuation character.

    This is intentionally weaker than the separate Rejection candle factor. It only
    prevents a simple level-take / pullback from being counted as a liquidity sweep.
    """

    if candle is None:
        return False

    open_price = float(candle["open"])
    high_price = float(candle["high"])
    low_price = float(candle["low"])
    close_price = float(candle["close"])

    candle_range = high_price - low_price

    if candle_range <= 0:
        return False

    midpoint = low_price + candle_range * 0.5
    upper_wick = high_price - max(open_price, close_price)

    bearish_body = close_price < open_price
    weak_close = close_price <= midpoint
    visible_upper_wick = upper_wick >= candle_range * LIQUIDITY_SWEEP_MIN_UPPER_WICK_RANGE_PCT

    return bool(bearish_body or weak_close or visible_upper_wick)


def candle_approached_level_from_below(df, candidate_index, level_price):
    if df is None or df.empty or candidate_index is None or level_price is None:
        return False

    candidate_index = int(candidate_index)

    if candidate_index <= 0 or candidate_index >= len(df):
        return False

    candidate = df.iloc[candidate_index]
    previous = df.iloc[candidate_index - 1]
    level_price = float(level_price)

    previous_close = float(previous["close"])
    candidate_open = float(candidate["open"])

    return previous_close < level_price and candidate_open < level_price


def level_was_unswept_before_candidate(df, level, candidate_index):
    if df is None or df.empty or not level or candidate_index is None:
        return False

    candidate_index = int(candidate_index)

    if candidate_index <= 0:
        return False

    level_price = float(level.get("price", 0.0))

    if level_price <= 0:
        return False

    work_df = df.copy().reset_index(drop=True)
    segment = work_df.iloc[:candidate_index].copy()

    source_time = level.get("source_time")

    if source_time is not None and "timestamp" in segment.columns:
        timestamps = pd.to_datetime(segment["timestamp"], errors="coerce")
        segment = segment[timestamps > pd.to_datetime(source_time)]
    else:
        source_index = level.get("source_index")

        if source_index is not None:
            try:
                segment = work_df.iloc[int(source_index) + 1:candidate_index].copy()
            except Exception:
                segment = work_df.iloc[:candidate_index].copy()

    if segment.empty:
        return True

    highs = pd.to_numeric(segment["high"], errors="coerce")
    closes = pd.to_numeric(segment["close"], errors="coerce")
    required_break_level = level_price * (1 + LIQUIDITY_SWEEP_MIN_BREAK_PCT)
    reclaim_level = level_price * (1 + LIQUIDITY_SWEEP_LIVE_INVALIDATION_PCT)

    if highs.notna().any() and bool((highs >= required_break_level).any()):
        return False

    if closes.notna().any() and bool((closes >= reclaim_level).any()):
        return False

    return True


def confirmed_sweep_conditions_ok(candle, level, confirm_tf, df_context, candidate_index):
    if candle is None or not level:
        return False

    level_price = float(level.get("price", 0.0))

    if level_price <= 0:
        return False

    if not is_confirmation_timeframe_allowed(level, confirm_tf):
        return False

    if not evaluate_sweep_against_previous_high(candle, level_price):
        return False

    if not candle_approached_level_from_below(df_context, candidate_index, level_price):
        return False

    if not level_was_unswept_before_candidate(df_context, level, candidate_index):
        return False

    if not candle_has_sweep_failure_character(candle):
        return False

    return True


def select_best_confirmed_swept_level(candle, levels, confirm_tf, df_context, candidate_index):
    swept_levels = []

    for level in levels or []:
        if confirmed_sweep_conditions_ok(
            candle=candle,
            level=level,
            confirm_tf=confirm_tf,
            df_context=df_context,
            candidate_index=candidate_index,
        ):
            swept_levels.append(level)

    if not swept_levels:
        return None

    return sorted(
        swept_levels,
        key=lambda item: (
            int(item.get("quality", 0)),
            int(item.get("touches", 1)),
            float(item.get("age_hours", 0.0)),
            float(item.get("price", 0.0)),
        ),
        reverse=True,
    )[0]


def liquidity_level_label(level):
    if not level:
        return "liquidity level"

    label = str(level.get("type", "liquidity level"))
    touches = int(level.get("touches", 1))

    if touches >= 2 and "equal" in label:
        return f"{label} x{touches}"

    return label


def liquidity_level_points(level):
    quality = int((level or {}).get("quality", 0))

    if quality >= 4:
        return 2

    return 1


def make_confirmed_sweep_factor(level, confirm_tf):
    swept_price = float(level["price"])
    level_label = liquidity_level_label(level)

    return make_factor(
        key="liquidity_sweep",
        label="Liquidity sweep",
        status="confirmed",
        points=liquidity_level_points(level),
        detail=f"{confirm_tf} closed below {level_label} {format_price_2(swept_price)}",
    )


def detect_liquidity_sweep(df_1h, df_4h=None, lookbacks=LIQUIDITY_SWEEP_LOOKBACKS):
    """
    Confirmed-only level-based liquidity sweep model.

    Current rule:
    - 1H liquidity level can be confirmed by a closed 1H or closed 4H candle.
    - 4H liquidity level can be confirmed only by a closed 4H candle.
    - Lower-TF closes do not confirm higher-TF sweeps.
    - Live/candidate sweeps are intentionally not reported or scored.

    A confirmed buy-side sweep requires:
    - old visible liquidity level;
    - approach from below;
    - first meaningful break above that level;
    - close back below;
    - minimal failed-continuation / rejection character.
    """

    if df_1h is None or df_1h.empty:
        return make_factor(
            key="liquidity_sweep",
            label="Liquidity sweep",
            status="not_enough_data",
            detail="1H: no candles",
        )

    df1 = df_1h.copy().reset_index(drop=True)

    if len(df1) < max(lookbacks.values()) + 1:
        return make_factor(
            key="liquidity_sweep",
            label="Liquidity sweep",
            status="not_enough_data",
            detail=f"1H: requires {max(lookbacks.values()) + 1} candles",
        )

    live_1h_candle, live_1h_index = get_live_candle_for_analysis(df1)
    closed_1h_candle, closed_1h_index = get_last_closed_candle_for_analysis(df1)
    confirmed_candidates = []

    # 1H levels confirmed by the last closed 1H candle.
    if closed_1h_candle is not None and closed_1h_index is not None:
        levels_1h = collect_liquidity_levels_for_timeframe(
            df1,
            candidate_index=closed_1h_index,
            timeframe="1H",
            include_rolling=True,
        )
        swept_1h = select_best_confirmed_swept_level(
            candle=closed_1h_candle,
            levels=levels_1h,
            confirm_tf="1H",
            df_context=df1,
            candidate_index=closed_1h_index,
        )

        if swept_1h is not None:
            confirmed_candidates.append((swept_1h, "1H"))

    # 4H closed candle can confirm 4H levels, and can also confirm older 1H levels.
    if df_4h is not None and not df_4h.empty:
        df4 = df_4h.copy().reset_index(drop=True)
        closed_4h_candle, closed_4h_index = get_last_closed_candle_for_analysis(df4)

        if closed_4h_candle is not None and closed_4h_index is not None:
            levels_4h = collect_liquidity_levels_for_timeframe(
                df4,
                candidate_index=closed_4h_index,
                timeframe="4H",
                include_rolling=False,
            )
            swept_4h_level = select_best_confirmed_swept_level(
                candle=closed_4h_candle,
                levels=levels_4h,
                confirm_tf="4H",
                df_context=df4,
                candidate_index=closed_4h_index,
            )

            if swept_4h_level is not None:
                confirmed_candidates.append((swept_4h_level, "4H"))

            # Optional stronger confirmation: a closed 4H candle can confirm pre-existing 1H liquidity levels.
            candle_4h_time = get_candle_timestamp_value(closed_4h_candle)
            candidate_1h_index_for_4h = get_index_before_time(df1, candle_4h_time)

            if candidate_1h_index_for_4h is not None and candidate_1h_index_for_4h > 0:
                levels_1h_for_4h = collect_liquidity_levels_for_timeframe(
                    df1,
                    candidate_index=candidate_1h_index_for_4h,
                    timeframe="1H",
                    include_rolling=True,
                )
                swept_1h_by_4h = select_best_confirmed_swept_level(
                    candle=closed_4h_candle,
                    levels=levels_1h_for_4h,
                    confirm_tf="4H",
                    df_context=df4,
                    candidate_index=closed_4h_index,
                )

                if swept_1h_by_4h is not None:
                    confirmed_candidates.append((swept_1h_by_4h, "4H"))

    if not confirmed_candidates:
        return make_factor(
            key="liquidity_sweep",
            label="Liquidity sweep",
            status="not_confirmed",
            detail="no confirmed 1H/4H liquidity sweep",
        )

    # Select strongest confirmed sweep across allowed confirmations.
    best_level, best_confirm_tf = sorted(
        confirmed_candidates,
        key=lambda item: (
            get_timeframe_hours(item[1]),
            int(item[0].get("quality", 0)),
            int(item[0].get("touches", 1)),
            float(item[0].get("age_hours", 0.0)),
            float(item[0].get("price", 0.0)),
        ),
        reverse=True,
    )[0]

    swept_price = float(best_level["price"])

    # If current/live price has already reclaimed the swept level, the sweep is not an active short-watch trigger.
    if live_1h_candle is not None and is_sweep_invalidated_by_live_candle(live_1h_candle, swept_price):
        return make_factor(
            key="liquidity_sweep",
            label="Liquidity sweep",
            status="not_confirmed",
            detail="no confirmed 1H/4H liquidity sweep",
        )

    return make_confirmed_sweep_factor(best_level, best_confirm_tf)

def classify_premium_position(position):
    if position is None or pd.isna(position):
        return "N/A"

    position = float(position)

    if position > 1.0:
        return "Above Range High"

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

            if label in ("Above Range High", "Extreme Premium"):
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
                detail=f"1H: new {label} high",
            )

    if checked_any:
        return make_factor(
            key="local_high_update",
            label="Local high update",
            status="not_confirmed",
            detail="1H: no 24H/48H/7D high update",
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
    """
    Detect a confirmed 1H rejection candle using only the last closed candle.

    Important:
    - Live candles are intentionally ignored for rejection confirmation.
    - This prevents a still-forming impulse candle from being counted as a confirmed short trigger.
    """

    if df_1h is None or df_1h.empty:
        return make_factor(
            key="rejection_candle",
            label="Rejection candle",
            status="not_enough_data",
            detail="1H: no candles",
        )

    df = df_1h.copy().reset_index(drop=True)
    closed_candle, closed_index = get_last_closed_candle_for_analysis(df)

    if closed_candle is None or closed_index is None:
        return make_factor(
            key="rejection_candle",
            label="Rejection candle",
            status="not_enough_data",
            detail="1H closed: no closed candle",
        )

    if closed_index < volume_lookback:
        return make_factor(
            key="rejection_candle",
            label="Rejection candle",
            status="not_enough_data",
            detail=f"requires {volume_lookback} closed 1H candles before trigger candle",
        )

    current = closed_candle

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
            detail="1H closed: invalid candle range",
        )

    body = abs(close_price - open_price)
    body_reference = max(body, candle_range * 0.05)
    upper_wick = high_price - max(open_price, close_price)
    midpoint = low_price + candle_range * 0.5

    volume_series = get_volume_series(df)
    current_volume_raw = volume_series.iloc[closed_index]
    previous_volume_window = volume_series.iloc[closed_index - volume_lookback:closed_index]

    current_volume = float(current_volume_raw) if not pd.isna(current_volume_raw) else np.nan
    previous_avg_volume = float(previous_volume_window.mean())

    if pd.isna(current_volume) or pd.isna(previous_avg_volume) or previous_avg_volume <= 0:
        return make_factor(
            key="rejection_candle",
            label="Rejection candle",
            status="not_enough_data",
            detail="1H closed: volume unavailable",
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
            detail=f"1H closed: upper wick + weak close + volume {current_volume / previous_avg_volume:.1f}x",
        )

    return make_factor(
        key="rejection_candle",
        label="Rejection candle",
        status="not_confirmed",
        detail="1H closed: no wick/weak-close/volume confirmation",
    )


def analyze_short_factors(df_1h, df_4h=None):
    premium_factor = detect_premium_zone(df_1h, df_4h)

    factors = [
        detect_liquidity_sweep(df_1h, df_4h=df_4h),
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
    rsi_1h_live_value = safe_float(rsi_1h_live)
    rsi_1h_closed_value = safe_float(rsi_1h_closed)
    rsi_4h_live_value = safe_float(rsi_4h_live)
    volume_24h_value = safe_float(exact_volume_24h)
    price_change_24h_value = safe_float(price_change_24h)

    pump_score = 0

    if price_change_24h_value >= 40:
        pump_score = 3
    elif price_change_24h_value >= EXTREME_PUMP_PRICE_CHANGE_24H:
        pump_score = 2
    elif price_change_24h_value >= PUMP_WATCH_PRICE_CHANGE_24H:
        pump_score = 1

    rsi_score = 0

    if rsi_1h_live_value >= EXTREME_PUMP_RSI_1H_LIVE:
        rsi_score += 2
    elif rsi_1h_live_value >= OVERHEAT_WATCH_RSI_1H_LIVE:
        rsi_score += 1

    if rsi_4h_live_value >= EXTREME_PUMP_RSI_4H_LIVE:
        rsi_score += 2
    elif rsi_4h_live_value >= 75:
        rsi_score += 1

    if rsi_1h_closed_value >= RSI_1H_CLOSED_CONFIRMATION:
        rsi_score += 1

    volume_score = 0
    volume_ok = volume_24h_value >= MIN_VOLUME_USD_24H

    if volume_ok:
        volume_score += 1

    vol_chg = None if volume_change_24h is None or pd.isna(volume_change_24h) else float(volume_change_24h)

    if vol_chg is not None:
        if vol_chg >= 300:
            volume_score += 3
        elif vol_chg >= 100:
            volume_score += 2
        elif vol_chg >= 50:
            volume_score += 1

    overheat_context = detect_overheat_watch_context(
        rsi_1h_live=rsi_1h_live_value,
        rsi_1h_closed=rsi_1h_closed_value,
        exact_volume_24h=volume_24h_value,
        price_change_24h=price_change_24h_value,
    )

    has_basic_pump_context = (
        price_change_24h_value >= PUMP_WATCH_PRICE_CHANGE_24H and
        volume_ok
    )

    has_strong_pump_context = (
        price_change_24h_value >= EXTREME_PUMP_PRICE_CHANGE_24H and
        volume_ok
    )

    has_extreme_pump_context = (
        price_change_24h_value >= 40 and
        volume_ok
    )

    final_score = pump_score + rsi_score + volume_score + short_setup_score

    return {
        "pump_score": pump_score,
        "rsi_score": rsi_score,
        "volume_score": volume_score,
        "short_setup_score": int(short_setup_score),
        "final_score": int(final_score),
        "volume_ok": bool(volume_ok),
        "has_basic_pump_context": bool(has_basic_pump_context),
        "has_strong_pump_context": bool(has_strong_pump_context),
        "has_extreme_pump_context": bool(has_extreme_pump_context),
        "has_overheat_context": bool(overheat_context["is_overheat"]),
        "overheat_reason": str(overheat_context["reason"]),
    }


def get_short_factor(short_factors, key):
    if not isinstance(short_factors, list):
        return None

    for factor in short_factors:
        if isinstance(factor, dict) and factor.get("key") == key:
            return factor

    return None


def is_factor_confirmed(short_factors, key):
    factor = get_short_factor(short_factors, key)
    return bool(factor and factor.get("status") == "confirmed")


def calculate_location_trigger_context(short_factors):
    """
    Split short analysis into location context and trigger confirmation.

    Location context:
    - Premium zone
    - Local high update

    Trigger / confirmation:
    - Liquidity sweep
    - Rejection candle
    """

    premium = get_short_factor(short_factors, "premium_zone") or {}
    local_high = get_short_factor(short_factors, "local_high_update") or {}

    liquidity = get_short_factor(short_factors, "liquidity_sweep") or {}
    liquidity_confirmed = is_factor_confirmed(short_factors, "liquidity_sweep")
    liquidity_candidate = liquidity.get("status") == "candidate"
    rejection_confirmed = is_factor_confirmed(short_factors, "rejection_candle")

    location_score = int(premium.get("points", 0)) + int(local_high.get("points", 0))
    trigger_count = int(liquidity_confirmed) + int(rejection_confirmed)
    early_trigger_count = int(liquidity_candidate)

    trigger_parts = []

    if liquidity_confirmed:
        trigger_parts.append("liquidity sweep")

    if liquidity_candidate:
        trigger_parts.append("live liquidity sweep candidate")

    if rejection_confirmed:
        trigger_parts.append("rejection candle")

    return {
        "location_score": location_score,
        "trigger_count": trigger_count,
        "early_trigger_count": early_trigger_count,
        "liquidity_confirmed": liquidity_confirmed,
        "liquidity_candidate": liquidity_candidate,
        "rejection_confirmed": rejection_confirmed,
        "trigger_parts": trigger_parts,
    }


def quality_location_label(score):
    score = int(score or 0)

    if score >= 5:
        return "Extreme"
    if score >= 3:
        return "Strong"
    if score >= 1:
        return "Moderate"
    return "None"


def quality_trigger_label_from_context(context):
    trigger_count = int(context.get("trigger_count", 0))
    liquidity_confirmed = bool(context.get("liquidity_confirmed"))
    rejection_confirmed = bool(context.get("rejection_confirmed"))

    if trigger_count >= 2:
        return "Confirmed — 1H liquidity sweep + 1H rejection"

    if liquidity_confirmed:
        return "Partial — 1H liquidity sweep only"

    if rejection_confirmed:
        return "Partial — 1H rejection only"

    if bool(context.get("liquidity_candidate")):
        return "Early — 1H live liquidity sweep candidate"

    return "None"


def build_setup_status(signal_level, scores, short_factors):
    rsi_score = int(scores.get("rsi_score", 0))
    context = calculate_location_trigger_context(short_factors)
    location_score = int(context.get("location_score", 0))
    trigger_count = int(context.get("trigger_count", 0))
    rejection_confirmed = bool(context.get("rejection_confirmed"))
    liquidity_candidate = bool(context.get("liquidity_candidate"))
    has_overheat_context = bool(scores.get("has_overheat_context", False))

    if signal_level == "HIGH_PRIORITY_SHORT_WATCH":
        return "High priority watch — strong heat, premium/location, and confirmed 1H trigger are aligned"

    if signal_level == "SHORT_WATCH":
        if trigger_count >= 2:
            return "Short watch — 1H liquidity sweep + 1H rejection detected"
        if rejection_confirmed:
            return "Short watch — 1H rejection detected, liquidity sweep not confirmed"
        return "Short watch — 1H liquidity sweep detected, rejection still missing"

    if signal_level == "OVERHEAT_WATCH":
        if location_score > 0:
            return "Overheat watch — RSI heat confirmed; waiting for confirmed 1H short trigger"
        return "Overheat watch — RSI heat confirmed; waiting for premium/location and 1H trigger"

    if signal_level == "PUMP_WATCH":
        missing = []

        if rsi_score < 1 and not has_overheat_context:
            missing.append("RSI heat")

        if location_score <= 0:
            missing.append("premium/location")

        if trigger_count == 0:
            if liquidity_candidate:
                missing.append("confirmed 1H trigger")
            else:
                missing.append("1H short trigger")
        elif not rejection_confirmed:
            missing.append("1H rejection")

        if missing:
            return "Watch only — missing " + " / ".join(missing)

        return "Watch only — setup quality is still insufficient"

    return "No watch setup"


def build_watch_reason(signal_level, scores, short_factors):
    pump_quality = quality_pump_label(scores.get("pump_score", 0))
    heat_quality = quality_heat_label(scores.get("rsi_score", 0))
    context = calculate_location_trigger_context(short_factors)
    location_quality = quality_location_label(context.get("location_score", 0))
    trigger_quality = quality_trigger_label_from_context(context)
    setup_status = build_setup_status(signal_level, scores, short_factors)

    if signal_level == "HIGH_PRIORITY_SHORT_WATCH":
        return f"{pump_quality} pump + {heat_quality} RSI heat + {location_quality} location + confirmed 1H trigger"

    if signal_level == "SHORT_WATCH":
        pump_part = "overheat" if scores.get("has_overheat_context") else f"{pump_quality} pump"
        return f"{pump_part} + {location_quality} location + {trigger_quality}. RSI heat: {heat_quality}"

    if signal_level == "OVERHEAT_WATCH":
        return str(scores.get("overheat_reason") or f"{pump_quality} pump + {heat_quality} RSI heat")

    if signal_level == "PUMP_WATCH":
        base_parts = [f"{pump_quality} pump"]

        if location_quality != "None":
            base_parts.append(f"{location_quality} location")

        if trigger_quality != "None":
            base_parts.append(trigger_quality)

        return " + ".join(base_parts) + ", but " + setup_status.replace("Watch only — ", "")

    return "Watch filters not passed"


def classify_watch_signal(scores, short_factors=None):
    """
    Short Watch Analysis model.

    A watchlist candidate must be one of:
    - pump context with incomplete but relevant location/heat/trigger context;
    - overheat context based on strict RSI + 24h change + 24h volume thresholds;
    - short-watch context with location and at least one confirmed 1H trigger.

    The function does not produce execution signals.
    """

    short_factors = short_factors or []

    pump_score = int(scores.get("pump_score", 0))
    rsi_score = int(scores.get("rsi_score", 0))

    context = calculate_location_trigger_context(short_factors)
    location_score = int(context.get("location_score", 0))
    trigger_count = int(context.get("trigger_count", 0))
    liquidity_candidate = bool(context.get("liquidity_candidate"))

    premium_factor = get_short_factor(short_factors, "premium_zone") or {}
    premium_points = int(premium_factor.get("points", 0))

    has_basic_pump = bool(scores.get("has_basic_pump_context", False)) or pump_score >= 1
    has_strong_pump = bool(scores.get("has_strong_pump_context", False)) or pump_score >= 2
    has_extreme_pump = bool(scores.get("has_extreme_pump_context", False)) or pump_score >= 3
    has_overheat_context = bool(scores.get("has_overheat_context", False))

    has_heat = rsi_score >= 1
    has_strong_heat = rsi_score >= 3 or has_overheat_context

    has_location = location_score >= 2
    has_strong_location = location_score >= 3 or premium_points >= 2
    has_premium_context = premium_points >= 1

    has_trigger = trigger_count >= 1

    pump_or_overheat = has_basic_pump or has_overheat_context

    if (
        (has_extreme_pump or has_strong_pump)
        and has_strong_heat
        and has_premium_context
        and has_strong_location
        and has_trigger
    ):
        signal_level = "HIGH_PRIORITY_SHORT_WATCH"
    elif pump_or_overheat and has_location and has_trigger:
        signal_level = "SHORT_WATCH"
    elif has_overheat_context:
        signal_level = "OVERHEAT_WATCH"
    elif has_basic_pump and has_location and (has_heat or liquidity_candidate):
        signal_level = "PUMP_WATCH"
    else:
        signal_level = "NO_SIGNAL"

    return {
        "signal_level": signal_level,
        "reason": build_watch_reason(signal_level, scores, short_factors),
        "location_score": int(location_score),
        "trigger_count": int(trigger_count),
        "setup_status": build_setup_status(signal_level, scores, short_factors),
    }


def classify_signal(
    rsi_1h_live,
    rsi_1h_closed,
    rsi_4h_live,
    exact_volume_24h,
    volume_change_24h,
    price_change_24h,
    short_setup_score=0,
    short_factors=None,
):
    """
    Watchlist signal model.

    This model separates:
    - pump / heat context;
    - location context;
    - short trigger confirmation.

    Closed 4H RSI is intentionally not used.
    """

    short_factors = short_factors or []

    scores = calculate_scores(
        rsi_1h_live=rsi_1h_live,
        rsi_1h_closed=rsi_1h_closed,
        rsi_4h_live=rsi_4h_live,
        exact_volume_24h=exact_volume_24h,
        volume_change_24h=volume_change_24h,
        price_change_24h=price_change_24h,
        short_setup_score=short_setup_score,
    )

    classification = classify_watch_signal(scores, short_factors=short_factors)
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
        short_factors=short_analysis["factors"],
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
        "location_score": classification.get("location_score", 0),
        "trigger_count": classification.get("trigger_count", 0),
        "setup_status": classification.get("setup_status", "N/A"),
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
        short_factors=short_analysis["factors"],
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
        "location_score": classification.get("location_score", 0),
        "trigger_count": classification.get("trigger_count", 0),
        "setup_status": classification.get("setup_status", "N/A"),
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
                "location_score": int(row.get("location_score", 0)),
                "trigger_count": int(row.get("trigger_count", 0)),
                "setup_status": str(row.get("setup_status", "N/A")),
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
    elif status == "candidate":
        icon = "⚠️"
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

    if score >= 3:
        return "Strong spike"
    if score == 2:
        return "Strong"
    if score == 1:
        return "Normal"
    return "Weak"


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
    short_factors = detail.get("short_factors", []) or []
    signal_level = str(detail.get("signal_level", "NO_SIGNAL"))

    context = calculate_location_trigger_context(short_factors)

    return {
        "pump_quality": quality_pump_label(pump_score),
        "heat_quality": quality_heat_label(rsi_score),
        "volume_quality": quality_volume_label(volume_score),
        "location_quality": quality_location_label(context.get("location_score", 0)),
        "trigger_quality": quality_trigger_label_from_context(context),
        "priority_quality": quality_priority_label(signal_level),
        "setup_status": str(detail.get("setup_status", build_setup_status(signal_level, detail, short_factors))),
    }



def compact_factor_summary(factors):
    factor_keys = [
        ("liquidity_sweep", "Sweep"),
        ("premium_zone", "Prem"),
        ("local_high_update", "High"),
        ("rejection_candle", "Rej"),
    ]

    parts = []

    for key, label in factor_keys:
        factor = find_factor(factors, key) or {}
        status = factor.get("status")

        if status == "confirmed":
            icon = "✅"
        elif status == "candidate":
            icon = "⚠️"
        elif status == "not_enough_data":
            icon = "⚪"
        else:
            icon = "❌"

        parts.append(f"{icon} {label}")

    return " | ".join(parts)


def compact_factor_detail(factors, key, default="N/A"):
    factor = find_factor(factors, key) or {}
    detail = str(factor.get("detail", "") or "").strip()

    if not detail:
        return default

    detail = detail.replace("Liquidity sweep", "Sweep")
    detail = detail.replace("Rejection candle", "Rej")
    detail = detail.replace("1H: ", "")
    detail = detail.replace("1H closed: ", "closed: ")

    return detail

def format_multi_provider_telegram(
    grouped_signals,
    okx_total,
    okx_prefiltered,
    okx_active,
    bitget_total,
    bitget_prefiltered,
    bitget_active,
):
    """
    Compact Telegram report with a single Short Watch Analysis block.
    """

    now_kyiv = datetime.now(KYIV_TZ).strftime("%Y-%m-%d %H:%M Kyiv")

    lines = []

    total_short_count = len(grouped_signals) if grouped_signals else 0
    display_groups = (grouped_signals or [])[:TELEGRAM_MAX_SIGNALS]
    displayed_count = len(display_groups)

    lines.append("📊 <b>Market Heat Scanner</b>")
    lines.append(f"🕒 <code>{html.escape(now_kyiv)}</code>")
    lines.append(f"🧩 <code>{html.escape(SCRIPT_VERSION)}</code>")

    if total_short_count > displayed_count:
        lines.append(f"Shown: <b>{displayed_count}</b>/<b>{total_short_count}</b>")

    if not display_groups:
        lines.append("✅ No short-watch signals.")
    else:
        for idx, group in enumerate(display_groups):
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
            setup_status = html.escape(str(detail.get("setup_status", "N/A")))

            short_factors = detail.get("short_factors", []) or []
            premium_info = format_premium_line_from_factors(short_factors)
            premium_1h = html.escape(premium_info["premium_1h"])
            premium_4h = html.escape(premium_info["premium_4h"])
            factor_summary = html.escape(compact_factor_summary(short_factors))
            sweep_detail = html.escape(compact_factor_detail(short_factors, "liquidity_sweep", "N/A"))

            lines.append("")
            lines.append(f"{idx + 1}) {signal_label} | <code>{symbol}</code>")
            lines.append(f"💵 <code>{price}</code> | 24h <code>{chg_24h}%</code> | Vol <code>{vol_24h}</code> | ΔVol <code>{vol_chg}%</code>")
            lines.append(f"📊 RSI 1H <code>{rsi_1h_live}/{rsi_1h_closed}</code> | 4H <code>{rsi_4h_live}</code>")
            lines.append(f"🗺 Prem: 1H <code>{premium_1h}</code> | 4H <code>{premium_4h}</code>")
            lines.append(f"⚙️ {factor_summary}")
            lines.append(f"🔎 Sweep: <code>{sweep_detail}</code>")
            lines.append(f"🧭 <i>{setup_status}</i>")

            if idx != len(display_groups) - 1:
                lines.append("────────────")

    lines.append("")
    lines.append("Planned: ⚪ OI/Funding | ⚪ Vol climax | ⚪ Failed BO | ⚪ MSS | ⚪ Div")

    return "\n".join(lines)


# ============================================================
# MULTI PROVIDER RUNNER
# ============================================================

def run_multi_provider_screener():
    print("\n" + "=" * 120)
    print("RUNNING MULTI-PROVIDER SCREENER")
    print("SCRIPT VERSION:", SCRIPT_VERSION)
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

    if SEND_MESSAGE_IF_NO_SIGNALS or grouped_signals:
        message = format_multi_provider_telegram(
            grouped_signals=grouped_signals,
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
