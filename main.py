import os
import sys
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

SCRIPT_VERSION = "p0-sweep-v4-compact-20260613-r31"

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

# Entry filter for Telegram short-list candidates.
# A confirmed sweep is not enough if RSI heat is too weak.
RSI_ENTRY_MIN_1H_LIVE = 65
RSI_ENTRY_MIN_1H_CLOSED = 65
RSI_ENTRY_MIN_4H_LIVE = 68

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
LIQUIDITY_SWEEP_MIN_LEVEL_AGE_BARS = 5 # swept high must be at least 5 bars old on its own timeframe
LIQUIDITY_SWEEP_MIN_REACTION_PCT = 0.015 # price must have reacted at least 1.5% from the level before sweep
LIQUIDITY_SWEEP_EQUAL_HIGH_TOLERANCE_PCT = 0.0025 # highs within 0.25% are treated as equal-high liquidity
LIQUIDITY_SWEEP_MIN_UPPER_WICK_RANGE_PCT = 0.30 # sweep candle should show rejection/failure character
LIQUIDITY_SWEEP_SWING_LEFT_BARS = 2
LIQUIDITY_SWEEP_SWING_RIGHT_BARS = 2
# A single swing high must stand out from nearby candles.
# This prevents tiny intrapump pivots from being shown as chart-level liquidity highs.
LIQUIDITY_SWEEP_MIN_SINGLE_HIGH_PROMINENCE_PCT = 0.0020 # 0.20% above nearby highs
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
# Local high is evaluated over the recent setup window, not only the latest live candle.
# This keeps the factor aligned with a sweep event that may have occurred a few candles before the scan.
LOCAL_HIGH_RECENT_WINDOW_BARS = 6

# Open levels are location/context amplifiers. They are not triggers, but they can strengthen signal_level when a confirmed sweep exists.
OPEN_LEVEL_NEAR_THRESHOLDS = {
    "D": 0.0015,   # 0.15% below daily open
    "W": 0.0035,   # 0.35% below weekly open
    "M": 0.0050,   # 0.50% below monthly open
    "Y": 0.0050,   # 0.50% below yearly open
}
# If price has reclaimed an open level, that level is no longer active resistance.
# This prevents old D/W/M/Y tests from being reported after price has already moved above the level.
OPEN_LEVEL_RECLAIM_INVALIDATION_PCT = 0.0  # any close back above the open level invalidates old resistance context
OPEN_LEVEL_CONTEXT_WEIGHTS = {
    "D": {"near": 0.50, "live_test": 0.75, "tested": 1.00},
    "W": {"near": 1.00, "live_test": 1.50, "tested": 2.00},
    "M": {"near": 1.00, "live_test": 1.50, "tested": 2.00},
    "Y": {"near": 1.00, "live_test": 1.50, "tested": 2.00},
}
# Open-level tests should be aligned with the recent setup window, not only the latest closed candle.
OPEN_LEVEL_RECENT_WINDOW_1H_BARS = 6
OPEN_LEVEL_RECENT_WINDOW_4H_BARS = 2

# Rejection Candle v1. Rejection is a trigger only when it happens at D/W/M/Y open-level resistance.
REJECTION_RECENT_WINDOW_1H_BARS = 6
REJECTION_RECENT_WINDOW_4H_BARS = 2
REJECTION_MIN_UPPER_WICK_RANGE_PCT = 0.35
REJECTION_MAX_CLOSE_POSITION_PCT = 0.55

# Open Interest context layer (r28).
# Context only: it is shown in Telegram but does not affect scoring yet.
OI_HISTORY_PERIOD = "1H"
OI_HISTORY_LOOKBACK_HOURS = 12
OI_CHANGE_FLAT_THRESHOLD_PCT = 2.0
OI_CHANGE_1H_ACTIVE_THRESHOLD_PCT = 3.0
OI_CHANGE_4H_ACTIVE_THRESHOLD_PCT = 5.0
OI_CHANGE_STRONG_4H_THRESHOLD_PCT = 10.0
OI_CHANGE_NEGATIVE_THRESHOLD_PCT = -5.0

PRE_FILTER_TOP_N = 40
FINAL_TOP_N = 30
TELEGRAM_MAX_SIGNALS = 10
CANDLE_LIMIT_1H = 500
CANDLE_LIMIT_4H = 500
CANDLE_LIMIT_1D = 300

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


def calculate_level_age_bars(candidate_index, level_index):
    """
    Return level age in bars of the level's own timeframe.

    This is intentionally not converted to hours. A 4H level with
    LIQUIDITY_SWEEP_MIN_LEVEL_AGE_BARS = 5 must be at least five
    completed 4H bars old, not merely five hours old.
    """

    if candidate_index is None or level_index is None:
        return 0

    try:
        return max(0, int(candidate_index) - int(level_index))
    except Exception:
        return 0


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
    prominence_pct=0.0,
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
    age_bars = calculate_level_age_bars(
        candidate_index=candidate_index,
        level_index=source_index,
    )

    return {
        "price": float(price),
        "type": str(level_type),
        "timeframe": str(timeframe),
        "source_index": int(source_index),
        "source_time": None if level_time is None else pd.to_datetime(level_time),
        "age_bars": int(age_bars),
        "age_hours": 0.0 if age_hours is None else float(age_hours),
        "touches": int(touches),
        "quality": int(quality),
        "reaction_pct": float(reaction_pct or 0.0),
        "prominence_pct": float(prominence_pct or 0.0),
    }


def liquidity_level_is_valid(level):
    if not level:
        return False

    if int(level.get("age_bars", 0)) < int(LIQUIDITY_SWEEP_MIN_LEVEL_AGE_BARS):
        return False

    if float(level.get("reaction_pct", 0.0)) < float(LIQUIDITY_SWEEP_MIN_REACTION_PCT):
        return False

    # For a single swing high, require that it is visually meaningful,
    # not just a tiny local bump inside the current pump. Equal-high clusters
    # are handled separately and can be valid through multiple touches.
    if "swing high" in str(level.get("type", "")):
        if float(level.get("prominence_pct", 0.0)) < float(LIQUIDITY_SWEEP_MIN_SINGLE_HIGH_PROMINENCE_PCT):
            return False

    if float(level.get("price", 0.0)) <= 0:
        return False

    return True


def calculate_swing_high_prominence_pct(high_value, left_window, right_window):
    """
    Estimate how visible a swing high is compared with nearby candles.

    A tiny pivot that is only a few ticks above its neighbours is often hard to
    verify visually and should not be reported as a liquidity high.
    """

    try:
        high_value = float(high_value)

        if high_value <= 0:
            return 0.0

        neighbour_values = []

        if left_window is not None and not left_window.empty:
            neighbour_values.append(float(pd.to_numeric(left_window, errors="coerce").max()))

        if right_window is not None and not right_window.empty:
            neighbour_values.append(float(pd.to_numeric(right_window, errors="coerce").max()))

        neighbour_values = [value for value in neighbour_values if not pd.isna(value)]

        if not neighbour_values:
            return 0.0

        neighbour_high = max(neighbour_values)

        if neighbour_high <= 0:
            return 0.0

        return max(0.0, (high_value - neighbour_high) / high_value)

    except Exception:
        return 0.0


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

        prominence_pct = calculate_swing_high_prominence_pct(
            high_value=float(high_value),
            left_window=left_window,
            right_window=right_window,
        )
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
            prominence_pct=prominence_pct,
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
        newest_member = max(members, key=lambda item: int(item.get("source_index", 0)))
        source_index = int(oldest_member["source_index"])
        source_time = oldest_member.get("source_time")
        # Equal-high liquidity is considered mature only after the latest touch is also old enough.
        age_bars = min(int(item.get("age_bars", 0)) for item in members)
        age_hours = min(float(item.get("age_hours", 0.0)) for item in members)
        reaction_pct = max(float(item.get("reaction_pct", 0.0)) for item in members)
        prominence_pct = max(float(item.get("prominence_pct", 0.0)) for item in members)
        quality = 5 if timeframe == "4H" else 4

        equal_level = {
            "price": float(price),
            "type": f"{timeframe} equal highs",
            "timeframe": str(timeframe),
            "source_index": int(source_index),
            "source_time": None if source_time is None else pd.to_datetime(source_time),
            "newest_source_index": int(newest_member.get("source_index", source_index)),
            "age_bars": int(age_bars),
            "age_hours": float(age_hours),
            "touches": int(len(members)),
            "quality": int(quality),
            "reaction_pct": float(reaction_pct),
            "prominence_pct": float(prominence_pct),
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

    # r19: keep sweep confirmation strictly on its own timeframe.
    # A 1H level needs a closed 1H candle.
    # A 4H level needs a closed 4H candle.
    # This avoids confusing messages like "1H high swept, 4H close below".
    return str(confirm_tf) == level_tf


def candle_has_sweep_failure_character(candle):
    """
    A buy-side sweep should show at least minimal failed-continuation character.

    This lightweight failure check prevents a simple level-take / pullback
    from being counted as a liquidity sweep.
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
            # First, prefer the more meaningful level type: equal highs > single highs.
            int(item.get("quality", 0)),
            int(item.get("touches", 1)),
            # If several levels are swept by the same candle, show the highest one.
            # This is usually the easiest level to verify on the chart.
            float(item.get("price", 0.0)),
            float(item.get("prominence_pct", 0.0)),
            int(item.get("age_bars", 0)),
        ),
        reverse=True,
    )[0]


def liquidity_level_label(level):
    """
    User-facing liquidity level label for Telegram.

    Keep it concise:
    - 1H swing high -> 1H high
    - 4H swing high -> 4H high
    - 1H equal highs + touches -> 1H equal highs x3
    """

    if not level:
        return "liquidity level"

    raw_label = str(level.get("type", "liquidity level"))
    timeframe = str(level.get("timeframe", "")).strip()
    touches = int(level.get("touches", 1))

    if "equal highs" in raw_label:
        base = f"{timeframe} equal highs" if timeframe else "equal highs"
        return f"{base} x{touches}" if touches >= 2 else base

    if "swing high" in raw_label:
        return f"{timeframe} high" if timeframe else "swing high"

    return raw_label


def liquidity_level_points(level):
    quality = int((level or {}).get("quality", 0))

    if quality >= 4:
        return 2

    return 1


def make_confirmed_sweep_factor(level, confirm_tf, confirm_candle=None):
    swept_price = float(level["price"])
    level_label = liquidity_level_label(level)

    factor = make_factor(
        key="liquidity_sweep",
        label="Liquidity sweep",
        status="confirmed",
        points=liquidity_level_points(level),
        detail=(
            f"{level_label} {format_price_2(swept_price)} swept, close below"
            if str(confirm_tf) == str(level.get("timeframe", ""))
            else f"{level_label} {format_price_2(swept_price)} swept, {confirm_tf} close below"
        ),
    )

    confirm_high = None
    confirm_close = None
    confirm_time = None

    if confirm_candle is not None:
        try:
            confirm_high = float(confirm_candle["high"])
            confirm_close = float(confirm_candle["close"])
            confirm_time = get_candle_timestamp_value(confirm_candle)
        except (KeyError, TypeError, ValueError):
            confirm_high = None
            confirm_close = None

    factor["debug"] = {
        "result": "accepted",
        "status": "confirmed",
        "level_tf": str(level.get("timeframe", "N/A")),
        "level_type": str(level.get("type", "N/A")),
        "level_price": swept_price,
        "source_index": int(level.get("source_index", -1)),
        "source_time": format_debug_value(level.get("source_time")),
        "age_bars": int(level.get("age_bars", 0)),
        "age_hours": float(level.get("age_hours", 0.0)),
        "touches": int(level.get("touches", 1)),
        "quality": int(level.get("quality", 0)),
        "reaction_pct": float(level.get("reaction_pct", 0.0)) * 100,
        "prominence_pct": float(level.get("prominence_pct", 0.0)) * 100,
        "confirm_tf": str(confirm_tf),
        "confirm_time": format_debug_value(confirm_time),
        "confirm_high": confirm_high,
        "confirm_close": confirm_close,
        "reason": "confirmed liquidity sweep",
    }

    return factor


def detect_liquidity_sweep(df_1h, df_4h=None, lookbacks=LIQUIDITY_SWEEP_LOOKBACKS):
    """
    Confirmed-only level-based liquidity sweep model.

    Current rule:
    - 1H liquidity level can be confirmed only by a closed 1H candle.
    - 4H liquidity level can be confirmed only by a closed 4H candle.
    - Cross-timeframe sweep confirmation is intentionally disabled.
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
            include_rolling=False,
        )
        swept_1h = select_best_confirmed_swept_level(
            candle=closed_1h_candle,
            levels=levels_1h,
            confirm_tf="1H",
            df_context=df1,
            candidate_index=closed_1h_index,
        )

        if swept_1h is not None:
            confirmed_candidates.append((swept_1h, "1H", closed_1h_candle))

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
                confirmed_candidates.append((swept_4h_level, "4H", closed_4h_candle))

            # r19: do not let a 4H close confirm a 1H liquidity level.
            # Keep sweep detection easy to verify on the same chart timeframe.

    if not confirmed_candidates:
        return make_factor(
            key="liquidity_sweep",
            label="Liquidity sweep",
            status="not_confirmed",
            detail="no confirmed 1H/4H liquidity sweep",
        )

    # Select strongest confirmed sweep across allowed confirmations.
    best_level, best_confirm_tf, best_confirm_candle = sorted(
        confirmed_candidates,
        key=lambda item: (
            get_timeframe_hours(item[1]),
            int(item[0].get("quality", 0)),
            int(item[0].get("touches", 1)),
            int(item[0].get("age_bars", 0)),
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

    return make_confirmed_sweep_factor(best_level, best_confirm_tf, best_confirm_candle)

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


def detect_local_high_update(df_1h, lookbacks=LOCAL_HIGH_LOOKBACKS, recent_window_bars=LOCAL_HIGH_RECENT_WINDOW_BARS):
    """
    Detect whether the recent setup window updated a local 24H / 48H / 7D high.

    Earlier versions checked only the very last 1H candle. That could miss a
    local-high update when the sweep/pump candle happened a few candles before
    the scan and the latest candle had already pulled back.

    This factor is still context only, not a trigger.
    """

    if df_1h is None or df_1h.empty:
        return make_factor(
            key="local_high_update",
            label="Local high update",
            status="not_enough_data",
            detail="no 1H candles",
        )

    if "high" not in df_1h.columns:
        return make_factor(
            key="local_high_update",
            label="Local high update",
            status="not_enough_data",
            detail="1H high column unavailable",
        )

    df = df_1h.copy().reset_index(drop=True)
    highs = pd.to_numeric(df["high"], errors="coerce")

    if highs.dropna().empty:
        return make_factor(
            key="local_high_update",
            label="Local high update",
            status="not_enough_data",
            detail="1H highs unavailable",
        )

    window = max(1, int(recent_window_bars or 1))
    window = min(window, len(df))
    setup_slice = highs.iloc[-window:]

    if setup_slice.dropna().empty:
        return make_factor(
            key="local_high_update",
            label="Local high update",
            status="not_enough_data",
            detail="recent 1H highs unavailable",
        )

    setup_high = float(setup_slice.max())

    # Check the highest-confidence tiers first. Points are not cumulative.
    tiers = [
        ("7D", lookbacks["7D"], 3),
        ("48H", lookbacks["48H"], 2),
        ("24H", lookbacks["24H"], 1),
    ]

    checked_any = False

    for label, lookback, points in tiers:
        # Need enough candles before the setup window to compare against.
        if len(df) < lookback + window:
            continue

        previous = highs.iloc[-(lookback + window):-window]

        if previous.dropna().empty:
            continue

        checked_any = True
        previous_high = float(previous.max())

        if setup_high > previous_high:
            factor = make_factor(
                key="local_high_update",
                label="Local high update",
                status="confirmed",
                points=points,
                detail=f"{label} high updated",
            )
            factor["setup_high"] = setup_high
            factor["previous_high"] = previous_high
            factor["recent_window_bars"] = int(window)
            return factor

    if checked_any:
        factor = make_factor(
            key="local_high_update",
            label="Local high update",
            status="not_confirmed",
            detail="no 24H/48H/7D high update",
        )
        factor["setup_high"] = setup_high
        factor["recent_window_bars"] = int(window)
        return factor

    return make_factor(
        key="local_high_update",
        label="Local high update",
        status="not_enough_data",
        detail="requires enough 1H candles before setup window",
    )



def normalize_daily_candles(df_1d):
    if df_1d is None or df_1d.empty:
        return pd.DataFrame()

    if "timestamp" not in df_1d.columns or "open" not in df_1d.columns:
        return pd.DataFrame()

    df = df_1d.copy().reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df = df.dropna(subset=["timestamp", "open"])

    if df.empty:
        return df

    return df.sort_values("timestamp").reset_index(drop=True)


def normalize_intraday_candles_for_open_levels(df):
    """Return clean intraday candles with UTC and Kyiv timestamps.

    Exchange candles are normalized as UTC timestamps in the dataframe.
    r25 uses intraday UTC period starts for D/W/M/Y opens because the
    TradingView open-level drawings shown in validation screenshots match
    exchange/TradingView UTC session opens better than OKX 1D candles or
    Kyiv-local period starts.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    if "timestamp" not in df.columns or "open" not in df.columns:
        return pd.DataFrame()

    work_df = df.copy().reset_index(drop=True)
    timestamps = pd.to_datetime(work_df["timestamp"], errors="coerce")

    try:
        # OKX/Bitget timestamps are UTC but stored as timezone-naive values.
        if timestamps.dt.tz is None:
            timestamps_utc = timestamps.dt.tz_localize("UTC")
        else:
            timestamps_utc = timestamps.dt.tz_convert("UTC")
    except Exception:
        return pd.DataFrame()

    work_df["timestamp_utc"] = timestamps_utc
    work_df["timestamp_kyiv"] = timestamps_utc.dt.tz_convert(KYIV_TZ)
    work_df["open"] = pd.to_numeric(work_df["open"], errors="coerce")
    work_df = work_df.dropna(subset=["timestamp_utc", "timestamp_kyiv", "open"])

    if work_df.empty:
        return work_df

    return work_df.sort_values("timestamp_utc").reset_index(drop=True)


def get_period_start(timestamp, period_label):
    ts = pd.to_datetime(timestamp)

    if period_label == "D":
        return ts.normalize()

    if period_label == "W":
        return (ts - pd.Timedelta(days=int(ts.weekday()))).normalize()

    if period_label == "M":
        return ts.replace(day=1).normalize()

    if period_label == "Y":
        return ts.replace(month=1, day=1).normalize()

    return None


def calculate_open_levels_from_daily(df_1d):
    """Legacy fallback: return D/W/M/Y open levels from 1D UTC candles."""

    df = normalize_daily_candles(df_1d)

    if df.empty:
        return {}

    reference_time = df.iloc[-1]["timestamp"]
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    levels = {}

    for label in ["D", "W", "M", "Y"]:
        start = get_period_start(reference_time, label)

        if start is None:
            continue

        period_df = df[timestamps >= start]

        if period_df.empty:
            continue

        first_row = period_df.iloc[0]
        open_price = safe_float(first_row.get("open"), default=np.nan)

        if open_price is None or pd.isna(open_price) or float(open_price) <= 0:
            continue

        levels[label] = {
            "label": label,
            "open": float(open_price),
            "source_time": first_row.get("timestamp"),
            "source": "1D_UTC_fallback",
        }

    return levels


def get_kyiv_period_start(reference_kyiv, period_label):
    ts = pd.to_datetime(reference_kyiv)

    if period_label == "D":
        return ts.normalize()

    if period_label == "W":
        return (ts - pd.Timedelta(days=int(ts.weekday()))).normalize()

    if period_label == "M":
        return ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    if period_label == "Y":
        return ts.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

    return None


def calculate_open_levels_from_intraday(df_intraday, source="1H_UTC"):
    """Return D/W/M/Y open levels from intraday candles aligned to UTC starts.

    source should describe the candle source, for example "1H_UTC" or
    "4H_UTC". 4H candles are especially useful for W/M/Y opens when the
    exchange only returns a limited number of 1H candles and the exact month
    start is not available in the 1H dataset.
    """

    df = normalize_intraday_candles_for_open_levels(df_intraday)

    if df.empty:
        return {}

    reference_utc = df.iloc[-1]["timestamp_utc"]
    levels = {}
    timestamps_utc = pd.to_datetime(df["timestamp_utc"], errors="coerce")

    for label in ["D", "W", "M", "Y"]:
        start_utc = get_period_start(reference_utc, label)

        if start_utc is None:
            continue

        exact_match = df[timestamps_utc == start_utc]

        # Use only the exact UTC period-start candle. If it is unavailable,
        # do not use the first visible candle inside the period because that
        # creates false W/M/Y opens from the middle of the period.
        if exact_match.empty:
            continue

        first_row = exact_match.iloc[0]
        open_price = safe_float(first_row.get("open"), default=np.nan)

        if open_price is None or pd.isna(open_price) or float(open_price) <= 0:
            continue

        levels[label] = {
            "label": label,
            "open": float(open_price),
            "source_time": first_row.get("timestamp"),
            "source_time_utc": first_row.get("timestamp_utc"),
            "source_time_kyiv": first_row.get("timestamp_kyiv"),
            "source": str(source),
        }

    return levels


def calculate_open_levels(df_1d=None, df_1h=None, df_4h=None):
    """Return D/W/M/Y open levels.

    Preferred order:
    1) exact 1H UTC period-start candle;
    2) exact 4H UTC period-start candle, useful for W/M/Y when 1H history is
       clipped by the exchange response limit;
    3) legacy 1D fallback only if no exact intraday start candle is available.

    The selected source is always exposed in debug so TradingView mismatches can
    be checked quickly.
    """

    intraday_1h_levels = calculate_open_levels_from_intraday(df_1h, source="1H_UTC")
    intraday_4h_levels = calculate_open_levels_from_intraday(df_4h, source="4H_UTC")
    daily_levels = calculate_open_levels_from_daily(df_1d)
    levels = {}

    for label in ["D", "W", "M", "Y"]:
        if label in intraday_1h_levels:
            levels[label] = intraday_1h_levels[label]
        elif label in intraday_4h_levels:
            levels[label] = intraday_4h_levels[label]
        elif label in daily_levels:
            levels[label] = daily_levels[label]

    if not any(label in levels for label in ["D", "W", "M", "Y"]):
        return {}

    selected_intraday = {
        label: item.get("open")
        for label, item in levels.items()
        if isinstance(item, dict) and str(item.get("source", "")).endswith("UTC") and "1D" not in str(item.get("source", ""))
    }

    levels["_debug"] = {
        "intraday": selected_intraday,
        "intraday_1h": {label: item.get("open") for label, item in intraday_1h_levels.items()},
        "intraday_4h": {label: item.get("open") for label, item in intraday_4h_levels.items()},
        "daily": {label: item.get("open") for label, item in daily_levels.items()},
        "source": {label: item.get("source") for label, item in levels.items() if isinstance(item, dict)},
    }

    return levels




def open_level_source_is_allowed_for_signal(label, level):
    """Return True when an open level source is reliable enough for Telegram signals.

    Intraday exact-start levels are preferred. 1D fallback can differ from the
    TradingView levels used for visual validation, especially for W/M/Y. To avoid
    false open-level rejection signals, W/M/Y levels from 1D fallback are kept
    in debug but are not allowed to create Open Levels / Rejection signals.
    """

    if not level or not isinstance(level, dict):
        return False

    source = str(level.get("source", ""))
    label = str(label)

    # Y open from 1D fallback has repeatedly mismatched TradingView/open-level
    # validation. Keep it in debug, but do not let it create active signals.
    # W/M remain allowed as fallback for now, because r26+r30 usually select
    # exact intraday W/M; blocking them would remove too many valid legacy tests.
    if label == "Y" and source == "1D_UTC_fallback":
        return False

    return True


def get_previous_candle(df, candle_index):
    if df is None or df.empty or candle_index is None:
        return None

    index = int(candle_index)

    if index <= 0 or index >= len(df):
        return None

    return df.iloc[index - 1]


def get_recent_closed_candles_for_analysis(df, window_bars):
    """
    Return recent closed candles as (candle, index) pairs.

    For providers with explicit confirm flag, use the latest confirm=1 candles.
    For normalized providers without confirm, treat the last row as live and use
    candles before it as closed.
    """

    if df is None or df.empty:
        return []

    work_df = df.copy().reset_index(drop=True)
    window = max(1, int(window_bars or 1))

    if "confirm" in work_df.columns:
        closed_indices = [int(idx) for idx in work_df.index[work_df["confirm"] == 1].tolist()]
    else:
        if len(work_df) < 2:
            return []
        closed_indices = list(range(0, len(work_df) - 1))

    if not closed_indices:
        return []

    selected = closed_indices[-window:]
    return [(work_df.iloc[index], int(index)) for index in selected]


def get_all_closed_candles_for_analysis(df):
    """Return all closed candles as (candle, index) pairs.

    This is used to validate whether an old open-level rejection is still active.
    A rejection is cancelled if price later closes back above the same open level.
    """

    if df is None or df.empty:
        return []

    work_df = df.copy().reset_index(drop=True)

    if "confirm" in work_df.columns:
        closed_indices = [int(idx) for idx in work_df.index[work_df["confirm"] == 1].tolist()]
    else:
        if len(work_df) < 2:
            return []
        closed_indices = list(range(0, len(work_df) - 1))

    return [(work_df.iloc[index], int(index)) for index in closed_indices]


def get_last_closed_close_for_analysis(df):
    closed = get_all_closed_candles_for_analysis(df)

    if not closed:
        return np.nan

    candle, _ = closed[-1]
    return safe_float(candle.get("close"), default=np.nan)


def event_confirm_time(event):
    try:
        value = (event or {}).get("confirm_time")

        if value is None or str(value).strip() == "":
            return None

        parsed = pd.to_datetime(value, errors="coerce")

        if pd.isna(parsed):
            return None

        return parsed
    except Exception:
        return None



def timeframe_to_timedelta(timeframe):
    if str(timeframe).upper() == "4H":
        return pd.Timedelta(hours=4)

    return pd.Timedelta(hours=1)


def get_candle_close_time_value(candle, timeframe):
    start_time = get_candle_timestamp_value(candle)

    if start_time is None:
        return None

    try:
        return pd.to_datetime(start_time) + timeframe_to_timedelta(timeframe)
    except Exception:
        return pd.to_datetime(start_time)


def closed_candle_reclaimed_level_after_event(df, level_price, event, same_timeframe=False, df_timeframe="1H"):
    """Return True if a closed candle after the event reclaimed the open level.

    Event timestamps are confirmation/close times, not candle open times. This
    matters for 4H rejection: 1H candles inside the same 4H candle must not
    invalidate the 4H event before that 4H candle has actually closed.
    """

    if df is None or df.empty or event is None or level_price is None:
        return False

    level_price = safe_float(level_price, default=np.nan)

    if pd.isna(level_price) or float(level_price) <= 0:
        return False

    event_time = event_confirm_time(event)
    event_index = (event or {}).get("confirm_index")
    closed = get_all_closed_candles_for_analysis(df)

    if not closed:
        return False

    for candle, candle_index in closed:
        include_candle = False

        if event_time is not None and "timestamp" in candle.index:
            candle_time = get_candle_close_time_value(candle, df_timeframe)

            if candle_time is not None:
                include_candle = pd.to_datetime(candle_time) > event_time
        elif same_timeframe and event_index is not None:
            try:
                include_candle = int(candle_index) > int(event_index)
            except Exception:
                include_candle = False

        if not include_candle:
            continue

        close_value = safe_float(candle.get("close"), default=np.nan)

        if not pd.isna(close_value) and float(close_value) >= float(level_price):
            return True

    return False


def open_level_event_is_active_resistance(event, level_price, current_price=None, df_1h=None, df_4h=None):
    """Validate that an open-level test/rejection is still active now.

    The bot should not keep reporting an old rejection after the level has been
    reclaimed by later closed candles. This prevents BANANAS/NEAR/USELESS-like
    false positives where an old M-open rejection remains in the recent window,
    but price has already closed above that M open afterwards.
    """

    if event is None or level_price is None:
        return False

    level_price = safe_float(level_price, default=np.nan)

    if pd.isna(level_price) or float(level_price) <= 0:
        return False

    # Current/live price must still be below the open level. Otherwise that
    # open is no longer resistance for a short-watch rejection.
    if open_level_reclaimed_by_current_price(current_price, level_price):
        return False

    # Last closed 1H should also remain below the level. If it closed above,
    # an older rejection has already been invalidated, even if the live price
    # later falls back below.
    last_closed_1h = get_last_closed_close_for_analysis(df_1h)
    if not pd.isna(last_closed_1h) and float(last_closed_1h) >= float(level_price):
        return False

    # Same for the latest closed 4H when available.
    last_closed_4h = get_last_closed_close_for_analysis(df_4h)
    if not pd.isna(last_closed_4h) and float(last_closed_4h) >= float(level_price):
        return False

    event_tf = str((event or {}).get("confirm_tf", ""))

    if closed_candle_reclaimed_level_after_event(df_1h, level_price, event, same_timeframe=(event_tf == "1H"), df_timeframe="1H"):
        return False

    if closed_candle_reclaimed_level_after_event(df_4h, level_price, event, same_timeframe=(event_tf == "4H"), df_timeframe="4H"):
        return False

    return True


def open_level_tested_from_below(confirm_candle, previous_candle, level_price):
    if confirm_candle is None or level_price is None:
        return False

    level_price = float(level_price)

    if level_price <= 0:
        return False

    candle_high = safe_float(confirm_candle.get("high"), default=np.nan)
    candle_close = safe_float(confirm_candle.get("close"), default=np.nan)
    candle_open = safe_float(confirm_candle.get("open"), default=np.nan)

    if pd.isna(candle_high) or pd.isna(candle_close):
        return False

    took_or_touched_level = float(candle_high) >= level_price
    closed_back_below = float(candle_close) < level_price

    if not took_or_touched_level or not closed_back_below:
        return False

    approached_from_below = False

    if not pd.isna(candle_open) and float(candle_open) < level_price:
        approached_from_below = True

    if previous_candle is not None:
        previous_close = safe_float(previous_candle.get("close"), default=np.nan)

        if not pd.isna(previous_close) and float(previous_close) < level_price:
            approached_from_below = True

    return bool(approached_from_below)


def open_level_reclaimed_by_current_price(current_price, level_price, reclaim_pct=OPEN_LEVEL_RECLAIM_INVALIDATION_PCT):
    """Return True when the latest price is already back above the open level.

    A D/W/M/Y open can be resistance only while price is below it. If price has
    reclaimed the level, an older test/rejection is no longer active short-watch
    context.
    """

    if current_price is None or level_price is None:
        return False

    current_price = safe_float(current_price, default=np.nan)
    level_price = safe_float(level_price, default=np.nan)

    if pd.isna(current_price) or pd.isna(level_price) or float(level_price) <= 0:
        return False

    reclaim_level = float(level_price) * (1 + float(reclaim_pct))
    return bool(float(current_price) >= reclaim_level)


def open_level_near_from_below(current_price, level_price, threshold_pct):
    if current_price is None or level_price is None:
        return False

    current_price = safe_float(current_price, default=np.nan)
    level_price = safe_float(level_price, default=np.nan)

    if pd.isna(current_price) or pd.isna(level_price) or float(level_price) <= 0:
        return False

    current_price = float(current_price)
    level_price = float(level_price)

    if current_price >= level_price:
        return False

    distance_pct = (level_price - current_price) / level_price

    return 0 <= distance_pct <= float(threshold_pct)


def open_level_live_test_from_below(live_candle, level_price):
    """
    Live open-level interaction.

    This is not a confirmed rejection. It is context only: the current live
    candle has already touched/pierced the D/W/M/Y open from below, while the
    latest live price remains below the level.
    """

    if live_candle is None or level_price is None:
        return False

    level_price = safe_float(level_price, default=np.nan)
    candle_open = safe_float(live_candle.get("open"), default=np.nan)
    candle_high = safe_float(live_candle.get("high"), default=np.nan)
    candle_low = safe_float(live_candle.get("low"), default=np.nan)
    candle_close = safe_float(live_candle.get("close"), default=np.nan)

    if any(pd.isna(value) for value in [level_price, candle_high, candle_close]):
        return False

    level_price = float(level_price)
    candle_high = float(candle_high)
    candle_close = float(candle_close)

    if level_price <= 0:
        return False

    touched_or_pierced = candle_high >= level_price
    price_back_below = candle_close < level_price

    if not touched_or_pierced or not price_back_below:
        return False

    approached_from_below = False

    if not pd.isna(candle_open) and float(candle_open) < level_price:
        approached_from_below = True

    if not pd.isna(candle_low) and float(candle_low) < level_price:
        approached_from_below = True

    return bool(approached_from_below)


def open_level_live_near_from_below(live_candle, level_price, threshold_pct):
    """
    Live candle approached an open level from below without touching it.

    This catches the common case where price stops just below an important
    open level and starts rotating lower before an actual touch.
    """

    if live_candle is None or level_price is None:
        return False

    level_price = safe_float(level_price, default=np.nan)
    candle_high = safe_float(live_candle.get("high"), default=np.nan)
    candle_close = safe_float(live_candle.get("close"), default=np.nan)
    candle_open = safe_float(live_candle.get("open"), default=np.nan)

    if any(pd.isna(value) for value in [level_price, candle_high, candle_close]):
        return False

    level_price = float(level_price)
    candle_high = float(candle_high)
    candle_close = float(candle_close)

    if level_price <= 0:
        return False

    if candle_high >= level_price or candle_close >= level_price:
        return False

    if not pd.isna(candle_open) and float(candle_open) >= level_price:
        return False

    distance_pct = (level_price - candle_high) / level_price
    return bool(0 <= distance_pct <= float(threshold_pct))


def build_open_test_event(label, level_price, confirm_candle, confirm_index, window_tf, window_bars):
    weights = OPEN_LEVEL_CONTEXT_WEIGHTS.get(label, {"near": 0.0, "tested": 0.0})
    event = {
        "label": label,
        "state": "tested",
        "weight": float(weights.get("tested", 0.0)),
        "open": float(level_price),
        "confirm_tf": str(window_tf),
        "window_bars": int(window_bars),
        "confirm_index": None if confirm_index is None else int(confirm_index),
    }

    if confirm_candle is not None:
        event["confirm_high"] = safe_float(confirm_candle.get("high"), default=np.nan)
        event["confirm_close"] = safe_float(confirm_candle.get("close"), default=np.nan)
        if "timestamp" in confirm_candle:
            event["confirm_start_time"] = str(confirm_candle.get("timestamp"))
            close_time = get_candle_close_time_value(confirm_candle, window_tf)
            event["confirm_time"] = str(close_time if close_time is not None else confirm_candle.get("timestamp"))

    return event


def build_open_live_event(label, state, level_price, live_candle, live_index, window_tf, window_bars):
    weights = OPEN_LEVEL_CONTEXT_WEIGHTS.get(label, {"near": 0.0, "live_test": 0.0, "tested": 0.0})
    event = {
        "label": label,
        "state": str(state),
        "weight": float(weights.get(str(state), weights.get("near", 0.0))),
        "open": float(level_price),
        "confirm_tf": str(window_tf),
        "window_bars": int(window_bars),
        "confirm_index": None if live_index is None else int(live_index),
        "is_live": True,
    }

    if live_candle is not None:
        event["confirm_high"] = safe_float(live_candle.get("high"), default=np.nan)
        event["confirm_close"] = safe_float(live_candle.get("close"), default=np.nan)
        if "timestamp" in live_candle:
            event["confirm_start_time"] = str(live_candle.get("timestamp"))
            event["confirm_time"] = str(live_candle.get("timestamp"))

    return event


def evaluate_open_level_context_over_window(
    label,
    level_price,
    current_price,
    closed_candles,
    df_for_previous,
    window_tf,
    window_bars,
    live_candle=None,
    live_index=None,
):
    """
    Evaluate open-level resistance over a recent setup window.

    Priority:
    1. confirmed test: recent closed candle touched/pierced the open from below
       and closed back below;
    2. live test: current live candle touched/pierced the open from below while
       the latest price is still below the level;
    3. near from below: live/current price action is very close below the open.

    Live states are context only and never counted as confirmed rejection.
    """

    threshold = OPEN_LEVEL_NEAR_THRESHOLDS.get(label, 0.0035)
    weights = OPEN_LEVEL_CONTEXT_WEIGHTS.get(label, {"near": 0.0, "live_test": 0.0, "tested": 0.0})

    if open_level_reclaimed_by_current_price(current_price, level_price):
        return None

    tested_candidates = []

    for candle, candle_index in closed_candles or []:
        previous_candle = get_previous_candle(df_for_previous, candle_index)
        if open_level_tested_from_below(candle, previous_candle, level_price):
            event = build_open_test_event(
                label=label,
                level_price=level_price,
                confirm_candle=candle,
                confirm_index=candle_index,
                window_tf=window_tf,
                window_bars=window_bars,
            )

            if open_level_event_is_active_resistance(
                event,
                level_price,
                current_price=current_price,
                df_1h=df_for_previous if str(window_tf) == "1H" else None,
                df_4h=df_for_previous if str(window_tf) == "4H" else None,
            ):
                tested_candidates.append(event)

    if tested_candidates:
        # Prefer the most recent active confirmed test in the setup window.
        return tested_candidates[-1]

    if open_level_live_test_from_below(live_candle, level_price):
        return build_open_live_event(
            label=label,
            state="live_test",
            level_price=level_price,
            live_candle=live_candle,
            live_index=live_index,
            window_tf=window_tf,
            window_bars=window_bars,
        )

    if open_level_live_near_from_below(live_candle, level_price, threshold):
        return build_open_live_event(
            label=label,
            state="near",
            level_price=level_price,
            live_candle=live_candle,
            live_index=live_index,
            window_tf=window_tf,
            window_bars=window_bars,
        )

    if open_level_near_from_below(current_price, level_price, threshold):
        return {
            "label": label,
            "state": "near",
            "weight": float(weights.get("near", 0.0)),
            "open": float(level_price),
            "confirm_tf": str(window_tf),
            "window_bars": int(window_bars),
        }

    return None


def open_level_event_rank(event):
    """Rank open-level context events for one D/W/M/Y level."""

    if not isinstance(event, dict):
        return (0, 0, -1)

    state_priority = {"tested": 3, "live_test": 2, "near": 1}
    timeframe_priority = {"4H": 2, "1H": 1}

    state = str(event.get("state", ""))
    timeframe = str(event.get("confirm_tf", ""))
    confirm_index = event.get("confirm_index")

    try:
        confirm_index_value = -1 if confirm_index is None else int(confirm_index)
    except Exception:
        confirm_index_value = -1

    return (
        int(state_priority.get(state, 0)),
        int(timeframe_priority.get(timeframe, 0)),
        int(confirm_index_value),
    )


def select_best_open_level_event(events):
    """Select the strongest event for one open level.

    For W/M/Y opens, r17 checks both 4H and 1H windows.
    A closed test wins over a live test; 4H wins over 1H only when
    the state is otherwise equal.
    """

    valid_events = [event for event in events or [] if isinstance(event, dict)]

    if not valid_events:
        return None

    return sorted(valid_events, key=open_level_event_rank, reverse=True)[0]


def evaluate_open_level_context(label, level_price, current_price, confirm_candle, previous_candle):
    """Backward-compatible single-candle evaluator used by older tests/helpers."""
    threshold = OPEN_LEVEL_NEAR_THRESHOLDS.get(label, 0.0035)
    weights = OPEN_LEVEL_CONTEXT_WEIGHTS.get(label, {"near": 0.0, "tested": 0.0})

    if open_level_tested_from_below(confirm_candle, previous_candle, level_price):
        return {
            "label": label,
            "state": "tested",
            "weight": float(weights.get("tested", 0.0)),
            "open": float(level_price),
        }

    if open_level_near_from_below(current_price, level_price, threshold):
        return {
            "label": label,
            "state": "near",
            "weight": float(weights.get("near", 0.0)),
            "open": float(level_price),
        }

    return None


def format_open_level_price_for_telegram(value):
    """Format D/W/M/Y open values for compact Telegram diagnostics.

    For normal-price coins three decimals are enough. For small tokens, keep
    more decimals so the level remains useful instead of becoming 0.000.
    """

    if value is None or pd.isna(value):
        return "N/A"

    value = float(value)
    abs_value = abs(value)

    if abs_value >= 1:
        return f"{value:.3f}"

    if abs_value >= 0.1:
        return f"{value:.3f}"

    if abs_value >= 0.01:
        return f"{value:.5f}"

    if abs_value >= 0.001:
        return f"{value:.6f}"

    return f"{value:.8f}"


def format_open_level_group_prices(events):
    """Return compact level values for a group of D/W/M/Y events."""

    order = {"D": 0, "W": 1, "M": 2, "Y": 3}
    clean_events = [event for event in (events or []) if isinstance(event, dict)]

    if not clean_events:
        return ""

    by_label = {}
    for event in clean_events:
        label = str(event.get("label", "")).strip()
        level_price = event.get("open")

        if not label or level_price is None or pd.isna(level_price):
            continue

        by_label[label] = float(level_price)

    if not by_label:
        return ""

    labels = sorted(by_label.keys(), key=lambda value: order.get(value, 99))
    values = [by_label[label] for label in labels]

    # If grouped labels have effectively the same level value, show it once.
    if len(values) > 1:
        reference = values[0]
        same_value = all(
            reference > 0 and abs(value - reference) / reference <= 0.00001
            for value in values
        )
        if same_value:
            return f"({format_open_level_price_for_telegram(reference)})"

    if len(values) == 1:
        return f"({format_open_level_price_for_telegram(values[0])})"

    parts = [
        f"{label} {format_open_level_price_for_telegram(by_label[label])}"
        for label in labels
    ]
    return f"({' / '.join(parts)})"


def build_open_levels_detail(events):
    if not events:
        return ""

    order = {"D": 0, "W": 1, "M": 2, "Y": 3}

    def labels_for(state):
        return sorted(
            [event["label"] for event in events if event.get("state") == state],
            key=lambda x: order.get(x, 99),
        )

    def events_for(state):
        return [event for event in events if event.get("state") == state]

    tested = labels_for("tested")
    live_test = labels_for("live_test")
    near = labels_for("near")

    parts = []

    if tested:
        prices = format_open_level_group_prices(events_for("tested"))
        suffix = f" {prices}" if prices else ""
        parts.append(f"{'/'.join(tested)} open tested, close below{suffix}")

    if live_test:
        prices = format_open_level_group_prices(events_for("live_test"))
        suffix = f" {prices}" if prices else ""
        parts.append(f"{'/'.join(live_test)} open live test, price below{suffix}")

    if near:
        prices = format_open_level_group_prices(events_for("near"))
        suffix = f" {prices}" if prices else ""
        parts.append(f"near {'/'.join(near)} open resistance{suffix}")

    return " | ".join(parts)


def detect_open_levels_context(df_1h=None, df_4h=None, df_1d=None, current_price=None):
    """
    D/W/M/Y open-level resistance context for short-watch setups.

    r17 evaluates open-level tests over recent closed candles and the current
    live candle. D open is checked on 1H. W/M/Y opens are checked on 4H first
    and also on 1H as a fallback, so a clean 1H reaction from monthly/yearly
    open resistance is not hidden while the 4H candle is still forming.

    States:
    - near from below: price/live candle is just below the open;
    - live test, price below: live candle touched/pierced the open from below,
      but the candle is not closed yet;
    - tested + close below: a recent closed candle touched/pierced the open and
      closed back below.

    Open levels are context only. They do not create a short-watch trigger by
    themselves.
    """

    levels = calculate_open_levels(df_1d=df_1d, df_1h=df_1h, df_4h=df_4h)

    if not levels:
        return make_factor(
            key="open_levels",
            label="Open levels",
            status="not_enough_data",
            points=0,
            detail="1D candles unavailable",
        )

    if current_price is None and df_1h is not None and not df_1h.empty:
        live_1h_candle, _ = get_live_candle_for_analysis(df_1h)

        if live_1h_candle is not None:
            current_price = safe_float(live_1h_candle.get("close"), default=np.nan)

    df1 = None if df_1h is None or df_1h.empty else df_1h.copy().reset_index(drop=True)
    df4 = None if df_4h is None or df_4h.empty else df_4h.copy().reset_index(drop=True)

    live_1h_candle, live_1h_index = get_live_candle_for_analysis(df1) if df1 is not None else (None, None)
    live_4h_candle, live_4h_index = get_live_candle_for_analysis(df4) if df4 is not None else (None, None)

    closed_1h_window = get_recent_closed_candles_for_analysis(
        df1,
        OPEN_LEVEL_RECENT_WINDOW_1H_BARS,
    ) if df1 is not None else []

    closed_4h_window = get_recent_closed_candles_for_analysis(
        df4,
        OPEN_LEVEL_RECENT_WINDOW_4H_BARS,
    ) if df4 is not None else []

    events = []

    for label in ["D", "W", "M", "Y"]:
        level = levels.get(label)

        if not level:
            continue

        if not open_level_source_is_allowed_for_signal(label, level):
            continue

        level_price = level.get("open")
        candidate_events = []

        # D open is a lower-timeframe session level, so 1H is enough.
        # W/M/Y opens are higher-timeframe levels; 4H gets priority, but a
        # closed 1H rejection/test is still valid context for a watchlist bot.
        event_1h = evaluate_open_level_context_over_window(
            label=label,
            level_price=level_price,
            current_price=current_price,
            closed_candles=closed_1h_window,
            df_for_previous=df1,
            window_tf="1H",
            window_bars=OPEN_LEVEL_RECENT_WINDOW_1H_BARS,
            live_candle=live_1h_candle,
            live_index=live_1h_index,
        )

        if event_1h is not None:
            candidate_events.append(event_1h)

        if label in ("W", "M", "Y"):
            event_4h = evaluate_open_level_context_over_window(
                label=label,
                level_price=level_price,
                current_price=current_price,
                closed_candles=closed_4h_window,
                df_for_previous=df4,
                window_tf="4H",
                window_bars=OPEN_LEVEL_RECENT_WINDOW_4H_BARS,
                live_candle=live_4h_candle,
                live_index=live_4h_index,
            )

            if event_4h is not None:
                candidate_events.append(event_4h)

        active_candidate_events = [
            event
            for event in candidate_events
            if open_level_event_is_active_resistance(
                event,
                level_price,
                current_price=current_price,
                df_1h=df1,
                df_4h=df4,
            )
        ]

        event = select_best_open_level_event(active_candidate_events)

        if event is not None:
            events.append(event)

    debug_levels = {
        label: float(level.get("open"))
        for label, level in levels.items()
        if isinstance(level, dict) and level.get("open") is not None
    }
    levels_debug_meta = levels.get("_debug", {}) if isinstance(levels, dict) else {}
    debug_levels_source = levels_debug_meta.get("source", {}) if isinstance(levels_debug_meta, dict) else {}
    debug_levels_intraday = levels_debug_meta.get("intraday", {}) if isinstance(levels_debug_meta, dict) else {}
    debug_levels_intraday_1h = levels_debug_meta.get("intraday_1h", {}) if isinstance(levels_debug_meta, dict) else {}
    debug_levels_intraday_4h = levels_debug_meta.get("intraday_4h", {}) if isinstance(levels_debug_meta, dict) else {}
    debug_levels_daily = levels_debug_meta.get("daily", {}) if isinstance(levels_debug_meta, dict) else {}

    window_debug = {
        "D_window_tf": "1H",
        "D_window_bars": int(OPEN_LEVEL_RECENT_WINDOW_1H_BARS),
        "HTF_window_tf": "4H+1H",
        "HTF_window_bars": f"{OPEN_LEVEL_RECENT_WINDOW_4H_BARS}+{OPEN_LEVEL_RECENT_WINDOW_1H_BARS}",
    }

    if not events:
        factor = make_factor(
            key="open_levels",
            label="Open levels",
            status="not_confirmed",
            points=0,
            detail="no open-level resistance nearby",
        )
        factor["open_context_score"] = 0.0
        factor["events"] = []
        factor["debug"] = {
            "status": "not_confirmed",
            "levels": debug_levels,
            "levels_source": debug_levels_source,
            "levels_intraday": debug_levels_intraday,
            "levels_intraday_1h": debug_levels_intraday_1h,
            "levels_intraday_4h": debug_levels_intraday_4h,
            "levels_daily": debug_levels_daily,
            "events": [],
            "score": 0.0,
            "reason": "no_open_resistance_nearby",
            **window_debug,
        }
        return factor

    detail = build_open_levels_detail(events)
    context_score = sum(float(event.get("weight", 0.0)) for event in events)

    factor = make_factor(
        key="open_levels",
        label="Open levels",
        status="confirmed",
        points=0,
        detail=detail,
    )
    factor["open_context_score"] = float(context_score)
    factor["events"] = events
    factor["debug"] = {
        "status": "confirmed",
        "levels": debug_levels,
        "levels_source": debug_levels_source,
        "levels_intraday": debug_levels_intraday,
        "levels_intraday_1h": debug_levels_intraday_1h,
        "levels_intraday_4h": debug_levels_intraday_4h,
        "levels_daily": debug_levels_daily,
        "events": events,
        "score": float(context_score),
        "reason": detail,
        **window_debug,
    }

    return factor



def candle_has_upper_rejection(candle):
    """Return True when a closed candle shows short-side rejection character."""

    if candle is None:
        return False

    open_price = safe_float(candle.get("open"), default=np.nan)
    high_price = safe_float(candle.get("high"), default=np.nan)
    low_price = safe_float(candle.get("low"), default=np.nan)
    close_price = safe_float(candle.get("close"), default=np.nan)

    if any(pd.isna(value) for value in [open_price, high_price, low_price, close_price]):
        return False

    open_price = float(open_price)
    high_price = float(high_price)
    low_price = float(low_price)
    close_price = float(close_price)

    candle_range = high_price - low_price
    if candle_range <= 0:
        return False

    upper_wick = high_price - max(open_price, close_price)
    upper_wick_pct = upper_wick / candle_range
    close_position = (close_price - low_price) / candle_range

    return bool(
        upper_wick_pct >= REJECTION_MIN_UPPER_WICK_RANGE_PCT
        and close_position <= REJECTION_MAX_CLOSE_POSITION_PCT
    )


def candle_near_open_level_from_below(candle, level_price, threshold_pct):
    """Return True when candle high approaches an open level from below without touching it."""

    if candle is None or level_price is None:
        return False

    level_price = safe_float(level_price, default=np.nan)
    candle_high = safe_float(candle.get("high"), default=np.nan)
    candle_close = safe_float(candle.get("close"), default=np.nan)
    candle_open = safe_float(candle.get("open"), default=np.nan)

    if any(pd.isna(value) for value in [level_price, candle_high, candle_close, candle_open]):
        return False

    level_price = float(level_price)
    candle_high = float(candle_high)
    candle_close = float(candle_close)
    candle_open = float(candle_open)

    if level_price <= 0:
        return False

    # The candle must remain below the open level and approach it from below.
    if candle_high >= level_price or candle_close >= level_price or candle_open >= level_price:
        return False

    distance_pct = (level_price - candle_high) / level_price
    return bool(0 <= distance_pct <= float(threshold_pct))


def candle_has_open_level_rejection_character(candle, level_price=None, state="tested"):
    """Return True when a closed candle shows failure at/near an open level.

    r19 keeps this simple for visual validation:
    - a clear upper-wick rejection is valid;
    - a bearish failed test is also valid, even if the upper wick is not large.
    """

    if candle is None:
        return False

    open_price = safe_float(candle.get("open"), default=np.nan)
    high_price = safe_float(candle.get("high"), default=np.nan)
    low_price = safe_float(candle.get("low"), default=np.nan)
    close_price = safe_float(candle.get("close"), default=np.nan)

    if any(pd.isna(value) for value in [open_price, high_price, low_price, close_price]):
        return False

    open_price = float(open_price)
    high_price = float(high_price)
    low_price = float(low_price)
    close_price = float(close_price)

    candle_range = high_price - low_price
    if candle_range <= 0:
        return False

    close_position = (close_price - low_price) / candle_range
    bearish_body = close_price < open_price

    if candle_has_upper_rejection(candle):
        return True

    # For a direct test of an open level, a bearish failed candle is acceptable
    # even without a very large upper wick.
    if str(state) == "tested":
        return bool(bearish_body and close_position <= 0.65)

    # For a near-level reaction, keep it stricter to avoid noise.
    return bool(bearish_body and close_position <= 0.55)


def build_rejection_event(label, state, level_price, candle, candle_index, window_tf, window_bars):
    open_price = safe_float(candle.get("open"), default=np.nan)
    high_price = safe_float(candle.get("high"), default=np.nan)
    low_price = safe_float(candle.get("low"), default=np.nan)
    close_price = safe_float(candle.get("close"), default=np.nan)

    candle_range = float(high_price) - float(low_price) if not any(pd.isna(v) for v in [high_price, low_price]) else np.nan
    upper_wick_pct = np.nan
    close_position = np.nan

    if candle_range and not pd.isna(candle_range) and candle_range > 0:
        upper_wick_pct = (float(high_price) - max(float(open_price), float(close_price))) / candle_range
        close_position = (float(close_price) - float(low_price)) / candle_range

    distance_to_level_pct = np.nan
    try:
        if not pd.isna(high_price) and float(level_price) > 0:
            distance_to_level_pct = ((float(level_price) - float(high_price)) / float(level_price)) * 100
    except Exception:
        distance_to_level_pct = np.nan

    event = {
        "label": str(label),
        "state": str(state),
        "open": float(level_price),
        "confirm_tf": str(window_tf),
        "window_bars": int(window_bars),
        "confirm_index": None if candle_index is None else int(candle_index),
        "confirm_high": high_price,
        "confirm_close": close_price,
        "distance_to_level_pct": distance_to_level_pct,
        "upper_wick_pct": upper_wick_pct,
        "close_position": close_position,
    }

    if "timestamp" in candle:
        event["confirm_start_time"] = str(candle.get("timestamp"))
        close_time = get_candle_close_time_value(candle, window_tf)
        event["confirm_time"] = str(close_time if close_time is not None else candle.get("timestamp"))

    return event


def rejection_event_rank(event):
    """Rank rejection events for one D/W/M/Y level.

    4H rejection has priority over 1H rejection because it is a stronger
    confirmation. Within the same timeframe, a direct test of the open level
    ranks above a near-open reaction.
    """

    if not isinstance(event, dict):
        return (0, 0, -1)

    state_priority = {"tested": 2, "near": 1}
    timeframe_priority = {"4H": 2, "1H": 1}

    state = str(event.get("state", ""))
    timeframe = str(event.get("confirm_tf", ""))
    confirm_index = event.get("confirm_index")

    try:
        confirm_index_value = -1 if confirm_index is None else int(confirm_index)
    except Exception:
        confirm_index_value = -1

    return (
        int(timeframe_priority.get(timeframe, 0)),
        int(state_priority.get(state, 0)),
        int(confirm_index_value),
    )


def select_best_rejection_event(events):
    valid_events = [event for event in events or [] if isinstance(event, dict)]

    if not valid_events:
        return None

    return sorted(valid_events, key=rejection_event_rank, reverse=True)[0]


def build_rejection_detail(events):
    """Build readable Telegram detail without mixing 1H and 4H labels."""

    if not events:
        return "none"

    label_order = {"D": 0, "W": 1, "M": 2, "Y": 3}
    timeframe_order = {"4H": 0, "1H": 1}
    parts = []

    for state in ["tested", "near"]:
        state_events = [event for event in events if event.get("state") == state]

        if not state_events:
            continue

        timeframes = sorted(
            {str(event.get("confirm_tf", "N/A")) for event in state_events},
            key=lambda value: timeframe_order.get(value, 99),
        )

        for timeframe in timeframes:
            tf_events = [event for event in state_events if str(event.get("confirm_tf", "N/A")) == timeframe]
            labels = "/".join(
                sorted(
                    {str(event.get("label")) for event in tf_events},
                    key=lambda value: label_order.get(value, 99),
                )
            )

            if not labels:
                continue

            prices = format_open_level_group_prices(tf_events)
            suffix = f" {prices}" if prices else ""

            if state == "tested":
                parts.append(f"{timeframe} rejection at {labels} open{suffix}")
            else:
                parts.append(f"{timeframe} rejection near {labels} open{suffix}")

    return " | ".join(parts) if parts else "none"


def detect_rejection_candle(df_1h=None, df_4h=None, df_1d=None, current_price=None):
    """
    Rejection Candle v1.

    A rejection candle is a trigger only when it occurs at D/W/M/Y open-level
    resistance. It uses closed candles only:
    - D open: recent closed 1H candles;
    - W/M/Y opens: recent closed 4H candles first, plus recent closed 1H
      candles as a valid watchlist trigger.

    Live candles never confirm rejection. They can only be open-level context.
    """

    levels = calculate_open_levels(df_1d=df_1d, df_1h=df_1h, df_4h=df_4h)

    if not levels:
        factor = make_factor(
            key="rejection_candle",
            label="Rejection candle",
            status="not_enough_data",
            points=0,
            detail="1D candles unavailable",
        )
        factor["events"] = []
        factor["debug"] = {"status": "not_enough_data", "reason": "1D_candles_unavailable"}
        return factor

    df1 = None if df_1h is None or df_1h.empty else df_1h.copy().reset_index(drop=True)
    df4 = None if df_4h is None or df_4h.empty else df_4h.copy().reset_index(drop=True)

    active_current_price = safe_float(current_price, default=np.nan)
    if pd.isna(active_current_price) and df1 is not None:
        live_1h_candle, _ = get_live_candle_for_analysis(df1)
        if live_1h_candle is not None:
            active_current_price = safe_float(live_1h_candle.get("close"), default=np.nan)

    closed_1h_window = get_recent_closed_candles_for_analysis(
        df1,
        REJECTION_RECENT_WINDOW_1H_BARS,
    ) if df1 is not None else []

    closed_4h_window = get_recent_closed_candles_for_analysis(
        df4,
        REJECTION_RECENT_WINDOW_4H_BARS,
    ) if df4 is not None else []

    def collect_rejection_events_for_window(label, level_price, closed_window, previous_df, window_tf, window_bars):
        threshold = OPEN_LEVEL_NEAR_THRESHOLDS.get(label, 0.0035)
        label_events = []

        if open_level_reclaimed_by_current_price(active_current_price, level_price):
            return label_events

        for candle, candle_index in closed_window or []:
            previous_candle = get_previous_candle(previous_df, candle_index)

            if open_level_tested_from_below(candle, previous_candle, level_price):
                if not candle_has_open_level_rejection_character(candle, level_price=level_price, state="tested"):
                    continue

                label_events.append(build_rejection_event(
                    label=label,
                    state="tested",
                    level_price=level_price,
                    candle=candle,
                    candle_index=candle_index,
                    window_tf=window_tf,
                    window_bars=window_bars,
                ))
            elif candle_near_open_level_from_below(candle, level_price, threshold):
                if not candle_has_open_level_rejection_character(candle, level_price=level_price, state="near"):
                    continue

                label_events.append(build_rejection_event(
                    label=label,
                    state="near",
                    level_price=level_price,
                    candle=candle,
                    candle_index=candle_index,
                    window_tf=window_tf,
                    window_bars=window_bars,
                ))

        return label_events

    events = []

    for label in ["D", "W", "M", "Y"]:
        level = levels.get(label)
        if not level:
            continue

        if not open_level_source_is_allowed_for_signal(label, level):
            continue

        level_price = level.get("open")
        candidate_events = []

        candidate_events.extend(collect_rejection_events_for_window(
            label=label,
            level_price=level_price,
            closed_window=closed_1h_window,
            previous_df=df1,
            window_tf="1H",
            window_bars=REJECTION_RECENT_WINDOW_1H_BARS,
        ))

        # Closed 4H rejection is valid for any D/W/M/Y open and has priority
        # over 1H when both are present.
        candidate_events.extend(collect_rejection_events_for_window(
            label=label,
            level_price=level_price,
            closed_window=closed_4h_window,
            previous_df=df4,
            window_tf="4H",
            window_bars=REJECTION_RECENT_WINDOW_4H_BARS,
        ))

        active_candidate_events = [
            event
            for event in candidate_events
            if open_level_event_is_active_resistance(
                event,
                level_price,
                current_price=active_current_price,
                df_1h=df1,
                df_4h=df4,
            )
        ]

        best_event = select_best_rejection_event(active_candidate_events)

        if best_event is not None:
            events.append(best_event)

    debug_levels = {
        label: float(level.get("open"))
        for label, level in levels.items()
        if isinstance(level, dict) and level.get("open") is not None
    }
    levels_debug_meta = levels.get("_debug", {}) if isinstance(levels, dict) else {}
    debug_levels_source = levels_debug_meta.get("source", {}) if isinstance(levels_debug_meta, dict) else {}
    debug_levels_intraday = levels_debug_meta.get("intraday", {}) if isinstance(levels_debug_meta, dict) else {}
    debug_levels_intraday_1h = levels_debug_meta.get("intraday_1h", {}) if isinstance(levels_debug_meta, dict) else {}
    debug_levels_intraday_4h = levels_debug_meta.get("intraday_4h", {}) if isinstance(levels_debug_meta, dict) else {}
    debug_levels_daily = levels_debug_meta.get("daily", {}) if isinstance(levels_debug_meta, dict) else {}

    if not events:
        factor = make_factor(
            key="rejection_candle",
            label="Rejection candle",
            status="not_confirmed",
            points=0,
            detail="none",
        )
        factor["events"] = []
        factor["debug"] = {
            "status": "not_confirmed",
            "levels": debug_levels,
            "levels_source": debug_levels_source,
            "levels_intraday": debug_levels_intraday,
            "levels_intraday_1h": debug_levels_intraday_1h,
            "levels_intraday_4h": debug_levels_intraday_4h,
            "levels_daily": debug_levels_daily,
            "events": [],
            "reason": "no_open_level_rejection",
        }
        return factor

    order = {"D": 0, "W": 1, "M": 2, "Y": 3}
    events = sorted(events, key=lambda item: order.get(str(item.get("label")), 99))
    detail = build_rejection_detail(events)

    factor = make_factor(
        key="rejection_candle",
        label="Rejection candle",
        status="confirmed",
        points=2,
        detail=detail,
    )
    factor["events"] = events
    factor["debug"] = {
        "status": "confirmed",
        "levels": debug_levels,
        "levels_source": debug_levels_source,
        "levels_intraday": debug_levels_intraday,
        "levels_intraday_1h": debug_levels_intraday_1h,
        "levels_intraday_4h": debug_levels_intraday_4h,
        "levels_daily": debug_levels_daily,
        "events": events,
        "reason": detail,
    }

    return factor



def analyze_short_factors(df_1h, df_4h=None, df_1d=None, current_price=None):
    """
    Current simplified short-watch factor set.

    Active factors:
    - Liquidity sweep: confirmed price-action trigger.
    - Open levels: D/W/M/Y resistance location/context.
    - Rejection candle: confirmed trigger only at D/W/M/Y open-level resistance.

    Removed from active model:
    - Premium/Discount: previous implementation was range-position, not a full ICT dealing range.
    - New High: too noisy for Telegram and classification.
    """

    open_levels_factor = detect_open_levels_context(
        df_1h=df_1h,
        df_4h=df_4h,
        df_1d=df_1d,
        current_price=current_price,
    )

    factors = [
        detect_liquidity_sweep(df_1h, df_4h=df_4h),
        open_levels_factor,
        detect_rejection_candle(df_1h=df_1h, df_4h=df_4h, df_1d=df_1d, current_price=current_price),
    ]

    score = sum(int(factor.get("points", 0)) for factor in factors)
    confirmed_count = sum(1 for factor in factors if factor.get("status") == "confirmed")
    total_count = len(factors)

    return {
        "factors": factors,
        "score": score,
        "confirmed_count": confirmed_count,
        "total_count": total_count,
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

    Location/context:
    - Open Levels D/W/M/Y resistance context.

    Trigger / confirmation:
    - Confirmed liquidity sweep.
    - Confirmed rejection candle at D/W/M/Y open-level resistance.
    """

    open_levels = get_short_factor(short_factors, "open_levels") or {}
    liquidity = get_short_factor(short_factors, "liquidity_sweep") or {}
    rejection = get_short_factor(short_factors, "rejection_candle") or {}

    liquidity_confirmed = is_factor_confirmed(short_factors, "liquidity_sweep")
    rejection_confirmed = is_factor_confirmed(short_factors, "rejection_candle")
    liquidity_candidate = liquidity.get("status") == "candidate"

    open_context_score = float(safe_float(open_levels.get("open_context_score", 0.0), default=0.0))

    open_events = open_levels.get("events", []) if isinstance(open_levels, dict) else []
    if not isinstance(open_events, list):
        open_events = []

    htf_open_context_score = sum(
        float(safe_float(event.get("weight", 0.0), default=0.0))
        for event in open_events
        if isinstance(event, dict) and event.get("label") in ("W", "M", "Y")
    )
    d_open_context_score = sum(
        float(safe_float(event.get("weight", 0.0), default=0.0))
        for event in open_events
        if isinstance(event, dict) and event.get("label") == "D"
    )

    tested_open_labels = [
        str(event.get("label"))
        for event in open_events
        if isinstance(event, dict) and event.get("state") == "tested"
    ]
    live_test_open_labels = [
        str(event.get("label"))
        for event in open_events
        if isinstance(event, dict) and event.get("state") == "live_test"
    ]
    near_open_labels = [
        str(event.get("label"))
        for event in open_events
        if isinstance(event, dict) and event.get("state") == "near"
    ]

    location_score = open_context_score
    trigger_count = int(liquidity_confirmed) + int(rejection_confirmed)
    early_trigger_count = int(liquidity_candidate)

    trigger_parts = []

    if liquidity_confirmed:
        trigger_parts.append("liquidity sweep")

    if rejection_confirmed:
        trigger_parts.append("open-level rejection")

    return {
        "location_score": float(location_score),
        "base_location_score": 0.0,
        "open_context_score": float(open_context_score),
        "htf_open_context_score": float(htf_open_context_score),
        "d_open_context_score": float(d_open_context_score),
        "tested_open_labels": tested_open_labels,
        "live_test_open_labels": live_test_open_labels,
        "near_open_labels": near_open_labels,
        "has_open_resistance": bool(open_context_score > 0),
        "has_htf_open_resistance": bool(htf_open_context_score > 0),
        "has_strong_open_resistance": bool(htf_open_context_score >= 2.0 or open_context_score >= 2.0),
        "trigger_count": trigger_count,
        "early_trigger_count": early_trigger_count,
        "liquidity_confirmed": liquidity_confirmed,
        "rejection_confirmed": rejection_confirmed,
        "liquidity_candidate": liquidity_candidate,
        "trigger_parts": trigger_parts,
    }


def quality_location_label(score):
    score = float(safe_float(score, default=0.0))

    if score >= 2:
        return "Strong open resistance"
    if score >= 1:
        return "Open resistance"
    if score > 0:
        return "Minor open resistance"
    return "None"


def quality_trigger_label_from_context(context):
    parts = context.get("trigger_parts", [])
    if parts:
        return " + ".join(str(part) for part in parts)
    return "None"


def build_setup_status(signal_level, scores, short_factors):
    context = calculate_location_trigger_context(short_factors)
    trigger_count = int(context.get("trigger_count", 0))
    has_overheat_context = bool(scores.get("has_overheat_context", False))
    has_open_resistance = bool(context.get("has_open_resistance", False))
    has_htf_open_resistance = bool(context.get("has_htf_open_resistance", False))
    liquidity_confirmed = bool(context.get("liquidity_confirmed", False))
    rejection_confirmed = bool(context.get("rejection_confirmed", False))

    if signal_level == "HIGH_PRIORITY_SHORT_WATCH":
        if liquidity_confirmed and rejection_confirmed:
            return "High priority watch — liquidity sweep + open-level rejection"
        if liquidity_confirmed and has_htf_open_resistance:
            return "High priority watch — liquidity sweep + HTF open resistance"
        if rejection_confirmed:
            return "High priority watch — open-level rejection"
        return "High priority watch — strong heat and confirmed short trigger"

    if signal_level == "SHORT_WATCH":
        if liquidity_confirmed and rejection_confirmed:
            return "Short watch — liquidity sweep + open-level rejection"
        if liquidity_confirmed:
            if has_htf_open_resistance:
                return "Short watch — liquidity sweep + HTF open resistance"
            if has_open_resistance:
                return "Short watch — liquidity sweep + open resistance"
            return "Short watch — confirmed liquidity sweep detected"
        if rejection_confirmed:
            return "Short watch — open-level rejection"
        return "Short watch — waiting for confirmed short trigger"

    if signal_level == "OVERHEAT_WATCH":
        if has_open_resistance:
            return "Overheat watch — RSI heat near open resistance; waiting for short trigger"
        return "Overheat watch — RSI heat confirmed; waiting for short trigger"

    return "No watch setup"


def build_watch_reason(signal_level, scores, short_factors):
    pump_quality = quality_pump_label(scores.get("pump_score", 0))
    heat_quality = quality_heat_label(scores.get("rsi_score", 0))
    context = calculate_location_trigger_context(short_factors)
    trigger_quality = quality_trigger_label_from_context(context)
    has_htf_open_resistance = bool(context.get("has_htf_open_resistance", False))
    has_open_resistance = bool(context.get("has_open_resistance", False))
    liquidity_confirmed = bool(context.get("liquidity_confirmed", False))
    rejection_confirmed = bool(context.get("rejection_confirmed", False))

    if signal_level == "HIGH_PRIORITY_SHORT_WATCH":
        parts = [f"{pump_quality} pump", f"{heat_quality} RSI heat"]
        if liquidity_confirmed:
            parts.append("liquidity sweep")
        if rejection_confirmed:
            parts.append("open-level rejection")
        elif has_htf_open_resistance:
            parts.append("HTF open resistance")
        elif has_open_resistance:
            parts.append("open resistance")
        return " + ".join(parts)

    if signal_level == "SHORT_WATCH":
        if rejection_confirmed and liquidity_confirmed:
            return "liquidity sweep + open-level rejection"
        if rejection_confirmed:
            return "open-level rejection"
        if liquidity_confirmed and has_htf_open_resistance:
            return "confirmed liquidity sweep + HTF open resistance"
        if liquidity_confirmed and has_open_resistance:
            return "confirmed liquidity sweep + open resistance"
        if liquidity_confirmed:
            return "confirmed liquidity sweep"
        return trigger_quality

    if signal_level == "OVERHEAT_WATCH":
        base = str(scores.get("overheat_reason") or f"{pump_quality} pump + {heat_quality} RSI heat")
        if has_open_resistance:
            return base + " + open resistance"
        return base

    return "Watch filters not passed"


def classify_watch_signal(scores, short_factors=None):
    """
    Simplified Short Watch model.

    Input filter is handled separately by RSI_ENTRY_FILTER.
    Location/context: Open Levels only.
    Triggers: confirmed Liquidity Sweep or confirmed Rejection at Open Levels.
    Telegram output levels: OVERHEAT, SHORT WATCH, HIGH PRIORITY SHORT WATCH.
    """

    short_factors = short_factors or []

    pump_score = int(scores.get("pump_score", 0))
    rsi_score = int(scores.get("rsi_score", 0))

    context = calculate_location_trigger_context(short_factors)
    location_score = float(context.get("location_score", 0.0))
    trigger_count = int(context.get("trigger_count", 0))
    open_context_score = float(context.get("open_context_score", 0.0))
    htf_open_context_score = float(context.get("htf_open_context_score", 0.0))
    has_open_resistance = bool(context.get("has_open_resistance", False))
    has_htf_open_resistance = bool(context.get("has_htf_open_resistance", False))
    liquidity_confirmed = bool(context.get("liquidity_confirmed", False))
    rejection_confirmed = bool(context.get("rejection_confirmed", False))

    has_basic_pump = bool(scores.get("has_basic_pump_context", False)) or pump_score >= 1
    has_strong_pump = bool(scores.get("has_strong_pump_context", False)) or pump_score >= 2
    has_extreme_pump = bool(scores.get("has_extreme_pump_context", False)) or pump_score >= 3
    has_overheat_context = bool(scores.get("has_overheat_context", False))

    has_strong_heat = rsi_score >= 3 or has_overheat_context
    has_trigger = trigger_count >= 1
    pump_or_overheat = has_basic_pump or has_overheat_context

    high_priority_context = (
        has_htf_open_resistance
        or rejection_confirmed
        or (has_open_resistance and liquidity_confirmed)
    )

    if (
        has_trigger
        and (has_extreme_pump or has_strong_pump)
        and has_strong_heat
        and high_priority_context
    ):
        signal_level = "HIGH_PRIORITY_SHORT_WATCH"
    elif pump_or_overheat and has_trigger:
        signal_level = "SHORT_WATCH"
    elif has_overheat_context:
        signal_level = "OVERHEAT_WATCH"
    else:
        signal_level = "NO_SIGNAL"

    return {
        "signal_level": signal_level,
        "reason": build_watch_reason(signal_level, scores, short_factors),
        "location_score": float(location_score),
        "open_context_score": float(open_context_score),
        "htf_open_context_score": float(htf_open_context_score),
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


def evaluate_rsi_entry_filter(rsi_1h_live, rsi_1h_closed, rsi_4h_live):
    """
    Decide whether a signal is hot enough to enter the Telegram short-list.

    This is a visibility filter for the shortlist. It does not change the
    underlying signal classification, but weak-RSI SHORT WATCH candidates are
    hidden from Telegram and grouped output.
    """

    live_1h = safe_float(rsi_1h_live, default=0.0)
    closed_1h = safe_float(rsi_1h_closed, default=0.0)
    live_4h = safe_float(rsi_4h_live, default=0.0)

    conditions = []

    if live_1h >= RSI_ENTRY_MIN_1H_LIVE:
        conditions.append(f"1H_live>={RSI_ENTRY_MIN_1H_LIVE}")

    if closed_1h >= RSI_ENTRY_MIN_1H_CLOSED:
        conditions.append(f"1H_closed>={RSI_ENTRY_MIN_1H_CLOSED}")

    if live_4h >= RSI_ENTRY_MIN_4H_LIVE:
        conditions.append(f"4H_live>={RSI_ENTRY_MIN_4H_LIVE}")

    passed = bool(conditions)

    return {
        "passed": passed,
        "reason": "+".join(conditions) if conditions else "RSI_below_entry_threshold",
        "rsi_1h_live": live_1h,
        "rsi_1h_closed": closed_1h,
        "rsi_4h_live": live_4h,
    }


def row_passes_rsi_entry_filter(row):
    result = evaluate_rsi_entry_filter(
        rsi_1h_live=row.get("rsi_1h_live"),
        rsi_1h_closed=row.get("rsi_1h_closed"),
        rsi_4h_live=row.get("rsi_4h_live"),
    )
    return bool(result.get("passed"))


def add_rsi_entry_filter_columns(df):
    if df is None or df.empty:
        return df

    work_df = df.copy()
    results = work_df.apply(
        lambda row: evaluate_rsi_entry_filter(
            rsi_1h_live=row.get("rsi_1h_live"),
            rsi_1h_closed=row.get("rsi_1h_closed"),
            rsi_4h_live=row.get("rsi_4h_live"),
        ),
        axis=1,
    )

    work_df["rsi_entry_passed"] = results.apply(lambda item: bool(item.get("passed")))
    work_df["rsi_entry_reason"] = results.apply(lambda item: str(item.get("reason", "N/A")))

    return work_df


def log_rsi_entry_debug(df):
    """Print RSI_ENTRY_DEBUG lines for active candidates before shortlist filtering."""

    if df is None or df.empty:
        return

    if "signal_level" not in df.columns:
        return

    visible_levels = {
        "HIGH_PRIORITY_SHORT_WATCH",
        "SHORT_WATCH",
        "OVERHEAT_WATCH",
    }

    for _, row in df.iterrows():
        signal_level = str(row.get("signal_level", "NO_SIGNAL"))
        if signal_level not in visible_levels:
            continue

        result = evaluate_rsi_entry_filter(
            rsi_1h_live=row.get("rsi_1h_live"),
            rsi_1h_closed=row.get("rsi_1h_closed"),
            rsi_4h_live=row.get("rsi_4h_live"),
        )

        fields = [
            "RSI_ENTRY_DEBUG",
            f"symbol={format_debug_value(row.get('symbol'))}",
            f"exchange={format_debug_value(row.get('exchange'))}",
            f"signal={format_debug_value(signal_level)}",
            f"passed={format_debug_value(result.get('passed'))}",
            f"reason={format_debug_value(result.get('reason'))}",
            f"rsi_1h_live={format_debug_value(result.get('rsi_1h_live'))}",
            f"rsi_1h_closed={format_debug_value(result.get('rsi_1h_closed'))}",
            f"rsi_4h_live={format_debug_value(result.get('rsi_4h_live'))}",
            f"thresholds=1H_live>={RSI_ENTRY_MIN_1H_LIVE}|1H_closed>={RSI_ENTRY_MIN_1H_CLOSED}|4H_live>={RSI_ENTRY_MIN_4H_LIVE}",
        ]

        print(" ".join(fields))


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
# OPEN INTEREST CONTEXT
# ============================================================

def percent_change_from_values(current_value, previous_value):
    current = safe_float(current_value, default=np.nan)
    previous = safe_float(previous_value, default=np.nan)

    if pd.isna(current) or pd.isna(previous) or float(previous) == 0:
        return None

    return float(((float(current) - float(previous)) / float(previous)) * 100)


def extract_timestamp_from_oi_row(row):
    if isinstance(row, dict):
        for key in ["ts", "timestamp", "time", "date"]:
            if key in row and row.get(key) is not None:
                try:
                    value = str(row.get(key)).strip()
                    if value.isdigit():
                        return pd.to_datetime(int(value), unit="ms")
                    return pd.to_datetime(value)
                except Exception:
                    pass
        return None

    if isinstance(row, (list, tuple)) and len(row) >= 1:
        try:
            value = str(row[0]).strip()
            if value.isdigit():
                return pd.to_datetime(int(value), unit="ms")
            return pd.to_datetime(value)
        except Exception:
            return None

    return None


def extract_oi_value_from_row(row):
    """Extract a comparable Open Interest value from flexible API responses."""

    preferred_keys = [
        "openInterest",
        "openInterestUsd",
        "openInterestValue",
        "oiUsd",
        "oiCcy",
        "oi",
        "amount",
        "value",
        "size",
    ]

    if isinstance(row, dict):
        # Bitget current OI response nests the actual value inside
        # data.openInterestList[0].size. Parse nested lists first instead of
        # trying to cast the whole list to float.
        nested_list = row.get("openInterestList")
        if isinstance(nested_list, list):
            for nested_item in nested_list:
                nested_value = extract_oi_value_from_row(nested_item)
                if nested_value is not None:
                    return nested_value

        # Some wrappers return a generic data list with OI rows.
        nested_data = row.get("data")
        if isinstance(nested_data, list):
            for nested_item in nested_data:
                nested_value = extract_oi_value_from_row(nested_item)
                if nested_value is not None:
                    return nested_value

        for key in preferred_keys:
            if key in row:
                value = safe_float(row.get(key), default=np.nan)
                if not pd.isna(value) and float(value) > 0:
                    return float(value)

        for key, raw_value in row.items():
            if str(key).lower() in {"ts", "timestamp", "time", "date"}:
                continue
            if isinstance(raw_value, (dict, list, tuple)):
                nested_value = extract_oi_value_from_row(raw_value)
                if nested_value is not None:
                    return nested_value
                continue
            value = safe_float(raw_value, default=np.nan)
            if not pd.isna(value) and float(value) > 0 and float(value) < 10**18:
                return float(value)

        return None

    if isinstance(row, (list, tuple)):
        for raw_value in list(row)[1:]:
            if isinstance(raw_value, (dict, list, tuple)):
                nested_value = extract_oi_value_from_row(raw_value)
                if nested_value is not None:
                    return nested_value
                continue
            value = safe_float(raw_value, default=np.nan)
            if not pd.isna(value) and float(value) > 0 and float(value) < 10**18:
                return float(value)

    return None


def normalize_oi_history_rows(rows):
    normalized = []

    for row in rows or []:
        timestamp = extract_timestamp_from_oi_row(row)
        value = extract_oi_value_from_row(row)

        if timestamp is None or value is None:
            continue

        normalized.append({
            "timestamp": pd.to_datetime(timestamp),
            "open_interest": float(value),
        })

    if not normalized:
        return pd.DataFrame(columns=["timestamp", "open_interest"])

    df = pd.DataFrame(normalized)
    df = df.dropna(subset=["timestamp", "open_interest"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def get_history_value_at_or_before(history, target_timestamp):
    if history is None or history.empty or target_timestamp is None:
        return np.nan

    try:
        timestamps = pd.to_datetime(history["timestamp"], errors="coerce")
        target = pd.to_datetime(target_timestamp)
        eligible = history[timestamps <= target]

        if eligible.empty:
            return np.nan

        return safe_float(eligible.iloc[-1].get("open_interest"), default=np.nan)
    except Exception:
        return np.nan


def build_open_interest_metrics(history_df=None, current_oi=None, provider="provider"):
    """
    Build Open Interest metrics.

    Important: current OI and historical Rubik OI can use different units/scales
    depending on endpoint. For percentage changes, use the historical series only
    when it is available. Current OI is kept as informational fallback/display.
    """

    history = history_df.copy() if isinstance(history_df, pd.DataFrame) else pd.DataFrame()

    if not history.empty:
        history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")
        history["open_interest"] = pd.to_numeric(history["open_interest"], errors="coerce")
        history = history.dropna(subset=["timestamp", "open_interest"])
        history = history[history["open_interest"] > 0]
        history = history.sort_values("timestamp").reset_index(drop=True)

    current_value = safe_float(current_oi, default=np.nan)
    has_current_value = not pd.isna(current_value) and float(current_value) > 0

    history_points = int(len(history)) if isinstance(history, pd.DataFrame) else 0
    history_latest_ts = None

    if not history.empty:
        latest_ts = pd.to_datetime(history.iloc[-1]["timestamp"])
        history_latest_ts = latest_ts
        history_current = float(history.iloc[-1]["open_interest"])
        previous_1h = get_history_value_at_or_before(history, latest_ts - pd.Timedelta(hours=1))
        previous_4h = get_history_value_at_or_before(history, latest_ts - pd.Timedelta(hours=4))

        change_1h = percent_change_from_values(history_current, previous_1h)
        change_4h = percent_change_from_values(history_current, previous_4h)

        return {
            "oi_status": "ok",
            "oi_current": float(current_value) if has_current_value else history_current,
            "oi_history_current": history_current,
            "oi_change_1h_percent": change_1h,
            "oi_change_4h_percent": change_4h,
            "oi_context": interpret_open_interest_context(change_1h, change_4h),
            "oi_source": str(provider),
            "oi_history_points": history_points,
            "oi_history_latest_ts": history_latest_ts,
        }

    if has_current_value:
        return {
            "oi_status": "current_only",
            "oi_current": float(current_value),
            "oi_history_current": None,
            "oi_change_1h_percent": None,
            "oi_change_4h_percent": None,
            "oi_context": "N/A",
            "oi_source": str(provider),
            "oi_history_points": 0,
            "oi_history_latest_ts": None,
        }

    return {
        "oi_status": "not_available",
        "oi_current": None,
        "oi_history_current": None,
        "oi_change_1h_percent": None,
        "oi_change_4h_percent": None,
        "oi_context": "N/A",
        "oi_source": str(provider),
        "oi_history_points": 0,
        "oi_history_latest_ts": None,
    }


def interpret_open_interest_context(oi_change_1h, oi_change_4h):
    change_1h = safe_float(oi_change_1h, default=np.nan)
    change_4h = safe_float(oi_change_4h, default=np.nan)

    has_1h = not pd.isna(change_1h)
    has_4h = not pd.isna(change_4h)

    if not has_1h and not has_4h:
        return "N/A"

    strongest_positive = max(
        float(change_1h) if has_1h else -999999.0,
        float(change_4h) if has_4h else -999999.0,
    )
    strongest_negative = min(
        float(change_1h) if has_1h else 999999.0,
        float(change_4h) if has_4h else 999999.0,
    )

    if has_4h and float(change_4h) >= OI_CHANGE_STRONG_4H_THRESHOLD_PCT:
        return "Long build-up"

    if strongest_positive >= OI_CHANGE_1H_ACTIVE_THRESHOLD_PCT:
        return "Long build-up"

    if strongest_negative <= OI_CHANGE_NEGATIVE_THRESHOLD_PCT:
        return "Short squeeze / OI unwind"

    flat_1h = (not has_1h) or abs(float(change_1h)) <= OI_CHANGE_FLAT_THRESHOLD_PCT
    flat_4h = (not has_4h) or abs(float(change_4h)) <= OI_CHANGE_FLAT_THRESHOLD_PCT

    if flat_1h and flat_4h:
        return "weak OI confirmation"

    return "mixed OI"


def format_oi_percent_for_telegram(value):
    value = safe_float(value, default=np.nan)

    if pd.isna(value):
        return "N/A"

    return f"{float(value):+.1f}%"


def format_oi_line_for_telegram(detail):
    if not isinstance(detail, dict):
        return ""

    change_1h = detail.get("oi_change_1h_percent")
    change_4h = detail.get("oi_change_4h_percent")
    current_oi = detail.get("oi_current")

    has_1h = change_1h is not None and not pd.isna(safe_float(change_1h, default=np.nan))
    has_4h = change_4h is not None and not pd.isna(safe_float(change_4h, default=np.nan))
    has_current = current_oi is not None and not pd.isna(safe_float(current_oi, default=np.nan))

    if not has_1h and not has_4h:
        if has_current:
            return f"OI: current {format_large_number(current_oi)} | history N/A"
        return ""

    context = str(detail.get("oi_context", "N/A") or "N/A")
    line = f"OI: 1H {format_oi_percent_for_telegram(change_1h)} | 4H {format_oi_percent_for_telegram(change_4h)}"

    if context and context != "N/A":
        line += f" | {context}"

    return line

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



def okx_get_open_interest(inst_id):
    data = safe_get_json(
        base_url=OKX_BASE_URL,
        endpoint="/api/v5/public/open-interest",
        params={
            "instType": OKX_INST_TYPE,
            "instId": inst_id,
        },
        provider_name="OKX OI",
    )

    if data.get("code") != "0":
        raise Exception(f"OKX OI API error: {data.get('msg')}")

    rows = data.get("data", [])
    if not rows:
        return None

    return extract_oi_value_from_row(rows[0])


def okx_base_ccy_from_inst_id(inst_id):
    try:
        return str(inst_id).split("-")[0].upper()
    except Exception:
        return ""


def okx_get_open_interest_history(inst_id, period=OI_HISTORY_PERIOD):
    base_ccy = okx_base_ccy_from_inst_id(inst_id)

    if not base_ccy:
        return pd.DataFrame(columns=["timestamp", "open_interest"])

    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    begin_ms = now_ms - int(OI_HISTORY_LOOKBACK_HOURS * 60 * 60 * 1000)

    # OKX historical contract OI is exposed through the Trading Data
    # "open-interest-volume" endpoint. r28 used a history endpoint that returned
    # current OI but not enough usable rows for 1H/4H changes in production.
    data = safe_get_json(
        base_url=OKX_BASE_URL,
        endpoint="/api/v5/rubik/stat/contracts/open-interest-volume",
        params={
            "ccy": base_ccy,
            "period": period,
            "begin": str(begin_ms),
            "end": str(now_ms),
        },
        provider_name="OKX OI history",
    )

    if data.get("code") != "0":
        raise Exception(f"OKX OI history API error: {data.get('msg')}")

    return normalize_oi_history_rows(data.get("data", []))


def okx_get_open_interest_metrics(inst_id):
    try:
        current_oi = okx_get_open_interest(inst_id)
    except Exception as e:
        print(f"OKX OI current unavailable for {inst_id}: {e}")
        current_oi = None

    try:
        history_df = okx_get_open_interest_history(inst_id)
    except Exception as e:
        print(f"OKX OI history unavailable for {inst_id}: {e}")
        history_df = pd.DataFrame(columns=["timestamp", "open_interest"])

    return build_open_interest_metrics(
        history_df=history_df,
        current_oi=current_oi,
        provider="OKX",
    )


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

    raw_1d = okx_get_candles(inst_id=inst_id, bar="1D", limit=CANDLE_LIMIT_1D)
    df_1d = okx_candles_to_dataframe(raw_1d)

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

    oi_metrics = okx_get_open_interest_metrics(inst_id)

    short_analysis = analyze_short_factors(df_1h, df_4h, df_1d=df_1d, current_price=price)

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
        "oi_current": oi_metrics.get("oi_current"),
        "oi_change_1h_percent": oi_metrics.get("oi_change_1h_percent"),
        "oi_change_4h_percent": oi_metrics.get("oi_change_4h_percent"),
        "oi_context": oi_metrics.get("oi_context", "N/A"),
        "oi_status": oi_metrics.get("oi_status", "not_available"),
        "oi_source": oi_metrics.get("oi_source", "OKX"),
        "oi_history_points": oi_metrics.get("oi_history_points", 0),
        "oi_history_latest_ts": oi_metrics.get("oi_history_latest_ts"),
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



def bitget_get_open_interest(symbol):
    data = safe_get_json(
        base_url=BITGET_BASE_URL,
        endpoint="/api/v2/mix/market/open-interest",
        params={
            "symbol": symbol,
            "productType": BITGET_PRODUCT_TYPE,
        },
        provider_name="Bitget OI",
    )

    if not bitget_success(data):
        raise Exception(f"Bitget OI API error: {data.get('msg')}")

    payload = data.get("data")

    if isinstance(payload, list) and payload:
        return extract_oi_value_from_row(payload[0])

    if isinstance(payload, dict):
        return extract_oi_value_from_row(payload)

    return None


def bitget_get_open_interest_metrics(symbol):
    # Bitget public Futures API exposes current OI. The official v2 Futures
    # market docs do not provide a per-symbol historical OI endpoint equivalent
    # to OKX Rubik history, so 1H/4H changes remain N/A until we add persistent
    # snapshots or a separate approved historical data source.
    try:
        current_oi = bitget_get_open_interest(symbol)
    except Exception as e:
        print(f"Bitget OI current unavailable for {symbol}: {e}")
        current_oi = None

    return build_open_interest_metrics(
        history_df=pd.DataFrame(columns=["timestamp", "open_interest"]),
        current_oi=current_oi,
        provider="Bitget_current",
    )


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

    raw_1d = bitget_get_candles(symbol=symbol, granularity="1D", limit=CANDLE_LIMIT_1D)
    df_1d = bitget_candles_to_dataframe(raw_1d)

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

    oi_metrics = bitget_get_open_interest_metrics(symbol)

    short_analysis = analyze_short_factors(df_1h, df_4h, df_1d=df_1d, current_price=price)

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
        "oi_current": oi_metrics.get("oi_current"),
        "oi_change_1h_percent": oi_metrics.get("oi_change_1h_percent"),
        "oi_change_4h_percent": oi_metrics.get("oi_change_4h_percent"),
        "oi_context": oi_metrics.get("oi_context", "N/A"),
        "oi_status": oi_metrics.get("oi_status", "not_available"),
        "oi_source": oi_metrics.get("oi_source", "Bitget"),
        "oi_history_points": oi_metrics.get("oi_history_points", 0),
        "oi_history_latest_ts": oi_metrics.get("oi_history_latest_ts"),
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
    if "oi_change_1h_percent" in df_out.columns:
        df_out["oi_1h_%"] = df_out["oi_change_1h_percent"].apply(format_oi_percent_for_telegram)
    else:
        df_out["oi_1h_%"] = "N/A"
    if "oi_change_4h_percent" in df_out.columns:
        df_out["oi_4h_%"] = df_out["oi_change_4h_percent"].apply(format_oi_percent_for_telegram)
    else:
        df_out["oi_4h_%"] = "N/A"

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
        "oi_1h_%",
        "oi_4h_%",
        "oi_context",
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
                "oi_current": row.get("oi_current"),
                "oi_change_1h_percent": row.get("oi_change_1h_percent"),
                "oi_change_4h_percent": row.get("oi_change_4h_percent"),
                "oi_context": str(row.get("oi_context", "N/A")),
                "oi_status": str(row.get("oi_status", "not_available")),
                "oi_source": str(row.get("oi_source", row.get("exchange", "N/A"))),
                "oi_history_points": row.get("oi_history_points", 0),
                "oi_history_latest_ts": row.get("oi_history_latest_ts"),
                "pump_score": int(row.get("pump_score", 0)),
                "rsi_score": int(row.get("rsi_score", 0)),
                "volume_score": int(row.get("volume_score", 0)),
                "short_setup_score": int(row.get("short_setup_score", 0)),
                "final_score": int(row.get("final_score", 0)),
                "location_score": float(safe_float(row.get("location_score", 0), default=0.0)),
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




def factor_status_icon(factor):
    status = (factor or {}).get("status")

    if status == "confirmed":
        return "✅"
    if status == "not_enough_data":
        return "⚪"
    return "❌"


def compact_factor_summary(factors):
    """
    Legacy compact summary helper kept for internal/debug use.
    Telegram output now uses user-facing factor lines instead.
    """

    factor_keys = [
        ("liquidity_sweep", "Sweep"),
        ("premium_zone", "Prem"),
        ("local_high_update", "LocalHigh"),
    ]

    parts = []

    for key, label in factor_keys:
        factor = find_factor(factors, key) or {}
        parts.append(f"{factor_status_icon(factor)} {label}")

    return " | ".join(parts)


def format_sweep_detail_for_telegram(factors):
    factor = find_factor(factors, "liquidity_sweep") or {}
    status = factor.get("status")
    detail = str(factor.get("detail", "") or "").strip()

    if status == "confirmed" and detail:
        return detail

    if status == "not_enough_data":
        return "Data unavailable"

    return "No confirmed sweep"


def format_open_levels_detail_for_telegram(factors):
    factor = find_factor(factors, "open_levels") or {}
    status = factor.get("status")
    detail = str(factor.get("detail", "") or "").strip()

    if status == "confirmed" and detail:
        return detail

    return ""


def format_rejection_detail_for_telegram(factors):
    factor = find_factor(factors, "rejection_candle") or {}
    status = factor.get("status")
    detail = str(factor.get("detail", "") or "").strip()

    if status == "confirmed" and detail:
        return detail

    if status == "not_enough_data":
        return "Data unavailable"

    return "none"


def format_local_high_detail_for_telegram(factors):
    factor = find_factor(factors, "local_high_update") or {}
    status = factor.get("status")
    detail = str(factor.get("detail", "") or "").strip()

    if status == "confirmed":
        cleaned = detail.replace("1H: ", "").strip()
        cleaned = cleaned.replace("new ", "")

        # Examples:
        # "24H high" -> "24H high updated"
        # "7D high"  -> "7D high updated"
        if cleaned:
            return f"{cleaned} updated"

        return "24H/48H/7D high updated"

    if status == "not_enough_data":
        return "Data unavailable"

    return "no 24H/48H/7D update"


def format_premium_value_for_telegram(value):
    """
    Convert internal premium labels into compact user-facing Telegram text.

    Examples:
    - "Premium / 0.75" -> "Premium 0.75"
    - "Extreme Premium / 0.80" -> "Extreme 0.80"
    - "Above Range High / 1.05" -> "Above Range 1.05"
    """

    text = str(value or "N/A").strip()

    if not text or text == "N/A":
        return "N/A"

    parts = [part.strip() for part in text.split("/")]
    label = parts[0] if parts else text
    number = parts[1] if len(parts) > 1 else ""

    label = label.replace("Extreme Premium", "Extreme")
    label = label.replace("Above Range High", "Above Range")

    if number:
        return f"{label} {number}"

    return label


def format_reason_for_telegram(setup_status):
    text = str(setup_status or "N/A").strip()

    replacements = {
        "High priority watch — ": "",
        "Short watch — ": "",
        "Overheat watch — ": "",
        "Watch only — ": "",
    }

    for prefix, replacement in replacements.items():
        if text.startswith(prefix):
            text = replacement + text[len(prefix):]
            break

    if text == "confirmed liquidity sweep detected":
        return "confirmed liquidity sweep"

    # Preserve common uppercase acronyms in user-facing reasons.
    if text.startswith(("RSI", "HTF", "MSS", "OI")):
        return text

    return text[:1].lower() + text[1:] if text else "N/A"


def compact_factor_detail(factors, key, default="N/A"):
    if key == "liquidity_sweep":
        return format_sweep_detail_for_telegram(factors)

    if key == "local_high_update":
        return format_local_high_detail_for_telegram(factors)

    factor = find_factor(factors, key) or {}
    detail = str(factor.get("detail", "") or "").strip()

    if not detail:
        return default

    return detail



def format_debug_value(value):
    if value is None:
        return "N/A"

    if isinstance(value, float):
        if pd.isna(value):
            return "N/A"
        return f"{value:.6g}"

    text = str(value).strip()

    if not text:
        return "N/A"

    return "_".join(text.split())



def log_oi_debug_for_grouped_signals(grouped_signals):
    """Print OI_DEBUG lines for Telegram-visible signals."""

    visible_levels = {
        "HIGH_PRIORITY_SHORT_WATCH",
        "SHORT_WATCH",
        "OVERHEAT_WATCH",
    }

    for group in grouped_signals or []:
        if str(group.get("signal_level")) not in visible_levels:
            continue

        detail = group.get("display_detail") or {}
        fields = [
            "OI_DEBUG",
            f"symbol={format_debug_value(group.get('symbol'))}",
            f"exchange={format_debug_value(detail.get('exchange'))}",
            f"signal={format_debug_value(detail.get('signal_level') or group.get('signal_level'))}",
            f"status={format_debug_value(detail.get('oi_status'))}",
            f"source={format_debug_value(detail.get('oi_source'))}",
            f"current={format_debug_value(detail.get('oi_current'))}",
            f"change_1h={format_debug_value(detail.get('oi_change_1h_percent'))}",
            f"change_4h={format_debug_value(detail.get('oi_change_4h_percent'))}",
            f"context={format_debug_value(detail.get('oi_context'))}",
            f"history_points={format_debug_value(detail.get('oi_history_points'))}",
            f"history_latest={format_debug_value(detail.get('oi_history_latest_ts'))}",
        ]
        print(" ".join(fields))


def log_sweep_debug_for_grouped_signals(grouped_signals):
    """
    Print one compact SWEEP_DEBUG line per Telegram-visible signal.

    This is intentionally written to GitHub Actions logs only. It does not
    change Telegram output and helps validate why the sweep factor accepted or
    rejected a level when we compare signals with charts.
    """

    visible_levels = {
        "HIGH_PRIORITY_SHORT_WATCH",
        "SHORT_WATCH",
        "OVERHEAT_WATCH",
    }

    for group in grouped_signals or []:
        if str(group.get("signal_level")) not in visible_levels:
            continue

        detail = group.get("display_detail") or {}
        short_factors = detail.get("short_factors", []) or []
        sweep_factor = find_factor(short_factors, "liquidity_sweep") or {}
        debug = sweep_factor.get("debug") or {}

        fields = [
            "SWEEP_DEBUG",
            f"symbol={format_debug_value(group.get('symbol'))}",
            f"exchange={format_debug_value(detail.get('exchange'))}",
            f"signal={format_debug_value(detail.get('signal_level') or group.get('signal_level'))}",
            f"status={format_debug_value(sweep_factor.get('status'))}",
        ]

        if debug:
            ordered_debug_keys = [
                "result",
                "level_tf",
                "level_type",
                "level_price",
                "source_index",
                "source_time",
                "age_bars",
                "age_hours",
                "touches",
                "quality",
                "reaction_pct",
                "prominence_pct",
                "confirm_tf",
                "confirm_time",
                "confirm_high",
                "confirm_close",
                "reason",
            ]

            for key in ordered_debug_keys:
                if key in debug:
                    fields.append(f"{key}={format_debug_value(debug.get(key))}")
        else:
            fields.append(f"detail={format_debug_value(sweep_factor.get('detail'))}")

        print(" ".join(fields))


def log_open_levels_debug_for_grouped_signals(grouped_signals):
    """
    Print one compact OPEN_LEVELS_DEBUG line per Telegram-visible signal.

    This stays in GitHub Actions logs only and helps validate D/W/M/Y open
    resistance detection without making Telegram noisy.
    """

    visible_levels = {
        "HIGH_PRIORITY_SHORT_WATCH",
        "SHORT_WATCH",
        "OVERHEAT_WATCH",
    }

    for group in grouped_signals or []:
        if str(group.get("signal_level")) not in visible_levels:
            continue

        detail = group.get("display_detail") or {}
        short_factors = detail.get("short_factors", []) or []
        open_factor = find_factor(short_factors, "open_levels") or {}
        debug = open_factor.get("debug") or {}
        events = open_factor.get("events") if isinstance(open_factor, dict) else []
        if not isinstance(events, list):
            events = []

        event_parts = []
        for event in events:
            if not isinstance(event, dict):
                continue
            label = format_debug_value(event.get("label"))
            state = format_debug_value(event.get("state"))
            weight = format_debug_value(event.get("weight"))
            open_price = format_debug_value(event.get("open"))
            confirm_tf = format_debug_value(event.get("confirm_tf"))
            window_bars = format_debug_value(event.get("window_bars"))
            confirm_high = format_debug_value(event.get("confirm_high"))
            confirm_close = format_debug_value(event.get("confirm_close"))
            event_parts.append(
                f"{label}:{state}:w{weight}:open{open_price}:tf{confirm_tf}:win{window_bars}:high{confirm_high}:close{confirm_close}"
            )

        levels = debug.get("levels") if isinstance(debug, dict) else {}
        if not isinstance(levels, dict):
            levels = {}

        level_parts = []
        for label in ["D", "W", "M", "Y"]:
            if label in levels:
                level_parts.append(f"{label}={format_debug_value(levels.get(label))}")

        sources = debug.get("levels_source") if isinstance(debug, dict) else {}
        if not isinstance(sources, dict):
            sources = {}
        source_parts = []
        for label in ["D", "W", "M", "Y"]:
            if label in sources:
                source_parts.append(f"{label}={format_debug_value(sources.get(label))}")

        intraday_levels = debug.get("levels_intraday") if isinstance(debug, dict) else {}
        if not isinstance(intraday_levels, dict):
            intraday_levels = {}
        intraday_parts = []
        for label in ["D", "W", "M", "Y"]:
            if label in intraday_levels:
                intraday_parts.append(f"{label}={format_debug_value(intraday_levels.get(label))}")

        intraday_1h_levels = debug.get("levels_intraday_1h") if isinstance(debug, dict) else {}
        if not isinstance(intraday_1h_levels, dict):
            intraday_1h_levels = {}
        intraday_1h_parts = []
        for label in ["D", "W", "M", "Y"]:
            if label in intraday_1h_levels:
                intraday_1h_parts.append(f"{label}={format_debug_value(intraday_1h_levels.get(label))}")

        intraday_4h_levels = debug.get("levels_intraday_4h") if isinstance(debug, dict) else {}
        if not isinstance(intraday_4h_levels, dict):
            intraday_4h_levels = {}
        intraday_4h_parts = []
        for label in ["D", "W", "M", "Y"]:
            if label in intraday_4h_levels:
                intraday_4h_parts.append(f"{label}={format_debug_value(intraday_4h_levels.get(label))}")

        daily_levels = debug.get("levels_daily") if isinstance(debug, dict) else {}
        if not isinstance(daily_levels, dict):
            daily_levels = {}
        daily_parts = []
        for label in ["D", "W", "M", "Y"]:
            if label in daily_levels:
                daily_parts.append(f"{label}={format_debug_value(daily_levels.get(label))}")

        fields = [
            "OPEN_LEVELS_DEBUG",
            f"symbol={format_debug_value(group.get('symbol'))}",
            f"exchange={format_debug_value(detail.get('exchange'))}",
            f"signal={format_debug_value(detail.get('signal_level') or group.get('signal_level'))}",
            f"status={format_debug_value(open_factor.get('status'))}",
            f"score={format_debug_value(open_factor.get('open_context_score', 0.0))}",
            f"events={format_debug_value('|'.join(event_parts) if event_parts else 'none')}",
            f"levels={format_debug_value('|'.join(level_parts) if level_parts else 'none')}",
            f"levels_source={format_debug_value('|'.join(source_parts) if source_parts else 'none')}",
            f"levels_intraday_selected={format_debug_value('|'.join(intraday_parts) if intraday_parts else 'none')}",
            f"levels_1h_utc={format_debug_value('|'.join(intraday_1h_parts) if intraday_1h_parts else 'none')}",
            f"levels_4h_utc={format_debug_value('|'.join(intraday_4h_parts) if intraday_4h_parts else 'none')}",
            f"levels_1d_utc={format_debug_value('|'.join(daily_parts) if daily_parts else 'none')}",
            f"D_window={format_debug_value(debug.get('D_window_tf'))}:{format_debug_value(debug.get('D_window_bars'))}",
            f"HTF_window={format_debug_value(debug.get('HTF_window_tf'))}:{format_debug_value(debug.get('HTF_window_bars'))}",
            f"detail={format_debug_value(open_factor.get('detail'))}",
        ]

        print(" ".join(fields))


def log_rejection_debug_for_grouped_signals(grouped_signals):
    """Print REJECTION_DEBUG lines for Telegram-visible signals."""

    visible_levels = {
        "HIGH_PRIORITY_SHORT_WATCH",
        "SHORT_WATCH",
        "OVERHEAT_WATCH",
    }

    for group in grouped_signals or []:
        if str(group.get("signal_level")) not in visible_levels:
            continue

        detail = group.get("display_detail") or {}
        short_factors = detail.get("short_factors", []) or []
        factor = find_factor(short_factors, "rejection_candle") or {}
        events = factor.get("events") if isinstance(factor, dict) else []
        if not isinstance(events, list):
            events = []

        event_parts = []
        for event in events:
            if not isinstance(event, dict):
                continue
            event_parts.append(
                ":".join([
                    format_debug_value(event.get("label")),
                    format_debug_value(event.get("state")),
                    format_debug_value(event.get("confirm_tf")),
                    f"open{format_debug_value(event.get('open'))}",
                    f"high{format_debug_value(event.get('confirm_high'))}",
                    f"close{format_debug_value(event.get('confirm_close'))}",
                    f"wick{format_debug_value(event.get('upper_wick_pct'))}",
                    f"closepos{format_debug_value(event.get('close_position'))}",
                ])
            )

        debug = factor.get("debug") if isinstance(factor, dict) else {}
        if not isinstance(debug, dict):
            debug = {}

        levels = debug.get("levels") if isinstance(debug, dict) else {}
        if not isinstance(levels, dict):
            levels = {}
        level_parts = []
        for label in ["D", "W", "M", "Y"]:
            if label in levels:
                level_parts.append(f"{label}={format_debug_value(levels.get(label))}")

        sources = debug.get("levels_source") if isinstance(debug, dict) else {}
        if not isinstance(sources, dict):
            sources = {}
        source_parts = []
        for label in ["D", "W", "M", "Y"]:
            if label in sources:
                source_parts.append(f"{label}={format_debug_value(sources.get(label))}")

        fields = [
            "REJECTION_DEBUG",
            f"symbol={format_debug_value(group.get('symbol'))}",
            f"exchange={format_debug_value(detail.get('exchange'))}",
            f"signal={format_debug_value(detail.get('signal_level') or group.get('signal_level'))}",
            f"status={format_debug_value(factor.get('status'))}",
            f"detail={format_debug_value(factor.get('detail'))}",
            f"events={format_debug_value('|'.join(event_parts) if event_parts else 'none')}",
            f"levels={format_debug_value('|'.join(level_parts) if level_parts else 'none')}",
            f"levels_source={format_debug_value('|'.join(source_parts) if source_parts else 'none')}",
        ]

        print(" ".join(fields))


def log_new_high_debug_for_grouped_signals(grouped_signals):
    """Print NEW_HIGH_DEBUG lines for Telegram-visible signals."""

    visible_levels = {
        "HIGH_PRIORITY_SHORT_WATCH",
        "SHORT_WATCH",
        "OVERHEAT_WATCH",
    }

    for group in grouped_signals or []:
        if str(group.get("signal_level")) not in visible_levels:
            continue

        detail = group.get("display_detail") or {}
        short_factors = detail.get("short_factors", []) or []
        factor = find_factor(short_factors, "local_high_update") or {}

        fields = [
            "NEW_HIGH_DEBUG",
            f"symbol={format_debug_value(group.get('symbol'))}",
            f"exchange={format_debug_value(detail.get('exchange'))}",
            f"signal={format_debug_value(detail.get('signal_level') or group.get('signal_level'))}",
            f"status={format_debug_value(factor.get('status'))}",
            f"detail={format_debug_value(factor.get('detail'))}",
            f"setup_high={format_debug_value(factor.get('setup_high'))}",
            f"previous_high={format_debug_value(factor.get('previous_high'))}",
            f"recent_window_bars={format_debug_value(factor.get('recent_window_bars'))}",
        ]

        print(" ".join(fields))


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
    Compact user-facing Telegram report.

    PUMP WATCH is intentionally hidden from Telegram. It remains available in
    internal scoring/output tables, but Telegram shows only actionable watchlist
    context: OVERHEAT, SHORT WATCH and HIGH PRIORITY SHORT WATCH.
    """

    now_kyiv = datetime.now(KYIV_TZ).strftime("%Y-%m-%d %H:%M Kyiv")

    visible_levels = {
        "HIGH_PRIORITY_SHORT_WATCH",
        "SHORT_WATCH",
        "OVERHEAT_WATCH",
    }

    visible_groups = [
        group for group in (grouped_signals or [])
        if str(group.get("signal_level")) in visible_levels
    ]

    total_visible_count = len(visible_groups)
    display_groups = visible_groups[:TELEGRAM_MAX_SIGNALS]
    displayed_count = len(display_groups)

    lines = []
    lines.append("📊 <b>Market Heat Scanner</b>")
    lines.append(f"{html.escape(now_kyiv)} | {html.escape(SCRIPT_VERSION)}")

    if total_visible_count > displayed_count:
        lines.append(f"Shown: <b>{displayed_count}</b>/<b>{total_visible_count}</b>")

    if not display_groups:
        lines.append("")
        lines.append("✅ No overheat or short-watch signals.")
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
            sweep_factor = find_factor(short_factors, "liquidity_sweep") or {}
            rejection_factor = find_factor(short_factors, "rejection_candle") or {}

            sweep_icon = factor_status_icon(sweep_factor)
            rejection_icon = factor_status_icon(rejection_factor)
            sweep_detail = html.escape(format_sweep_detail_for_telegram(short_factors))
            rejection_detail = html.escape(format_rejection_detail_for_telegram(short_factors))

            reason = html.escape(format_reason_for_telegram(setup_status))

            lines.append("")
            lines.append(f"{idx + 1}) {signal_label} — <code>{symbol}</code>")
            lines.append(f"Price {price} | 24h {chg_24h}% | Vol {vol_24h} | ΔVol {vol_chg}%")
            lines.append(f"RSI 1H Live {rsi_1h_live} | Closed {rsi_1h_closed} | 4H Live {rsi_4h_live}")
            oi_line = html.escape(format_oi_line_for_telegram(detail))
            if oi_line:
                lines.append(oi_line)
            open_levels_detail = html.escape(format_open_levels_detail_for_telegram(short_factors))
            if open_levels_detail:
                lines.append(f"Open Levels: {open_levels_detail}")

            lines.append("")
            lines.append(f"Sweep: {sweep_icon} {sweep_detail}")
            lines.append(f"Rejection: {rejection_icon} {rejection_detail}")
            lines.append(f"Reason: {reason}")

            if idx != len(display_groups) - 1:
                lines.append("────────────")

    lines.append("")
    lines.append("Planned: ⚪ Funding | ⚪ Vol climax | ⚪ Failed BO | ⚪ MSS | ⚪ Div")

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
        rsi_filtered_out_count = 0
    else:
        df_all["signal_rank"] = df_all["signal_level"].apply(get_signal_rank)
        df_all = add_rsi_entry_filter_columns(df_all)

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

        df_active_before_rsi_filter = df_all[df_all["signal_level"] != "NO_SIGNAL"].copy()
        log_rsi_entry_debug(df_active_before_rsi_filter)
        rsi_filtered_out_count = len(df_active_before_rsi_filter[~df_active_before_rsi_filter["rsi_entry_passed"]])
        df_active = df_active_before_rsi_filter[df_active_before_rsi_filter["rsi_entry_passed"]].copy()

    print("\n" + "=" * 120)
    print("MULTI-PROVIDER SUMMARY")
    print("=" * 120)
    print("OKX total universe:", okx_total)
    print("OKX prefiltered:", okx_prefiltered)
    print("OKX active:", okx_active)
    print("Bitget total universe:", bitget_total)
    print("Bitget prefiltered:", bitget_prefiltered)
    print("Bitget active:", bitget_active)
    print("RSI entry filtered out:", rsi_filtered_out_count)
    print("Total active signals after RSI entry filter:", len(df_active))

    df_active_output = prepare_active_output_table(df_active)
    grouped_signals = prepare_grouped_active_signals(df_active, max_groups=FINAL_TOP_N)
    log_oi_debug_for_grouped_signals(grouped_signals)
    log_sweep_debug_for_grouped_signals(grouped_signals)
    log_open_levels_debug_for_grouped_signals(grouped_signals)
    log_rejection_debug_for_grouped_signals(grouped_signals)
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



# ============================================================
# SELF TESTS
# ============================================================

def make_synthetic_ohlcv(rows, freq="1h"):
    timestamps = pd.date_range("2026-01-01", periods=len(rows), freq=freq)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "confirm"])
    df["timestamp"] = timestamps
    df["volume"] = 1.0
    df["quote_volume"] = 1.0

    return df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "confirm"]]


def make_flat_synthetic_1h(rows_count=31, price=90.0):
    rows = []

    for i in range(rows_count):
        confirm = 0 if i == rows_count - 1 else 1
        rows.append((price, price + 0.5, price - 0.5, price, confirm))

    return make_synthetic_ohlcv(rows, freq="1h")


def make_flat_synthetic_4h(rows_count=15, price=90.0):
    rows = []

    for i in range(rows_count):
        confirm = 0 if i == rows_count - 1 else 1
        rows.append((price, price + 0.5, price - 0.5, price, confirm))

    return make_synthetic_ohlcv(rows, freq="4h")


def make_4h_swing_sweep_case(level_age_bars=5, confirm_with_4h=True):
    level_index = 4
    candidate_index = level_index + int(level_age_bars)
    rows_count = candidate_index + 2
    rows = []

    for i in range(rows_count):
        if i == level_index:
            rows.append((98.0, 100.0, 97.0, 98.0, 1))
        elif i == candidate_index and confirm_with_4h:
            rows.append((99.0, 100.3, 92.0, 95.0, 1))
        elif i == rows_count - 1:
            rows.append((94.0, 95.0, 93.0, 94.0, 0))
        elif i < level_index:
            base = 95.0 + i * 0.5
            rows.append((base, base + 1.0, 94.0, 95.0, 1))
        else:
            rows.append((94.0, 97.0, 90.0, 94.0, 1))

    return make_synthetic_ohlcv(rows, freq="4h")


def make_4h_level_without_4h_sweep_case():
    rows = []

    for i in range(12):
        if i == 4:
            rows.append((98.0, 100.0, 97.0, 98.0, 1))
        elif i == 9:
            rows.append((95.0, 97.0, 92.0, 94.0, 1))
        elif i == 11:
            rows.append((94.0, 95.0, 93.0, 94.0, 0))
        elif i < 4:
            base = 95.0 + i * 0.5
            rows.append((base, base + 1.0, 94.0, 95.0, 1))
        else:
            rows.append((94.0, 97.0, 90.0, 94.0, 1))

    return make_synthetic_ohlcv(rows, freq="4h")


def make_1h_intrabar_take_of_4h_level_case():
    rows = []

    for i in range(31):
        if i == 29:
            rows.append((99.0, 100.3, 94.0, 95.0, 1))
        elif i == 30:
            rows.append((94.0, 95.0, 93.0, 94.0, 0))
        else:
            rows.append((90.0, 90.5, 89.5, 90.0, 1))

    return make_synthetic_ohlcv(rows, freq="1h")


def make_1h_equal_high_sweep_case():
    rows = []

    for i in range(31):
        if i == 5:
            rows.append((98.0, 100.0, 97.0, 98.0, 1))
        elif i == 12:
            rows.append((98.0, 100.1, 97.0, 98.0, 1))
        elif i == 29:
            rows.append((99.0, 100.4, 92.0, 95.0, 1))
        elif i == 30:
            rows.append((94.0, 95.0, 93.0, 94.0, 0))
        elif i in [3, 4, 6, 7, 10, 11, 13, 14]:
            rows.append((94.0, 97.0, 90.0, 94.0, 1))
        else:
            rows.append((93.0, 96.0, 90.0, 93.0, 1))

    return make_synthetic_ohlcv(rows, freq="1h")


def make_rolling_only_high_take_case():
    rows = []

    for i in range(31):
        if i == 28:
            rows.append((98.0, 100.0, 97.0, 99.0, 1))
        elif i == 29:
            rows.append((99.0, 100.3, 94.0, 95.0, 1))
        elif i == 30:
            rows.append((94.0, 95.0, 93.0, 94.0, 0))
        else:
            rows.append((90.0, 95.0, 89.0, 90.0, 1))

    return make_synthetic_ohlcv(rows, freq="1h")




def make_1h_minor_micro_high_sweep_case():
    """A tiny local bump should not be reported as a visible 1H high."""
    rows = []

    for i in range(31):
        if i == 5:
            # Only 0.05% above neighbours, below the 0.20% visibility threshold.
            rows.append((99.8, 100.05, 98.8, 99.6, 1))
        elif i in [3, 4, 6, 7]:
            rows.append((99.5, 100.00, 98.5, 99.2, 1))
        elif i in [8, 9, 10, 11, 12]:
            rows.append((97.5, 98.0, 96.0, 97.0, 1))
        elif i == 29:
            rows.append((99.0, 100.25, 96.0, 99.7, 1))
        elif i == 30:
            rows.append((99.6, 99.8, 99.0, 99.5, 0))
        else:
            rows.append((98.0, 99.0, 96.0, 98.0, 1))

    return make_synthetic_ohlcv(rows, freq="1h")

def make_1h_level_with_4h_close_only_case():
    """A visible 1H level exists, but only the 4H candle confirms it.

    r19 should reject this: 1H levels require a closed 1H confirmation.
    """
    rows = []

    for i in range(31):
        if i == 5:
            rows.append((98.0, 100.0, 97.0, 98.0, 1))
        elif i in [6, 7, 8, 9, 10, 11, 12]:
            rows.append((94.0, 96.0, 90.0, 94.0, 1))
        elif i == 29:
            # No 1H sweep here: the high stays below the 1H level.
            rows.append((94.0, 96.0, 93.0, 95.0, 1))
        elif i == 30:
            rows.append((95.0, 96.0, 94.0, 95.0, 0))
        else:
            rows.append((93.0, 96.0, 90.0, 93.0, 1))

    df_1h = make_synthetic_ohlcv(rows, freq="1h")

    df_4h = make_synthetic_ohlcv([
        (94.0, 96.0, 90.0, 94.0, 1),
        (94.0, 96.0, 90.0, 94.0, 1),
        (94.0, 96.0, 90.0, 94.0, 1),
        (94.0, 96.0, 90.0, 94.0, 1),
        (94.0, 96.0, 90.0, 94.0, 1),
        (99.0, 100.5, 92.0, 95.0, 1),
        (95.0, 96.0, 94.0, 95.0, 0),
    ], freq="4h")

    return df_1h, df_4h


def make_open_levels_1d_case():
    dates = pd.to_datetime([
        "2026-01-01 00:00:00",
        "2026-06-08 00:00:00",
        "2026-06-09 00:00:00",
        "2026-06-10 00:00:00",
        "2026-06-11 00:00:00",
    ])

    return pd.DataFrame({
        "timestamp": dates,
        "open": [80.0, 100.0, 98.0, 97.0, 94.0],
        "high": [82.0, 101.0, 99.0, 98.0, 95.0],
        "low": [79.0, 95.0, 96.0, 95.0, 93.5],
        "close": [81.0, 98.0, 97.0, 96.0, 94.5],
    })


def make_open_levels_1h_confirm_case():
    timestamps = pd.date_range("2026-06-11 00:00:00", periods=8, freq="1h")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [95.0, 95.5, 95.7, 95.8, 95.9, 95.7, 95.6, 95.5],
        "high": [95.8, 96.0, 96.1, 96.0, 96.2, 96.0, 96.0, 96.0],
        "low": [94.9, 95.2, 95.4, 95.5, 95.4, 95.3, 95.2, 95.2],
        "close": [95.4, 95.7, 95.8, 95.9, 95.7, 95.6, 95.5, 95.4],
    })
    return df


def make_open_levels_1h_near_d_rejection_case():
    """Price fails just under D open; this should read as near-D rejection."""
    timestamps = pd.date_range("2026-06-11 00:00:00", periods=8, freq="1h")
    df = pd.DataFrame({
        "timestamp": timestamps,
        # First 1H candle defines the UTC D open used by r25.
        "open": [94.0, 91.5, 92.0, 92.5, 93.0, 93.4, 93.8, 93.3],
        "high": [94.2, 92.1, 92.6, 93.0, 93.4, 93.7, 93.90, 93.5],
        "low": [90.8, 91.2, 91.7, 92.2, 92.7, 93.0, 93.1, 93.0],
        "close": [91.4, 91.9, 92.4, 92.8, 93.2, 93.5, 93.25, 93.2],
    })
    return df


def make_open_levels_d_rejected_then_reclaimed_case():
    """D open was rejected earlier, but latest price has already reclaimed it."""
    timestamps = pd.date_range("2026-06-11 00:00:00", periods=8, freq="1h")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [92.0, 92.2, 92.4, 92.6, 92.8, 93.0, 93.5, 95.2],
        "high": [92.6, 92.8, 93.0, 93.2, 93.4, 93.6, 94.4, 96.2],
        "low": [91.8, 92.0, 92.2, 92.4, 92.6, 92.8, 93.0, 94.8],
        "close": [92.2, 92.4, 92.6, 92.8, 93.0, 93.2, 93.4, 96.0],
    })
    return df


def make_open_levels_4h_test_case():
    timestamps = pd.date_range("2026-06-10 00:00:00", periods=8, freq="4h")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 99.4, 99.2],
        "high": [95.0, 96.0, 97.0, 98.0, 99.0, 100.2, 100.5, 99.5],
        "low": [93.5, 94.5, 95.5, 96.5, 97.5, 98.4, 98.8, 98.7],
        "close": [94.8, 95.8, 96.8, 97.8, 98.8, 99.4, 99.5, 99.0],
    })
    return df


def make_open_levels_4h_near_case():
    timestamps = pd.date_range("2026-06-10 00:00:00", periods=8, freq="4h")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 99.2, 99.4],
        "high": [95.0, 96.0, 97.0, 98.0, 99.0, 99.4, 99.6, 99.85],
        "low": [93.5, 94.5, 95.5, 96.5, 97.5, 98.7, 98.9, 99.1],
        "close": [94.8, 95.8, 96.8, 97.8, 98.8, 99.1, 99.3, 99.7],
    })
    return df


def make_open_levels_4h_far_case():
    timestamps = pd.date_range("2026-06-10 00:00:00", periods=8, freq="4h")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [90.0, 90.5, 91.0, 91.5, 92.0, 92.5, 93.0, 93.5],
        "high": [91.0, 91.5, 92.0, 92.5, 93.0, 93.5, 94.0, 94.5],
        "low": [89.5, 90.0, 90.5, 91.0, 91.5, 92.0, 92.5, 93.0],
        "close": [90.5, 91.0, 91.5, 92.0, 92.5, 93.0, 93.5, 94.0],
    })
    return df



def make_open_levels_4h_recent_window_test_case():
    """4H open was tested one closed candle before the latest closed candle."""
    timestamps = pd.date_range("2026-06-10 00:00:00", periods=8, freq="4h")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 98.5, 98.0],
        "high": [95.0, 96.0, 97.0, 98.0, 99.0, 100.3, 99.2, 98.5],
        "low": [93.5, 94.5, 95.5, 96.5, 97.5, 98.2, 97.8, 97.5],
        "close": [94.8, 95.8, 96.8, 97.8, 98.8, 99.2, 98.6, 98.1],
    })
    return df



def make_open_levels_4h_live_test_case():
    """Live 4H candle is testing W/M open from below, but closed 4H candles did not confirm yet."""
    timestamps = pd.date_range("2026-06-10 00:00:00", periods=8, freq="4h")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [94.0, 95.0, 96.0, 97.0, 98.0, 98.5, 98.7, 99.0],
        "high": [95.0, 96.0, 97.0, 98.0, 99.0, 99.2, 99.4, 100.3],
        "low": [93.5, 94.5, 95.5, 96.5, 97.5, 98.1, 98.2, 98.5],
        "close": [94.8, 95.8, 96.8, 97.8, 98.8, 98.9, 99.0, 99.4],
    })
    return df

def make_open_levels_1h_htf_rejection_case():
    """Closed 1H candle rejects W/M open while 4H has not confirmed it."""
    timestamps = pd.date_range("2026-06-11 00:00:00", periods=8, freq="1h")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [95.0, 96.0, 97.0, 98.0, 98.6, 99.0, 99.1, 99.0],
        "high": [95.8, 96.8, 97.8, 98.8, 99.4, 99.6, 100.5, 99.6],
        "low": [94.8, 95.8, 96.8, 97.8, 98.2, 98.7, 98.8, 98.7],
        "close": [95.5, 96.5, 97.5, 98.5, 99.0, 99.2, 99.3, 99.1],
    })
    return df


def make_open_levels_1h_htf_live_test_case():
    """Live 1H candle is testing W/M open, but no closed 1H rejection exists yet."""
    timestamps = pd.date_range("2026-06-11 00:00:00", periods=8, freq="1h")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [95.0, 96.0, 97.0, 98.0, 98.4, 98.7, 99.0, 99.1],
        "high": [95.8, 96.8, 97.8, 98.8, 99.1, 99.3, 99.5, 100.5],
        "low": [94.8, 95.8, 96.8, 97.8, 98.1, 98.4, 98.7, 98.8],
        "close": [95.5, 96.5, 97.5, 98.5, 98.9, 99.0, 99.2, 99.3],
    })
    return df


def run_open_levels_self_tests():
    print("\n" + "=" * 120)
    print("RUNNING OPEN LEVELS SELF TESTS")
    print("SCRIPT VERSION:", SCRIPT_VERSION)
    print("=" * 120)

    df_1d = make_open_levels_1d_case()

    tests = []

    tests.append((
        "W/M open tested by 4H close below",
        detect_open_levels_context(
            df_1h=make_open_levels_1h_confirm_case(),
            df_4h=make_open_levels_4h_test_case(),
            df_1d=df_1d,
            current_price=99.4,
        ),
        "confirmed",
        "W/M open tested, close below",
    ))

    tests.append((
        "W/M open near from below",
        detect_open_levels_context(
            df_1h=make_open_levels_1h_confirm_case(),
            df_4h=make_open_levels_4h_near_case(),
            df_1d=df_1d,
            current_price=99.7,
        ),
        "confirmed",
        "near W/M open resistance",
    ))
    tests.append((
        "W/M open tested in recent 4H setup window",
        detect_open_levels_context(
            df_1h=make_open_levels_1h_confirm_case(),
            df_4h=make_open_levels_4h_recent_window_test_case(),
            df_1d=df_1d,
            current_price=98.6,
        ),
        "confirmed",
        "W/M open tested, close below",
    ))

    tests.append((
        "W/M open live test before 4H close",
        detect_open_levels_context(
            df_1h=make_open_levels_1h_confirm_case(),
            df_4h=make_open_levels_4h_live_test_case(),
            df_1d=df_1d,
            current_price=99.4,
        ),
        "confirmed",
        "W/M open live test, price below",
    ))

    tests.append((
        "W/M open tested by 1H close below",
        detect_open_levels_context(
            df_1h=make_open_levels_1h_htf_rejection_case(),
            df_4h=make_open_levels_4h_far_case(),
            df_1d=df_1d,
            current_price=99.3,
        ),
        "confirmed",
        "W/M open tested, close below",
    ))

    tests.append((
        "W/M open live test by 1H before close",
        detect_open_levels_context(
            df_1h=make_open_levels_1h_htf_live_test_case(),
            df_4h=make_open_levels_4h_far_case(),
            df_1d=df_1d,
            current_price=99.3,
        ),
        "confirmed",
        "W/M open live test, price below",
    ))

    tests.append((
        "D open test invalidated after price reclaimed the level",
        detect_open_levels_context(
            df_1h=make_open_levels_d_rejected_then_reclaimed_case(),
            df_4h=make_open_levels_4h_far_case(),
            df_1d=df_1d,
            current_price=96.0,
        ),
        "not_confirmed",
        "no open-level resistance nearby",
    ))

    tests.append((
        "TAO/ZKP-like M open is too far above price",
        detect_open_levels_context(
            df_1h=make_open_levels_far_below_month_case(),
            df_4h=make_open_levels_far_below_month_case(),
            df_1d=make_open_levels_month_only_1d_case(),
            current_price=95.0,
        ),
        "not_confirmed",
        "no open-level resistance nearby",
    ))

    tests.append((
        "NEAR-like M open reclaimed by current price",
        detect_open_levels_context(
            df_1h=make_open_levels_month_reclaimed_case(),
            df_4h=make_open_levels_month_reclaimed_case(),
            df_1d=make_open_levels_month_only_1d_case(),
            current_price=101.5,
        ),
        "not_confirmed",
        "no open-level resistance nearby",
    ))

    tests.append((
        "BANANAS-like old M test invalid after later closes above M",
        detect_open_levels_context(
            df_1h=make_open_levels_old_month_rejection_then_closed_above_case(),
            df_4h=make_open_levels_far_below_month_case(),
            df_1d=make_open_levels_month_only_1d_case(),
            current_price=99.4,
        ),
        "not_confirmed",
        "no open-level resistance nearby",
    ))

    tests.append((
        "USELESS-like near M invalid after later reclaim",
        detect_open_levels_context(
            df_1h=make_open_levels_useless_like_case(),
            df_4h=make_open_levels_far_below_month_case(),
            df_1d=make_open_levels_month_only_1d_case(),
            current_price=100.2,
        ),
        "not_confirmed",
        "no open-level resistance nearby",
    ))

    tests.append((
        "VVV-like W reclaimed and M not reached",
        detect_open_levels_context(
            df_1h=make_open_levels_w_reclaimed_m_not_reached_case(),
            df_4h=make_open_levels_w_reclaimed_m_not_reached_case(),
            df_1d=make_open_levels_vvv_1d_case(),
            current_price=102.0,
        ),
        "not_confirmed",
        "no open-level resistance nearby",
    ))

    tests.append((
        "No open resistance nearby",
        detect_open_levels_context(
            df_1h=make_open_levels_1h_confirm_case(),
            df_4h=make_open_levels_4h_far_case(),
            df_1d=df_1d,
            current_price=97.0,
        ),
        "not_confirmed",
        "no open-level resistance nearby",
    ))

    failed = 0

    for name, result, expected_status, expected_detail in tests:
        actual_status = str(result.get("status"))
        detail = str(result.get("detail", ""))
        ok = actual_status == expected_status and expected_detail in detail
        status_text = "PASS" if ok else "FAIL"
        print(f"{status_text} | {name} | expected={expected_status} actual={actual_status} | detail={detail}")

        if not ok:
            failed += 1

    if failed > 0:
        raise AssertionError(f"Open levels self-tests failed: {failed}/{len(tests)}")

    print(f"OPEN LEVELS SELF TESTS PASSED: {len(tests)}/{len(tests)}")


def make_classification_open_factor(labels, states):
    events = []

    for label, state in zip(labels, states):
        weights = OPEN_LEVEL_CONTEXT_WEIGHTS.get(label, {"near": 0.0, "tested": 0.0})
        events.append({
            "label": label,
            "state": state,
            "weight": float(weights.get(state, 0.0)),
            "open": 100.0,
        })

    factor = make_factor(
        key="open_levels",
        label="Open levels",
        status="confirmed" if events else "not_confirmed",
        points=0,
        detail=build_open_levels_detail(events),
    )
    factor["open_context_score"] = float(sum(event["weight"] for event in events))
    factor["events"] = events
    return factor


def make_classification_sweep_factor():
    return make_factor(
        key="liquidity_sweep",
        label="Liquidity sweep",
        status="confirmed",
        points=2,
        detail="Swept 1H equal highs x2 100.00, close below",
    )


def make_classification_premium_factor(points=0):
    factor = make_factor(
        key="premium_zone",
        label="Premium zone",
        status="confirmed" if points > 0 else "not_confirmed",
        points=points,
        detail="1H: Neutral | 4H: Neutral",
    )
    factor["premium_1h_label"] = "Neutral"
    factor["premium_1h_position"] = 0.50
    factor["premium_4h_label"] = "Neutral"
    factor["premium_4h_position"] = 0.50
    return factor


def make_classification_local_high_factor(points=0):
    return make_factor(
        key="local_high_update",
        label="Local high update",
        status="confirmed" if points > 0 else "not_confirmed",
        points=points,
        detail="1H: no 24H/48H/7D high update",
    )


def run_open_levels_classification_self_tests():
    print("\n" + "=" * 120)
    print("RUNNING OPEN LEVELS CLASSIFICATION SELF TESTS")
    print("SCRIPT VERSION:", SCRIPT_VERSION)
    print("=" * 120)

    base_scores = {
        "pump_score": 2,
        "rsi_score": 0,
        "volume_score": 1,
        "short_setup_score": 2,
        "final_score": 5,
        "volume_ok": True,
        "has_basic_pump_context": True,
        "has_strong_pump_context": True,
        "has_extreme_pump_context": False,
        "has_overheat_context": False,
        "overheat_reason": "N/A",
    }

    strong_heat_scores = dict(base_scores)
    strong_heat_scores["rsi_score"] = 3
    strong_heat_scores["final_score"] = 8

    tests = [
        (
            "W open tested + sweep can provide location and create SHORT WATCH",
            base_scores,
            [
                make_classification_sweep_factor(),
                make_classification_premium_factor(points=0),
                make_classification_local_high_factor(points=0),
                make_classification_open_factor(["W"], ["tested"]),
            ],
            "SHORT_WATCH",
        ),
        (
            "W open tested without sweep cannot create SHORT WATCH",
            base_scores,
            [
                make_classification_premium_factor(points=0),
                make_classification_local_high_factor(points=0),
                make_classification_open_factor(["W"], ["tested"]),
            ],
            "NO_SIGNAL",
        ),
        (
            "Strong heat + sweep + W tested can upgrade to HIGH PRIORITY",
            strong_heat_scores,
            [
                make_classification_sweep_factor(),
                make_classification_premium_factor(points=0),
                make_classification_local_high_factor(points=0),
                make_classification_open_factor(["W"], ["tested"]),
            ],
            "HIGH_PRIORITY_SHORT_WATCH",
        ),
        (
            "Confirmed sweep remains SHORT WATCH even with minor D near context",
            base_scores,
            [
                make_classification_sweep_factor(),
                make_classification_open_factor(["D"], ["near"]),
            ],
            "SHORT_WATCH",
        ),
    ]

    failed = 0

    for name, scores, factors, expected_signal in tests:
        result = classify_watch_signal(scores, short_factors=factors)
        actual_signal = str(result.get("signal_level"))
        ok = actual_signal == expected_signal
        status_text = "PASS" if ok else "FAIL"
        print(f"{status_text} | {name} | expected={expected_signal} actual={actual_signal} | location={result.get('location_score')} open={result.get('open_context_score')}")

        if not ok:
            failed += 1

    if failed > 0:
        raise AssertionError(f"Open levels classification self-tests failed: {failed}/{len(tests)}")

    print(f"OPEN LEVELS CLASSIFICATION SELF TESTS PASSED: {len(tests)}/{len(tests)}")


def make_local_high_recent_update_case(updated=True):
    rows = []

    # Build enough 1H candles to validate 24H local-high update over a recent setup window.
    for i in range(35):
        confirm = 0 if i == 34 else 1
        high = 100.0
        open_price = 95.0
        low = 94.0
        close = 96.0

        if i >= 29:
            high = 98.0

        # Setup candle is not the latest candle. This catches the false-negative
        # case where the last candle has already pulled back.
        if updated and i == 31:
            high = 110.0
            open_price = 101.0
            low = 99.0
            close = 104.0

        rows.append((open_price, high, low, close, confirm))

    return make_synthetic_ohlcv(rows, freq="1h")


def run_local_high_self_tests():
    print("\n" + "=" * 120)
    print("RUNNING LOCAL HIGH SELF TESTS")
    print("SCRIPT VERSION:", SCRIPT_VERSION)
    print("=" * 120)

    tests = [
        (
            "recent setup high updated even if latest candle pulled back",
            detect_local_high_update(make_local_high_recent_update_case(updated=True)),
            "confirmed",
            "24H high updated",
        ),
        (
            "recent setup window without new high",
            detect_local_high_update(make_local_high_recent_update_case(updated=False)),
            "not_confirmed",
            "no 24H/48H/7D high update",
        ),
    ]

    failed = 0

    for name, result, expected_status, expected_detail in tests:
        actual_status = str(result.get("status"))
        detail = str(result.get("detail", ""))
        ok = actual_status == expected_status and expected_detail in detail
        status_text = "PASS" if ok else "FAIL"
        print(f"{status_text} | {name} | expected={expected_status} actual={actual_status} | detail={detail}")

        if not ok:
            failed += 1

    if failed > 0:
        raise AssertionError(f"Local high self-tests failed: {failed}/{len(tests)}")

    print(f"LOCAL HIGH SELF TESTS PASSED: {len(tests)}/{len(tests)}")


def run_sweep_self_tests():
    print("\n" + "=" * 120)
    print("RUNNING SWEEP SELF TESTS")
    print("SCRIPT VERSION:", SCRIPT_VERSION)
    print("=" * 120)

    tests = []

    tests.append((
        "4H level age 4 bars -> no sweep",
        detect_liquidity_sweep(
            df_1h=make_flat_synthetic_1h(),
            df_4h=make_4h_swing_sweep_case(level_age_bars=4),
        ),
        "not_confirmed",
    ))

    tests.append((
        "4H level age 5 bars -> confirmed",
        detect_liquidity_sweep(
            df_1h=make_flat_synthetic_1h(),
            df_4h=make_4h_swing_sweep_case(level_age_bars=5),
        ),
        "confirmed",
    ))

    tests.append((
        "4H level + 1H close only -> no sweep",
        detect_liquidity_sweep(
            df_1h=make_1h_intrabar_take_of_4h_level_case(),
            df_4h=make_4h_level_without_4h_sweep_case(),
        ),
        "not_confirmed",
    ))

    tests.append((
        "1H equal highs + 1H close below -> confirmed",
        detect_liquidity_sweep(
            df_1h=make_1h_equal_high_sweep_case(),
            df_4h=make_flat_synthetic_4h(),
        ),
        "confirmed",
    ))

    tests.append((
        "rolling-only 12H/24H take -> no sweep",
        detect_liquidity_sweep(
            df_1h=make_rolling_only_high_take_case(),
            df_4h=make_flat_synthetic_4h(),
        ),
        "not_confirmed",
    ))

    tests.append((
        "minor micro 1H high -> no sweep",
        detect_liquidity_sweep(
            df_1h=make_1h_minor_micro_high_sweep_case(),
            df_4h=make_flat_synthetic_4h(),
        ),
        "not_confirmed",
    ))

    df_1h_cross_tf, df_4h_cross_tf = make_1h_level_with_4h_close_only_case()
    tests.append((
        "1H level + 4H close only -> no sweep",
        detect_liquidity_sweep(
            df_1h=df_1h_cross_tf,
            df_4h=df_4h_cross_tf,
        ),
        "not_confirmed",
    ))

    failed = 0

    for name, result, expected_status in tests:
        actual_status = str(result.get("status"))
        detail = str(result.get("detail", ""))
        ok = actual_status == expected_status
        status_text = "PASS" if ok else "FAIL"
        print(f"{status_text} | {name} | expected={expected_status} actual={actual_status} | detail={detail}")

        if not ok:
            failed += 1

    if failed > 0:
        raise AssertionError(f"Sweep self-tests failed: {failed}/{len(tests)}")

    print(f"SWEEP SELF TESTS PASSED: {len(tests)}/{len(tests)}")





def make_open_levels_4h_test_without_rejection_case():
    timestamps = pd.date_range("2026-06-10 00:00:00", periods=8, freq="4h")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 99.4, 99.2],
        "high": [95.0, 96.0, 97.0, 98.0, 99.0, 100.2, 100.5, 99.5],
        "low": [93.5, 94.5, 95.5, 96.5, 97.5, 98.4, 98.8, 98.7],
        "close": [94.8, 95.8, 96.8, 97.8, 98.8, 99.4, 100.1, 99.0],
    })
    return df



def make_open_levels_month_only_1d_case():
    """D/W are below current price, M is the only relevant higher open."""
    dates = pd.to_datetime([
        "2026-01-01 00:00:00",
        "2026-06-01 00:00:00",
        "2026-06-08 00:00:00",
        "2026-06-15 00:00:00",
    ])

    return pd.DataFrame({
        "timestamp": dates,
        "open": [80.0, 100.0, 90.0, 90.0],
        "high": [82.0, 101.0, 91.0, 91.0],
        "low": [79.0, 89.0, 89.0, 89.0],
        "close": [81.0, 90.0, 90.0, 90.0],
    })


def make_open_levels_far_below_month_case():
    """TAO/ZKP-like: price is below M open but far outside the near-zone."""
    timestamps = pd.date_range("2026-06-15 00:00:00", periods=8, freq="1h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [92.0, 93.0, 94.0, 95.0, 95.5, 95.7, 95.6, 95.4],
        "high": [93.0, 94.0, 95.0, 96.0, 96.2, 96.1, 96.0, 95.7],
        "low": [91.5, 92.5, 93.5, 94.5, 95.0, 95.0, 94.8, 94.7],
        "close": [92.8, 93.8, 94.8, 95.5, 95.4, 95.2, 95.1, 95.0],
    })


def make_open_levels_month_reclaimed_case():
    """NEAR-like: an older M-open rejection existed, but price is now above M."""
    timestamps = pd.date_range("2026-06-15 00:00:00", periods=8, freq="1h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [96.0, 97.0, 98.0, 99.0, 99.2, 99.4, 99.0, 101.0],
        "high": [97.0, 98.0, 99.0, 100.3, 100.1, 99.8, 99.5, 102.0],
        "low": [95.5, 96.5, 97.5, 98.5, 98.7, 98.8, 98.6, 100.5],
        "close": [96.8, 97.8, 98.8, 99.2, 99.1, 99.0, 98.9, 101.5],
    })


def make_open_levels_old_month_rejection_then_closed_above_case():
    """BANANAS-like: older M-open rejection, then later closed candles reclaimed M."""
    timestamps = pd.date_range("2026-06-15 00:00:00", periods=9, freq="1h")
    return pd.DataFrame({
        "timestamp": timestamps,
        # M open in make_open_levels_month_only_1d_case is 100.
        "open":  [96.0, 97.0, 98.0, 99.0, 99.4, 100.2, 100.5, 100.4, 99.6],
        "high":  [97.0, 98.0, 99.0, 100.4, 99.8, 100.8, 100.9, 100.7, 99.9],
        "low":   [95.5, 96.5, 97.5, 98.4, 98.8, 99.8, 100.1, 100.0, 99.2],
        "close": [96.8, 97.8, 98.8, 99.1, 99.2, 100.4, 100.6, 100.3, 99.4],
    })


def make_open_levels_useless_like_case():
    """USELESS-like: M open is already reclaimed, so it is not resistance."""
    timestamps = pd.date_range("2026-06-15 00:00:00", periods=9, freq="1h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "open":  [96.0, 97.0, 98.0, 99.0, 99.5, 100.2, 100.4, 100.1, 100.1],
        "high":  [97.0, 98.0, 99.0, 99.85, 99.7, 100.6, 100.5, 100.2, 100.4],
        "low":   [95.5, 96.5, 97.5, 98.4, 98.8, 99.7, 99.9, 99.7, 99.8],
        "close": [96.8, 97.8, 98.8, 99.2, 99.1, 100.3, 100.2, 99.9, 100.2],
    })


def make_open_levels_w_reclaimed_m_not_reached_case():
    """VVV-like: W/D are already reclaimed, while M is still too far above price."""
    timestamps = pd.date_range("2026-06-15 00:00:00", periods=8, freq="1h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [98.0, 99.0, 100.0, 101.0, 101.5, 102.0, 102.2, 102.0],
        "high": [99.0, 100.0, 101.0, 102.0, 102.5, 102.8, 102.6, 102.4],
        "low": [97.5, 98.5, 99.5, 100.5, 101.0, 101.5, 101.8, 101.7],
        "close": [98.8, 99.8, 100.8, 101.6, 102.0, 102.2, 102.1, 102.0],
    })


def make_open_levels_vvv_1d_case():
    dates = pd.to_datetime([
        "2026-01-01 00:00:00",
        "2026-06-01 00:00:00",
        "2026-06-08 00:00:00",
        "2026-06-15 00:00:00",
    ])

    return pd.DataFrame({
        "timestamp": dates,
        "open": [80.0, 110.0, 100.0, 100.0],
        "high": [82.0, 111.0, 101.0, 101.0],
        "low": [79.0, 99.0, 99.0, 99.0],
        "close": [81.0, 100.0, 100.0, 100.0],
    })


def make_open_levels_grass_1h_case():
    """1H confirms M open, but 4H also confirms and must be preferred."""
    timestamps = pd.date_range("2026-06-15 00:00:00", periods=8, freq="1h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [92.0, 94.0, 96.0, 98.0, 99.0, 99.4, 99.5, 99.0],
        "high": [93.0, 95.0, 97.0, 99.0, 99.7, 100.3, 100.2, 99.5],
        "low": [91.5, 93.5, 95.5, 97.5, 98.6, 98.9, 98.7, 98.5],
        "close": [92.8, 94.8, 96.8, 98.5, 99.2, 99.1, 98.9, 98.8],
    })


def make_open_levels_grass_4h_case():
    """Closed 4H candle rejects M open; latest row is live/current."""
    timestamps = pd.date_range("2026-06-14 00:00:00", periods=8, freq="4h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [92.0, 94.0, 96.0, 98.0, 99.0, 99.4, 99.6, 99.0],
        "high": [93.0, 95.0, 97.0, 99.0, 99.5, 100.5, 100.4, 99.4],
        "low": [91.5, 93.5, 95.5, 97.5, 98.5, 98.7, 98.5, 98.4],
        "close": [92.8, 94.8, 96.8, 98.7, 99.1, 99.0, 98.8, 98.7],
    })

def run_rejection_self_tests():
    print("\n" + "=" * 120)
    print("RUNNING REJECTION SELF TESTS")
    print("SCRIPT VERSION:", SCRIPT_VERSION)
    print("=" * 120)

    df_1d = make_open_levels_1d_case()

    tests = [
        (
            "4H upper rejection at W/M open",
            detect_rejection_candle(
                df_1h=make_open_levels_1h_confirm_case(),
                df_4h=make_open_levels_4h_test_case(),
                df_1d=df_1d,
                current_price=99.4,
            ),
            "confirmed",
            "4H rejection at W/M open",
        ),
        (
            "Open test without rejection candle is ignored",
            detect_rejection_candle(
                df_1h=make_open_levels_1h_confirm_case(),
                df_4h=make_open_levels_4h_test_without_rejection_case(),
                df_1d=df_1d,
                current_price=99.4,
            ),
            "not_confirmed",
            "none",
        ),
        (
            "1H upper rejection at W/M open",
            detect_rejection_candle(
                df_1h=make_open_levels_1h_htf_rejection_case(),
                df_4h=make_open_levels_4h_far_case(),
                df_1d=df_1d,
                current_price=99.3,
            ),
            "confirmed",
            "1H rejection at W/M open",
        ),
        (
            "D open rejection invalidated after price reclaimed the level",
            detect_rejection_candle(
                df_1h=make_open_levels_d_rejected_then_reclaimed_case(),
                df_4h=make_open_levels_4h_far_case(),
                df_1d=df_1d,
                current_price=96.0,
            ),
            "not_confirmed",
            "none",
        ),
        (
            "1H bearish failure near D open",
            detect_rejection_candle(
                df_1h=make_open_levels_1h_near_d_rejection_case(),
                df_4h=make_open_levels_4h_far_case(),
                df_1d=df_1d,
                current_price=93.2,
            ),
            "confirmed",
            "1H rejection near D open",
        ),
        (
            "Live 1H test at W/M open is not closed rejection",
            detect_rejection_candle(
                df_1h=make_open_levels_1h_htf_live_test_case(),
                df_4h=make_open_levels_4h_far_case(),
                df_1d=df_1d,
                current_price=99.3,
            ),
            "not_confirmed",
            "none",
        ),
        (
            "TAO/ZKP-like M open too far above high",
            detect_rejection_candle(
                df_1h=make_open_levels_far_below_month_case(),
                df_4h=make_open_levels_far_below_month_case(),
                df_1d=make_open_levels_month_only_1d_case(),
                current_price=95.0,
            ),
            "not_confirmed",
            "none",
        ),
        (
            "NEAR-like M open rejection invalid after reclaim",
            detect_rejection_candle(
                df_1h=make_open_levels_month_reclaimed_case(),
                df_4h=make_open_levels_month_reclaimed_case(),
                df_1d=make_open_levels_month_only_1d_case(),
                current_price=101.5,
            ),
            "not_confirmed",
            "none",
        ),
        (
            "BANANAS-like old M rejection invalid after later closes above M",
            detect_rejection_candle(
                df_1h=make_open_levels_old_month_rejection_then_closed_above_case(),
                df_4h=make_open_levels_far_below_month_case(),
                df_1d=make_open_levels_month_only_1d_case(),
                current_price=99.4,
            ),
            "not_confirmed",
            "none",
        ),
        (
            "USELESS-like near M rejection invalid after later reclaim",
            detect_rejection_candle(
                df_1h=make_open_levels_useless_like_case(),
                df_4h=make_open_levels_far_below_month_case(),
                df_1d=make_open_levels_month_only_1d_case(),
                current_price=99.4,
            ),
            "not_confirmed",
            "none",
        ),
        (
            "VVV-like W reclaimed and M not reached",
            detect_rejection_candle(
                df_1h=make_open_levels_w_reclaimed_m_not_reached_case(),
                df_4h=make_open_levels_w_reclaimed_m_not_reached_case(),
                df_1d=make_open_levels_vvv_1d_case(),
                current_price=102.0,
            ),
            "not_confirmed",
            "none",
        ),
        (
            "GRASS-like 4H rejection has priority over 1H",
            detect_rejection_candle(
                df_1h=make_open_levels_grass_1h_case(),
                df_4h=make_open_levels_grass_4h_case(),
                df_1d=make_open_levels_month_only_1d_case(),
                current_price=98.8,
            ),
            "confirmed",
            "4H rejection at M open",
        ),
    ]

    failed = 0

    for name, result, expected_status, expected_detail in tests:
        actual_status = str(result.get("status"))
        detail = str(result.get("detail", ""))
        ok = actual_status == expected_status and expected_detail in detail
        status_text = "PASS" if ok else "FAIL"
        print(f"{status_text} | {name} | expected={expected_status} actual={actual_status} | detail={detail}")

        if not ok:
            failed += 1

    if failed > 0:
        raise AssertionError(f"Rejection self-tests failed: {failed}/{len(tests)}")

    print(f"REJECTION SELF TESTS PASSED: {len(tests)}/{len(tests)}")


def run_rsi_entry_filter_self_tests():
    print("\n" + "=" * 120)
    print("RUNNING RSI ENTRY FILTER SELF TESTS")
    print("SCRIPT VERSION:", SCRIPT_VERSION)
    print("=" * 120)

    tests = [
        ("1H live passes", 65.0, 50.0, 50.0, True),
        ("1H closed passes", 50.0, 65.0, 50.0, True),
        ("4H live passes", 50.0, 50.0, 68.0, True),
        ("weak RSI rejected", 64.9, 64.9, 67.9, False),
    ]

    failed = 0

    for name, rsi_1h_live, rsi_1h_closed, rsi_4h_live, expected in tests:
        result = evaluate_rsi_entry_filter(rsi_1h_live, rsi_1h_closed, rsi_4h_live)
        actual = bool(result.get("passed"))
        ok = actual == expected
        status_text = "PASS" if ok else "FAIL"
        print(f"{status_text} | {name} | expected={expected} actual={actual} | reason={result.get('reason')}")

        if not ok:
            failed += 1

    if failed > 0:
        raise AssertionError(f"RSI entry filter self-tests failed: {failed}/{len(tests)}")

    print(f"RSI ENTRY FILTER SELF TESTS PASSED: {len(tests)}/{len(tests)}")



def run_open_interest_self_tests():
    print("RUNNING OPEN INTEREST SELF TESTS")
    print("SCRIPT VERSION:", SCRIPT_VERSION)

    tests = []
    tests.append(("long_build_up_4h", interpret_open_interest_context(2.5, 12.0) == "Long build-up"))
    tests.append(("short_squeeze_unwind", interpret_open_interest_context(-1.0, -7.0) == "Short squeeze / OI unwind"))
    tests.append(("weak_confirmation", interpret_open_interest_context(0.7, 1.5) == "weak OI confirmation"))
    tests.append((
        "format_line",
        format_oi_line_for_telegram({
            "oi_change_1h_percent": 8.44,
            "oi_change_4h_percent": 21.74,
            "oi_context": "Long build-up",
        }) == "OI: 1H +8.4% | 4H +21.7% | Long build-up",
    ))
    tests.append((
        "format_current_only_line",
        format_oi_line_for_telegram({
            "oi_current": 3293570,
            "oi_change_1h_percent": None,
            "oi_change_4h_percent": None,
            "oi_context": "N/A",
        }) == "OI: current 3.29M | history N/A",
    ))
    bitget_sample = {
        "openInterestList": [
            {"symbol": "BTCUSDT", "size": "34278.06"},
        ],
        "ts": "1695796781616",
    }
    tests.append((
        "bitget_nested_current_oi",
        abs(extract_oi_value_from_row(bitget_sample) - 34278.06) < 0.0001,
    ))

    history = pd.DataFrame([
        {"timestamp": pd.Timestamp("2026-06-15 10:00:00"), "open_interest": 100.0},
        {"timestamp": pd.Timestamp("2026-06-15 11:00:00"), "open_interest": 110.0},
        {"timestamp": pd.Timestamp("2026-06-15 12:00:00"), "open_interest": 121.0},
        {"timestamp": pd.Timestamp("2026-06-15 13:00:00"), "open_interest": 130.0},
        {"timestamp": pd.Timestamp("2026-06-15 14:00:00"), "open_interest": 150.0},
    ])
    metrics = build_open_interest_metrics(history_df=history, current_oi=999999.0, provider="TEST")
    tests.append((
        "history_change_1h",
        metrics.get("oi_change_1h_percent") is not None and abs(metrics.get("oi_change_1h_percent") - 15.3846153846) < 0.0001,
    ))
    tests.append((
        "history_change_4h",
        metrics.get("oi_change_4h_percent") is not None and abs(metrics.get("oi_change_4h_percent") - 50.0) < 0.0001,
    ))

    failed = 0
    for name, passed in tests:
        if passed:
            print(f"✅ {name}")
        else:
            failed += 1
            print(f"❌ {name}")

    if failed:
        raise AssertionError(f"Open Interest self-tests failed: {failed}/{len(tests)}")

    print(f"OPEN INTEREST SELF TESTS PASSED: {len(tests)}/{len(tests)}")


def main():
    if "--self-test" in sys.argv:
        run_sweep_self_tests()
        run_open_levels_self_tests()
        run_open_levels_classification_self_tests()
        run_rejection_self_tests()
        run_local_high_self_tests()
        run_rsi_entry_filter_self_tests()
        run_open_interest_self_tests()
        return

    get_telegram_credentials()
    run_multi_provider_screener()


if __name__ == "__main__":
    main()
