import pandas as pd
import numpy as np

from screener.common import (
    get_candle_timestamp_value,
    get_live_candle_for_analysis,
    make_factor,
    safe_float,
)
from screener.config import (
    KYIV_TZ,
    OPEN_LEVEL_CONTEXT_WEIGHTS,
    OPEN_LEVEL_NEAR_THRESHOLDS,
    OPEN_LEVEL_RECENT_WINDOW_1H_BARS,
    OPEN_LEVEL_RECENT_WINDOW_4H_BARS,
    OPEN_LEVEL_RECLAIM_INVALIDATION_PCT,
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
    exchange/TradingView UTC session opens better than exchange 1D candles or
    Kyiv-local period starts.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    if "timestamp" not in df.columns or "open" not in df.columns:
        return pd.DataFrame()

    work_df = df.copy().reset_index(drop=True)
    timestamps = pd.to_datetime(work_df["timestamp"], errors="coerce")

    try:
        # Bybit timestamps are UTC but stored as timezone-naive values.
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

    With an explicit confirm flag, use the latest confirm=1 candles.
    Without confirm (Bybit), treat the last row as live and use
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
