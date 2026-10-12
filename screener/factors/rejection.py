import pandas as pd
import numpy as np

from screener.common import (
    get_live_candle_for_analysis,
    make_factor,
    safe_float,
)
from screener.config import (
    OPEN_LEVEL_NEAR_THRESHOLDS,
    REJECTION_MAX_CLOSE_POSITION_PCT,
    REJECTION_MIN_UPPER_WICK_RANGE_PCT,
    REJECTION_RECENT_WINDOW_1H_BARS,
    REJECTION_RECENT_WINDOW_4H_BARS,
)
from screener.factors.open_levels import (
    calculate_open_levels,
    format_open_level_group_prices,
    get_candle_close_time_value,
    get_previous_candle,
    get_recent_closed_candles_for_analysis,
    open_level_event_is_active_resistance,
    open_level_reclaimed_by_current_price,
    open_level_source_is_allowed_for_signal,
    open_level_tested_from_below,
)


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
