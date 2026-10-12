import pandas as pd
import numpy as np

from screener.common import (
    format_debug_value,
    format_price_2,
    get_candle_time,
    get_candle_timestamp_value,
    get_last_closed_candle_for_analysis,
    get_live_candle_for_analysis,
    get_timeframe_hours,
    make_factor,
)
from screener.config import (
    LIQUIDITY_SWEEP_EQUAL_HIGH_TOLERANCE_PCT,
    LIQUIDITY_SWEEP_LEVEL_LOOKBACK_1H,
    LIQUIDITY_SWEEP_LEVEL_LOOKBACK_4H,
    LIQUIDITY_SWEEP_LIVE_INVALIDATION_PCT,
    LIQUIDITY_SWEEP_LOOKBACKS,
    LIQUIDITY_SWEEP_MIN_BREAK_PCT,
    LIQUIDITY_SWEEP_MIN_CLOSE_BACK_PCT,
    LIQUIDITY_SWEEP_MIN_LEVEL_AGE_BARS,
    LIQUIDITY_SWEEP_MIN_REACTION_PCT,
    LIQUIDITY_SWEEP_MIN_SINGLE_HIGH_PROMINENCE_PCT,
    LIQUIDITY_SWEEP_MIN_UPPER_WICK_RANGE_PCT,
    LIQUIDITY_SWEEP_SWING_LEFT_BARS,
    LIQUIDITY_SWEEP_SWING_RIGHT_BARS,
)


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
