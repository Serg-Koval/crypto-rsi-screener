import pandas as pd
import numpy as np

from screener.common import (
    get_history_value_at_or_before,
    get_short_factor,
    get_timeframe_hours,
    make_factor,
    percent_change_from_values,
    safe_float,
)
from screener.config import (
    OI_DIVERGENCE_EQUAL_HIGH_TOLERANCE_PCT,
    OI_DIVERGENCE_LOOKBACK_1H,
    OI_DIVERGENCE_LOOKBACK_4H,
    OI_DIVERGENCE_MIN_OI_DROP_PCT,
    OI_DIVERGENCE_MIN_PRICE_HH_PCT,
    OI_DIVERGENCE_MIN_SWING_DISTANCE_1H,
    OI_DIVERGENCE_MIN_SWING_DISTANCE_4H,
    OI_DIVERGENCE_RECENT_WINDOW_1H,
    OI_DIVERGENCE_RECENT_WINDOW_4H,
    OI_DIVERGENCE_WEAK_SUPPORT_MAX_OI_CHANGE_PCT,
    RSI_DIVERGENCE_EQUAL_HIGH_TOLERANCE_PCT,
    RSI_DIVERGENCE_LOOKBACK_1H,
    RSI_DIVERGENCE_LOOKBACK_4H,
    RSI_DIVERGENCE_MIN_PRICE_HH_PCT,
    RSI_DIVERGENCE_MIN_RSI_DROP,
    RSI_DIVERGENCE_MIN_SWING_DISTANCE_1H,
    RSI_DIVERGENCE_MIN_SWING_DISTANCE_4H,
    RSI_DIVERGENCE_RECENT_WINDOW_1H,
    RSI_DIVERGENCE_RECENT_WINDOW_4H,
    RSI_DIVERGENCE_SWING_LEFT_BARS,
    RSI_DIVERGENCE_SWING_RIGHT_BARS,
)


def get_closed_candles_dataframe_for_analysis(df):
    """
    Return a dataframe of closed candles only.

    If a confirm column exists, confirm=1 marks closed candles. Bybit candles
    have no such column, so the latest row is treated as live and excluded.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    work_df = df.copy().reset_index(drop=True)

    if "confirm" in work_df.columns:
        closed_df = work_df[work_df["confirm"] == 1].copy()
    else:
        if len(work_df) < 2:
            return pd.DataFrame()
        closed_df = work_df.iloc[:-1].copy()

    return closed_df.reset_index(drop=True)


def collect_rsi_price_swing_highs(df, lookback=None):
    """
    Collect confirmed price swing highs with their RSI values.

    This is intentionally simple and conservative. It is used only as context,
    not as a trade trigger.
    """

    if df is None or df.empty:
        return []

    required_columns = {"high", "rsi"}
    if not required_columns.issubset(set(df.columns)):
        return []

    work_df = df.copy().reset_index(drop=True)
    if lookback is not None and len(work_df) > int(lookback):
        offset = len(work_df) - int(lookback)
        work_df = work_df.iloc[offset:].copy().reset_index(drop=True)
    else:
        offset = 0

    left = int(RSI_DIVERGENCE_SWING_LEFT_BARS)
    right = int(RSI_DIVERGENCE_SWING_RIGHT_BARS)

    if len(work_df) <= left + right + 1:
        return []

    highs = pd.to_numeric(work_df["high"], errors="coerce")
    rsis = pd.to_numeric(work_df["rsi"], errors="coerce")
    swings = []

    for i in range(left, len(work_df) - right):
        high_value = highs.iloc[i]
        rsi_value = rsis.iloc[i]

        if pd.isna(high_value) or pd.isna(rsi_value):
            continue

        left_window = highs.iloc[i - left:i]
        right_window = highs.iloc[i + 1:i + right + 1]

        if left_window.dropna().empty or right_window.dropna().empty:
            continue

        is_swing_high = (
            float(high_value) >= float(left_window.max()) and
            float(high_value) > float(right_window.max())
        )

        if not is_swing_high:
            continue

        timestamp = None
        if "timestamp" in work_df.columns:
            try:
                timestamp = pd.to_datetime(work_df.iloc[i]["timestamp"])
            except Exception:
                timestamp = None

        swings.append({
            "index": int(i + offset),
            "local_index": int(i),
            "timestamp": timestamp,
            "price_high": float(high_value),
            "rsi_high": float(rsi_value),
        })

    return swings


def get_divergence_min_swing_distance(timeframe):
    return int(RSI_DIVERGENCE_MIN_SWING_DISTANCE_4H if str(timeframe) == "4H" else RSI_DIVERGENCE_MIN_SWING_DISTANCE_1H)


def price_high_relation(candidate_high, prior_high, min_hh_pct=None, equal_tolerance_pct=None):
    if min_hh_pct is None:
        min_hh_pct = RSI_DIVERGENCE_MIN_PRICE_HH_PCT
    if equal_tolerance_pct is None:
        equal_tolerance_pct = RSI_DIVERGENCE_EQUAL_HIGH_TOLERANCE_PCT

    candidate = safe_float(candidate_high, default=np.nan)
    prior = safe_float(prior_high, default=np.nan)

    if pd.isna(candidate) or pd.isna(prior) or float(prior) <= 0:
        return "none"

    candidate = float(candidate)
    prior = float(prior)

    if candidate >= prior * (1 + float(min_hh_pct)):
        return "higher_high"

    if abs(candidate - prior) / prior <= float(equal_tolerance_pct):
        return "equal_high"

    return "none"


def detect_rsi_bearish_divergence_for_timeframe(df, timeframe="1H"):
    """
    Detect bearish RSI divergence:
    - classic: price higher high + RSI lower high;
    - equal-high: price equal/near-equal high + RSI lower high.

    The two highs must be separated by a minimum number of candles so that the
    model does not call adjacent-candle noise a divergence.
    """

    closed_df = get_closed_candles_dataframe_for_analysis(df)

    if closed_df is None or closed_df.empty:
        return {"status": "not_enough_data", "timeframe": str(timeframe), "detail": "no closed candles"}

    if "high" not in closed_df.columns or "rsi" not in closed_df.columns:
        return {"status": "not_enough_data", "timeframe": str(timeframe), "detail": "high/RSI unavailable"}

    lookback = RSI_DIVERGENCE_LOOKBACK_4H if str(timeframe) == "4H" else RSI_DIVERGENCE_LOOKBACK_1H
    recent_window = RSI_DIVERGENCE_RECENT_WINDOW_4H if str(timeframe) == "4H" else RSI_DIVERGENCE_RECENT_WINDOW_1H
    min_distance = get_divergence_min_swing_distance(timeframe)

    if len(closed_df) < max(20, recent_window + RSI_DIVERGENCE_SWING_LEFT_BARS + RSI_DIVERGENCE_SWING_RIGHT_BARS + min_distance + 1):
        return {"status": "not_enough_data", "timeframe": str(timeframe), "detail": "not enough closed candles"}

    work_df = closed_df.copy().reset_index(drop=True)
    if len(work_df) > int(lookback):
        base_offset = len(work_df) - int(lookback)
        analysis_df = work_df.iloc[base_offset:].copy().reset_index(drop=True)
    else:
        base_offset = 0
        analysis_df = work_df.copy().reset_index(drop=True)

    if analysis_df.empty:
        return {"status": "not_enough_data", "timeframe": str(timeframe), "detail": "analysis window empty"}

    highs = pd.to_numeric(analysis_df["high"], errors="coerce")
    rsis = pd.to_numeric(analysis_df["rsi"], errors="coerce")

    window = min(max(1, int(recent_window)), len(analysis_df))
    recent_slice = highs.iloc[-window:]

    if recent_slice.dropna().empty:
        return {"status": "not_enough_data", "timeframe": str(timeframe), "detail": "recent highs unavailable"}

    candidate_local_index = int(recent_slice.idxmax())
    candidate_high = safe_float(highs.iloc[candidate_local_index], default=np.nan)
    candidate_rsi = safe_float(rsis.iloc[candidate_local_index], default=np.nan)

    if pd.isna(candidate_high) or pd.isna(candidate_rsi):
        return {"status": "not_enough_data", "timeframe": str(timeframe), "detail": "recent high/RSI unavailable"}

    candidate_global_index = int(candidate_local_index + base_offset)
    candidate_time = None
    if "timestamp" in analysis_df.columns:
        try:
            candidate_time = pd.to_datetime(analysis_df.iloc[candidate_local_index]["timestamp"])
        except Exception:
            candidate_time = None

    swings = collect_rsi_price_swing_highs(analysis_df, lookback=None)
    prior_swings = [
        swing for swing in swings
        if int(swing.get("local_index", -1)) <= candidate_local_index - int(min_distance)
    ]

    if not prior_swings:
        return {
            "status": "not_confirmed",
            "timeframe": str(timeframe),
            "detail": "no prior swing high with required distance",
            "candidate_price_high": float(candidate_high),
            "candidate_rsi_high": float(candidate_rsi),
            "min_distance_bars": int(min_distance),
        }

    prior_swings = sorted(prior_swings, key=lambda item: int(item.get("local_index", 0)), reverse=True)

    for prior in prior_swings:
        prior_high = safe_float(prior.get("price_high"), default=np.nan)
        prior_rsi = safe_float(prior.get("rsi_high"), default=np.nan)

        if pd.isna(prior_high) or pd.isna(prior_rsi) or float(prior_high) <= 0:
            continue

        relation = price_high_relation(candidate_high, prior_high)
        rsi_lower_high = float(candidate_rsi) <= float(prior_rsi) - float(RSI_DIVERGENCE_MIN_RSI_DROP)

        if relation in ("higher_high", "equal_high") and rsi_lower_high:
            detail = f"{timeframe} RSI bearish" if relation == "higher_high" else f"{timeframe} RSI bearish equal high"
            return {
                "status": "confirmed",
                "timeframe": str(timeframe),
                "detail": detail,
                "divergence_type": str(relation),
                "candidate_index": int(candidate_global_index),
                "candidate_time": candidate_time,
                "candidate_price_high": float(candidate_high),
                "candidate_rsi_high": float(candidate_rsi),
                "prior_index": int(prior.get("index", 0) + base_offset),
                "prior_time": prior.get("timestamp"),
                "prior_price_high": float(prior_high),
                "prior_rsi_high": float(prior_rsi),
                "distance_bars": int(candidate_local_index - int(prior.get("local_index", 0))),
                "min_distance_bars": int(min_distance),
                "price_change_pct": float(((float(candidate_high) - float(prior_high)) / float(prior_high)) * 100),
                "rsi_drop": float(float(prior_rsi) - float(candidate_rsi)),
            }

    return {
        "status": "not_confirmed",
        "timeframe": str(timeframe),
        "detail": "no bearish RSI divergence",
        "candidate_price_high": float(candidate_high),
        "candidate_rsi_high": float(candidate_rsi),
        "min_distance_bars": int(min_distance),
    }


def detect_rsi_bearish_divergence(df_1h=None, df_4h=None):
    item_1h = detect_rsi_bearish_divergence_for_timeframe(df_1h, timeframe="1H") if df_1h is not None else {
        "status": "not_enough_data", "timeframe": "1H", "detail": "1H unavailable"
    }
    item_4h = detect_rsi_bearish_divergence_for_timeframe(df_4h, timeframe="4H") if df_4h is not None else {
        "status": "not_enough_data", "timeframe": "4H", "detail": "4H unavailable"
    }

    confirmed = [item for item in [item_4h, item_1h] if item.get("status") == "confirmed"]

    if confirmed:
        detail = " | ".join(str(item.get("detail", "")).strip() for item in confirmed if item.get("detail"))
        factor = make_factor("rsi_divergence", "RSI divergence", "confirmed", points=0, detail=detail)
        factor["events"] = confirmed
        return factor

    if item_1h.get("status") == "not_enough_data" and item_4h.get("status") == "not_enough_data":
        return make_factor("rsi_divergence", "RSI divergence", "not_enough_data", points=0, detail="data unavailable")

    factor = make_factor("rsi_divergence", "RSI divergence", "not_confirmed", points=0, detail="none")
    factor["events"] = [item_4h, item_1h]
    return factor


def has_confirmed_rsi_divergence(short_factors):
    factor = get_short_factor(short_factors, "rsi_divergence") or {}
    return bool(factor.get("status") == "confirmed")


def get_oi_divergence_min_swing_distance(timeframe):
    return int(OI_DIVERGENCE_MIN_SWING_DISTANCE_4H if str(timeframe) == "4H" else OI_DIVERGENCE_MIN_SWING_DISTANCE_1H)


def get_oi_history_value_for_candle(history, candle_time, timeframe="1H"):
    if candle_time is None:
        return np.nan
    try:
        # Use the candle close time for OI alignment. OI snapshots are not
        # guaranteed to be exactly at candle open time.
        target = pd.to_datetime(candle_time) + pd.Timedelta(hours=get_timeframe_hours(timeframe))
        return get_history_value_at_or_before(history, target)
    except Exception:
        return np.nan


def collect_price_oi_swing_highs(df, oi_history, timeframe="1H", lookback=None):
    if df is None or df.empty or oi_history is None or oi_history.empty:
        return []

    if "high" not in df.columns:
        return []

    work_df = df.copy().reset_index(drop=True)
    if lookback is not None and len(work_df) > int(lookback):
        offset = len(work_df) - int(lookback)
        work_df = work_df.iloc[offset:].copy().reset_index(drop=True)
    else:
        offset = 0

    left = int(RSI_DIVERGENCE_SWING_LEFT_BARS)
    right = int(RSI_DIVERGENCE_SWING_RIGHT_BARS)

    if len(work_df) <= left + right + 1:
        return []

    highs = pd.to_numeric(work_df["high"], errors="coerce")
    swings = []

    for i in range(left, len(work_df) - right):
        high_value = highs.iloc[i]

        if pd.isna(high_value):
            continue

        left_window = highs.iloc[i - left:i]
        right_window = highs.iloc[i + 1:i + right + 1]

        if left_window.dropna().empty or right_window.dropna().empty:
            continue

        is_swing_high = (
            float(high_value) >= float(left_window.max()) and
            float(high_value) > float(right_window.max())
        )

        if not is_swing_high:
            continue

        timestamp = None
        if "timestamp" in work_df.columns:
            try:
                timestamp = pd.to_datetime(work_df.iloc[i]["timestamp"])
            except Exception:
                timestamp = None

        oi_value = get_oi_history_value_for_candle(oi_history, timestamp, timeframe=timeframe)

        if pd.isna(oi_value):
            continue

        swings.append({
            "index": int(i + offset),
            "local_index": int(i),
            "timestamp": timestamp,
            "price_high": float(high_value),
            "oi_high": float(oi_value),
        })

    return swings


def detect_oi_bearish_divergence_for_timeframe(df, oi_history, timeframe="1H"):
    """Detect price high without OI confirmation.

    Strong OI bearish divergence:
    - price higher/equal high;
    - OI at the second high is lower by at least OI_DIVERGENCE_MIN_OI_DROP_PCT.

    Weak OI support:
    - price higher/equal high;
    - OI increased less than OI_DIVERGENCE_WEAK_SUPPORT_MAX_OI_CHANGE_PCT.
      This is shown in Telegram but does not count as a strong divergence for
      status upgrades.
    """

    closed_df = get_closed_candles_dataframe_for_analysis(df)

    if closed_df is None or closed_df.empty:
        return {"status": "not_enough_data", "timeframe": str(timeframe), "detail": "no closed candles"}

    history = oi_history.copy() if isinstance(oi_history, pd.DataFrame) else pd.DataFrame()
    if history.empty:
        return {"status": "not_available", "timeframe": str(timeframe), "detail": "OI history unavailable"}

    if "high" not in closed_df.columns:
        return {"status": "not_enough_data", "timeframe": str(timeframe), "detail": "high unavailable"}

    lookback = OI_DIVERGENCE_LOOKBACK_4H if str(timeframe) == "4H" else OI_DIVERGENCE_LOOKBACK_1H
    recent_window = OI_DIVERGENCE_RECENT_WINDOW_4H if str(timeframe) == "4H" else OI_DIVERGENCE_RECENT_WINDOW_1H
    min_distance = get_oi_divergence_min_swing_distance(timeframe)

    if len(closed_df) < max(20, recent_window + RSI_DIVERGENCE_SWING_LEFT_BARS + RSI_DIVERGENCE_SWING_RIGHT_BARS + min_distance + 1):
        return {"status": "not_enough_data", "timeframe": str(timeframe), "detail": "not enough closed candles"}

    work_df = closed_df.copy().reset_index(drop=True)
    if len(work_df) > int(lookback):
        base_offset = len(work_df) - int(lookback)
        analysis_df = work_df.iloc[base_offset:].copy().reset_index(drop=True)
    else:
        base_offset = 0
        analysis_df = work_df.copy().reset_index(drop=True)

    highs = pd.to_numeric(analysis_df["high"], errors="coerce")
    window = min(max(1, int(recent_window)), len(analysis_df))
    recent_slice = highs.iloc[-window:]

    if recent_slice.dropna().empty:
        return {"status": "not_enough_data", "timeframe": str(timeframe), "detail": "recent highs unavailable"}

    candidate_local_index = int(recent_slice.idxmax())
    candidate_high = safe_float(highs.iloc[candidate_local_index], default=np.nan)

    if pd.isna(candidate_high):
        return {"status": "not_enough_data", "timeframe": str(timeframe), "detail": "candidate high unavailable"}

    candidate_time = None
    if "timestamp" in analysis_df.columns:
        try:
            candidate_time = pd.to_datetime(analysis_df.iloc[candidate_local_index]["timestamp"])
        except Exception:
            candidate_time = None

    candidate_oi = get_oi_history_value_for_candle(history, candidate_time, timeframe=timeframe)

    if pd.isna(candidate_oi):
        return {"status": "not_enough_data", "timeframe": str(timeframe), "detail": "candidate OI unavailable"}

    swings = collect_price_oi_swing_highs(analysis_df, history, timeframe=timeframe, lookback=None)
    prior_swings = [
        swing for swing in swings
        if int(swing.get("local_index", -1)) <= candidate_local_index - int(min_distance)
    ]

    if not prior_swings:
        return {
            "status": "not_confirmed",
            "timeframe": str(timeframe),
            "detail": "no prior swing high with OI",
            "candidate_price_high": float(candidate_high),
            "candidate_oi_high": float(candidate_oi),
            "min_distance_bars": int(min_distance),
        }

    prior_swings = sorted(prior_swings, key=lambda item: int(item.get("local_index", 0)), reverse=True)
    weak_candidate = None

    for prior in prior_swings:
        prior_high = safe_float(prior.get("price_high"), default=np.nan)
        prior_oi = safe_float(prior.get("oi_high"), default=np.nan)

        if pd.isna(prior_high) or pd.isna(prior_oi) or float(prior_high) <= 0 or float(prior_oi) <= 0:
            continue

        relation = price_high_relation(
            candidate_high,
            prior_high,
            min_hh_pct=OI_DIVERGENCE_MIN_PRICE_HH_PCT,
            equal_tolerance_pct=OI_DIVERGENCE_EQUAL_HIGH_TOLERANCE_PCT,
        )

        if relation not in ("higher_high", "equal_high"):
            continue

        oi_change_pct = percent_change_from_values(candidate_oi, prior_oi)

        if oi_change_pct is None:
            continue

        base_event = {
            "timeframe": str(timeframe),
            "divergence_type": str(relation),
            "candidate_index": int(candidate_local_index + base_offset),
            "candidate_time": candidate_time,
            "candidate_price_high": float(candidate_high),
            "candidate_oi_high": float(candidate_oi),
            "prior_index": int(prior.get("index", 0) + base_offset),
            "prior_time": prior.get("timestamp"),
            "prior_price_high": float(prior_high),
            "prior_oi_high": float(prior_oi),
            "distance_bars": int(candidate_local_index - int(prior.get("local_index", 0))),
            "min_distance_bars": int(min_distance),
            "price_change_pct": float(((float(candidate_high) - float(prior_high)) / float(prior_high)) * 100),
            "oi_change_pct": float(oi_change_pct),
        }

        if float(oi_change_pct) <= -float(OI_DIVERGENCE_MIN_OI_DROP_PCT):
            detail = f"{timeframe} OI bearish" if relation == "higher_high" else f"{timeframe} OI bearish equal high"
            base_event.update({"status": "confirmed", "detail": detail})
            return base_event

        if float(oi_change_pct) <= float(OI_DIVERGENCE_WEAK_SUPPORT_MAX_OI_CHANGE_PCT) and weak_candidate is None:
            detail = f"{timeframe} weak OI support"
            base_event.update({"status": "weak", "detail": detail})
            weak_candidate = base_event

    if weak_candidate is not None:
        return weak_candidate

    return {
        "status": "not_confirmed",
        "timeframe": str(timeframe),
        "detail": "no OI divergence",
        "candidate_price_high": float(candidate_high),
        "candidate_oi_high": float(candidate_oi),
        "min_distance_bars": int(min_distance),
    }


def detect_oi_bearish_divergence(df_1h=None, df_4h=None, oi_history=None, provider="provider"):
    item_1h = detect_oi_bearish_divergence_for_timeframe(df_1h, oi_history, timeframe="1H") if df_1h is not None else {
        "status": "not_enough_data", "timeframe": "1H", "detail": "1H unavailable"
    }
    item_4h = detect_oi_bearish_divergence_for_timeframe(df_4h, oi_history, timeframe="4H") if df_4h is not None else {
        "status": "not_enough_data", "timeframe": "4H", "detail": "4H unavailable"
    }

    confirmed = [item for item in [item_4h, item_1h] if item.get("status") == "confirmed"]
    weak = [item for item in [item_4h, item_1h] if item.get("status") == "weak"]

    if confirmed:
        detail = " | ".join(str(item.get("detail", "")).strip() for item in confirmed if item.get("detail"))
        factor = make_factor("oi_divergence", "OI divergence", "confirmed", points=0, detail=detail)
        factor["events"] = confirmed + weak
        factor["provider"] = str(provider)
        return factor

    if weak:
        detail = " | ".join(str(item.get("detail", "")).strip() for item in weak if item.get("detail"))
        factor = make_factor("oi_divergence", "OI divergence", "weak", points=0, detail=detail)
        factor["events"] = weak
        factor["provider"] = str(provider)
        return factor

    if item_1h.get("status") in ("not_available", "not_enough_data") and item_4h.get("status") in ("not_available", "not_enough_data"):
        factor = make_factor("oi_divergence", "OI divergence", "not_enough_data", points=0, detail="OI history unavailable")
        factor["events"] = [item_4h, item_1h]
        factor["provider"] = str(provider)
        return factor

    factor = make_factor("oi_divergence", "OI divergence", "not_confirmed", points=0, detail="none")
    factor["events"] = [item_4h, item_1h]
    factor["provider"] = str(provider)
    return factor


def has_confirmed_oi_divergence(short_factors):
    factor = get_short_factor(short_factors, "oi_divergence") or {}
    return bool(factor.get("status") == "confirmed")


def has_any_bearish_divergence(short_factors):
    return bool(has_confirmed_rsi_divergence(short_factors) or has_confirmed_oi_divergence(short_factors))
