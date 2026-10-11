import pandas as pd
import numpy as np

from screener.common import (
    safe_float,
    percent_change_from_values,
    get_history_value_at_or_before,
)
from screener.config import (
    OI_CHANGE_FLAT_THRESHOLD_PCT,
    OI_CHANGE_1H_ACTIVE_THRESHOLD_PCT,
    OI_CHANGE_STRONG_4H_THRESHOLD_PCT,
    OI_CHANGE_NEGATIVE_THRESHOLD_PCT,
)


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
            "oi_history_df": history,
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
            "oi_history_df": pd.DataFrame(columns=["timestamp", "open_interest"]),
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
        "oi_history_df": pd.DataFrame(columns=["timestamp", "open_interest"]),
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

    context = str(detail.get("oi_context", "N/A") or "N/A")
    change_1h = detail.get("oi_change_1h_percent")
    change_4h = detail.get("oi_change_4h_percent")

    # Raw weak/mixed OI lines are hidden because they add noise. Show only
    # actionable context that can help interpret a pump.
    if context == "Long build-up":
        return f"OI: Long build-up ({format_oi_percent_for_telegram(change_1h)} 1H / {format_oi_percent_for_telegram(change_4h)} 4H)"

    if context == "Short squeeze / OI unwind":
        return f"OI: OI unwind ({format_oi_percent_for_telegram(change_1h)} 1H / {format_oi_percent_for_telegram(change_4h)} 4H)"

    return ""
