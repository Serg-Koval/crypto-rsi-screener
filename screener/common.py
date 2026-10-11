import requests
import pandas as pd
import numpy as np

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from screener.config import RSI_PERIOD


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


def safe_get_json(base_url, endpoint, params=None, provider_name="provider", headers=None):
    if params is None:
        params = {}

    url = base_url + endpoint
    response = SESSION.get(url, params=params, headers=headers, timeout=20)

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


def safe_float(value, default=np.nan):
    try:
        if value is None or pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


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


def get_last_closed_candle_for_analysis(df):
    """
    Return the last closed candle for factor analysis.

    A dataframe may carry an explicit confirm flag:
    - confirm = 0 means live candle;
    - confirm = 1 means closed candle.

    Bybit candles do not expose confirm in the normalized dataframe.
    In that case, the safe assumption is:
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


def get_candle_timestamp_value(candle):
    try:
        if candle is not None and "timestamp" in candle.index and not pd.isna(candle["timestamp"]):
            return pd.to_datetime(candle["timestamp"])
    except Exception:
        return None

    return None


def get_short_factor(short_factors, key):
    if not isinstance(short_factors, list):
        return None

    for factor in short_factors:
        if isinstance(factor, dict) and factor.get("key") == key:
            return factor

    return None


def percent_change_from_values(current_value, previous_value):
    current = safe_float(current_value, default=np.nan)
    previous = safe_float(previous_value, default=np.nan)

    if pd.isna(current) or pd.isna(previous) or float(previous) == 0:
        return None

    return float(((float(current) - float(previous)) / float(previous)) * 100)


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
