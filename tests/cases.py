import pandas as pd

from screener.common import make_factor
from screener.config import OPEN_LEVEL_CONTEXT_WEIGHTS
from screener.factors.open_levels import build_open_levels_detail


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


def make_classification_rsi_divergence_factor():
    factor = make_factor(
        key="rsi_divergence",
        label="RSI divergence",
        status="confirmed",
        points=0,
        detail="1H RSI bearish",
    )
    factor["events"] = [{"status": "confirmed", "timeframe": "1H", "detail": "1H RSI bearish"}]
    return factor


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
