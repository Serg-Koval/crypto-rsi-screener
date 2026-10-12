"""Boundary self-tests for calculation thresholds (r37_16).

Each threshold is checked with one point slightly inside and one slightly
outside the boundary (never exactly on it, to stay clear of float noise).
"""

import pandas as pd

from screener.classification import calculate_scores
from screener.config import SCRIPT_VERSION
from screener.factors.open_levels import (
    detect_open_levels_context,
    evaluate_open_level_context_over_window,
)
from screener.factors.rejection import (
    candle_has_open_level_rejection_character,
    candle_has_upper_rejection,
    detect_rejection_candle,
)
from screener.factors.sweep import candle_has_sweep_failure_character


def _candle(open_price, high, low, close):
    return pd.Series({"open": open_price, "high": high, "low": low, "close": close})


def _scores(**overrides):
    """calculate_scores with neutral arguments; override only what is tested."""

    args = {
        "rsi_1h_live": 50,
        "rsi_1h_closed": 50,
        "rsi_4h_live": 50,
        "exact_volume_24h": 0,
        "volume_change_24h": 0,
        "price_change_24h": 0,
        "short_setup_score": 0,
    }
    args.update(overrides)
    return calculate_scores(**args)


def _near_event_state(label, distance_pct):
    event = evaluate_open_level_context_over_window(
        label=label,
        level_price=100.0,
        current_price=100.0 * (1 - distance_pct / 100.0),
        closed_candles=[],
        df_for_previous=None,
        window_tf="1H",
        window_bars=6,
        live_candle=None,
        live_index=None,
    )
    return None if event is None else event.get("state")


# Synthetic series for the freshness-window tests.
# Level (period open) = 100. The qualifying candle touches the level from
# below and closes under it with a bearish body and a long upper wick, so it
# is both an open-level "tested" candle and a rejection candle.
_FILLER = (97.9, 98.2, 97.8, 98.0)
_QUALIFIED = (99.5, 100.5, 98.5, 99.0)
_LEVEL_CANDLE = (100.0, 100.0, 97.9, 98.0)  # first candle: defines the open = 100


def _window_df(start, freq, total, qualified_back):
    """Build a df of `total` candles; last one is live.

    qualified_back = position of the qualifying candle counted from the end
    among CLOSED candles (1 = last closed, 6 = sixth from the end).
    """

    rows = [_FILLER] * total
    rows[0] = _LEVEL_CANDLE
    rows[total - 1 - qualified_back] = _QUALIFIED
    rows[total - 1] = (98.0, 98.4, 97.8, 98.0)  # live candle, far below the level
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=total, freq=freq),
        "open": [r[0] for r in rows],
        "high": [r[1] for r in rows],
        "low": [r[2] for r in rows],
        "close": [r[3] for r in rows],
    })


def run_threshold_self_tests():
    print("\n" + "=" * 120)
    print("RUNNING THRESHOLD SELF TESTS")
    print("SCRIPT VERSION:", SCRIPT_VERSION)
    print("=" * 120)

    # (name, actual, expected)
    tests = []

    # A. Sweep failure character: upper wick >= 30% of range.
    # Bullish candle (close > open) with close above the midpoint, range 100.
    tests.append((
        "A sweep failure: bullish candle, upper wick 31% -> True",
        candle_has_sweep_failure_character(_candle(120, 200, 100, 169)),
        True,
    ))
    tests.append((
        "A sweep failure: bullish candle, upper wick 29% -> False",
        candle_has_sweep_failure_character(_candle(120, 200, 100, 171)),
        False,
    ))

    # B. Rejection: upper wick >= 35% and close in lower <= 55% of range.
    tests.append((
        "B upper rejection: wick 35.5%, close at 50% -> True",
        candle_has_upper_rejection(_candle(164.5, 200, 100, 150)),
        True,
    ))
    tests.append((
        "B upper rejection: wick 34.5%, close at 50% -> False",
        candle_has_upper_rejection(_candle(165.5, 200, 100, 150)),
        False,
    ))
    tests.append((
        "B upper rejection: wick 40%, close at 54% -> True",
        candle_has_upper_rejection(_candle(160, 200, 100, 154)),
        True,
    ))
    tests.append((
        "B upper rejection: wick 40%, close at 56% -> False",
        candle_has_upper_rejection(_candle(160, 200, 100, 156)),
        False,
    ))
    # Bearish candles with a small (20%) upper wick, so only the
    # bearish-close rule can make them pass.
    tests.append((
        "B open-level character tested: bearish, close at 64.5% -> True",
        candle_has_open_level_rejection_character(_candle(180, 200, 100, 164.5), state="tested"),
        True,
    ))
    tests.append((
        "B open-level character tested: bearish, close at 65.5% -> False",
        candle_has_open_level_rejection_character(_candle(180, 200, 100, 165.5), state="tested"),
        False,
    ))
    tests.append((
        "B open-level character near: bearish, close at 54.5% -> True",
        candle_has_open_level_rejection_character(_candle(180, 200, 100, 154.5), state="near"),
        True,
    ))
    tests.append((
        "B open-level character near: bearish, close at 55.5% -> False",
        candle_has_open_level_rejection_character(_candle(180, 200, 100, 155.5), state="near"),
        False,
    ))

    # C. Near thresholds per open level (distance below the level, in %).
    for label, inside, outside in [
        ("D", 0.145, 0.155),
        ("W", 0.345, 0.355),
        ("M", 0.495, 0.505),
        ("Y", 0.495, 0.505),
    ]:
        tests.append((
            f"C near {label}: {inside}% below -> near",
            _near_event_state(label, inside),
            "near",
        ))
        tests.append((
            f"C near {label}: {outside}% below -> None",
            _near_event_state(label, outside),
            None,
        ))

    # D. Freshness windows, end to end. The last candle is live.
    # 1H: D open is the first 1H candle of the UTC day (2026-06-11 00:00).
    # 4H: W open is the first 4H candle of the UTC week (Monday 2026-06-08).
    for back, expected in [(6, "confirmed"), (7, "not_confirmed")]:
        df_1h = _window_df("2026-06-11 00:00:00", "1h", 10, back)
        tests.append((
            f"D open levels 1H window: qualified candle #{back} from end -> {expected}",
            detect_open_levels_context(df_1h=df_1h, current_price=98.0).get("status"),
            expected,
        ))
        tests.append((
            f"D rejection 1H window: qualified candle #{back} from end -> {expected}",
            detect_rejection_candle(df_1h=df_1h, current_price=98.0).get("status"),
            expected,
        ))

    for back, expected in [(2, "confirmed"), (3, "not_confirmed")]:
        df_4h = _window_df("2026-06-08 00:00:00", "4h", 8, back)
        tests.append((
            f"D open levels 4H window: qualified candle #{back} from end -> {expected}",
            detect_open_levels_context(df_4h=df_4h, current_price=98.0).get("status"),
            expected,
        ))
        tests.append((
            f"D rejection 4H window: qualified candle #{back} from end -> {expected}",
            detect_rejection_candle(df_4h=df_4h, current_price=98.0).get("status"),
            expected,
        ))

    # E. Scores.
    for value, expected in [(9.9, 0), (10.1, 1), (19.9, 1), (20.1, 2), (39.9, 2), (40.1, 3)]:
        tests.append((
            f"E pump_score: 24h change {value} -> {expected}",
            _scores(price_change_24h=value)["pump_score"],
            expected,
        ))

    for field, value, expected in [
        ("rsi_1h_live", 81.9, 0), ("rsi_1h_live", 82.1, 1),
        ("rsi_1h_live", 84.9, 1), ("rsi_1h_live", 85.1, 2),
        ("rsi_4h_live", 74.9, 0), ("rsi_4h_live", 75.1, 1),
        ("rsi_4h_live", 79.9, 1), ("rsi_4h_live", 80.1, 2),
        ("rsi_1h_closed", 79.9, 0), ("rsi_1h_closed", 80.1, 1),
    ]:
        tests.append((
            f"E rsi_score: {field} {value} -> {expected}",
            _scores(**{field: value})["rsi_score"],
            expected,
        ))

    for value, expected in [(4_999_999, 0), (5_000_000, 1)]:
        tests.append((
            f"E volume_score: turnover {value} -> {expected}",
            _scores(exact_volume_24h=value, volume_change_24h=None)["volume_score"],
            expected,
        ))

    for value, expected in [
        (49.9, 0), (50.1, 1), (99.9, 1), (100.1, 2), (299.9, 2), (300.1, 3), (None, 0),
    ]:
        tests.append((
            f"E volume_score: volume change {value} -> {expected}",
            _scores(exact_volume_24h=0, volume_change_24h=value)["volume_score"],
            expected,
        ))

    failed = 0

    for name, actual, expected in tests:
        ok = actual == expected
        status_text = "PASS" if ok else "FAIL"
        print(f"{status_text} | {name} | expected={expected} actual={actual}")

        if not ok:
            failed += 1

    if failed > 0:
        raise AssertionError(f"Threshold self-tests failed: {failed}/{len(tests)}")

    print(f"THRESHOLD SELF TESTS PASSED: {len(tests)}/{len(tests)}")
