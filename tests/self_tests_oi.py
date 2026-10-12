import pandas as pd

from screener.config import SCRIPT_VERSION
from screener.oi_context import (
    build_open_interest_metrics,
    format_oi_line_for_telegram,
    interpret_open_interest_context,
    normalize_oi_history_rows,
)


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
        }) == "OI: Long build-up (+8.4% 1H / +21.7% 4H)",
    ))
    tests.append((
        "format_current_only_line_is_hidden",
        format_oi_line_for_telegram({
            "oi_current": 3293570,
            "oi_change_1h_percent": None,
            "oi_change_4h_percent": None,
            "oi_context": "N/A",
            "oi_source": "Bybit",
        }) == "",
    ))
    tests.append((
        "format_unwind_line",
        format_oi_line_for_telegram({
            "oi_change_1h_percent": -6.0,
            "oi_change_4h_percent": -9.5,
            "oi_context": "Short squeeze / OI unwind",
            "oi_source": "Bybit",
        }) == "OI: OI unwind (-6.0% 1H / -9.5% 4H)",
    ))
    bybit_oi_rows = [
        {"openInterest": "130.5", "timestamp": "1781532000000"},
        {"openInterest": "120.0", "timestamp": "1781528400000"},
        {"openInterest": "100.0", "timestamp": "1781524800000"},
    ]
    bybit_oi_history = normalize_oi_history_rows(bybit_oi_rows)
    tests.append((
        "bybit_oi_history_sorted_oldest_first",
        list(bybit_oi_history["open_interest"]) == [100.0, 120.0, 130.5]
        and bool(bybit_oi_history["timestamp"].is_monotonic_increasing),
    ))
    tests.append((
        "bybit_oi_history_timestamp_ms",
        str(bybit_oi_history["timestamp"].iloc[-1]) == str(pd.to_datetime(1781532000000, unit="ms")),
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
