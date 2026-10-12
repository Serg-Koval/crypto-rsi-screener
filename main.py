import os
import sys
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

from screener.config import (
    SCRIPT_VERSION,
)
from screener.oi_context import (
    normalize_oi_history_rows,
    build_open_interest_metrics,
    interpret_open_interest_context,
    format_oi_line_for_telegram,
)
from screener.factors.sweep import detect_liquidity_sweep
from screener.factors.divergence import (
    detect_rsi_bearish_divergence,
    detect_oi_bearish_divergence,
    detect_rsi_bearish_divergence_for_timeframe,
    detect_oi_bearish_divergence_for_timeframe,
)
from screener.factors.open_levels import (
    detect_open_levels_context,
)
from screener.factors.rejection import detect_rejection_candle
from screener.classification import (
    evaluate_rsi_entry_filter,
    classify_watch_signal,
    detect_overheat_watch_context,
)
from screener.bybit import (
    get_bybit_proxy_config,
    redact_proxy_details,
    run_bybit_screener,
    bybit_analyze_candidate,
    bybit_build_market_universe,
    bybit_candles_to_dataframe,
    bybit_closed_24h_quote_volume,
    bybit_get,
    bybit_get_last_closed_candle,
    bybit_get_last_live_candle,
    bybit_instruments_to_dataframe,
    bybit_prefilter_candidates,
    bybit_tickers_to_dataframe,
    bybit_volume_change_24h,
)
from screener.telegram import (
    get_telegram_credentials,
)
from screener.report import (
    format_telegram_report,
)
from screener.runner import (
    classify_data_quality,
    format_data_warning_line,
    run_screener,
)
from tests.cases import (
    make_flat_synthetic_1h,
    make_flat_synthetic_4h,
    make_4h_swing_sweep_case,
    make_4h_level_without_4h_sweep_case,
    make_1h_intrabar_take_of_4h_level_case,
    make_1h_equal_high_sweep_case,
    make_rolling_only_high_take_case,
    make_1h_minor_micro_high_sweep_case,
    make_1h_level_with_4h_close_only_case,
    make_open_levels_1d_case,
    make_open_levels_1h_confirm_case,
    make_open_levels_1h_near_d_rejection_case,
    make_open_levels_d_rejected_then_reclaimed_case,
    make_open_levels_4h_test_case,
    make_open_levels_4h_near_case,
    make_open_levels_4h_far_case,
    make_open_levels_4h_recent_window_test_case,
    make_open_levels_4h_live_test_case,
    make_open_levels_1h_htf_rejection_case,
    make_open_levels_1h_htf_live_test_case,
    make_classification_open_factor,
    make_classification_sweep_factor,
    make_classification_rsi_divergence_factor,
    make_open_levels_4h_test_without_rejection_case,
    make_open_levels_month_only_1d_case,
    make_open_levels_far_below_month_case,
    make_open_levels_month_reclaimed_case,
    make_open_levels_old_month_rejection_then_closed_above_case,
    make_open_levels_useless_like_case,
    make_open_levels_w_reclaimed_m_not_reached_case,
    make_open_levels_vvv_1d_case,
    make_open_levels_grass_1h_case,
    make_open_levels_grass_4h_case,
)


# ============================================================
# HTTP SESSION
# ============================================================

# ============================================================
# COMMON HELPERS
# ============================================================


# ============================================================
# RSI DIVERGENCE
# ============================================================


# ============================================================
# BYBIT PROVIDER (via Cloudflare Worker proxy)
# ============================================================


# ============================================================
# TELEGRAM
# ============================================================


# ============================================================
# DATA QUALITY (incomplete data must never look like a calm market)
# ============================================================

# ============================================================
# SCREENER RUNNER
# ============================================================


# ============================================================
# SELF TESTS
# ============================================================

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
                make_classification_open_factor(["W"], ["tested"]),
            ],
            "SHORT_WATCH",
        ),
        (
            "W open tested without sweep cannot create SHORT WATCH",
            base_scores,
            [
                make_classification_open_factor(["W"], ["tested"]),
            ],
            "NO_SIGNAL",
        ),
        (
            "Strong heat + sweep + W tested remains SHORT WATCH without divergence",
            strong_heat_scores,
            [
                make_classification_sweep_factor(),
                make_classification_open_factor(["W"], ["tested"]),
            ],
            "SHORT_WATCH",
        ),
        (
            "Strong heat + sweep + divergence upgrades to HIGH PRIORITY",
            {**strong_heat_scores, "has_overheat_context": True},
            [
                make_classification_sweep_factor(),
                make_classification_rsi_divergence_factor(),
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


def run_overheat_context_self_tests():
    print("RUNNING OVERHEAT CONTEXT SELF TESTS")
    print("SCRIPT VERSION:", SCRIPT_VERSION)

    tests = [
        (
            "4H RSI heat passes even after 1H cools",
            detect_overheat_watch_context(
                rsi_1h_live=66,
                rsi_1h_closed=72,
                rsi_4h_live=80.5,
                exact_volume_24h=10_000_000,
                price_change_24h=16,
            ).get("is_overheat") is True,
        ),
        (
            "1H RSI heat still passes",
            detect_overheat_watch_context(
                rsi_1h_live=83,
                rsi_1h_closed=81,
                rsi_4h_live=70,
                exact_volume_24h=10_000_000,
                price_change_24h=16,
            ).get("is_overheat") is True,
        ),
        (
            "RSI heat blocked without pump context",
            detect_overheat_watch_context(
                rsi_1h_live=83,
                rsi_1h_closed=81,
                rsi_4h_live=82,
                exact_volume_24h=10_000_000,
                price_change_24h=9,
            ).get("is_overheat") is False,
        ),
    ]

    failed = 0
    for name, ok in tests:
        if ok:
            print(f"PASS | {name}")
        else:
            failed += 1
            print(f"FAIL | {name}")

    if failed:
        raise AssertionError(f"Overheat context self-tests failed: {failed}/{len(tests)}")

    print(f"OVERHEAT CONTEXT SELF TESTS PASSED: {len(tests)}/{len(tests)}")

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


def run_rsi_divergence_self_tests():
    print("RUNNING RSI DIVERGENCE SELF TESTS")

    def make_divergence_df():
        rows = []
        base_time = pd.Timestamp("2026-01-01 00:00:00")
        highs = [100, 102, 104, 106, 108, 111, 115, 110, 107, 105, 106, 108, 110, 112, 114, 116, 118, 122, 119, 117, 116, 115, 114, 113]
        rsis =  [45,  48,  52,  55,  60,  68,  78,  70,  61,  55,  54,  56,  58,  60,  61,  62,  64,  70,  68,  66,  64,  62,  60,  58]
        for i, (high, rsi) in enumerate(zip(highs, rsis)):
            close = high - 1
            rows.append({
                "timestamp": base_time + pd.Timedelta(hours=i),
                "open": close - 0.5,
                "high": float(high),
                "low": close - 2,
                "close": float(close),
                "rsi": float(rsi),
                "confirm": 1,
            })
        return pd.DataFrame(rows)

    def make_no_divergence_df():
        df = make_divergence_df()
        # Candidate high is at index 17 in this synthetic series.
        df.loc[17, "rsi"] = 82.0
        return df

    div_df = make_divergence_df()
    no_div_df = make_no_divergence_df()

    tests = [
        ("1H bearish divergence detected", lambda: detect_rsi_bearish_divergence_for_timeframe(div_df, timeframe="1H").get("status") == "confirmed"),
        ("1H no divergence when RSI confirms high", lambda: detect_rsi_bearish_divergence_for_timeframe(no_div_df, timeframe="1H").get("status") == "not_confirmed"),
        ("combined factor reports 1H divergence", lambda: detect_rsi_bearish_divergence(df_1h=div_df, df_4h=no_div_df).get("status") == "confirmed"),
        ("combined factor reports no divergence", lambda: detect_rsi_bearish_divergence(df_1h=no_div_df, df_4h=no_div_df).get("status") == "not_confirmed"),
    ]

    failed = 0
    for name, check in tests:
        try:
            ok = bool(check())
        except Exception as exc:
            ok = False
            print(f"❌ {name}: {exc}")

        if not ok:
            failed += 1
            print(f"❌ {name}")

    if failed:
        raise AssertionError(f"RSI divergence self-tests failed: {failed}/{len(tests)}")

    print(f"RSI DIVERGENCE SELF TESTS PASSED: {len(tests)}/{len(tests)}")


def run_oi_divergence_self_tests():
    print("RUNNING OI DIVERGENCE SELF TESTS")
    print("SCRIPT VERSION:", SCRIPT_VERSION)

    def make_price_df():
        rows = []
        base_time = pd.Timestamp("2026-01-01 00:00:00")
        highs = [100, 102, 104, 106, 108, 111, 115, 110, 107, 105, 106, 108, 110, 112, 114, 116, 118, 122, 119, 117, 116, 115, 114, 113]
        for i, high in enumerate(highs):
            close = high - 1
            rows.append({
                "timestamp": base_time + pd.Timedelta(hours=i),
                "open": close - 0.5,
                "high": float(high),
                "low": close - 2,
                "close": float(close),
                "confirm": 1,
            })
        return pd.DataFrame(rows)

    def make_oi_history(bearish=True):
        rows = []
        base_time = pd.Timestamp("2026-01-01 00:00:00")
        for i in range(0, 30):
            value = 100.0 + i
            rows.append({"timestamp": base_time + pd.Timedelta(hours=i), "open_interest": value})
        # Prior price swing high is around candle 6 close time; candidate high is around candle 17 close time.
        # Bearish case: OI at candidate high is below prior swing high OI.
        rows[7]["open_interest"] = 120.0
        rows[18]["open_interest"] = 112.0 if bearish else 135.0
        return pd.DataFrame(rows)

    price_df = make_price_df()
    bearish_oi = make_oi_history(bearish=True)
    confirming_oi = make_oi_history(bearish=False)

    tests = [
        (
            "1H OI bearish divergence detected",
            lambda: detect_oi_bearish_divergence_for_timeframe(price_df, bearish_oi, timeframe="1H").get("status") == "confirmed",
        ),
        (
            "1H no OI divergence when OI confirms high",
            lambda: detect_oi_bearish_divergence_for_timeframe(price_df, confirming_oi, timeframe="1H").get("status") in ("not_confirmed", "weak"),
        ),
        (
            "combined OI divergence factor confirms Bybit history",
            lambda: detect_oi_bearish_divergence(df_1h=price_df, df_4h=price_df, oi_history=bearish_oi, provider="Bybit").get("status") == "confirmed",
        ),
        (
            "OI divergence is available for Bybit (no provider restriction)",
            lambda: detect_oi_bearish_divergence(df_1h=price_df, df_4h=price_df, oi_history=bearish_oi, provider="Bybit").get("status") != "not_available",
        ),
        (
            "Bybit-style raw OI rows (newest first) give a confirmed divergence",
            lambda: detect_oi_bearish_divergence(
                df_1h=price_df,
                df_4h=price_df,
                oi_history=normalize_oi_history_rows(
                    [
                        {"openInterest": str(row["open_interest"]), "timestamp": str(int(row["timestamp"].timestamp() * 1000))}
                        for _, row in bearish_oi.sort_values("timestamp", ascending=False).iterrows()
                    ]
                ),
                provider="Bybit",
            ).get("status") == "confirmed",
        ),
    ]

    failed = 0
    for name, check in tests:
        try:
            ok = bool(check())
        except Exception as exc:
            ok = False
            print(f"❌ {name}: {exc}")

        if not ok:
            failed += 1
            print(f"❌ {name}")

    if failed:
        raise AssertionError(f"OI divergence self-tests failed: {failed}/{len(tests)}")

    print(f"OI DIVERGENCE SELF TESTS PASSED: {len(tests)}/{len(tests)}")


def run_bybit_self_tests():
    print("RUNNING BYBIT SELF TESTS")
    print("SCRIPT VERSION:", SCRIPT_VERSION)

    import contextlib
    import io

    @contextlib.contextmanager
    def patched(**replacements):
        # Each replaced name must be patched in the module whose functions read it.
        import screener.bybit as bybit_module
        import screener.common as common_module
        import screener.runner as runner_module
        owner_modules = {
            "SESSION": common_module,
            "bybit_get_candles": bybit_module,
            "bybit_get_open_interest_history": bybit_module,
            "bybit_get_instruments": bybit_module,
            "bybit_get_tickers": bybit_module,
            "bybit_build_market_universe": bybit_module,
            "bybit_prefilter_candidates": bybit_module,
            "bybit_analyze_candidate": bybit_module,
            "REQUEST_DELAY_SECONDS": bybit_module,
            "run_bybit_screener": runner_module,
            "send_telegram_message_safe": runner_module,
        }

        targets = {}
        for name in replacements:
            if name not in owner_modules:
                raise KeyError(
                    f"patched(): no owner module configured for '{name}'. "
                    "Add it to owner_modules in run_bybit_self_tests."
                )
            targets[name] = owner_modules[name]

        saved = {name: getattr(targets[name], name) for name in replacements}
        for name, value in replacements.items():
            setattr(targets[name], name, value)
        try:
            yield
        finally:
            for name, value in saved.items():
                setattr(targets[name], name, value)

    @contextlib.contextmanager
    def proxy_env(url="https://proxy.example.com/", secret="test-secret-value"):
        names = ["BYBIT_PROXY_URL", "BYBIT_PROXY_SECRET"]
        saved = {name: os.environ.get(name) for name in names}
        for name, value in zip(names, [url, secret]):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        try:
            yield
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    class FakeResponse:
        def __init__(self, payload, status_code=200):
            self.payload = payload
            self.status_code = status_code
            self.text = str(payload)

        def json(self):
            return self.payload

    class FakeSession:
        def __init__(self, payload, status_code=200):
            self.payload = payload
            self.status_code = status_code
            self.calls = []

        def get(self, url, params=None, headers=None, timeout=None):
            self.calls.append({"url": url, "params": params, "headers": headers})
            return FakeResponse(self.payload, self.status_code)

    def quiet(func, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return func(*args, **kwargs)

    def raises(func, *args, **kwargs):
        try:
            quiet(func, *args, **kwargs)
        except BaseException as exc:
            return exc
        return None

    # Bybit kline rows: newest first, strings, [start ms, o, h, l, c, volume, turnover].
    kline_rows = [
        ["1781532000000", "10.5", "11.0", "10.0", "10.8", "100", "1080"],
        ["1781528400000", "10.0", "10.6", "9.9", "10.5", "90", "950"],
        ["1781524800000", "9.8", "10.1", "9.7", "10.0", "80", "790"],
    ]
    df_k = bybit_candles_to_dataframe(kline_rows)

    # Synthetic API payloads for the full candidate pipeline.
    def to_bybit_rows(df):
        rows = []
        for _, r in df.iloc[::-1].iterrows():
            rows.append([
                str(int(r["timestamp"].timestamp() * 1000)),
                str(r["open"]), str(r["high"]), str(r["low"]), str(r["close"]),
                str(r["volume"]), str(r["quote_volume"]),
            ])
        return rows

    rng = np.random.default_rng(7)
    n = 400
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.004, size=n)))
    closes[-30:] *= np.linspace(1.0, 1.25, 30)
    base_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1h"),
        "open": np.concatenate([[100.0], closes[:-1]]),
        "high": closes * 1.004,
        "low": closes * 0.996,
        "close": closes,
        "volume": 1000.0,
        "quote_volume": 1000.0 * closes,
    })
    market_1h = base_df
    market_4h = base_df.iloc[::4].reset_index(drop=True)
    market_1d = base_df.iloc[::24].reset_index(drop=True)
    market_oi = [
        {"openInterest": str(5000 + i), "timestamp": str(int(ts.timestamp() * 1000))}
        for i, ts in list(enumerate(base_df["timestamp"].tail(200)))[::-1]
    ]

    def fake_candles(symbol, interval, limit):
        source = {"60": market_1h, "240": market_4h, "D": market_1d}[interval]
        return to_bybit_rows(source)

    def fake_oi_history(symbol):
        return normalize_oi_history_rows(market_oi)

    ticker_row = pd.Series({"symbol": "TESTUSDT", "price_change_24h_percent": 25.0})

    with patched(bybit_get_candles=fake_candles, bybit_get_open_interest_history=fake_oi_history):
        candidate = quiet(bybit_analyze_candidate, "TESTUSDT", ticker_row)

    instruments = [
        {"symbol": "AAAUSDT", "contractType": "LinearPerpetual", "quoteCoin": "USDT", "status": "Trading"},
        {"symbol": "BBBUSDT", "contractType": "LinearFutures", "quoteCoin": "USDT", "status": "Trading"},
        {"symbol": "CCCUSDC", "contractType": "LinearPerpetual", "quoteCoin": "USDC", "status": "Trading"},
        {"symbol": "DDDUSDT", "contractType": "LinearPerpetual", "quoteCoin": "USDT", "status": "Closed"},
        {"symbol": "EEEUSDT", "contractType": "LinearPerpetual", "quoteCoin": "USDT", "status": "Trading"},
    ]
    tickers = [
        {"symbol": "AAAUSDT", "lastPrice": "2.5", "price24hPcnt": "0.1273", "turnover24h": "74340000"},
        {"symbol": "EEEUSDT", "lastPrice": "1.0", "price24hPcnt": "0.05", "turnover24h": "90000000"},
        {"symbol": "BBBUSDT", "lastPrice": "1.0", "price24hPcnt": "0.50", "turnover24h": "90000000"},
    ]
    df_inst = bybit_instruments_to_dataframe(instruments)
    df_tick = bybit_tickers_to_dataframe(tickers)

    with patched(bybit_get_instruments=lambda: instruments, bybit_get_tickers=lambda: tickers):
        df_universe = bybit_build_market_universe()
        df_pre = bybit_prefilter_candidates(df_universe)

    # Data quality / error accounting.
    def run_with_errors(failing_symbols):
        candidates = pd.DataFrame({
            "symbol": [f"S{i}USDT" for i in range(10)],
            "price": 1.0,
            "price_change_24h_percent": 20.0,
            "volume_usd_24h_est": 10_000_000.0,
        })

        def fake_analyze(symbol, row):
            if symbol in failing_symbols:
                raise Exception("boom")
            return {"signal_level": "NO_SIGNAL", "symbol": symbol}

        with patched(
            bybit_build_market_universe=lambda: candidates,
            bybit_prefilter_candidates=lambda df: df,
            bybit_analyze_candidate=fake_analyze,
            REQUEST_DELAY_SECONDS=0,
        ):
            return quiet(run_bybit_screener)

    _, stats_none = run_with_errors(set())
    _, stats_three = run_with_errors({"S0USDT", "S1USDT", "S2USDT"})

    sent_messages = []

    def fake_send(text):
        sent_messages.append(text)
        return True

    def run_screener_with(result=None, error=None):
        sent_messages.clear()

        def fake_run():
            if error is not None:
                raise error
            return result

        with patched(run_bybit_screener=fake_run, send_telegram_message_safe=fake_send):
            return raises(run_screener)

    def stats(analyzed, errors):
        return {"total_universe": 100, "prefiltered": analyzed + errors, "analyzed": analyzed, "errors": errors, "active": 0}

    exit_incomplete = run_screener_with(result=(pd.DataFrame(), stats(4, 6)))
    incomplete_messages = list(sent_messages)
    exit_fetch_error = run_screener_with(error=Exception("Bybit HTTP error: 403"))
    fetch_error_messages = list(sent_messages)
    exit_partial = run_screener_with(result=(pd.DataFrame(), stats(8, 2)))
    partial_messages = list(sent_messages)
    exit_ok = run_screener_with(result=(pd.DataFrame(), stats(10, 0)))
    ok_messages = list(sent_messages)

    def call_bybit_with(session):
        with patched(SESSION=session), proxy_env():
            return bybit_get("/v5/market/kline", {"category": "linear", "symbol": "XUSDT"}, label="Bybit kline")

    ok_session = FakeSession({"retCode": 0, "retMsg": "OK", "result": {"list": [1]}})
    call_bybit_with(ok_session)

    with proxy_env():
        configured_url, configured_secret = get_bybit_proxy_config()
    with proxy_env(url="https://my-worker.example.workers.dev", secret="s3cret-xyz"):
        redacted_text = redact_proxy_details(
            "failed https://my-worker.example.workers.dev/bybit/v5 host=my-worker.example.workers.dev token=s3cret-xyz"
        )
    with proxy_env(url=None, secret=None):
        missing_both = raises(get_bybit_proxy_config)
    with proxy_env(url="https://leak-check.example.com", secret=None):
        missing_secret = raises(get_bybit_proxy_config)

    tests = [
        # --- configuration and requests
        ("config: both variables missing -> clear error", isinstance(missing_both, RuntimeError)
            and "BYBIT_PROXY_URL" in str(missing_both) and "BYBIT_PROXY_SECRET" in str(missing_both)),
        ("config: error does not show configured values", isinstance(missing_secret, RuntimeError)
            and "leak-check" not in str(missing_secret) and "BYBIT_PROXY_SECRET" in str(missing_secret)),
        ("config: values are read, trailing slash is removed", configured_url == "https://proxy.example.com" and configured_secret == "test-secret-value"),
        ("request: URL goes through proxy /bybit prefix", ok_session.calls[0]["url"] == "https://proxy.example.com/bybit/v5/market/kline"),
        ("request: X-Proxy-Secret header is sent", ok_session.calls[0]["headers"] == {"X-Proxy-Secret": "test-secret-value"}),
        ("request: query params are passed", ok_session.calls[0]["params"] == {"category": "linear", "symbol": "XUSDT"}),
        ("request: retCode != 0 raises", raises(call_bybit_with, FakeSession({"retCode": 10001, "retMsg": "bad", "result": {}})) is not None),
        ("request: HTTP 401 raises", raises(call_bybit_with, FakeSession({"error": "unauthorized"}, status_code=401)) is not None),
        ("redaction: URL, host and secret are hidden in log text", redacted_text == "failed ***/bybit/v5 host=*** token=***"),
        ("request: errors never contain the secret", "test-secret-value" not in str(raises(call_bybit_with, FakeSession({"retCode": 10001, "retMsg": "bad", "result": {}})))),

        # --- candles
        ("candles: sorted oldest first", list(df_k["close"]) == [10.0, 10.5, 10.8]),
        ("candles: columns", list(df_k.columns) == ["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]),
        ("candles: no confirm column", "confirm" not in df_k.columns),
        ("candles: quote_volume is turnover", list(df_k["quote_volume"]) == [790.0, 950.0, 1080.0]),
        ("candles: numeric dtypes", all(str(df_k[c].dtype).startswith("float") for c in ["open", "high", "low", "close", "volume", "quote_volume"])),
        ("candles: timestamp from ms", df_k["timestamp"].iloc[0] == pd.to_datetime(1781524800000, unit="ms")),
        ("candles: last row is live", float(bybit_get_last_live_candle(df_k)["close"]) == 10.8),
        ("candles: previous row is last closed", float(bybit_get_last_closed_candle(df_k)["close"]) == 10.5),
        ("candles: too few columns raises", raises(bybit_candles_to_dataframe, [["1781532000000", "1", "2", "3", "4", "5"]]) is not None),
        ("candles: empty rows give empty dataframe", bybit_candles_to_dataframe([]).empty),
        ("volume: 24h sum excludes live candle", bybit_closed_24h_quote_volume(pd.DataFrame({"quote_volume": [1.0] * 24 + [999.0]})) == 24.0),
        ("volume: change vs previous 24h", abs(bybit_volume_change_24h(pd.DataFrame({"quote_volume": [1.0] * 24 + [2.0] * 24 + [999.0]})) - 100.0) < 1e-9),
        ("volume: not enough candles -> None", bybit_closed_24h_quote_volume(pd.DataFrame({"quote_volume": [1.0] * 10})) is None),

        # --- universe
        ("instruments: only LinearPerpetual USDT Trading", list(df_inst["symbol"]) == ["AAAUSDT", "EEEUSDT"]),
        ("tickers: 24h change ratio -> percent", abs(float(df_tick.loc[df_tick["symbol"] == "AAAUSDT", "price_change_24h_percent"].iloc[0]) - 12.73) < 1e-9),
        ("tickers: turnover is USDT volume", float(df_tick.loc[df_tick["symbol"] == "AAAUSDT", "volume_usd_24h_est"].iloc[0]) == 74340000.0),
        ("universe: dated futures are excluded", "BBBUSDT" not in list(df_universe["symbol"])),
        ("prefilter: change >= 8% and volume >= 5M", list(df_pre["symbol"]) == ["AAAUSDT"]),

        # --- full candidate pipeline on fake API data
        ("candidate: exchange and provider are Bybit", candidate["exchange"] == "Bybit" and candidate["provider"] == "Bybit"),
        ("candidate: report symbol has .P suffix", candidate["symbol"] == "TESTUSDT.P"),
        ("candidate: OI source is Bybit with history", candidate["oi_source"] == "Bybit" and candidate["oi_history_points"] == 200),
        ("candidate: all five factors analyzed", candidate["total_short_factors_count"] == 5),
        ("candidate: price change comes from ticker", candidate["price_change_24h_percent"] == 25.0),

        # --- 2.5 data quality
        ("quality: no errors -> ok", classify_data_quality(10, 0) == "ok"),
        ("quality: no candidates -> ok", classify_data_quality(0, 0) == "ok"),
        ("quality: 1 of 10 failed -> partial", classify_data_quality(10, 1) == "partial"),
        ("quality: exactly half failed -> partial", classify_data_quality(10, 5) == "partial"),
        ("quality: more than half failed -> incomplete", classify_data_quality(10, 6) == "incomplete"),
        ("quality: all failed -> incomplete", classify_data_quality(5, 5) == "incomplete"),
        ("runner: no errors counted", stats_none["errors"] == 0 and stats_none["analyzed"] == 10),
        ("runner: errors are counted per symbol", stats_three["errors"] == 3 and stats_three["analyzed"] == 7),
        ("warning line text", format_data_warning_line(7, 10, 3) == "⚠️ Проаналізовано 7 з 10 кандидатів, 3 помилок"),
        ("report: warning line is shown", "⚠️ Проаналізовано 7 з 10 кандидатів, 3 помилок" in format_telegram_report([], data_warning=format_data_warning_line(7, 10, 3))),
        ("report: no warning without errors", "⚠️" not in format_telegram_report([])),
        ("run: >half failed -> exit code 1", isinstance(exit_incomplete, SystemExit) and exit_incomplete.code == 1),
        ("run: >half failed -> 'data incomplete' message, not 'No signals'",
            len(incomplete_messages) == 1 and "Дані неповні" in incomplete_messages[0] and "No overheat" not in incomplete_messages[0]),
        ("run: market data failure -> exit code 1 and warning",
            isinstance(exit_fetch_error, SystemExit) and exit_fetch_error.code == 1
            and len(fetch_error_messages) == 1 and "список монет" in fetch_error_messages[0]),
        ("run: some errors -> normal report with warning line, exit 0",
            exit_partial is None and len(partial_messages) == 1
            and "Проаналізовано 8 з 10 кандидатів, 2 помилок" in partial_messages[0]
            and "No overheat or short-watch signals" in partial_messages[0]),
        ("run: no errors -> normal report without warning",
            exit_ok is None and len(ok_messages) == 1 and "⚠️" not in ok_messages[0]),
    ]

    failed = 0
    for name, passed in tests:
        status = "PASS" if passed else "FAIL"
        print(f"{status} | {name}")

        if not passed:
            failed += 1

    if failed:
        raise AssertionError(f"Bybit self-tests failed: {failed}/{len(tests)}")

    print(f"BYBIT SELF TESTS PASSED: {len(tests)}/{len(tests)}")


def main():
    if "--self-test" in sys.argv:
        run_sweep_self_tests()
        run_open_levels_self_tests()
        run_open_levels_classification_self_tests()
        run_rejection_self_tests()
        run_rsi_entry_filter_self_tests()
        run_overheat_context_self_tests()
        run_open_interest_self_tests()
        run_rsi_divergence_self_tests()
        run_oi_divergence_self_tests()
        run_bybit_self_tests()
        return

    get_telegram_credentials()

    try:
        get_bybit_proxy_config()
    except RuntimeError as e:
        raise SystemExit(f"ERROR: {e}")

    run_screener()


if __name__ == "__main__":
    main()
