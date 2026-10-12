import pandas as pd

from screener.config import SCRIPT_VERSION
from screener.oi_context import normalize_oi_history_rows
from screener.factors.sweep import detect_liquidity_sweep
from screener.factors.divergence import (
    detect_oi_bearish_divergence,
    detect_oi_bearish_divergence_for_timeframe,
    detect_rsi_bearish_divergence,
    detect_rsi_bearish_divergence_for_timeframe,
)
from screener.factors.open_levels import detect_open_levels_context
from screener.factors.rejection import detect_rejection_candle
from tests.cases import (
    make_1h_equal_high_sweep_case,
    make_1h_intrabar_take_of_4h_level_case,
    make_1h_level_with_4h_close_only_case,
    make_1h_minor_micro_high_sweep_case,
    make_4h_level_without_4h_sweep_case,
    make_4h_swing_sweep_case,
    make_flat_synthetic_1h,
    make_flat_synthetic_4h,
    make_open_levels_1d_case,
    make_open_levels_1h_confirm_case,
    make_open_levels_1h_htf_live_test_case,
    make_open_levels_1h_htf_rejection_case,
    make_open_levels_1h_near_d_rejection_case,
    make_open_levels_4h_far_case,
    make_open_levels_4h_live_test_case,
    make_open_levels_4h_near_case,
    make_open_levels_4h_recent_window_test_case,
    make_open_levels_4h_test_case,
    make_open_levels_4h_test_without_rejection_case,
    make_open_levels_d_rejected_then_reclaimed_case,
    make_open_levels_far_below_month_case,
    make_open_levels_grass_1h_case,
    make_open_levels_grass_4h_case,
    make_open_levels_month_only_1d_case,
    make_open_levels_month_reclaimed_case,
    make_open_levels_old_month_rejection_then_closed_above_case,
    make_open_levels_useless_like_case,
    make_open_levels_vvv_1d_case,
    make_open_levels_w_reclaimed_m_not_reached_case,
    make_rolling_only_high_take_case,
)


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
