import os
import sys
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

from screener.config import (
    SCRIPT_VERSION,
    OPEN_LEVEL_CONTEXT_WEIGHTS,
    FINAL_TOP_N,
    SEND_MESSAGE_IF_NO_SIGNALS,
    DATA_ERROR_MAX_SHARE,
)
from screener.common import (
    make_factor,
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
    build_open_levels_detail,
)
from screener.factors.rejection import detect_rejection_candle
from screener.factors.analysis import analyze_short_factors  # noqa: F401  (used by tests/golden_test.py)
from screener.classification import classify_signal  # noqa: F401  (used by tests/golden_test.py)
from screener.common import calculate_rsi  # noqa: F401  (used by tests/golden_test.py)
from screener.config import RSI_PERIOD  # noqa: F401  (used by tests/golden_test.py)
from screener.classification import (
    get_signal_rank,
    add_rsi_entry_filter_columns,
    log_rsi_entry_debug,
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
    send_telegram_message_safe,
)
from screener.report import (
    format_telegram_report,
    prepare_active_output_table,
    prepare_grouped_active_signals,
    prepare_grouped_output_table,
    log_oi_debug_for_grouped_signals,
    log_sweep_debug_for_grouped_signals,
    log_rsi_divergence_debug_for_grouped_signals,
    log_open_levels_debug_for_grouped_signals,
    log_rejection_debug_for_grouped_signals,
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

def classify_data_quality(candidates_total, errors):
    """Return "ok", "partial" or "incomplete".

    - ok: no candidate failed;
    - partial: some candidates failed, but not more than half;
    - incomplete: more than half of the candidates failed.
    """

    total = int(candidates_total or 0)
    failed = int(errors or 0)

    if failed <= 0 or total <= 0:
        return "ok"

    if failed / total > DATA_ERROR_MAX_SHARE:
        return "incomplete"

    return "partial"


def format_data_warning_line(analyzed, candidates_total, errors):
    return f"⚠️ Проаналізовано {int(analyzed)} з {int(candidates_total)} кандидатів, {int(errors)} помилок"


def format_incomplete_data_message(analyzed, candidates_total, errors):
    return (
        f"⚠️ Дані неповні: проаналізовано {int(analyzed)} з {int(candidates_total)} кандидатів, "
        f"{int(errors)} помилок. Звіт не сформовано."
    )


def format_market_data_error_message():
    return "⚠️ Не вдалося отримати список монет або тікери з Bybit. Звіт не сформовано."


# ============================================================
# SCREENER RUNNER
# ============================================================

def run_screener():
    print("\n" + "=" * 120)
    print("RUNNING BYBIT SCREENER")
    print("SCRIPT VERSION:", SCRIPT_VERSION)
    print("=" * 120)

    try:
        df_all, stats = run_bybit_screener()
    except Exception as e:
        print(f"Bybit market data unavailable: {redact_proxy_details(e)}")
        send_telegram_message_safe(format_market_data_error_message())
        sys.exit(1)

    candidates_total = stats["analyzed"] + stats["errors"]
    data_quality = classify_data_quality(candidates_total, stats["errors"])

    if data_quality == "incomplete":
        print(
            "Bybit data is incomplete:",
            f"analyzed={stats['analyzed']}",
            f"candidates={candidates_total}",
            f"errors={stats['errors']}",
        )
        send_telegram_message_safe(
            format_incomplete_data_message(stats["analyzed"], candidates_total, stats["errors"])
        )
        sys.exit(1)

    data_warning = None

    if data_quality == "partial":
        data_warning = format_data_warning_line(stats["analyzed"], candidates_total, stats["errors"])

    if df_all.empty:
        df_active = pd.DataFrame()
        rsi_filtered_out_count = 0
    else:
        df_all["signal_rank"] = df_all["signal_level"].apply(get_signal_rank)
        df_all = add_rsi_entry_filter_columns(df_all)

        df_all = df_all.sort_values(
            by=[
                "signal_rank",
                "final_score",
                "short_setup_score",
                "rsi_4h_live",
                "rsi_1h_live",
                "rsi_1h_closed",
                "price_change_24h_percent",
                "volume_usd_24h_exact",
            ],
            ascending=[False, False, False, False, False, False, False, False],
        ).reset_index(drop=True)

        df_active_before_rsi_filter = df_all[df_all["signal_level"] != "NO_SIGNAL"].copy()
        log_rsi_entry_debug(df_active_before_rsi_filter)
        rsi_filtered_out_count = len(df_active_before_rsi_filter[~df_active_before_rsi_filter["rsi_entry_passed"]])
        df_active = df_active_before_rsi_filter[df_active_before_rsi_filter["rsi_entry_passed"]].copy()

    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print("Bybit total universe:", stats["total_universe"])
    print("Bybit prefiltered:", stats["prefiltered"])
    print("Bybit analyzed:", stats["analyzed"])
    print("Bybit errors:", stats["errors"])
    print("Bybit active:", stats["active"])
    print("RSI entry filtered out:", rsi_filtered_out_count)
    print("Total active signals after RSI entry filter:", len(df_active))

    df_active_output = prepare_active_output_table(df_active)
    grouped_signals = prepare_grouped_active_signals(df_active, max_groups=FINAL_TOP_N)
    log_oi_debug_for_grouped_signals(grouped_signals)
    log_rsi_divergence_debug_for_grouped_signals(grouped_signals)
    log_sweep_debug_for_grouped_signals(grouped_signals)
    log_open_levels_debug_for_grouped_signals(grouped_signals)
    log_rejection_debug_for_grouped_signals(grouped_signals)
    df_grouped_output = prepare_grouped_output_table(grouped_signals)

    if df_active_output.empty:
        print("No active signals.")
    else:
        print("\n" + "=" * 120)
        print("ACTIVE SIGNALS")
        print("=" * 120)
        print(df_active_output.head(FINAL_TOP_N).to_string(index=False))

        print("\n" + "=" * 120)
        print("ACTIVE SIGNALS GROUPED BY SYMBOL")
        print("=" * 120)
        print(df_grouped_output.head(FINAL_TOP_N).to_string(index=False))

    if SEND_MESSAGE_IF_NO_SIGNALS or grouped_signals:
        message = format_telegram_report(
            grouped_signals=grouped_signals,
            data_warning=data_warning,
        )

        send_telegram_message_safe(message)



# ============================================================
# SELF TESTS
# ============================================================

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

        main_module = sys.modules[__name__]
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
            "run_bybit_screener": main_module,
            "send_telegram_message_safe": main_module,
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
