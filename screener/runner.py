import sys
import pandas as pd

from screener.config import (
    DATA_ERROR_MAX_SHARE,
    FINAL_TOP_N,
    SCRIPT_VERSION,
    SEND_MESSAGE_IF_NO_SIGNALS,
)
from screener.classification import (
    add_rsi_entry_filter_columns,
    get_signal_rank,
    log_rsi_entry_debug,
)
from screener.bybit import (
    redact_proxy_details,
    run_bybit_screener,
)
from screener.report import (
    format_telegram_report,
    log_oi_debug_for_grouped_signals,
    log_open_levels_debug_for_grouped_signals,
    log_rejection_debug_for_grouped_signals,
    log_rsi_divergence_debug_for_grouped_signals,
    log_sweep_debug_for_grouped_signals,
    prepare_active_output_table,
    prepare_grouped_active_signals,
    prepare_grouped_output_table,
)
from screener.telegram import send_telegram_message_safe


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
