import html
import pandas as pd

from datetime import datetime

from screener.common import (
    format_debug_value,
    format_large_number,
    format_percent_2,
    format_price_2,
    safe_float,
)
from screener.config import (
    FINAL_TOP_N,
    KYIV_TZ,
    SCRIPT_VERSION,
    TELEGRAM_MAX_SIGNALS,
)
from screener.classification import get_signal_rank
from screener.oi_context import (
    format_oi_line_for_telegram,
    format_oi_percent_for_telegram,
)
from screener.telegram import telegram_signal_label


def prepare_active_output_table(df_active):
    if df_active is None or df_active.empty:
        return pd.DataFrame()

    df_out = df_active.copy()

    df_out["price"] = df_out["price"].apply(format_price_2)
    df_out["rsi_1h_live"] = df_out["rsi_1h_live"].round(2)
    df_out["rsi_1h_closed"] = df_out["rsi_1h_closed"].round(2)
    df_out["rsi_4h_live"] = df_out["rsi_4h_live"].round(2)
    df_out["chg_24h_%"] = df_out["price_change_24h_percent"].apply(format_percent_2)
    df_out["vol_24h"] = df_out["volume_usd_24h_exact"].apply(format_large_number)
    df_out["vol_chg_24h_%"] = df_out["volume_change_24h_percent"].apply(format_percent_2)
    if "oi_change_1h_percent" in df_out.columns:
        df_out["oi_1h_%"] = df_out["oi_change_1h_percent"].apply(format_oi_percent_for_telegram)
    else:
        df_out["oi_1h_%"] = "N/A"
    if "oi_change_4h_percent" in df_out.columns:
        df_out["oi_4h_%"] = df_out["oi_change_4h_percent"].apply(format_oi_percent_for_telegram)
    else:
        df_out["oi_4h_%"] = "N/A"

    columns = [
        "exchange",
        "signal_level",
        "symbol",
        "price",
        "rsi_1h_live",
        "rsi_1h_closed",
        "rsi_4h_live",
        "chg_24h_%",
        "vol_24h",
        "vol_chg_24h_%",
        "oi_1h_%",
        "oi_4h_%",
        "oi_context",
        "pump_score",
        "rsi_score",
        "volume_score",
        "short_setup_score",
        "final_score",
        "confirmed_short_factors_count",
        "total_short_factors_count",
        "reason",
    ]

    return df_out[columns]


def prepare_grouped_active_signals(df_active, max_groups=FINAL_TOP_N):
    """
    Group active signals by symbol.

    Bybit is the only data source, so each symbol appears once.
    Ranking uses live RSI fields, not closed 4H RSI.
    """

    if df_active is None or df_active.empty:
        return []

    df = df_active.copy()
    df["signal_rank"] = df["signal_level"].apply(get_signal_rank)
    df["sort_volume"] = df["volume_usd_24h_exact"].fillna(0)
    df["sort_vol_chg"] = df["volume_change_24h_percent"].fillna(-999999)

    groups = []

    for symbol, group in df.groupby("symbol", sort=False):
        group_sorted = group.sort_values(
            by=[
                "signal_rank",
                "final_score",
                "short_setup_score",
                "rsi_4h_live",
                "rsi_1h_live",
                "rsi_1h_closed",
                "price_change_24h_percent",
                "sort_volume",
            ],
            ascending=[False, False, False, False, False, False, False, False],
        ).reset_index(drop=True)

        top_row = group_sorted.iloc[0]

        details = []

        for _, row in group_sorted.iterrows():
            details.append({
                "exchange": str(row["exchange"]),
                "signal_level": str(row["signal_level"]),
                "reason": str(row["reason"]),
                "price": format_price_2(row["price"]),
                "rsi_1h_live": f"{float(row['rsi_1h_live']):.2f}",
                "rsi_1h_closed": f"{float(row['rsi_1h_closed']):.2f}",
                "rsi_4h_live": f"{float(row['rsi_4h_live']):.2f}",
                "chg_24h_%": format_percent_2(row["price_change_24h_percent"]),
                "vol_24h": format_large_number(row["volume_usd_24h_exact"]),
                "vol_chg_24h_%": format_percent_2(row["volume_change_24h_percent"]),
                "oi_current": row.get("oi_current"),
                "oi_change_1h_percent": row.get("oi_change_1h_percent"),
                "oi_change_4h_percent": row.get("oi_change_4h_percent"),
                "oi_context": str(row.get("oi_context", "N/A")),
                "oi_status": str(row.get("oi_status", "not_available")),
                "oi_source": str(row.get("oi_source", row.get("exchange", "N/A"))),
                "oi_history_points": row.get("oi_history_points", 0),
                "oi_history_latest_ts": row.get("oi_history_latest_ts"),
                "pump_score": int(row.get("pump_score", 0)),
                "rsi_score": int(row.get("rsi_score", 0)),
                "volume_score": int(row.get("volume_score", 0)),
                "short_setup_score": int(row.get("short_setup_score", 0)),
                "final_score": int(row.get("final_score", 0)),
                "location_score": float(safe_float(row.get("location_score", 0), default=0.0)),
                "trigger_count": int(row.get("trigger_count", 0)),
                "setup_status": str(row.get("setup_status", "N/A")),
                "short_factors": row.get("short_factors", []),
                "confirmed_short_factors_count": int(row.get("confirmed_short_factors_count", 0)),
                "total_short_factors_count": int(row.get("total_short_factors_count", 0)),
            })

        unique_reasons = []

        for reason in group_sorted["reason"].astype(str).tolist():
            if reason not in unique_reasons:
                unique_reasons.append(reason)

        max_price_change = group_sorted["price_change_24h_percent"].max()
        max_volume_change = group_sorted["volume_change_24h_percent"].max()
        max_volume = group_sorted["volume_usd_24h_exact"].max()
        max_final_score = group_sorted["final_score"].max() if "final_score" in group_sorted.columns else 0
        max_short_setup_score = group_sorted["short_setup_score"].max() if "short_setup_score" in group_sorted.columns else 0

        display_detail = details[0] if details else {}
        display_signal_level = str(display_detail.get("signal_level", top_row["signal_level"]))
        display_signal_rank = get_signal_rank(display_signal_level)

        groups.append({
            "symbol": str(symbol),
            "signal_level": display_signal_level,
            "signal_rank": int(display_signal_rank),
            "reason": str(display_detail.get("reason", top_row["reason"])),
            "reasons": unique_reasons,
            "best_rsi_1h_live": float(group_sorted["rsi_1h_live"].max()),
            "best_rsi_1h_closed": float(group_sorted["rsi_1h_closed"].max()),
            "best_rsi_4h_live": float(group_sorted["rsi_4h_live"].max()),
            "best_price_change_24h_percent": float(max_price_change),
            "best_volume_usd_24h_exact": None if pd.isna(max_volume) else float(max_volume),
            "best_volume_change_24h_percent": None if pd.isna(max_volume_change) else float(max_volume_change),
            "best_final_score": int(max_final_score),
            "best_short_setup_score": int(max_short_setup_score),
            "details": details,
            "display_detail": display_detail,
        })

    groups = sorted(
        groups,
        key=lambda item: (
            item["signal_rank"],
            item.get("best_final_score", 0),
            item.get("best_short_setup_score", 0),
            item["best_rsi_4h_live"],
            item["best_rsi_1h_live"],
            item["best_rsi_1h_closed"],
            item["best_price_change_24h_percent"],
            item["best_volume_usd_24h_exact"] or 0,
        ),
        reverse=True,
    )

    return groups[:max_groups]


def prepare_grouped_output_table(grouped_signals):
    if not grouped_signals:
        return pd.DataFrame()

    rows = []

    for item in grouped_signals:
        rows.append({
            "signal_level": item["signal_level"],
            "symbol": item["symbol"],
            "best_rsi_1h_live": round(item["best_rsi_1h_live"], 2),
            "best_rsi_1h_closed": round(item["best_rsi_1h_closed"], 2),
            "best_rsi_4h_live": round(item["best_rsi_4h_live"], 2),
            "best_chg_24h_%": format_percent_2(item["best_price_change_24h_percent"]),
            "best_vol_24h": format_large_number(item["best_volume_usd_24h_exact"]),
            "best_vol_chg_24h_%": format_percent_2(item["best_volume_change_24h_percent"]),
            "final_score": item.get("best_final_score", 0),
            "short_setup_score": item.get("best_short_setup_score", 0),
            "reason": item["reason"],
        })

    return pd.DataFrame(rows)


def find_factor(factors, key):
    if not factors:
        return None

    for factor in factors:
        if factor.get("key") == key:
            return factor

    return None


def factor_status_icon(factor):
    status = (factor or {}).get("status")

    if status == "confirmed":
        return "✅"
    if status == "not_enough_data":
        return "⚪"
    return "❌"


def format_sweep_detail_for_telegram(factors):
    factor = find_factor(factors, "liquidity_sweep") or {}
    status = factor.get("status")
    detail = str(factor.get("detail", "") or "").strip()

    if status == "confirmed" and detail:
        return detail

    if status == "not_enough_data":
        return "Data unavailable"

    return "No confirmed sweep"


def format_open_levels_detail_for_telegram(factors):
    factor = find_factor(factors, "open_levels") or {}
    status = factor.get("status")
    detail = str(factor.get("detail", "") or "").strip()

    if status == "confirmed" and detail:
        return detail

    return ""


def format_rejection_detail_for_telegram(factors):
    factor = find_factor(factors, "rejection_candle") or {}
    status = factor.get("status")
    detail = str(factor.get("detail", "") or "").strip()

    if status == "confirmed" and detail:
        return detail

    if status == "not_enough_data":
        return "Data unavailable"

    return "none"


def format_rsi_divergence_detail_for_telegram(factors):
    factor = find_factor(factors, "rsi_divergence") or {}
    status = factor.get("status")
    detail = str(factor.get("detail", "") or "").strip()

    if status == "confirmed" and detail:
        return detail

    return ""


def format_oi_divergence_detail_for_telegram(factors):
    factor = find_factor(factors, "oi_divergence") or {}
    status = str(factor.get("status", "") or "")
    detail = str(factor.get("detail", "") or "").strip()

    if status in ("confirmed", "weak") and detail:
        return detail

    return ""


def format_divergence_detail_for_telegram(factors):
    parts = []
    rsi_detail = format_rsi_divergence_detail_for_telegram(factors)
    oi_detail = format_oi_divergence_detail_for_telegram(factors)

    if rsi_detail:
        parts.append(rsi_detail)

    if oi_detail:
        parts.append(oi_detail)

    return " | ".join(parts)


def format_reason_for_telegram(setup_status):
    text = str(setup_status or "N/A").strip()

    replacements = {
        "High priority watch — ": "",
        "Short watch — ": "",
        "Overheat watch — ": "",
        "Watch only — ": "",
    }

    for prefix, replacement in replacements.items():
        if text.startswith(prefix):
            text = replacement + text[len(prefix):]
            break

    if text == "confirmed liquidity sweep detected":
        return "confirmed liquidity sweep"

    # Preserve common uppercase acronyms in user-facing reasons.
    if text.startswith(("RSI", "HTF", "MSS", "OI")):
        return text

    return text[:1].lower() + text[1:] if text else "N/A"


def log_oi_debug_for_grouped_signals(grouped_signals):
    """Print OI_DEBUG lines for Telegram-visible signals."""

    visible_levels = {
        "HIGH_PRIORITY_SHORT_WATCH",
        "SHORT_WATCH",
        "OVERHEAT_WATCH",
    }

    for group in grouped_signals or []:
        if str(group.get("signal_level")) not in visible_levels:
            continue

        detail = group.get("display_detail") or {}
        fields = [
            "OI_DEBUG",
            f"symbol={format_debug_value(group.get('symbol'))}",
            f"exchange={format_debug_value(detail.get('exchange'))}",
            f"signal={format_debug_value(detail.get('signal_level') or group.get('signal_level'))}",
            f"status={format_debug_value(detail.get('oi_status'))}",
            f"source={format_debug_value(detail.get('oi_source'))}",
            f"current={format_debug_value(detail.get('oi_current'))}",
            f"change_1h={format_debug_value(detail.get('oi_change_1h_percent'))}",
            f"change_4h={format_debug_value(detail.get('oi_change_4h_percent'))}",
            f"context={format_debug_value(detail.get('oi_context'))}",
            f"history_points={format_debug_value(detail.get('oi_history_points'))}",
            f"history_latest={format_debug_value(detail.get('oi_history_latest_ts'))}",
        ]
        print(" ".join(fields))


def log_sweep_debug_for_grouped_signals(grouped_signals):
    """
    Print one compact SWEEP_DEBUG line per Telegram-visible signal.

    This is intentionally written to GitHub Actions logs only. It does not
    change Telegram output and helps validate why the sweep factor accepted or
    rejected a level when we compare signals with charts.
    """

    visible_levels = {
        "HIGH_PRIORITY_SHORT_WATCH",
        "SHORT_WATCH",
        "OVERHEAT_WATCH",
    }

    for group in grouped_signals or []:
        if str(group.get("signal_level")) not in visible_levels:
            continue

        detail = group.get("display_detail") or {}
        short_factors = detail.get("short_factors", []) or []
        sweep_factor = find_factor(short_factors, "liquidity_sweep") or {}
        debug = sweep_factor.get("debug") or {}

        fields = [
            "SWEEP_DEBUG",
            f"symbol={format_debug_value(group.get('symbol'))}",
            f"exchange={format_debug_value(detail.get('exchange'))}",
            f"signal={format_debug_value(detail.get('signal_level') or group.get('signal_level'))}",
            f"status={format_debug_value(sweep_factor.get('status'))}",
        ]

        if debug:
            ordered_debug_keys = [
                "result",
                "level_tf",
                "level_type",
                "level_price",
                "source_index",
                "source_time",
                "age_bars",
                "age_hours",
                "touches",
                "quality",
                "reaction_pct",
                "prominence_pct",
                "confirm_tf",
                "confirm_time",
                "confirm_high",
                "confirm_close",
                "reason",
            ]

            for key in ordered_debug_keys:
                if key in debug:
                    fields.append(f"{key}={format_debug_value(debug.get(key))}")
        else:
            fields.append(f"detail={format_debug_value(sweep_factor.get('detail'))}")

        print(" ".join(fields))


def log_rsi_divergence_debug_for_grouped_signals(grouped_signals):
    """Print RSI_DIVERGENCE_DEBUG lines for Telegram-visible signals."""

    visible_levels = {
        "HIGH_PRIORITY_SHORT_WATCH",
        "SHORT_WATCH",
        "OVERHEAT_WATCH",
    }

    for group in grouped_signals or []:
        if str(group.get("signal_level")) not in visible_levels:
            continue

        detail = group.get("display_detail") or {}
        short_factors = detail.get("short_factors", []) or []
        factor = find_factor(short_factors, "rsi_divergence") or {}
        events = factor.get("events", []) if isinstance(factor, dict) else []

        event_parts = []
        for event in events or []:
            if not isinstance(event, dict):
                continue
            event_parts.append(
                ":".join([
                    str(event.get("timeframe", "N/A")),
                    str(event.get("status", "N/A")),
                    f"priorHigh{format_debug_value(event.get('prior_price_high'))}",
                    f"newHigh{format_debug_value(event.get('candidate_price_high'))}",
                    f"priorRSI{format_debug_value(event.get('prior_rsi_high'))}",
                    f"newRSI{format_debug_value(event.get('candidate_rsi_high'))}",
                    f"rsiDrop{format_debug_value(event.get('rsi_drop'))}",
                ])
            )

        fields = [
            "RSI_DIVERGENCE_DEBUG",
            f"symbol={format_debug_value(group.get('symbol'))}",
            f"exchange={format_debug_value(detail.get('exchange'))}",
            f"signal={format_debug_value(detail.get('signal_level') or group.get('signal_level'))}",
            f"status={format_debug_value(factor.get('status'))}",
            f"detail={format_debug_value(factor.get('detail'))}",
            f"events={'|'.join(event_parts) if event_parts else 'none'}",
        ]
        print(" ".join(fields))


def log_open_levels_debug_for_grouped_signals(grouped_signals):
    """
    Print one compact OPEN_LEVELS_DEBUG line per Telegram-visible signal.

    This stays in GitHub Actions logs only and helps validate D/W/M/Y open
    resistance detection without making Telegram noisy.
    """

    visible_levels = {
        "HIGH_PRIORITY_SHORT_WATCH",
        "SHORT_WATCH",
        "OVERHEAT_WATCH",
    }

    for group in grouped_signals or []:
        if str(group.get("signal_level")) not in visible_levels:
            continue

        detail = group.get("display_detail") or {}
        short_factors = detail.get("short_factors", []) or []
        open_factor = find_factor(short_factors, "open_levels") or {}
        debug = open_factor.get("debug") or {}
        events = open_factor.get("events") if isinstance(open_factor, dict) else []
        if not isinstance(events, list):
            events = []

        event_parts = []
        for event in events:
            if not isinstance(event, dict):
                continue
            label = format_debug_value(event.get("label"))
            state = format_debug_value(event.get("state"))
            weight = format_debug_value(event.get("weight"))
            open_price = format_debug_value(event.get("open"))
            confirm_tf = format_debug_value(event.get("confirm_tf"))
            window_bars = format_debug_value(event.get("window_bars"))
            confirm_high = format_debug_value(event.get("confirm_high"))
            confirm_close = format_debug_value(event.get("confirm_close"))
            event_parts.append(
                f"{label}:{state}:w{weight}:open{open_price}:tf{confirm_tf}:win{window_bars}:high{confirm_high}:close{confirm_close}"
            )

        levels = debug.get("levels") if isinstance(debug, dict) else {}
        if not isinstance(levels, dict):
            levels = {}

        level_parts = []
        for label in ["D", "W", "M", "Y"]:
            if label in levels:
                level_parts.append(f"{label}={format_debug_value(levels.get(label))}")

        sources = debug.get("levels_source") if isinstance(debug, dict) else {}
        if not isinstance(sources, dict):
            sources = {}
        source_parts = []
        for label in ["D", "W", "M", "Y"]:
            if label in sources:
                source_parts.append(f"{label}={format_debug_value(sources.get(label))}")

        intraday_levels = debug.get("levels_intraday") if isinstance(debug, dict) else {}
        if not isinstance(intraday_levels, dict):
            intraday_levels = {}
        intraday_parts = []
        for label in ["D", "W", "M", "Y"]:
            if label in intraday_levels:
                intraday_parts.append(f"{label}={format_debug_value(intraday_levels.get(label))}")

        intraday_1h_levels = debug.get("levels_intraday_1h") if isinstance(debug, dict) else {}
        if not isinstance(intraday_1h_levels, dict):
            intraday_1h_levels = {}
        intraday_1h_parts = []
        for label in ["D", "W", "M", "Y"]:
            if label in intraday_1h_levels:
                intraday_1h_parts.append(f"{label}={format_debug_value(intraday_1h_levels.get(label))}")

        intraday_4h_levels = debug.get("levels_intraday_4h") if isinstance(debug, dict) else {}
        if not isinstance(intraday_4h_levels, dict):
            intraday_4h_levels = {}
        intraday_4h_parts = []
        for label in ["D", "W", "M", "Y"]:
            if label in intraday_4h_levels:
                intraday_4h_parts.append(f"{label}={format_debug_value(intraday_4h_levels.get(label))}")

        daily_levels = debug.get("levels_daily") if isinstance(debug, dict) else {}
        if not isinstance(daily_levels, dict):
            daily_levels = {}
        daily_parts = []
        for label in ["D", "W", "M", "Y"]:
            if label in daily_levels:
                daily_parts.append(f"{label}={format_debug_value(daily_levels.get(label))}")

        fields = [
            "OPEN_LEVELS_DEBUG",
            f"symbol={format_debug_value(group.get('symbol'))}",
            f"exchange={format_debug_value(detail.get('exchange'))}",
            f"signal={format_debug_value(detail.get('signal_level') or group.get('signal_level'))}",
            f"status={format_debug_value(open_factor.get('status'))}",
            f"score={format_debug_value(open_factor.get('open_context_score', 0.0))}",
            f"events={format_debug_value('|'.join(event_parts) if event_parts else 'none')}",
            f"levels={format_debug_value('|'.join(level_parts) if level_parts else 'none')}",
            f"levels_source={format_debug_value('|'.join(source_parts) if source_parts else 'none')}",
            f"levels_intraday_selected={format_debug_value('|'.join(intraday_parts) if intraday_parts else 'none')}",
            f"levels_1h_utc={format_debug_value('|'.join(intraday_1h_parts) if intraday_1h_parts else 'none')}",
            f"levels_4h_utc={format_debug_value('|'.join(intraday_4h_parts) if intraday_4h_parts else 'none')}",
            f"levels_1d_utc={format_debug_value('|'.join(daily_parts) if daily_parts else 'none')}",
            f"D_window={format_debug_value(debug.get('D_window_tf'))}:{format_debug_value(debug.get('D_window_bars'))}",
            f"HTF_window={format_debug_value(debug.get('HTF_window_tf'))}:{format_debug_value(debug.get('HTF_window_bars'))}",
            f"detail={format_debug_value(open_factor.get('detail'))}",
        ]

        print(" ".join(fields))


def log_rejection_debug_for_grouped_signals(grouped_signals):
    """Print REJECTION_DEBUG lines for Telegram-visible signals."""

    visible_levels = {
        "HIGH_PRIORITY_SHORT_WATCH",
        "SHORT_WATCH",
        "OVERHEAT_WATCH",
    }

    for group in grouped_signals or []:
        if str(group.get("signal_level")) not in visible_levels:
            continue

        detail = group.get("display_detail") or {}
        short_factors = detail.get("short_factors", []) or []
        factor = find_factor(short_factors, "rejection_candle") or {}
        events = factor.get("events") if isinstance(factor, dict) else []
        if not isinstance(events, list):
            events = []

        event_parts = []
        for event in events:
            if not isinstance(event, dict):
                continue
            event_parts.append(
                ":".join([
                    format_debug_value(event.get("label")),
                    format_debug_value(event.get("state")),
                    format_debug_value(event.get("confirm_tf")),
                    f"open{format_debug_value(event.get('open'))}",
                    f"high{format_debug_value(event.get('confirm_high'))}",
                    f"close{format_debug_value(event.get('confirm_close'))}",
                    f"wick{format_debug_value(event.get('upper_wick_pct'))}",
                    f"closepos{format_debug_value(event.get('close_position'))}",
                ])
            )

        debug = factor.get("debug") if isinstance(factor, dict) else {}
        if not isinstance(debug, dict):
            debug = {}

        levels = debug.get("levels") if isinstance(debug, dict) else {}
        if not isinstance(levels, dict):
            levels = {}
        level_parts = []
        for label in ["D", "W", "M", "Y"]:
            if label in levels:
                level_parts.append(f"{label}={format_debug_value(levels.get(label))}")

        sources = debug.get("levels_source") if isinstance(debug, dict) else {}
        if not isinstance(sources, dict):
            sources = {}
        source_parts = []
        for label in ["D", "W", "M", "Y"]:
            if label in sources:
                source_parts.append(f"{label}={format_debug_value(sources.get(label))}")

        fields = [
            "REJECTION_DEBUG",
            f"symbol={format_debug_value(group.get('symbol'))}",
            f"exchange={format_debug_value(detail.get('exchange'))}",
            f"signal={format_debug_value(detail.get('signal_level') or group.get('signal_level'))}",
            f"status={format_debug_value(factor.get('status'))}",
            f"detail={format_debug_value(factor.get('detail'))}",
            f"events={format_debug_value('|'.join(event_parts) if event_parts else 'none')}",
            f"levels={format_debug_value('|'.join(level_parts) if level_parts else 'none')}",
            f"levels_source={format_debug_value('|'.join(source_parts) if source_parts else 'none')}",
        ]

        print(" ".join(fields))


def format_telegram_report(grouped_signals, data_warning=None):
    """
    Compact user-facing Telegram report.

    PUMP WATCH is intentionally hidden from Telegram. It remains available in
    internal scoring/output tables, but Telegram shows only actionable watchlist
    context: OVERHEAT, SHORT WATCH and HIGH PRIORITY SHORT WATCH.
    """

    now_kyiv = datetime.now(KYIV_TZ).strftime("%Y-%m-%d %H:%M Kyiv")

    visible_levels = {
        "HIGH_PRIORITY_SHORT_WATCH",
        "SHORT_WATCH",
        "OVERHEAT_WATCH",
    }

    visible_groups = [
        group for group in (grouped_signals or [])
        if str(group.get("signal_level")) in visible_levels
    ]

    total_visible_count = len(visible_groups)
    display_groups = visible_groups[:TELEGRAM_MAX_SIGNALS]
    displayed_count = len(display_groups)

    lines = []
    lines.append("📊 <b>Market Heat Scanner</b>")
    lines.append(f"{html.escape(now_kyiv)} | {html.escape(SCRIPT_VERSION)}")

    if data_warning:
        lines.append(html.escape(str(data_warning)))

    if total_visible_count > displayed_count:
        lines.append(f"Shown: <b>{displayed_count}</b>/<b>{total_visible_count}</b>")

    if not display_groups:
        lines.append("")
        lines.append("✅ No overheat or short-watch signals.")
    else:
        for idx, group in enumerate(display_groups):
            signal_label = telegram_signal_label(group["signal_level"])
            symbol = html.escape(str(group["symbol"]))
            detail = group.get("display_detail") or {}

            price = html.escape(str(detail.get("price", "N/A")))
            rsi_1h_live = html.escape(str(detail.get("rsi_1h_live", "N/A")))
            rsi_1h_closed = html.escape(str(detail.get("rsi_1h_closed", "N/A")))
            rsi_4h_live = html.escape(str(detail.get("rsi_4h_live", "N/A")))
            chg_24h = html.escape(str(detail.get("chg_24h_%", "N/A")))
            vol_24h = html.escape(str(detail.get("vol_24h", "N/A")))
            vol_chg = html.escape(str(detail.get("vol_chg_24h_%", "N/A")))
            setup_status = html.escape(str(detail.get("setup_status", "N/A")))

            short_factors = detail.get("short_factors", []) or []
            sweep_factor = find_factor(short_factors, "liquidity_sweep") or {}
            rejection_factor = find_factor(short_factors, "rejection_candle") or {}

            sweep_icon = factor_status_icon(sweep_factor)
            rejection_icon = factor_status_icon(rejection_factor)
            sweep_detail = html.escape(format_sweep_detail_for_telegram(short_factors))
            rejection_detail = html.escape(format_rejection_detail_for_telegram(short_factors))

            reason_source = detail.get("reason") or setup_status
            reason = html.escape(format_reason_for_telegram(reason_source))

            lines.append("")
            lines.append(f"{idx + 1}) {signal_label} — <code>{symbol}</code>")
            lines.append(f"Price {price} | 24h {chg_24h}% | Vol {vol_24h} | ΔVol {vol_chg}%")
            lines.append(f"RSI 1H Live {rsi_1h_live} | Closed {rsi_1h_closed} | 4H Live {rsi_4h_live}")
            oi_line = html.escape(format_oi_line_for_telegram(detail))
            if oi_line:
                lines.append(oi_line)
            divergence_detail = html.escape(format_divergence_detail_for_telegram(short_factors))
            if divergence_detail:
                lines.append(f"Div: {divergence_detail}")
            open_levels_detail = html.escape(format_open_levels_detail_for_telegram(short_factors))
            if open_levels_detail:
                lines.append(f"Open Levels: {open_levels_detail}")

            lines.append("")
            lines.append(f"Sweep: {sweep_icon} {sweep_detail}")
            lines.append(f"Rejection: {rejection_icon} {rejection_detail}")
            lines.append(f"Reason: {reason}")

            if idx != len(display_groups) - 1:
                lines.append("────────────")

    lines.append("")
    lines.append("Planned: ⚪ Funding | ⚪ Vol climax | ⚪ Failed BO | ⚪ MSS")

    return "\n".join(lines)
