import pandas as pd

from screener.common import (
    format_large_number,
    get_short_factor,
    safe_float,
)
from screener.config import (
    EXTREME_PUMP_PRICE_CHANGE_24H,
    EXTREME_PUMP_RSI_1H_LIVE,
    EXTREME_PUMP_RSI_4H_LIVE,
    MIN_VOLUME_USD_24H,
    OVERHEAT_WATCH_MIN_VOLUME_USD_24H,
    OVERHEAT_WATCH_RSI_1H_CLOSED,
    OVERHEAT_WATCH_RSI_1H_LIVE,
    OVERHEAT_WATCH_RSI_4H_LIVE,
    PUMP_WATCH_PRICE_CHANGE_24H,
    RSI_1H_CLOSED_CONFIRMATION,
)
from screener.factors.divergence import (
    has_any_bearish_divergence,
    has_confirmed_oi_divergence,
    has_confirmed_rsi_divergence,
)


def detect_overheat_watch_context(
    rsi_1h_live,
    rsi_1h_closed,
    exact_volume_24h,
    price_change_24h,  # kept for callers; no longer affects the overheat status
    rsi_4h_live=None,
):
    """Detect pump + RSI heat context for watchlist visibility.

    r36 update: 4H RSI heat can validate the overheat context even when
    the 1H RSI has already cooled after the first impulse. This catches
    cases like UNI where the 4H RSI is still overheated but the 1H live RSI
    has pulled back below the strict 1H heat pair.
    """

    rsi_1h_live_value = safe_float(rsi_1h_live)
    rsi_1h_closed_value = safe_float(rsi_1h_closed)
    rsi_4h_live_value = safe_float(rsi_4h_live)
    volume_24h_value = safe_float(exact_volume_24h)

    has_1h_heat = (
        rsi_1h_live_value >= OVERHEAT_WATCH_RSI_1H_LIVE
        and rsi_1h_closed_value >= OVERHEAT_WATCH_RSI_1H_CLOSED
    )
    has_4h_heat = rsi_4h_live_value >= OVERHEAT_WATCH_RSI_4H_LIVE
    has_rsi_heat = bool(has_1h_heat or has_4h_heat)

    common_checks = [
        (
            volume_24h_value >= OVERHEAT_WATCH_MIN_VOLUME_USD_24H,
            f"24h volume >= {format_large_number(OVERHEAT_WATCH_MIN_VOLUME_USD_24H)}",
        ),
    ]

    rsi_passed = []
    if has_1h_heat:
        rsi_passed.append(f"1H RSI heat: live >= {OVERHEAT_WATCH_RSI_1H_LIVE} and closed >= {OVERHEAT_WATCH_RSI_1H_CLOSED}")
    if has_4h_heat:
        rsi_passed.append(f"4H RSI heat >= {OVERHEAT_WATCH_RSI_4H_LIVE}")

    passed = [label for ok, label in common_checks if ok] + rsi_passed
    missing = [label for ok, label in common_checks if not ok]

    if not has_rsi_heat:
        missing.append(
            f"RSI heat: 1H live/closed >= {OVERHEAT_WATCH_RSI_1H_LIVE}/{OVERHEAT_WATCH_RSI_1H_CLOSED} "
            f"or 4H live >= {OVERHEAT_WATCH_RSI_4H_LIVE}"
        )

    is_overheat = len(missing) == 0

    return {
        "is_overheat": is_overheat,
        "passed": passed,
        "missing": missing,
        "has_1h_heat": bool(has_1h_heat),
        "has_4h_heat": bool(has_4h_heat),
        "reason": " + ".join(passed) if is_overheat else "Missing " + " / ".join(missing),
    }


def calculate_scores(
    rsi_1h_live,
    rsi_1h_closed,
    rsi_4h_live,
    exact_volume_24h,
    volume_change_24h,
    price_change_24h,
    short_setup_score,
):
    rsi_1h_live_value = safe_float(rsi_1h_live)
    rsi_1h_closed_value = safe_float(rsi_1h_closed)
    rsi_4h_live_value = safe_float(rsi_4h_live)
    volume_24h_value = safe_float(exact_volume_24h)
    price_change_24h_value = safe_float(price_change_24h)

    pump_score = 0

    if price_change_24h_value >= 40:
        pump_score = 3
    elif price_change_24h_value >= EXTREME_PUMP_PRICE_CHANGE_24H:
        pump_score = 2
    elif price_change_24h_value >= PUMP_WATCH_PRICE_CHANGE_24H:
        pump_score = 1

    rsi_score = 0

    if rsi_1h_live_value >= EXTREME_PUMP_RSI_1H_LIVE:
        rsi_score += 2
    elif rsi_1h_live_value >= OVERHEAT_WATCH_RSI_1H_LIVE:
        rsi_score += 1

    if rsi_4h_live_value >= EXTREME_PUMP_RSI_4H_LIVE:
        rsi_score += 2
    elif rsi_4h_live_value >= 75:
        rsi_score += 1

    if rsi_1h_closed_value >= RSI_1H_CLOSED_CONFIRMATION:
        rsi_score += 1

    volume_score = 0
    volume_ok = volume_24h_value >= MIN_VOLUME_USD_24H

    if volume_ok:
        volume_score += 1

    vol_chg = None if volume_change_24h is None or pd.isna(volume_change_24h) else float(volume_change_24h)

    if vol_chg is not None:
        if vol_chg >= 300:
            volume_score += 3
        elif vol_chg >= 100:
            volume_score += 2
        elif vol_chg >= 50:
            volume_score += 1

    overheat_context = detect_overheat_watch_context(
        rsi_1h_live=rsi_1h_live_value,
        rsi_1h_closed=rsi_1h_closed_value,
        rsi_4h_live=rsi_4h_live_value,
        exact_volume_24h=volume_24h_value,
        price_change_24h=price_change_24h_value,
    )

    has_basic_pump_context = (
        price_change_24h_value >= PUMP_WATCH_PRICE_CHANGE_24H and
        volume_ok
    )

    has_strong_pump_context = (
        price_change_24h_value >= EXTREME_PUMP_PRICE_CHANGE_24H and
        volume_ok
    )

    has_extreme_pump_context = (
        price_change_24h_value >= 40 and
        volume_ok
    )

    final_score = pump_score + rsi_score + volume_score + short_setup_score

    return {
        "pump_score": pump_score,
        "rsi_score": rsi_score,
        "volume_score": volume_score,
        "short_setup_score": int(short_setup_score),
        "final_score": int(final_score),
        "volume_ok": bool(volume_ok),
        "has_basic_pump_context": bool(has_basic_pump_context),
        "has_strong_pump_context": bool(has_strong_pump_context),
        "has_extreme_pump_context": bool(has_extreme_pump_context),
        "has_overheat_context": bool(overheat_context["is_overheat"]),
        "has_rsi_heat": bool(overheat_context["has_1h_heat"] or overheat_context["has_4h_heat"]),
        "overheat_reason": str(overheat_context["reason"]),
    }


def is_factor_confirmed(short_factors, key):
    factor = get_short_factor(short_factors, key)
    return bool(factor and factor.get("status") == "confirmed")


def calculate_location_trigger_context(short_factors):
    """
    Split short analysis into location context and trigger confirmation.

    Location/context:
    - Open Levels D/W/M/Y resistance context.

    Trigger / confirmation:
    - Confirmed liquidity sweep.
    - Confirmed rejection candle at D/W/M/Y open-level resistance.
    """

    open_levels = get_short_factor(short_factors, "open_levels") or {}
    liquidity = get_short_factor(short_factors, "liquidity_sweep") or {}

    liquidity_confirmed = is_factor_confirmed(short_factors, "liquidity_sweep")
    rejection_confirmed = is_factor_confirmed(short_factors, "rejection_candle")
    liquidity_candidate = liquidity.get("status") == "candidate"

    open_context_score = float(safe_float(open_levels.get("open_context_score", 0.0), default=0.0))

    open_events = open_levels.get("events", []) if isinstance(open_levels, dict) else []
    if not isinstance(open_events, list):
        open_events = []

    htf_open_context_score = sum(
        float(safe_float(event.get("weight", 0.0), default=0.0))
        for event in open_events
        if isinstance(event, dict) and event.get("label") in ("W", "M", "Y")
    )
    d_open_context_score = sum(
        float(safe_float(event.get("weight", 0.0), default=0.0))
        for event in open_events
        if isinstance(event, dict) and event.get("label") == "D"
    )

    tested_open_labels = [
        str(event.get("label"))
        for event in open_events
        if isinstance(event, dict) and event.get("state") == "tested"
    ]
    live_test_open_labels = [
        str(event.get("label"))
        for event in open_events
        if isinstance(event, dict) and event.get("state") == "live_test"
    ]
    near_open_labels = [
        str(event.get("label"))
        for event in open_events
        if isinstance(event, dict) and event.get("state") == "near"
    ]

    location_score = open_context_score
    trigger_count = int(liquidity_confirmed) + int(rejection_confirmed)
    early_trigger_count = int(liquidity_candidate)

    trigger_parts = []

    if liquidity_confirmed:
        trigger_parts.append("liquidity sweep")

    if rejection_confirmed:
        trigger_parts.append("open-level rejection")

    return {
        "location_score": float(location_score),
        "base_location_score": 0.0,
        "open_context_score": float(open_context_score),
        "htf_open_context_score": float(htf_open_context_score),
        "d_open_context_score": float(d_open_context_score),
        "tested_open_labels": tested_open_labels,
        "live_test_open_labels": live_test_open_labels,
        "near_open_labels": near_open_labels,
        "has_open_resistance": bool(open_context_score > 0),
        "has_htf_open_resistance": bool(htf_open_context_score > 0),
        "has_strong_open_resistance": bool(htf_open_context_score >= 2.0 or open_context_score >= 2.0),
        "trigger_count": trigger_count,
        "early_trigger_count": early_trigger_count,
        "liquidity_confirmed": liquidity_confirmed,
        "rejection_confirmed": rejection_confirmed,
        "liquidity_candidate": liquidity_candidate,
        "trigger_parts": trigger_parts,
    }


def divergence_summary_for_reason(short_factors):
    rsi = has_confirmed_rsi_divergence(short_factors)
    oi = has_confirmed_oi_divergence(short_factors)

    if rsi and oi:
        return "RSI/OI divergence"
    if rsi:
        return "RSI divergence"
    if oi:
        return "OI divergence"
    return "bearish divergence"


def build_setup_status(signal_level, scores, short_factors):
    context = calculate_location_trigger_context(short_factors)
    has_overheat_context = bool(scores.get("has_overheat_context", False))
    has_open_resistance = bool(context.get("has_open_resistance", False))
    liquidity_confirmed = bool(context.get("liquidity_confirmed", False))
    rejection_confirmed = bool(context.get("rejection_confirmed", False))
    has_divergence = has_any_bearish_divergence(short_factors)
    div_text = divergence_summary_for_reason(short_factors)

    if signal_level == "HIGH_PRIORITY_SHORT_WATCH":
        if liquidity_confirmed and rejection_confirmed:
            return f"High priority watch — liquidity sweep + open-level rejection + {div_text}"
        if rejection_confirmed:
            return f"High priority watch — open-level rejection + {div_text}"
        if liquidity_confirmed:
            return f"High priority watch — liquidity sweep + {div_text}"
        return f"High priority watch — RSI heat + {div_text}"

    if signal_level == "SHORT_WATCH":
        if has_overheat_context and has_divergence and not (liquidity_confirmed or rejection_confirmed):
            return f"Short watch — RSI heat + {div_text}; waiting for short trigger"
        if liquidity_confirmed and rejection_confirmed:
            if has_divergence:
                return f"Short watch — liquidity sweep + open-level rejection + {div_text}"
            return "Short watch — liquidity sweep + open-level rejection"
        if liquidity_confirmed:
            if has_divergence:
                return f"Short watch — liquidity sweep + {div_text}"
            if has_open_resistance:
                return "Short watch — liquidity sweep + open resistance"
            return "Short watch — confirmed liquidity sweep detected"
        if rejection_confirmed:
            if has_divergence:
                return f"Short watch — open-level rejection + {div_text}"
            return "Short watch — open-level rejection"
        return "Short watch — watchlist context"

    if signal_level == "OVERHEAT_WATCH":
        if has_open_resistance:
            return "Overheat watch — RSI heat near open resistance; waiting for short trigger"
        return "Overheat watch — RSI heat confirmed; waiting for short trigger"

    return "No watch setup"


def build_watch_reason(signal_level, scores, short_factors):
    context = calculate_location_trigger_context(short_factors)
    has_open_resistance = bool(context.get("has_open_resistance", False))
    liquidity_confirmed = bool(context.get("liquidity_confirmed", False))
    rejection_confirmed = bool(context.get("rejection_confirmed", False))
    has_divergence = has_any_bearish_divergence(short_factors)
    div_text = divergence_summary_for_reason(short_factors)

    if signal_level == "HIGH_PRIORITY_SHORT_WATCH":
        parts = []
        if liquidity_confirmed:
            parts.append("liquidity sweep")
        if rejection_confirmed:
            parts.append("open-level rejection")
        if has_divergence:
            parts.append(div_text)
        return " + ".join(parts) if parts else "high priority short-watch context"

    if signal_level == "SHORT_WATCH":
        if bool(scores.get("has_overheat_context", False)) and has_divergence and not (liquidity_confirmed or rejection_confirmed):
            return f"RSI heat + {div_text}"
        if rejection_confirmed and liquidity_confirmed:
            return "liquidity sweep + open-level rejection"
        if rejection_confirmed:
            if has_divergence:
                return f"open-level rejection + {div_text}"
            return "open-level rejection"
        if liquidity_confirmed:
            if has_divergence:
                return f"liquidity sweep + {div_text}"
            if has_open_resistance:
                return "confirmed liquidity sweep + open resistance"
            return "confirmed liquidity sweep"
        if has_divergence:
            return div_text
        return "short-watch context"

    if signal_level == "OVERHEAT_WATCH":
        return "RSI heat confirmed; waiting for short trigger"

    return "Watch filters not passed"


def classify_watch_signal(scores, short_factors=None):
    """
    r33 classification.

    - 🟡 OVERHEAT WATCH: pump + RSI heat, no divergence/trigger.
    - 🟠 SHORT WATCH: sweep/rejection + RSI heat (trigger alone is not enough;
      it needs RSI heat) OR pump + RSI heat + RSI/OI divergence.
    - 🔴 HIGH PRIORITY: pump + RSI heat + RSI/OI divergence + sweep/rejection.

    All statuses remain watchlist statuses, not entry signals.
    """

    short_factors = short_factors or []

    context = calculate_location_trigger_context(short_factors)
    location_score = float(context.get("location_score", 0.0))
    trigger_count = int(context.get("trigger_count", 0))
    open_context_score = float(context.get("open_context_score", 0.0))
    htf_open_context_score = float(context.get("htf_open_context_score", 0.0))

    has_overheat_context = bool(scores.get("has_overheat_context", False))
    has_rsi_heat = bool(scores.get("has_rsi_heat", False))
    has_trigger = trigger_count >= 1
    has_divergence = has_any_bearish_divergence(short_factors)

    # Main decision tree agreed in r33.
    if has_overheat_context and has_divergence and has_trigger:
        signal_level = "HIGH_PRIORITY_SHORT_WATCH"
    elif has_trigger and has_rsi_heat:
        # Trigger alone is not enough; it needs RSI heat.
        signal_level = "SHORT_WATCH"
    elif has_overheat_context and has_divergence:
        signal_level = "SHORT_WATCH"
    elif has_overheat_context:
        signal_level = "OVERHEAT_WATCH"
    else:
        signal_level = "NO_SIGNAL"

    return {
        "signal_level": signal_level,
        "reason": build_watch_reason(signal_level, scores, short_factors),
        "location_score": float(location_score),
        "open_context_score": float(open_context_score),
        "htf_open_context_score": float(htf_open_context_score),
        "trigger_count": int(trigger_count),
        "setup_status": build_setup_status(signal_level, scores, short_factors),
    }


def classify_signal(
    rsi_1h_live,
    rsi_1h_closed,
    rsi_4h_live,
    exact_volume_24h,
    volume_change_24h,
    price_change_24h,
    short_setup_score=0,
    short_factors=None,
):
    """
    Watchlist signal model.

    This model separates:
    - pump / heat context;
    - location context;
    - short trigger confirmation.

    Closed 4H RSI is intentionally not used.
    """

    short_factors = short_factors or []

    scores = calculate_scores(
        rsi_1h_live=rsi_1h_live,
        rsi_1h_closed=rsi_1h_closed,
        rsi_4h_live=rsi_4h_live,
        exact_volume_24h=exact_volume_24h,
        volume_change_24h=volume_change_24h,
        price_change_24h=price_change_24h,
        short_setup_score=short_setup_score,
    )

    classification = classify_watch_signal(scores, short_factors=short_factors)
    classification.update(scores)

    return classification


def get_signal_rank(signal_level):
    ranks = {
        "HIGH_PRIORITY_SHORT_WATCH": 4,
        "SHORT_WATCH": 3,
        "OVERHEAT_WATCH": 2,
        "PUMP_WATCH": 1,
        "NO_SIGNAL": 0,
    }

    return ranks.get(signal_level, 0)
