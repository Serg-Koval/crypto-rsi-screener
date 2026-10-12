from screener.factors.sweep import detect_liquidity_sweep
from screener.factors.open_levels import detect_open_levels_context
from screener.factors.rejection import detect_rejection_candle
from screener.factors.divergence import (
    detect_rsi_bearish_divergence,
    detect_oi_bearish_divergence,
)


def analyze_short_factors(df_1h, df_4h=None, df_1d=None, current_price=None, oi_history=None, provider="provider"):
    """
    Current simplified short-watch factor set.

    Active factors:
    - Liquidity sweep: confirmed price-action trigger.
    - Open levels: D/W/M/Y resistance location/context.
    - Rejection candle: confirmed trigger only at D/W/M/Y open-level resistance.
    - RSI/OI bearish divergence: context layer and status modifier.

    Removed from active model:
    - Premium/Discount: previous implementation was range-position, not a full ICT dealing range.
    - New High: too noisy for Telegram and classification.
    """

    open_levels_factor = detect_open_levels_context(
        df_1h=df_1h,
        df_4h=df_4h,
        df_1d=df_1d,
        current_price=current_price,
    )

    factors = [
        detect_liquidity_sweep(df_1h, df_4h=df_4h),
        open_levels_factor,
        detect_rejection_candle(df_1h=df_1h, df_4h=df_4h, df_1d=df_1d, current_price=current_price),
        detect_rsi_bearish_divergence(df_1h=df_1h, df_4h=df_4h),
        detect_oi_bearish_divergence(df_1h=df_1h, df_4h=df_4h, oi_history=oi_history, provider=provider),
    ]

    score = sum(int(factor.get("points", 0)) for factor in factors)
    confirmed_count = sum(1 for factor in factors if factor.get("status") == "confirmed")
    total_count = len(factors)

    return {
        "factors": factors,
        "score": score,
        "confirmed_count": confirmed_count,
        "total_count": total_count,
    }
