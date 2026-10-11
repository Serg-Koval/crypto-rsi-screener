from zoneinfo import ZoneInfo


KYIV_TZ = ZoneInfo("Europe/Kyiv")

SCRIPT_VERSION = "bybit-proxy-v1-20261010-r37_1"

RSI_PERIOD = 14

# Watch thresholds.
# Closed 4H RSI is intentionally not used in signal classification.
EARLY_PUMP_PRICE_CHANGE_24H = 8

PUMP_WATCH_PRICE_CHANGE_24H = 10

OVERHEAT_WATCH_RSI_1H_LIVE = 82
OVERHEAT_WATCH_RSI_1H_CLOSED = 80
OVERHEAT_WATCH_RSI_4H_LIVE = 80
OVERHEAT_WATCH_PRICE_CHANGE_24H = 15
OVERHEAT_WATCH_MIN_VOLUME_USD_24H = 5_000_000

EXTREME_PUMP_RSI_1H_LIVE = 85
EXTREME_PUMP_RSI_4H_LIVE = 80
EXTREME_PUMP_PRICE_CHANGE_24H = 20

RSI_1H_CLOSED_CONFIRMATION = OVERHEAT_WATCH_RSI_1H_CLOSED

# Entry filter for Telegram short-list candidates.
# A confirmed sweep is not enough if RSI heat is too weak.
RSI_ENTRY_MIN_1H_LIVE = 65
RSI_ENTRY_MIN_1H_CLOSED = 65
RSI_ENTRY_MIN_4H_LIVE = 68

MIN_PRICE_CHANGE_24H = EARLY_PUMP_PRICE_CHANGE_24H
MIN_VOLUME_USD_24H = OVERHEAT_WATCH_MIN_VOLUME_USD_24H

# Short-analysis settings based on 1H OHLCV only.
LIQUIDITY_SWEEP_LOOKBACKS = {
    "24H": 24,
    "12H": 12,
}
# Require a real sweep, not a one-tick wick above the previous high.
LIQUIDITY_SWEEP_MIN_BREAK_PCT = 0.001      # 0.10% above previous high
LIQUIDITY_SWEEP_MIN_CLOSE_BACK_PCT = 0.0005 # 0.05% close back below previous high
LIQUIDITY_SWEEP_LIVE_INVALIDATION_PCT = 0.0005 # current/live candle reclaimed level
LIQUIDITY_SWEEP_MIN_LEVEL_AGE_BARS = 5 # swept high must be at least 5 bars old on its own timeframe
LIQUIDITY_SWEEP_MIN_REACTION_PCT = 0.015 # price must have reacted at least 1.5% from the level before sweep
LIQUIDITY_SWEEP_EQUAL_HIGH_TOLERANCE_PCT = 0.0025 # highs within 0.25% are treated as equal-high liquidity
LIQUIDITY_SWEEP_MIN_UPPER_WICK_RANGE_PCT = 0.30 # sweep candle should show rejection/failure character
LIQUIDITY_SWEEP_SWING_LEFT_BARS = 2
LIQUIDITY_SWEEP_SWING_RIGHT_BARS = 2
# A single swing high must stand out from nearby candles.
# This prevents tiny intrapump pivots from being shown as chart-level liquidity highs.
LIQUIDITY_SWEEP_MIN_SINGLE_HIGH_PROMINENCE_PCT = 0.0020 # 0.20% above nearby highs
LIQUIDITY_SWEEP_LEVEL_LOOKBACK_1H = 240
LIQUIDITY_SWEEP_LEVEL_LOOKBACK_4H = 120

# Open levels are location/context amplifiers. They are not triggers, but they can strengthen signal_level when a confirmed sweep exists.
OPEN_LEVEL_NEAR_THRESHOLDS = {
    "D": 0.0015,   # 0.15% below daily open
    "W": 0.0035,   # 0.35% below weekly open
    "M": 0.0050,   # 0.50% below monthly open
    "Y": 0.0050,   # 0.50% below yearly open
}
# If price has reclaimed an open level, that level is no longer active resistance.
# This prevents old D/W/M/Y tests from being reported after price has already moved above the level.
OPEN_LEVEL_RECLAIM_INVALIDATION_PCT = 0.0  # any close back above the open level invalidates old resistance context
OPEN_LEVEL_CONTEXT_WEIGHTS = {
    "D": {"near": 0.50, "live_test": 0.75, "tested": 1.00},
    "W": {"near": 1.00, "live_test": 1.50, "tested": 2.00},
    "M": {"near": 1.00, "live_test": 1.50, "tested": 2.00},
    "Y": {"near": 1.00, "live_test": 1.50, "tested": 2.00},
}
# Open-level tests should be aligned with the recent setup window, not only the latest closed candle.
OPEN_LEVEL_RECENT_WINDOW_1H_BARS = 6
OPEN_LEVEL_RECENT_WINDOW_4H_BARS = 2

# Rejection Candle v1. Rejection is a trigger only when it happens at D/W/M/Y open-level resistance.
REJECTION_RECENT_WINDOW_1H_BARS = 6
REJECTION_RECENT_WINDOW_4H_BARS = 2
REJECTION_MIN_UPPER_WICK_RANGE_PCT = 0.35
REJECTION_MAX_CLOSE_POSITION_PCT = 0.55

# Open Interest context layer (r28).
# Context only: it is shown in Telegram but does not affect scoring yet.
OI_CHANGE_FLAT_THRESHOLD_PCT = 2.0
OI_CHANGE_1H_ACTIVE_THRESHOLD_PCT = 3.0
OI_CHANGE_STRONG_4H_THRESHOLD_PCT = 10.0
OI_CHANGE_NEGATIVE_THRESHOLD_PCT = -5.0

# Divergence context layer (r33).
# Divergence is now a real status modifier:
# - pump + RSI heat + RSI/OI divergence => SHORT WATCH;
# - pump + RSI heat + RSI/OI divergence + sweep/rejection => HIGH PRIORITY.
RSI_DIVERGENCE_SWING_LEFT_BARS = 2
RSI_DIVERGENCE_SWING_RIGHT_BARS = 2
RSI_DIVERGENCE_LOOKBACK_1H = 96
RSI_DIVERGENCE_LOOKBACK_4H = 60
RSI_DIVERGENCE_RECENT_WINDOW_1H = 12
RSI_DIVERGENCE_RECENT_WINDOW_4H = 8
RSI_DIVERGENCE_MIN_SWING_DISTANCE_1H = 5
RSI_DIVERGENCE_MIN_SWING_DISTANCE_4H = 3
RSI_DIVERGENCE_MIN_PRICE_HH_PCT = 0.002  # 0.20% higher high
RSI_DIVERGENCE_EQUAL_HIGH_TOLERANCE_PCT = 0.003  # +/-0.30% equal high
RSI_DIVERGENCE_MIN_RSI_DROP = 2.0        # RSI lower high by at least 2 points

OI_DIVERGENCE_LOOKBACK_1H = 96
OI_DIVERGENCE_LOOKBACK_4H = 60
OI_DIVERGENCE_RECENT_WINDOW_1H = 12
OI_DIVERGENCE_RECENT_WINDOW_4H = 8
OI_DIVERGENCE_MIN_SWING_DISTANCE_1H = 5
OI_DIVERGENCE_MIN_SWING_DISTANCE_4H = 3
OI_DIVERGENCE_MIN_PRICE_HH_PCT = 0.002
OI_DIVERGENCE_EQUAL_HIGH_TOLERANCE_PCT = 0.003
OI_DIVERGENCE_MIN_OI_DROP_PCT = 2.0
OI_DIVERGENCE_WEAK_SUPPORT_MAX_OI_CHANGE_PCT = 1.0

PRE_FILTER_TOP_N = 40
FINAL_TOP_N = 30
TELEGRAM_MAX_SIGNALS = 10
CANDLE_LIMIT_1H = 500
CANDLE_LIMIT_4H = 500
CANDLE_LIMIT_1D = 300

REQUEST_DELAY_SECONDS = 0.15

SEND_MESSAGE_IF_NO_SIGNALS = True

# Telegram delivery is best-effort. A temporary Telegram API timeout must not
# fail the whole GitHub Actions run after the screener has already completed.
TELEGRAM_SEND_TIMEOUT_SECONDS = 10
TELEGRAM_SEND_MAX_ATTEMPTS = 1
TELEGRAM_SEND_RETRY_SLEEP_SECONDS = 2

# Bybit data comes through the Cloudflare Worker proxy (GitHub Actions cannot
# reach api.bybit.com directly). Requests: <BYBIT_PROXY_URL>/bybit/v5/market/...
# with the X-Proxy-Secret header; both values come from environment variables.
BYBIT_PROXY_PATH_PREFIX = "/bybit"
BYBIT_PROXY_SECRET_HEADER = "X-Proxy-Secret"
BYBIT_CATEGORY = "linear"
BYBIT_CONTRACT_TYPE = "LinearPerpetual"
BYBIT_QUOTE_COIN = "USDT"
BYBIT_STATUS = "Trading"
BYBIT_INTERVALS = {"1H": "60", "4H": "240", "1D": "D"}
BYBIT_OI_INTERVAL = "1h"
BYBIT_OI_HISTORY_LIMIT = 200

# Data quality: if more than this share of candidates failed to analyze,
# the report is not sent and the run fails (see classify_data_quality).
DATA_ERROR_MAX_SHARE = 0.5
