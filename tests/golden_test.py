"""
Golden regression test for Market Heat Scanner.

Builds deterministic synthetic data (fixed seed + the make_*_case helpers from
main.py), runs analyze_short_factors and classify_signal on it and compares the
result with tests/golden_expected.json.

Usage:
    python tests/golden_test.py            # compare with the saved expectation
    python tests/golden_test.py --update   # (re)write golden_expected.json
                                           # ONLY on a version of main.py whose
                                           # behaviour is known to be correct
"""

import contextlib
import io
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import main as m  # noqa: E402

EXPECTED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_expected.json")
RANDOM_SEEDS = list(range(40))


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def clean(value):
    """Convert numpy/pandas values into stable JSON-friendly values."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return round(value, 9)
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def summarize_factors(analysis):
    return {
        "score": analysis["score"],
        "confirmed_count": analysis["confirmed_count"],
        "total_count": analysis["total_count"],
        "factors": [
            {
                "key": f.get("key"),
                "status": f.get("status"),
                "points": f.get("points"),
                "detail": f.get("detail"),
            }
            for f in analysis["factors"]
        ],
    }


def summarize_classification(result):
    keys = [
        "signal_level", "reason", "setup_status", "final_score",
        "pump_score", "rsi_score", "volume_score", "short_setup_score",
        "location_score", "trigger_count",
    ]
    return {k: result.get(k) for k in keys}


def run_quietly(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def safe_run(func):
    try:
        return func()
    except Exception as exc:  # recorded so that a changed failure is also detected
        return {"error": type(exc).__name__}


def with_rsi(df):
    if df is None:
        return None
    return m.calculate_rsi(df, period=m.RSI_PERIOD)


def last_rsi(df, live=True):
    if df is None or "rsi" not in df.columns:
        return 50.0
    series = df["rsi"].dropna()
    if series.empty:
        return 50.0
    return float(series.iloc[-1] if live else series.iloc[-2 if len(series) > 1 else -1])


def analyze_and_classify(df_1h, df_4h, df_1d, price, oi_history, price_change=25.0):
    df_1h = with_rsi(df_1h)
    df_4h = with_rsi(df_4h)

    analysis = m.analyze_short_factors(
        df_1h, df_4h, df_1d=df_1d, current_price=price,
        oi_history=oi_history, provider="Bybit",
    )

    classification = m.classify_signal(
        rsi_1h_live=last_rsi(df_1h, True),
        rsi_1h_closed=last_rsi(df_1h, False),
        rsi_4h_live=last_rsi(df_4h, True),
        exact_volume_24h=10_000_000,
        volume_change_24h=40.0,
        price_change_24h=price_change,
        short_setup_score=analysis["score"],
        short_factors=analysis["factors"],
    )

    return {
        "analysis": summarize_factors(analysis),
        "classification": summarize_classification(classification),
        "rsi": [round(last_rsi(df_1h, True), 6), round(last_rsi(df_1h, False), 6), round(last_rsi(df_4h, True), 6)],
    }


# ------------------------------------------------------------
# Scenarios from the existing make_*_case helpers
# ------------------------------------------------------------

def natural_role(df):
    """Detect candle timeframe of a synthetic frame: 1h / 4h / 1d."""
    if len(df) < 2:
        return "1h"
    step = (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds()
    if step >= 86400:
        return "1d"
    if step >= 4 * 3600:
        return "4h"
    return "1h"


def build_case_scenarios():
    scenarios = {}

    case_functions = {
        "4h_swing_sweep_age5": lambda: m.make_4h_swing_sweep_case(level_age_bars=5),
        "4h_swing_sweep_age4": lambda: m.make_4h_swing_sweep_case(level_age_bars=4),
        "4h_swing_sweep_age8_no4hconfirm": lambda: m.make_4h_swing_sweep_case(level_age_bars=8, confirm_with_4h=False),
        "4h_level_without_4h_sweep": m.make_4h_level_without_4h_sweep_case,
        "1h_intrabar_take_of_4h_level": m.make_1h_intrabar_take_of_4h_level_case,
        "1h_equal_high_sweep": m.make_1h_equal_high_sweep_case,
        "rolling_only_high_take": m.make_rolling_only_high_take_case,
        "1h_minor_micro_high_sweep": m.make_1h_minor_micro_high_sweep_case,
        "1h_level_with_4h_close_only": m.make_1h_level_with_4h_close_only_case,
    }

    flat_1h = m.make_flat_synthetic_1h()
    flat_4h = m.make_flat_synthetic_4h()

    for name, factory in case_functions.items():
        def run(factory=factory):
            df = factory()
            # Some helpers return a tuple (1h, 4h).
            if isinstance(df, tuple):
                df_1h, df_4h = df[0], df[1]
            elif natural_role(df) == "4h":
                df_1h, df_4h = flat_1h, df
            else:
                df_1h, df_4h = df, flat_4h
            price = float(df_1h["close"].iloc[-1])
            return analyze_and_classify(df_1h, df_4h, None, price, None)

        scenarios[f"case/{name}"] = safe_run(lambda run=run: run_quietly(run))

    # Open-level cases: cross product of 1H and 4H helpers with the 1D helper.
    ol_1h = {
        "1h_confirm": m.make_open_levels_1h_confirm_case,
        "1h_near_d_rejection": m.make_open_levels_1h_near_d_rejection_case,
        "1h_htf_rejection": m.make_open_levels_1h_htf_rejection_case,
        "1h_htf_live_test": m.make_open_levels_1h_htf_live_test_case,
        "1h_grass": m.make_open_levels_grass_1h_case,
    }
    ol_4h = {
        "4h_test": m.make_open_levels_4h_test_case,
        "4h_near": m.make_open_levels_4h_near_case,
        "4h_far": m.make_open_levels_4h_far_case,
        "4h_recent_window": m.make_open_levels_4h_recent_window_test_case,
        "4h_live_test": m.make_open_levels_4h_live_test_case,
        "4h_test_without_rejection": m.make_open_levels_4h_test_without_rejection_case,
        "4h_grass": m.make_open_levels_grass_4h_case,
    }
    ol_1d = {
        "1d": m.make_open_levels_1d_case,
        "1d_d_rejected_reclaimed": m.make_open_levels_d_rejected_then_reclaimed_case,
        "1d_month_only": m.make_open_levels_month_only_1d_case,
        "1d_far_below_month": m.make_open_levels_far_below_month_case,
        "1d_month_reclaimed": m.make_open_levels_month_reclaimed_case,
        "1d_old_month_rejection": m.make_open_levels_old_month_rejection_then_closed_above_case,
        "1d_useless_like": m.make_open_levels_useless_like_case,
        "1d_w_reclaimed_m_not_reached": m.make_open_levels_w_reclaimed_m_not_reached_case,
        "1d_vvv": m.make_open_levels_vvv_1d_case,
    }

    for n1, f1 in ol_1h.items():
        for n4, f4 in ol_4h.items():
            for nd, fd in ol_1d.items():
                def run(f1=f1, f4=f4, fd=fd):
                    df_1h, df_4h, df_1d = f1(), f4(), fd()
                    price = float(df_1h["close"].iloc[-1])
                    return analyze_and_classify(df_1h, df_4h, df_1d, price, None)

                scenarios[f"open_levels/{n1}|{n4}|{nd}"] = safe_run(lambda run=run: run_quietly(run))

    return scenarios


# ------------------------------------------------------------
# Seeded random market scenarios (1H / 4H / 1D + OI history)
# ------------------------------------------------------------

def make_random_market(seed):
    rng = np.random.default_rng(seed)
    n = 24 * 60  # 60 days of 1H candles
    start = pd.Timestamp("2026-03-01 00:00:00")
    timestamps = pd.date_range(start, periods=n, freq="1h")

    drift = rng.normal(0.0, 0.004, size=n)
    # Pump phase near the end with a random strength and a possible reversal.
    pump_len = int(rng.integers(10, 40))
    drift[-pump_len:] += rng.uniform(0.002, 0.012)
    if seed % 3 == 0:
        drift[-int(rng.integers(1, 4)):] -= 0.02

    close = 100.0 * np.exp(np.cumsum(drift))
    open_ = np.concatenate([[100.0], close[:-1]])
    wick_up = np.abs(rng.normal(0.0, 0.004, size=n)) * close
    wick_dn = np.abs(rng.normal(0.0, 0.004, size=n)) * close
    high = np.maximum(open_, close) + wick_up
    low = np.minimum(open_, close) - wick_dn
    volume = rng.uniform(1000, 5000, size=n)

    df_1h = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "quote_volume": volume * close,
    })

    def resample(df, rule):
        agg = df.set_index("timestamp").resample(rule).agg({
            "open": "first", "high": "max", "low": "min", "close": "last",
            "volume": "sum", "quote_volume": "sum",
        }).dropna().reset_index()
        return agg

    df_4h = resample(df_1h, "4h")
    df_1d = resample(df_1h, "1D")

    for df in (df_1h, df_4h, df_1d):
        df["confirm"] = 1
        df.loc[df.index[-1], "confirm"] = 0

    oi_values = 1000.0 + np.cumsum(rng.normal(0.0, 15.0, size=n))
    oi_history = pd.DataFrame({
        "timestamp": timestamps[-300:],
        "open_interest": oi_values[-300:],
    })

    return df_1h, df_4h, df_1d, oi_history


def build_random_scenarios():
    scenarios = {}

    for seed in RANDOM_SEEDS:
        def run(seed=seed):
            df_1h, df_4h, df_1d, oi_history = make_random_market(seed)
            price = float(df_1h["close"].iloc[-1])
            price_change = float(round((df_1h["close"].iloc[-1] / df_1h["close"].iloc[-25] - 1) * 100, 6))
            return analyze_and_classify(df_1h, df_4h, df_1d, price, oi_history, price_change=price_change)

        scenarios[f"random/seed_{seed}"] = safe_run(lambda run=run: run_quietly(run))

    return scenarios


# ------------------------------------------------------------
# Classification grid (thresholds / decision tree)
# ------------------------------------------------------------

def build_classification_grid(factor_sets):
    grid = {}

    for set_name, factors in factor_sets.items():
        for rsi_1h_live in (60.0, 82.0, 86.0):
            for rsi_1h_closed in (70.0, 81.0):
                for rsi_4h_live in (70.0, 76.0, 81.0):
                    for price_change in (5.0, 12.0, 25.0, 45.0):
                        for volume_change in (-20.0, 50.0):
                            key = (
                                f"grid/{set_name}|1h{rsi_1h_live}|c{rsi_1h_closed}|"
                                f"4h{rsi_4h_live}|p{price_change}|v{volume_change}"
                            )
                            result = m.classify_signal(
                                rsi_1h_live=rsi_1h_live,
                                rsi_1h_closed=rsi_1h_closed,
                                rsi_4h_live=rsi_4h_live,
                                exact_volume_24h=10_000_000,
                                volume_change_24h=volume_change,
                                price_change_24h=price_change,
                                short_setup_score=sum(int(f.get("points", 0)) for f in factors),
                                short_factors=factors,
                            )
                            grid[key] = summarize_classification(result)

    return grid


def collect_factor_sets():
    """Factor lists produced by a few representative scenarios (plus empty)."""
    sets = {"empty": []}

    def sweep_factors():
        df_1h = m.make_1h_equal_high_sweep_case()
        df_1h = with_rsi(df_1h)
        return m.analyze_short_factors(
            df_1h, with_rsi(m.make_flat_synthetic_4h()), df_1d=None,
            current_price=float(df_1h["close"].iloc[-1]), oi_history=None, provider="Bybit",
        )["factors"]

    def open_levels_factors():
        df_1h = with_rsi(m.make_open_levels_1h_htf_rejection_case())
        return m.analyze_short_factors(
            df_1h, with_rsi(m.make_open_levels_4h_test_case()),
            df_1d=m.make_open_levels_1d_case(),
            current_price=float(df_1h["close"].iloc[-1]), oi_history=None, provider="Bybit",
        )["factors"]

    def random_factors(seed):
        df_1h, df_4h, df_1d, oi_history = make_random_market(seed)
        return m.analyze_short_factors(
            with_rsi(df_1h), with_rsi(df_4h), df_1d=df_1d,
            current_price=float(df_1h["close"].iloc[-1]), oi_history=oi_history, provider="Bybit",
        )["factors"]

    sets["sweep"] = run_quietly(sweep_factors)
    sets["open_levels"] = run_quietly(open_levels_factors)
    sets["random3"] = run_quietly(random_factors, 3)
    sets["random7"] = run_quietly(random_factors, 7)

    return sets


# ------------------------------------------------------------
# Run / compare
# ------------------------------------------------------------

def build_results():
    results = {}
    results.update(build_case_scenarios())
    results.update(build_random_scenarios())
    results.update(build_classification_grid(collect_factor_sets()))
    return clean(results)


def diff_results(expected, actual):
    differences = []

    for key in sorted(set(expected) | set(actual)):
        if key not in actual:
            differences.append(f"MISSING in actual: {key}")
        elif key not in expected:
            differences.append(f"NEW in actual: {key}")
        elif expected[key] != actual[key]:
            differences.append(f"CHANGED: {key}\n    expected: {json.dumps(expected[key], ensure_ascii=False)[:600]}\n    actual:   {json.dumps(actual[key], ensure_ascii=False)[:600]}")

    return differences


def check_no_confirm_parity():
    """Bybit candles have no confirm column (last row = live candle).

    The golden scenarios carry confirm=1 for closed candles and confirm=0 for
    the live one. Dropping the column must not change any factor result.
    """

    def factors_json(df_1h, df_4h, df_1d, oi_history, strip):
        frames = [df_1h.copy(), df_4h.copy(), df_1d.copy() if df_1d is not None else None]

        if strip:
            frames = [f.drop(columns=["confirm"]) if f is not None and "confirm" in f.columns else f for f in frames]

        a = with_rsi(frames[0])
        b = with_rsi(frames[1])
        analysis = run_quietly(
            m.analyze_short_factors, a, b, df_1d=frames[2],
            current_price=float(a["close"].iloc[-1]), oi_history=oi_history, provider="Bybit",
        )
        return json.dumps(clean(summarize_factors(analysis)), sort_keys=True)

    mismatches = []

    for seed in RANDOM_SEEDS:
        df_1h, df_4h, df_1d, oi_history = make_random_market(seed)

        if factors_json(df_1h, df_4h, df_1d, oi_history, False) != factors_json(df_1h, df_4h, df_1d, oi_history, True):
            mismatches.append(f"random/seed_{seed}")

    sweep_cases = {
        "case/1h_equal_high_sweep": (m.make_1h_equal_high_sweep_case(), m.make_flat_synthetic_4h()),
        "case/4h_swing_sweep": (m.make_flat_synthetic_1h(), m.make_4h_swing_sweep_case(level_age_bars=5)),
        "case/rolling_only_high_take": (m.make_rolling_only_high_take_case(), m.make_flat_synthetic_4h()),
    }

    for name, (df_1h, df_4h) in sweep_cases.items():
        if factors_json(df_1h, df_4h, None, None, False) != factors_json(df_1h, df_4h, None, None, True):
            mismatches.append(name)

    return mismatches


def main():
    actual = build_results()

    if "--update" in sys.argv:
        with open(EXPECTED_PATH, "w", encoding="utf-8") as f:
            json.dump(actual, f, ensure_ascii=False, indent=1, sort_keys=True)
            f.write("\n")
        print(f"Golden file written: {EXPECTED_PATH} ({len(actual)} scenarios)")
        return 0

    if not os.path.exists(EXPECTED_PATH):
        print(f"Golden file not found: {EXPECTED_PATH}. Run with --update on a known-good version.")
        return 2

    with open(EXPECTED_PATH, "r", encoding="utf-8") as f:
        expected = json.load(f)

    differences = diff_results(expected, actual)

    parity_mismatches = check_no_confirm_parity()

    if parity_mismatches:
        print("GOLDEN TEST: шлях без колонки confirm (Bybit) дає інший результат:", parity_mismatches[:10])
        return 1

    if differences:
        print(f"GOLDEN TEST: ВІДМІННОСТІ ЗНАЙДЕНО ({len(differences)} з {len(expected)} сценаріїв)")
        for item in differences[:20]:
            print(" -", item)
        return 1

    print(f"GOLDEN TEST: без відмінностей ({len(expected)} сценаріїв; шлях без confirm збігається)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
