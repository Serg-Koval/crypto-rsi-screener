from screener.common import make_factor
from screener.config import SCRIPT_VERSION
from screener.classification import (
    calculate_scores,
    classify_signal,
    classify_watch_signal,
    detect_overheat_watch_context,
)
from tests.cases import (
    make_classification_open_factor,
    make_classification_rsi_divergence_factor,
    make_classification_sweep_factor,
)


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
        "has_rsi_heat": False,
        "overheat_reason": "N/A",
    }

    # RSI heat without the full overheat context (e.g. 24h growth < 15%).
    heat_scores = {**base_scores, "has_rsi_heat": True}

    strong_heat_scores = dict(base_scores)
    strong_heat_scores["has_rsi_heat"] = True
    strong_heat_scores["rsi_score"] = 3
    strong_heat_scores["final_score"] = 8

    tests = [
        (
            "W open tested + sweep can provide location and create SHORT WATCH",
            heat_scores,
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
            heat_scores,
            [
                make_classification_sweep_factor(),
                make_classification_open_factor(["D"], ["near"]),
            ],
            "SHORT_WATCH",
        ),
        (
            "Sweep without RSI heat is NO_SIGNAL",
            base_scores,
            [make_classification_sweep_factor()],
            "NO_SIGNAL",
        ),
        (
            "Rejection without RSI heat is NO_SIGNAL",
            base_scores,
            [make_classification_rejection_factor()],
            "NO_SIGNAL",
        ),
        (
            "Sweep + RSI heat without overheat context is SHORT WATCH",
            heat_scores,
            [make_classification_sweep_factor()],
            "SHORT_WATCH",
        ),
        (
            "Rejection + RSI heat without overheat context is SHORT WATCH",
            heat_scores,
            [make_classification_rejection_factor()],
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


def make_classification_rejection_factor():
    return make_factor(
        key="rejection_candle",
        label="Rejection candle",
        status="confirmed",
        points=2,
        detail="1H rejection at D open",
    )


def run_rsi_heat_gate_self_tests():
    print("\n" + "=" * 120)
    print("RUNNING RSI HEAT GATE SELF TESTS")
    print("SCRIPT VERSION:", SCRIPT_VERSION)
    print("=" * 120)

    # (name, 1H live, 1H closed, 4H live, expected has_rsi_heat). Growth stays
    # below 15% so only the RSI part of the heat context is exercised.
    cases = [
        ("1H live 82 + closed 80 -> heat", 82.0, 80.0, 50.0, True),
        ("1H live 81.99 -> no heat", 81.99, 80.0, 50.0, False),
        ("1H live 82 + closed 79.99 -> no heat (both required)", 82.0, 79.99, 50.0, False),
        ("4H live 78 -> heat", 50.0, 50.0, 78.0, True),
        ("4H live 77.99 -> no heat", 50.0, 50.0, 77.99, False),
    ]

    failed = 0

    for name, live, closed, live_4h, expected_heat in cases:
        scores = calculate_scores(live, closed, live_4h, 10_000_000, 0, 10.0, 2)
        result = classify_signal(
            live, closed, live_4h, 10_000_000, 0, 10.0,
            short_factors=[make_classification_sweep_factor()],
        )
        expected_signal = "SHORT_WATCH" if expected_heat else "NO_SIGNAL"
        ok = (
            scores["has_rsi_heat"] is expected_heat
            and scores["has_overheat_context"] is False
            and result["signal_level"] == expected_signal
        )
        print(f"{'PASS' if ok else 'FAIL'} | {name} | has_rsi_heat={scores['has_rsi_heat']} signal={result['signal_level']}")

        if not ok:
            failed += 1

    if failed > 0:
        raise AssertionError(f"RSI heat gate self-tests failed: {failed}/{len(cases)}")

    print(f"RSI HEAT GATE SELF TESTS PASSED: {len(cases)}/{len(cases)}")


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
            "4H live 78.0 passes (threshold is inclusive)",
            detect_overheat_watch_context(
                rsi_1h_live=66,
                rsi_1h_closed=72,
                rsi_4h_live=78.0,
                exact_volume_24h=10_000_000,
                price_change_24h=16,
            ).get("is_overheat") is True,
        ),
        (
            "4H live 77.99 fails",
            detect_overheat_watch_context(
                rsi_1h_live=66,
                rsi_1h_closed=72,
                rsi_4h_live=77.99,
                exact_volume_24h=10_000_000,
                price_change_24h=16,
            ).get("is_overheat") is False,
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


def run_rsi_4h_threshold_end_to_end_self_tests():
    print("\n" + "=" * 120)
    print("RUNNING 4H RSI 78 END-TO-END SELF TESTS")
    print("SCRIPT VERSION:", SCRIPT_VERSION)
    print("=" * 120)

    sweep = [make_classification_sweep_factor()]

    # (name, price_change_24h, rsi_4h_live, short_factors, expected signal_level)
    cases = [
        ("pump >= 15%, 4H 78.0, no trigger -> OVERHEAT_WATCH", 16.0, 78.0, [], "OVERHEAT_WATCH"),
        ("pump >= 15%, 4H 77.99, no trigger -> NO_SIGNAL", 16.0, 77.99, [], "NO_SIGNAL"),
        ("sweep, pump < 15%, 4H 78.0 -> SHORT_WATCH", 10.0, 78.0, sweep, "SHORT_WATCH"),
        ("sweep, pump < 15%, 4H 77.99 -> NO_SIGNAL", 10.0, 77.99, sweep, "NO_SIGNAL"),
    ]

    failed = 0

    for name, price_change, rsi_4h, factors, expected in cases:
        # 1H RSI stays below every 1H heat threshold.
        result = classify_signal(
            60.0, 60.0, rsi_4h, 10_000_000, 0, price_change,
            short_factors=factors,
        )
        ok = result["signal_level"] == expected
        print(f"{'PASS' if ok else 'FAIL'} | {name} | signal={result['signal_level']}")

        if not ok:
            failed += 1

    if failed > 0:
        raise AssertionError(f"4H RSI 78 end-to-end self-tests failed: {failed}/{len(cases)}")

    print(f"4H RSI 78 END-TO-END SELF TESTS PASSED: {len(cases)}/{len(cases)}")
