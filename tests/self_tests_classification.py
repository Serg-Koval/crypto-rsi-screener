from screener.config import SCRIPT_VERSION
from screener.classification import (
    classify_watch_signal,
    detect_overheat_watch_context,
    evaluate_rsi_entry_filter,
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
