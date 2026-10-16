from tests.self_tests_bybit import run_bybit_self_tests, run_tradfi_filter_self_tests
from tests.self_tests_classification import (
    run_open_levels_classification_self_tests,
    run_rsi_heat_gate_self_tests,
    run_overheat_context_self_tests,
    run_qusdt_style_end_to_end_self_tests,
    run_rsi_4h_threshold_end_to_end_self_tests,
)
from tests.self_tests_factors import (
    run_oi_divergence_self_tests,
    run_open_levels_self_tests,
    run_rejection_self_tests,
    run_rsi_divergence_self_tests,
    run_sweep_self_tests,
)
from tests.self_tests_oi import run_open_interest_self_tests
from tests.self_tests_security import run_security_self_tests
from tests.self_tests_thresholds import run_threshold_self_tests


def run_all():
    run_sweep_self_tests()
    run_open_levels_self_tests()
    run_open_levels_classification_self_tests()
    run_rsi_heat_gate_self_tests()
    run_rejection_self_tests()
    run_rsi_4h_threshold_end_to_end_self_tests()
    run_qusdt_style_end_to_end_self_tests()
    run_overheat_context_self_tests()
    run_open_interest_self_tests()
    run_rsi_divergence_self_tests()
    run_oi_divergence_self_tests()
    run_bybit_self_tests()
    run_tradfi_filter_self_tests()
    run_threshold_self_tests()
    run_security_self_tests()
