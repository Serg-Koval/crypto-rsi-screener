"""Self-tests for hiding the Telegram bot token in logs (r37_17)."""

import os

from screener.config import SCRIPT_VERSION
from screener.telegram import redact_telegram_token


TEST_TOKEN = "123456:ABC-test"
TOKEN_ENV = "TELEGRAM_BOT_TOKEN"


def _with_token(token, func):
    """Run func with TELEGRAM_BOT_TOKEN set (or removed if None); restore it after."""

    previous = os.environ.get(TOKEN_ENV)

    try:
        if token is None:
            os.environ.pop(TOKEN_ENV, None)
        else:
            os.environ[TOKEN_ENV] = token

        return func()
    finally:
        if previous is None:
            os.environ.pop(TOKEN_ENV, None)
        else:
            os.environ[TOKEN_ENV] = previous


def run_security_self_tests():
    print("\n" + "=" * 120)
    print("RUNNING SECURITY SELF TESTS")
    print("SCRIPT VERSION:", SCRIPT_VERSION)
    print("=" * 120)

    error_text = repr(ConnectionError(
        f"HTTPSConnectionPool(host='api.telegram.org'): Max retries exceeded "
        f"with url: /bot{TEST_TOKEN}/sendMessage"
    ))
    plain_text = "Telegram error response: chat not found"

    # (name, actual, expected)
    tests = []

    redacted = _with_token(TEST_TOKEN, lambda: redact_telegram_token(error_text))
    tests.append((
        "token is hidden in a requests-style error text",
        TEST_TOKEN in redacted,
        False,
    ))
    tests.append((
        "bot<token> is replaced with ***",
        "/***/sendMessage" in redacted,
        True,
    ))
    tests.append((
        "text without the token stays unchanged when token is set",
        _with_token(TEST_TOKEN, lambda: redact_telegram_token(plain_text)),
        plain_text,
    ))
    tests.append((
        "no token in environment -> text unchanged, no error",
        _with_token(None, lambda: redact_telegram_token(error_text)),
        error_text,
    ))
    tests.append((
        "empty token in environment -> text unchanged",
        _with_token("  ", lambda: redact_telegram_token(error_text)),
        error_text,
    ))

    previous = os.environ.get(TOKEN_ENV)
    _with_token(TEST_TOKEN, lambda: None)
    tests.append((
        "environment is restored after the tests",
        os.environ.get(TOKEN_ENV),
        previous,
    ))

    failed = 0

    for name, actual, expected in tests:
        ok = actual == expected
        status_text = "PASS" if ok else "FAIL"
        print(f"{status_text} | {name} | expected={expected!r} actual={actual!r}")

        if not ok:
            failed += 1

    if failed > 0:
        raise AssertionError(f"Security self-tests failed: {failed}/{len(tests)}")

    print(f"SECURITY SELF TESTS PASSED: {len(tests)}/{len(tests)}")
