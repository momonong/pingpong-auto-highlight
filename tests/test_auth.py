from fastapi import Request

from pingpong_highlight.auth import (
    generate_session_token,
    hash_password,
    hash_session_token,
    normalize_username,
    verify_password,
)
from pingpong_highlight.web import LoginRateLimiter, _login_client_address


def test_login_rate_limiter_isolated_by_client_address() -> None:
    limiter = LoginRateLimiter(window_seconds=60, attempt_limit=2)

    assert limiter.consume("host:alice") is None
    assert limiter.consume("host:alice") is None
    assert limiter.consume("host:alice") == 60
    assert limiter.consume("host:bob") is None
    assert limiter.consume("host:bob") is None
    assert limiter.consume("host:bob") == 60


def test_login_client_address_only_trusts_the_configured_tunnel_header() -> None:
    scope = {
        "type": "http",
        "client": ("172.19.0.4", 1234),
        "headers": [
            (b"cf-connecting-ip", b"198.51.100.20"),
            (b"x-forwarded-for", b"203.0.113.10, 198.51.100.30"),
        ],
    }
    request = Request(scope)

    assert _login_client_address(request, proxy_provider="none") == "172.19.0.4"
    assert _login_client_address(request, proxy_provider="cloudflare") == "198.51.100.20"
    assert _login_client_address(request, proxy_provider="ngrok") == "198.51.100.30"


def test_scrypt_password_round_trip_uses_a_random_salt() -> None:
    first = hash_password("correct horse battery staple")
    second = hash_password("correct horse battery staple")

    assert first.startswith("scrypt$1$")
    assert second != first
    assert verify_password("correct horse battery staple", first)
    assert not verify_password("wrong password", first)


def test_password_verification_rejects_malformed_or_untrusted_parameters() -> None:
    encoded = hash_password("password")

    assert not verify_password("password", "not-a-password-hash")
    assert not verify_password("password", encoded.replace("$16384$", "$32768$"))
    assert not verify_password("password", encoded[:-2] + "!!")


def test_session_token_is_only_persisted_as_a_stable_sha256_hash() -> None:
    token = generate_session_token()
    digest = hash_session_token(token)

    assert token not in digest
    assert len(digest) == 64
    assert hash_session_token(token) == digest


def test_username_normalization_is_case_insensitive_and_rejects_unsafe_values() -> None:
    assert normalize_username(" Alice.Example ") == "alice.example"

    for value in ("ab", "has spaces", "../escape", "ümlaut"):
        try:
            normalize_username(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected {value!r} to be rejected")
