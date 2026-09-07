from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets

_PASSWORD_SCHEME = "scrypt"
_PASSWORD_VERSION = 1
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SALT_BYTES = 16
_SESSION_TOKEN_BYTES = 32
_USERNAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.b64decode(value + padding, altchars=b"-_", validate=True)


def normalize_username(username: str) -> str:
    """Return the canonical, case-insensitive form used for storage and login."""

    normalized = username.strip().casefold()
    if not _USERNAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "username must be 3-64 characters using lowercase letters, numbers, '.', '_', or '-'"
        )
    return normalized


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """Hash a password into a self-describing, versioned scrypt string."""

    if not isinstance(password, str) or not password:
        raise ValueError("password must not be empty")
    if salt is None:
        salt = secrets.token_bytes(_SALT_BYTES)
    if len(salt) < _SALT_BYTES:
        raise ValueError(f"password salt must be at least {_SALT_BYTES} bytes")
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
    )
    return "$".join(
        (
            _PASSWORD_SCHEME,
            str(_PASSWORD_VERSION),
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            _encode(salt),
            _encode(digest),
        )
    )


def verify_password(password: str, encoded: str) -> bool:
    """Verify a password without leaking malformed hashes or comparison timing."""

    if not isinstance(password, str) or not isinstance(encoded, str):
        return False
    try:
        scheme, version, n, r, p, salt_value, digest_value = encoded.split("$")
        if scheme != _PASSWORD_SCHEME or int(version) != _PASSWORD_VERSION:
            return False
        cost_n, block_r, parallel_p = int(n), int(r), int(p)
        if (cost_n, block_r, parallel_p) != (_SCRYPT_N, _SCRYPT_R, _SCRYPT_P):
            return False
        salt = _decode(salt_value)
        expected = _decode(digest_value)
        if len(salt) < _SALT_BYTES or len(expected) != _SCRYPT_DKLEN:
            return False
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=cost_n,
            r=block_r,
            p=parallel_p,
            dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def generate_session_token() -> str:
    """Create the opaque credential returned to a signed-in client."""

    return secrets.token_urlsafe(_SESSION_TOKEN_BYTES)


def hash_session_token(token: str) -> str:
    """Hash an opaque session token for safe database storage and lookup."""

    if not isinstance(token, str) or not token:
        raise ValueError("session token must not be empty")
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
