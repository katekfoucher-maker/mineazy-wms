""""Sign up with Google" (see /signup, wms/web/routes.py).

The authorization-code flow against Google's OAuth 2.0 endpoints, using the
Client ID/Secret configured in Settings (wms.config). No JWT library is
needed to verify the returned ID token - Google's own ``tokeninfo`` endpoint
validates the signature and expiry server-side and just hands back the
decoded claims, which is enough at this app's scale.
"""
from __future__ import annotations

import secrets
from urllib.parse import urlencode

import httpx

from wms.config import get_settings

_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"
_VALID_ISSUERS = {"accounts.google.com", "https://accounts.google.com"}


def configured() -> bool:
    s = get_settings()
    return bool(s.google_client_id and s.google_client_secret)


def new_state() -> str:
    """A random per-attempt token, stashed in the session and checked on the
    callback so a forged/replayed redirect can't log someone in as someone
    else (CSRF on the OAuth callback)."""
    return secrets.token_urlsafe(24)


def auth_url(redirect_uri: str, state: str) -> str:
    s = get_settings()
    params = {
        "client_id": s.google_client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
    }
    return f"{_AUTH_URL}?{urlencode(params)}"


class GoogleAuthError(Exception):
    pass


def exchange_code(code: str, redirect_uri: str) -> dict:
    """code -> verified claims ``{sub, email, email_verified, name}``.
    Raises :class:`GoogleAuthError` (a message safe to flash to the user) on
    any failure - a bad/expired code, an unverified email, a network hiccup."""
    s = get_settings()
    try:
        with httpx.Client(timeout=10) as client:
            tok = client.post(_TOKEN_URL, data={
                "client_id": s.google_client_id,
                "client_secret": s.google_client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
            })
            if tok.status_code != 200:
                raise GoogleAuthError("Google sign-in failed (couldn't exchange the code). Try again.")
            id_token = tok.json().get("id_token")
            if not id_token:
                raise GoogleAuthError("Google sign-in failed (no ID token returned). Try again.")

            info = client.get(_TOKENINFO_URL, params={"id_token": id_token})
    except httpx.HTTPError:
        raise GoogleAuthError("Couldn't reach Google to complete sign-in. Try again.")

    if info.status_code != 200:
        raise GoogleAuthError("Google sign-in failed (invalid ID token). Try again.")
    claims = info.json()

    if claims.get("aud") != s.google_client_id:
        raise GoogleAuthError("Google sign-in failed (token wasn't issued for this site).")
    if claims.get("iss") not in _VALID_ISSUERS:
        raise GoogleAuthError("Google sign-in failed (unrecognised issuer).")
    if str(claims.get("email_verified", "")).lower() != "true":
        raise GoogleAuthError("That Google account's email isn't verified - verify it with Google first.")
    sub, email = claims.get("sub"), claims.get("email")
    if not sub or not email:
        raise GoogleAuthError("Google sign-in failed (missing account info).")

    return {"sub": sub, "email": email,
            "name": claims.get("name") or email.split("@")[0]}
