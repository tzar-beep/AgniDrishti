"""Shared helpers: logging setup and HTTP GET with retries."""

import logging
import os
import time

import requests

log = logging.getLogger(__name__)


class FetchError(RuntimeError):
    """Raised when a remote resource cannot be fetched after all retries."""


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging with a compact timestamped format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def redact(text: str) -> str:
    """Remove secrets (e.g. the FIRMS MAP_KEY) from a string before logging it."""
    key = os.environ.get("FIRMS_MAP_KEY")
    return text.replace(key, "***") if key else text


def _rate_limit_wait(resp: requests.Response) -> float | None:
    """Seconds to wait after HTTP 429, or None if the limit will not reset soon."""
    if resp.headers.get("Retry-After", "").isdigit():
        return float(resp.headers["Retry-After"])
    body = resp.text.lower()
    if "daily" in body:
        return None
    if "hourly" in body:
        return 3600.0
    return 61.0


def http_get(
    url: str,
    params: dict | None = None,
    *,
    timeout: float = 120,
    retries: int = 5,
    backoff: float = 2.0,
) -> requests.Response:
    """GET a URL, retrying on timeouts, connection errors, 5xx and rate limits.

    Raises:
        FetchError: with a clear message when the request ultimately fails.
    """
    last = ""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code == 200:
                return resp
            last = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code == 429:
                wait = _rate_limit_wait(resp)
                if wait is None:
                    raise FetchError(
                        redact(f"Daily rate limit hit for {url} ({last}). "
                               "Rerun tomorrow; cached responses are reused.")
                    )
                log.warning("Rate limited (%s); sleeping %.0fs", redact(last), wait)
                time.sleep(wait)
                continue
            if resp.status_code < 500:
                raise FetchError(redact(f"Request to {url} failed: {last}"))
        delay = backoff**attempt
        log.warning(
            "Attempt %d/%d failed for %s (%s); retrying in %.0fs",
            attempt, retries, redact(url), redact(last), delay,
        )
        time.sleep(delay)
    raise FetchError(redact(f"Giving up on {url} after {retries} attempts: {last}"))
