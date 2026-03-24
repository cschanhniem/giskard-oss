"""Retry eligibility check for LLM provider errors."""

RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504, 520, 524, 529})


def should_retry(status_code: int) -> bool:
    """Return True if *status_code* indicates a transient error worth retrying."""
    return status_code in RETRYABLE_STATUS_CODES
