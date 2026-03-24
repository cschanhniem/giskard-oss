from giskard.llm.errors import (
    AuthenticationError,
    LLMError,
    ProviderNotAvailableError,
    RateLimitError,
    ServerError,
)


def test_llm_error_attributes():
    err = LLMError(429, "rate limited", "openai")
    assert err.status_code == 429
    assert err.message == "rate limited"
    assert err.provider == "openai"
    assert "[openai] 429" in str(err)


def test_subclass_hierarchy():
    err = RateLimitError(429, "slow down", "gemini")
    assert isinstance(err, LLMError)
    assert isinstance(err, RateLimitError)
    assert err.status_code == 429


def test_provider_not_available_error():
    err = ProviderNotAvailableError("openai", "openai")
    assert err.status_code == 0
    assert "pip install giskard-llm[openai]" in str(err)
    assert isinstance(err, LLMError)


def test_error_chaining():
    original = ValueError("original error")
    try:
        raise ServerError(500, "internal", "openai") from original
    except LLMError as e:
        assert e.__cause__ is original
        assert e.status_code == 500


def test_all_error_types_are_llm_error():
    for cls in [AuthenticationError, RateLimitError, ServerError]:
        err = cls(400, "test", "test")
        assert isinstance(err, LLMError)
