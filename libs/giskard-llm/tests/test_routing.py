import pytest
from giskard.llm.routing import _parse_model_string


@pytest.mark.parametrize(
    "model_str, expected",
    [
        ("openai/gpt-4o", ("openai", "gpt-4o")),
        ("gemini/gemini-2.0-flash", ("gemini", "gemini-2.0-flash")),
        ("anthropic/claude-opus-4-6", ("anthropic", "claude-opus-4-6")),
        ("openai/gpt-4o-mini", ("openai", "gpt-4o-mini")),
    ],
    ids=["openai", "gemini", "anthropic", "openai-mini"],
)
def test_parse_model_string_valid(model_str: str, expected: tuple[str, str]):
    assert _parse_model_string(model_str) == expected


@pytest.mark.parametrize(
    "model_str",
    ["gpt-4o", "", "openai/", "/gpt-4o"],
    ids=["no-slash", "empty", "no-model", "no-provider"],
)
def test_parse_model_string_invalid(model_str: str):
    with pytest.raises(ValueError, match="Invalid model string"):
        _parse_model_string(model_str)
