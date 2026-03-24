import pytest
from giskard.llm.retry import should_retry


@pytest.mark.parametrize(
    "status_code, expected",
    [
        (408, True),
        (429, True),
        (500, True),
        (502, True),
        (503, True),
        (504, True),
        (520, True),
        (524, True),
        (529, True),
        (200, False),
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (0, False),
    ],
    ids=[
        "timeout-408",
        "rate-limit-429",
        "internal-500",
        "bad-gateway-502",
        "unavailable-503",
        "gateway-timeout-504",
        "cloudflare-520",
        "cloudflare-524",
        "overloaded-529",
        "ok-200",
        "bad-request-400",
        "unauthorized-401",
        "forbidden-403",
        "not-found-404",
        "zero",
    ],
)
def test_should_retry(status_code: int, expected: bool):
    assert should_retry(status_code) is expected
