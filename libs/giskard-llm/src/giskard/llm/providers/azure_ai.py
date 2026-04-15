"""Azure AI Foundry provider using the ``openai`` SDK (OpenAI-compatible endpoint).

Routing prefix: ``azure_ai/``

Authentication:
    - Env: ``AZURE_AI_API_KEY``, ``AZURE_AI_ENDPOINT``
    - Kwargs: ``api_key``, ``base_url``

Role mapping:
    Same as OpenAI — all canonical roles passed through as-is. Azure AI
    Foundry endpoints expose an OpenAI-compatible API.

Message constraints:
    Same as OpenAI.

Tool call format:
    Same as OpenAI.

Error mapping:
    Same as OpenAI.

Supported features:
    - Completion: yes
    - Embeddings: depends on deployed model
    - Structured output (response_format): depends on deployed model

Provider-specific kwargs:
    - ``base_url``: Azure AI Foundry endpoint URL
"""

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportImplicitRelativeImport=false, reportMissingSuperCall=false

import logging
import os
from typing import Any

from ..errors import ProviderNotAvailableError
from ..utils import compact
from .openai import OpenAIProvider

logger = logging.getLogger(__name__)

PROVIDER = "azure_ai"


class AzureAIProvider(OpenAIProvider):
    _PROVIDER = "azure_ai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        **_kwargs: Any,
    ) -> None:
        if _kwargs:
            logger.warning(
                "%s provider: ignoring unknown kwargs: %s", PROVIDER, sorted(_kwargs)
            )
        try:
            import openai
        except ImportError as exc:
            raise ProviderNotAvailableError(PROVIDER, "openai", extra="azure") from exc

        resolved_key = api_key or os.environ.get("AZURE_AI_API_KEY")
        resolved_base = base_url or os.environ.get("AZURE_AI_ENDPOINT")

        self._client = openai.AsyncOpenAI(
            **compact(api_key=resolved_key, base_url=resolved_base, timeout=timeout)
        )
