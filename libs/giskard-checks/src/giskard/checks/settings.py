import os

from giskard.agents.embeddings import BaseEmbeddingModel, EmbeddingModel
from giskard.agents.embeddings.base import EmbeddingParams
from giskard.agents.generators import BaseGenerator, Generator

# Global default generator
_default_generator: BaseGenerator | None = None
_default_embedding_model: BaseEmbeddingModel | None = None


def set_default_generator(generator: "BaseGenerator") -> None:
    """Set the default LLM generator for all checks.

    Parameters
    ----------
    generator : BaseGenerator
        The generator to use as default for all LLM checks.
    """
    global _default_generator
    _default_generator = generator


def get_default_generator() -> BaseGenerator:
    """Get the current default generator.

    Returns
    -------
    BaseGenerator
        The current default generator, or ``TEST_MODEL`` (default
        ``gemini/gemini-2.0-flash``) if none has been set.
    """
    return _default_generator or Generator(
        model=os.getenv("TEST_MODEL", "gemini/gemini-2.0-flash")
    )


def set_default_embedding_model(embedding_model: "BaseEmbeddingModel") -> None:
    """Set the default embedding model for all checks.

    Parameters
    ----------
    embedding_model : BaseEmbeddingModel
        The embedding model to use as default for all embedding checks.
    """
    global _default_embedding_model
    _default_embedding_model = embedding_model


def get_default_embedding_model() -> BaseEmbeddingModel:
    """Get the current default embedding model.

    Returns
    -------
    BaseEmbeddingModel
        The current default embedding model, or ``TEST_EMBEDDING_MODEL`` (default
        ``gemini/gemini-embedding-001``) if none has been set.
    """
    return _default_embedding_model or EmbeddingModel(
        model=os.getenv("TEST_EMBEDDING_MODEL", "gemini/gemini-embedding-001"),
        params=EmbeddingParams(dimensions=1536),
    )
