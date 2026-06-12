"""Knowledge base primitives for document-grounded scan generators."""

from typing import Self

import numpy as np
from giskard.checks.core.mixin import WithEmbeddingMixin
from pydantic import BaseModel, model_validator


class EmbeddedDocument(BaseModel):
    """Document stored in a scan knowledge base.

    Attributes:
        content: Text content used for question generation and grounding.
        embeddings: Optional embedding vector. Missing vectors are computed
            lazily when nearest-neighbor retrieval is requested.
        tags: Optional document labels carried by the caller.
    """

    content: str
    embeddings: list[float] | None = None
    tags: list[str] | None = None


class KnowledgeBase(WithEmbeddingMixin):
    """Collection of documents used by knowledge-base scenario generators.

    Embeddings are not computed when the knowledge base is created. They are
    filled lazily, in batches, the first time nearest-neighbor retrieval needs
    them.
    """

    documents: list[EmbeddedDocument]

    @classmethod
    def from_texts(cls, texts: list[str]) -> Self:
        """Create a knowledge base from raw text documents.

        Args:
            texts: Text chunks to wrap as :class:`EmbeddedDocument` objects.

        Returns:
            A knowledge base containing one document per input text.
        """
        return cls(documents=[EmbeddedDocument(content=text) for text in texts])

    @model_validator(mode="after")
    def _validate_documents(self) -> Self:
        self.documents = [doc for doc in self.documents if doc.content.strip()]
        if not self.documents:
            raise ValueError(
                "KnowledgeBase must contain at least one non-empty document"
            )
        return self

    async def ensure_embeddings(self) -> None:
        """Ensure every document has embeddings from the same model.

        If any document is missing embeddings, all embeddings are recomputed in
        one batch. This avoids mixing vectors that may have been produced by
        different embedding models.
        """
        if all(doc.embeddings is not None for doc in self.documents):
            return

        embeddings = await self._embedding_model.embed(
            [document.content for document in self.documents]
        )
        if len(embeddings) != len(self.documents):
            raise ValueError(
                "Embedding model returned a different number of vectors than documents"
            )

        for document, embedding in zip(self.documents, embeddings):
            document.embeddings = [
                float(value) for value in np.asarray(embedding, dtype=float).tolist()
            ]

    async def closest_documents(
        self, seed_index: int, max_documents: int
    ) -> list[EmbeddedDocument]:
        """Return the documents closest to a seed document by cosine similarity.

        Args:
            seed_index: Index of the seed document in ``documents``.
            max_documents: Maximum number of documents to return, including the
                seed document itself.

        Returns:
            Documents sorted from highest to lowest cosine similarity.
        """
        if not 0 <= seed_index < len(self.documents):
            raise IndexError(f"seed_index out of range: {seed_index}")
        if max_documents <= 0:
            return []

        await self.ensure_embeddings()
        matrix = self._embedding_matrix()
        similarities = self._cosine_similarity(matrix[seed_index], matrix)
        indices = np.argsort(-similarities)[:max_documents]
        return [self.documents[int(index)] for index in indices]

    def _embedding_matrix(self) -> np.ndarray:
        embeddings = [doc.embeddings for doc in self.documents]
        if any(embedding is None for embedding in embeddings):
            raise ValueError("KnowledgeBase embeddings are incomplete")

        matrix = np.asarray(embeddings, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("KnowledgeBase embeddings must be a 2D matrix")
        if np.any(np.linalg.norm(matrix, axis=1) == 0):
            raise ValueError("KnowledgeBase embeddings must not contain zero vectors")
        return matrix

    @staticmethod
    def _cosine_similarity(seed: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        return matrix @ seed / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(seed))


def normalize_knowledge_base(
    knowledge_base: KnowledgeBase | list[str] | None,
) -> KnowledgeBase | None:
    """Normalize supported knowledge base inputs.

    Args:
        knowledge_base: Either an existing knowledge base, raw text documents,
            or ``None``.

    Returns:
        A :class:`KnowledgeBase` instance, or ``None`` when no input was
        provided.
    """
    if knowledge_base is None or isinstance(knowledge_base, KnowledgeBase):
        return knowledge_base
    return KnowledgeBase.from_texts(knowledge_base)
