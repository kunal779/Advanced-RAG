from __future__ import annotations

from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

QDRANT_PATH = Path("data/qdrant")
COLLECTION_NAME = "rag_documents"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

VECTOR_SIZE = 384

DEFAULT_TOP_K = 20


# ============================================================
# RETRIEVER
# ============================================================

class Retriever:
    """
    First-stage dense retriever.

    Question
        ↓
    Embedding
        ↓
    Qdrant
        ↓
    Top-K candidates
    """

    def __init__(
        self,
        qdrant_path: Path = QDRANT_PATH,
        collection_name: str = COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:

        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.model_name = model_name

        self._validate_qdrant_path()

        print(
            f"Opening Qdrant database: "
            f"{self.qdrant_path}"
        )

        self.client = QdrantClient(
            path=str(self.qdrant_path)
        )

        print(
            f"Loading embedding model: "
            f"{self.model_name}"
        )

        self.embedding_model = SentenceTransformer(
            self.model_name
        )

        self._validate_collection()

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_qdrant_path(self) -> None:

        if not self.qdrant_path.exists():
            raise FileNotFoundError(
                f"Qdrant database not found: "
                f"{self.qdrant_path}"
            )

    def _validate_collection(self) -> None:

        if not self.client.collection_exists(
            collection_name=self.collection_name
        ):
            raise ValueError(
                f"Collection '{self.collection_name}' "
                f"does not exist."
            )

        info = self.client.get_collection(
            collection_name=self.collection_name
        )

        vector_config = info.config.params.vectors

        if vector_config.size != VECTOR_SIZE:
            raise ValueError(
                "Vector size mismatch. "
                f"Expected {VECTOR_SIZE}, "
                f"got {vector_config.size}."
            )

        if info.points_count == 0:
            raise ValueError(
                f"Collection '{self.collection_name}' "
                f"contains no points."
            )

        print("\nRetriever validation: PASSED")
        print(f"Collection : {self.collection_name}")
        print(f"Points     : {info.points_count}")
        print(f"Vector size: {vector_config.size}")

    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    def _embed_query(
        self,
        question: str,
    ) -> list[float]:

        embedding = self.embedding_model.encode(
            question,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        if embedding.shape[0] != VECTOR_SIZE:
            raise ValueError(
                "Invalid query vector size. "
                f"Expected {VECTOR_SIZE}, "
                f"got {embedding.shape[0]}."
            )

        return embedding.tolist()

    # ========================================================
    # RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
    ):

        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        print(
            "\nCreating query embedding..."
        )

        query_vector = self._embed_query(
            question
        )

        print(
            f"Query vector: "
            f"{len(query_vector)} dimensions"
        )

        print(
            f"\nRetrieving top {top_k} "
            f"candidates from Qdrant..."
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        candidates = response.points

        if not candidates:
            raise RuntimeError(
                "Qdrant returned no candidates."
            )

        print(
            f"Retrieved "
            f"{len(candidates)} candidates."
        )

        return candidates

    # ========================================================
    # DEBUG OUTPUT
    # ========================================================

    def print_candidates(
        self,
        candidates,
    ) -> None:

        print("\n" + "=" * 80)
        print("DENSE RETRIEVAL RESULTS")
        print("=" * 80)

        for rank, result in enumerate(
            candidates,
            start=1,
        ):

            payload = result.payload or {}

            print(
                f"#{rank:02d} | "
                f"Qdrant={float(result.score):.4f} | "
                f"Chunk={payload.get('chunk_id')} | "
                f"Pages="
                f"{payload.get('page_start')}-"
                f"{payload.get('page_end')}"
            )