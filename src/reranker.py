from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sentence_transformers import CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

RERANK_BATCH_SIZE = 8

DEFAULT_TOP_K = 5


# ============================================================
# RESULT STRUCTURE
# ============================================================

@dataclass
class RankedChunk:
    chunk_id: int

    page_start: int
    page_end: int

    text: str

    retrieval_rank: int
    retrieval_score: float

    rerank_rank: int
    rerank_score: float


# ============================================================
# RERANKER
# ============================================================

class Reranker:
    """
    Second-stage CrossEncoder reranker.

    Input:
        user question
        Qdrant candidates

    Output:
        ranked chunks
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL_NAME,
        batch_size: int = RERANK_BATCH_SIZE,
    ) -> None:

        self.model_name = model_name
        self.batch_size = batch_size

        print(
            f"\nLoading reranker model: "
            f"{self.model_name}"
        )

        self.model = CrossEncoder(
            self.model_name
        )

    # ========================================================
    # RERANK
    # ========================================================

    def rerank(
        self,
        question: str,
        candidates: list[Any],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[RankedChunk]:

        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        if not candidates:
            raise ValueError(
                "No candidates were provided."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        # ----------------------------------------------------
        # Build CrossEncoder pairs
        # ----------------------------------------------------

        pairs = []

        for result in candidates:

            payload = result.payload or {}

            text = str(
                payload.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                raise ValueError(
                    "Candidate contains empty text."
                )

            pairs.append(
                (
                    question,
                    text,
                )
            )

        # ----------------------------------------------------
        # Score candidates
        # ----------------------------------------------------

        print(
            "\nRunning BGE reranker..."
        )

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=True,
        )

        # ----------------------------------------------------
        # Combine retrieval + reranking information
        # ----------------------------------------------------

        combined = []

        for retrieval_rank, (
            result,
            score,
        ) in enumerate(
            zip(candidates, scores),
            start=1,
        ):

            combined.append(
                {
                    "result": result,
                    "retrieval_rank": retrieval_rank,
                    "retrieval_score": float(
                        result.score
                    ),
                    "rerank_score": float(
                        score
                    ),
                }
            )

        # ----------------------------------------------------
        # Sort by BGE relevance
        # ----------------------------------------------------

        combined.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        # ----------------------------------------------------
        # Build final objects
        # ----------------------------------------------------

        ranked_chunks = []

        for rerank_rank, item in enumerate(
            combined,
            start=1,
        ):

            result = item["result"]
            payload = result.payload or {}

            ranked_chunks.append(
                RankedChunk(
                    chunk_id=int(
                        payload["chunk_id"]
                    ),
                    page_start=int(
                        payload["page_start"]
                    ),
                    page_end=int(
                        payload["page_end"]
                    ),
                    text=str(
                        payload["text"]
                    ).strip(),
                    retrieval_rank=int(
                        item["retrieval_rank"]
                    ),
                    retrieval_score=float(
                        item["retrieval_score"]
                    ),
                    rerank_rank=rerank_rank,
                    rerank_score=float(
                        item["rerank_score"]
                    ),
                )
            )

        return ranked_chunks[:top_k]

    # ========================================================
    # DEBUG OUTPUT
    # ========================================================

    @staticmethod
    def print_results(
        chunks: list[RankedChunk],
    ) -> None:

        print("\n" + "=" * 80)
        print("BGE RERANKED RESULTS")
        print("=" * 80)

        print(
            f"{'BGE':>4} | "
            f"{'QDR':>4} | "
            f"{'BGE SCORE':>10} | "
            f"{'QDR SCORE':>10} | "
            f"{'CHUNK':>7} | "
            f"PAGES"
        )

        print("-" * 80)

        for chunk in chunks:

            print(
                f"{chunk.rerank_rank:>4} | "
                f"{chunk.retrieval_rank:>4} | "
                f"{chunk.rerank_score:>10.4f} | "
                f"{chunk.retrieval_score:>10.4f} | "
                f"{chunk.chunk_id:>7} | "
                f"{chunk.page_start}-"
                f"{chunk.page_end}"
            )