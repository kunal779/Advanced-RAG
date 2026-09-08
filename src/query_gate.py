from __future__ import annotations

from dataclasses import dataclass

from .reranker import RankedChunk, Reranker
from .retriever import Retriever


# ============================================================
# CONFIGURATION
# ============================================================

QUERY_GATE_TOP_K = 20

# Initial threshold based on our current test set.
QUERY_GATE_THRESHOLD = 0.40


# ============================================================
# RESULT
# ============================================================

@dataclass
class QueryGateResult:
    question: str

    passed: bool

    top_score: float

    candidate_count: int

    ranked_chunks: list[RankedChunk]

    reason: str


# ============================================================
# QUERY GATE
# ============================================================

class QueryGate:
    """
    Corpus-grounded relevance gate.

    It uses the actual knowledge base to determine whether
    the user's question has sufficiently relevant evidence.

    It does NOT use an LLM classifier.

    Workflow:

        Question
            ↓
        MiniLM embedding
            ↓
        Qdrant
            ↓
        BGE
            ↓
        Relevance decision
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        top_k: int = QUERY_GATE_TOP_K,
        threshold: float = QUERY_GATE_THRESHOLD,
    ) -> None:

        self.retriever = retriever
        self.reranker = reranker

        self.top_k = top_k
        self.threshold = threshold

        if self.top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if self.threshold < 0:
            raise ValueError(
                "threshold cannot be negative."
            )

    # ========================================================
    # CHECK
    # ========================================================

    def check(
        self,
        question: str,
    ) -> QueryGateResult:

        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        print(
            "\n" + "=" * 80
        )
        print(
            "QUERY RELEVANCE GATE"
        )
        print(
            "=" * 80
        )

        print(
            f"Question: {question}"
        )

        # ----------------------------------------------------
        # Stage 1:
        # Dense retrieval
        # ----------------------------------------------------

        print(
            "\n[Gate 1/2] Searching knowledge base..."
        )

        candidates = self.retriever.retrieve(
            question,
            self.top_k,
        )

        if not candidates:

            print(
                "\nNo candidates found."
            )

            return QueryGateResult(
                question=question,
                passed=False,
                top_score=0.0,
                candidate_count=0,
                ranked_chunks=[],
                reason=(
                    "No relevant evidence was found "
                    "in the knowledge base."
                ),
            )

        # ----------------------------------------------------
        # Stage 2:
        # BGE reranking
        # ----------------------------------------------------

        print(
            "\n[Gate 2/2] Checking semantic relevance..."
        )

        ranked_chunks = self.reranker.rerank(
            question,
            candidates,
            top_k=len(candidates),
        )

        if not ranked_chunks:

            return QueryGateResult(
                question=question,
                passed=False,
                top_score=0.0,
                candidate_count=len(candidates),
                ranked_chunks=[],
                reason=(
                    "BGE returned no ranked evidence."
                ),
            )

        # ----------------------------------------------------
        # Best evidence score
        # ----------------------------------------------------

        top_score = float(
            ranked_chunks[0].rerank_score
        )

        passed = (
            top_score >= self.threshold
        )

        # ----------------------------------------------------
        # Display decision
        # ----------------------------------------------------

        print(
            "\n" + "-" * 80
        )

        print(
            "QUERY GATE DECISION"
        )

        print(
            "-" * 80
        )

        print(
            f"Top BGE score : {top_score:.4f}"
        )

        print(
            f"Threshold     : {self.threshold:.4f}"
        )

        print(
            f"Candidates    : {len(candidates)}"
        )

        print(
            "\nTop evidence:"
        )

        for chunk in ranked_chunks[:5]:

            print(
                f"  Chunk {chunk.chunk_id} | "
                f"BGE={chunk.rerank_score:.4f} | "
                f"Pages="
                f"{chunk.page_start}-"
                f"{chunk.page_end}"
            )

        # ----------------------------------------------------
        # Decision
        # ----------------------------------------------------

        if passed:

            print(
                "\nDecision: IN_SCOPE ✅"
            )

            reason = (
                "The knowledge base contains "
                "sufficiently relevant evidence."
            )

        else:

            print(
                "\nDecision: OUT_OF_SCOPE ❌"
            )

            reason = (
                "The knowledge base does not contain "
                "sufficiently relevant evidence."
            )

        return QueryGateResult(
            question=question,
            passed=passed,
            top_score=top_score,
            candidate_count=len(candidates),
            ranked_chunks=ranked_chunks,
            reason=reason,
        )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    retriever = Retriever()

    reranker = Reranker()

    gate = QueryGate(
        retriever=retriever,
        reranker=reranker,
    )

    test_questions = [
        "What is Retrieval-Augmented Generation?",
        "What is dense retrieval?",
        "What is reranking?",
        "What is LangChain?",
        "Do you know about GTA 6?",
        "How much salt should I add to my dish?",
        "How to feed medicine to dogs?",
    ]

    for question in test_questions:

        result = gate.check(
            question
        )

        print(
            f"\nQUESTION : {question}"
        )

        print(
            "RESULT   : "
            + (
                "IN_SCOPE ✅"
                if result.passed
                else "OUT_OF_SCOPE ❌"
            )
        )

        print(
            f"TOP SCORE: {result.top_score:.4f}"
        )

        print(
            f"REASON   : {result.reason}"
        )