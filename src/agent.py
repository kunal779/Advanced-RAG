from __future__ import annotations

from dataclasses import dataclass

from .generator import (
    ABSTAIN_MESSAGE,
    GenerationResult,
    Generator,
    print_generation_result,
)

from .query_gate import (
    QUERY_GATE_THRESHOLD,
    QUERY_GATE_TOP_K,
    QueryGate,
)

from .reranker import (
    DEFAULT_TOP_K as RERANK_TOP_K,
    RankedChunk,
    Reranker,
)

from .retriever import (
    DEFAULT_TOP_K as RETRIEVAL_TOP_K,
    Retriever,
)


# ============================================================
# FINAL CONFIGURATION
# ============================================================

# Initial evidence relevance threshold.
RELEVANCE_THRESHOLD = QUERY_GATE_THRESHOLD

# Number of final chunks sent to the generator.
FINAL_CONTEXT_TOP_K = RERANK_TOP_K


# ============================================================
# RESULT
# ============================================================

@dataclass
class AgentResult:
    question: str

    retrieved_count: int

    ranked_count: int

    final_context_count: int

    relevance_passed: bool

    top_relevance_score: float

    generation: GenerationResult


# ============================================================
# RAG AGENT
# ============================================================

class RAGAgent:
    """
    Complete RAG orchestrator.

    Workflow:

        Question
            ↓
        Query Gate
            ↓
        Qdrant
            ↓
        BGE
            ↓
        Relevance Decision
            │
            ├── FAIL → ABSTAIN
            │
            └── PASS
                  ↓
              Top 5 chunks
                  ↓
              Qwen3 4B
                  ↓
              Final answer
    """

    def __init__(
        self,
        retrieval_top_k: int = RETRIEVAL_TOP_K,
        final_top_k: int = FINAL_CONTEXT_TOP_K,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
    ) -> None:

        self.retrieval_top_k = retrieval_top_k

        self.final_top_k = final_top_k

        self.relevance_threshold = (
            relevance_threshold
        )

        print(
            "\n" + "=" * 80
        )

        print(
            "INITIALIZING RAG AGENT"
        )

        print(
            "=" * 80
        )

        print(
            f"Retrieval Top-K    : "
            f"{self.retrieval_top_k}"
        )

        print(
            f"Final Context Top-K: "
            f"{self.final_top_k}"
        )

        print(
            f"Relevance threshold: "
            f"{self.relevance_threshold:.4f}"
        )

        # ----------------------------------------------------
        # Load each model only ONCE.
        # ----------------------------------------------------

        self.retriever = Retriever()

        self.reranker = Reranker()

        self.query_gate = QueryGate(
            retriever=self.retriever,
            reranker=self.reranker,
            top_k=self.retrieval_top_k,
            threshold=self.relevance_threshold,
        )

        self.generator = Generator()

        print(
            "\nRAG Agent initialized successfully."
        )

    # ========================================================
    # BUILD ABSTENTION
    # ========================================================

    @staticmethod
    def _build_abstention_result(
        question: str,
        retrieved_count: int,
        ranked_count: int,
        top_score: float,
    ) -> AgentResult:

        generation = GenerationResult(
            answer=ABSTAIN_MESSAGE,
            citations=[],
            grounded=False,
            abstained=True,
        )

        return AgentResult(
            question=question,
            retrieved_count=retrieved_count,
            ranked_count=ranked_count,
            final_context_count=0,
            relevance_passed=False,
            top_relevance_score=top_score,
            generation=generation,
        )

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
        question: str,
    ) -> AgentResult:

        question = question.strip()

        if not question:

            raise ValueError(
                "Question cannot be empty."
            )

        print(
            "\n" + "=" * 80
        )

        print(
            "RUNNING RAG AGENT"
        )

        print(
            "=" * 80
        )

        print(
            f"\nQuestion: {question}"
        )

        # ====================================================
        # STEP 1 — QUERY GATE
        # ====================================================

        print(
            "\n[1/2] Checking question relevance..."
        )

        gate_result = self.query_gate.check(
            question
        )

        # ----------------------------------------------------
        # Reject before LLM
        # ----------------------------------------------------

        if not gate_result.passed:

            print(
                "\nLLM generation skipped."
            )

            print(
                "Reason:"
            )

            print(
                gate_result.reason
            )

            result = (
                self._build_abstention_result(
                    question=question,
                    retrieved_count=(
                        gate_result.candidate_count
                    ),
                    ranked_count=len(
                        gate_result.ranked_chunks
                    ),
                    top_score=(
                        gate_result.top_score
                    ),
                )
            )

            print_generation_result(
                result.generation
            )

            return result

        # ====================================================
        # STEP 2 — FINAL GENERATION
        # ====================================================

        print(
            "\n[2/2] Preparing final context..."
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We DO NOT run Qdrant again.
        # We DO NOT run BGE again.
        #
        # The gate already returned all ranked chunks.
        # We simply take the best final_top_k.
        # ----------------------------------------------------

        final_chunks = (
            gate_result.ranked_chunks[
                :self.final_top_k
            ]
        )

        if not final_chunks:

            result = (
                self._build_abstention_result(
                    question=question,
                    retrieved_count=(
                        gate_result.candidate_count
                    ),
                    ranked_count=len(
                        gate_result.ranked_chunks
                    ),
                    top_score=(
                        gate_result.top_score
                    ),
                )
            )

            print_generation_result(
                result.generation
            )

            return result

        print(
            "\nFinal context chunks:"
        )

        for chunk in final_chunks:

            print(
                f"  Chunk {chunk.chunk_id} | "
                f"BGE={chunk.rerank_score:.4f} | "
                f"Pages="
                f"{chunk.page_start}-"
                f"{chunk.page_end}"
            )

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        print(
            "\nGenerating grounded answer..."
        )

        generation_result = (
            self.generator.generate(
                question,
                final_chunks,
            )
        )

        print_generation_result(
            generation_result
        )

        print(
            "\n" + "=" * 80
        )

        print(
            "RAG AGENT COMPLETED"
        )

        print(
            "=" * 80
        )

        return AgentResult(
            question=question,
            retrieved_count=(
                gate_result.candidate_count
            ),
            ranked_count=len(
                gate_result.ranked_chunks
            ),
            final_context_count=len(
                final_chunks
            ),
            relevance_passed=True,
            top_relevance_score=(
                gate_result.top_score
            ),
            generation=generation_result,
        )


# ============================================================
# FACTORY
# ============================================================

def create_agent() -> RAGAgent:
    """
    Create one RAG agent.
    """

    return RAGAgent()


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    agent = create_agent()

    while True:

        question = input(
            "\nEnter your question: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
        }:

            break

        if not question:

            continue

        try:

            agent.run(
                question
            )

        except Exception as exc:

            print(
                "\nAgent error:"
            )

            print(
                exc
            )