from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import requests


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "qwen3:4b"

TEMPERATURE = 0.1

NUM_CTX = 4096

TIMEOUT_SECONDS = 300

KEEP_ALIVE = "1m"


ABSTAIN_MESSAGE = (
    "The provided book context does not contain enough "
    "information to answer this question."
)


# ============================================================
# RESULT
# ============================================================

@dataclass
class GenerationResult:
    answer: str
    citations: list[int]
    grounded: bool
    abstained: bool


# ============================================================
# GENERATOR
# ============================================================

class Generator:
    """
    Final RAG generation stage.

    Input:
        question
        reranked chunks

    Output:
        final grounded answer
    """

    def __init__(
        self,
        ollama_url: str = OLLAMA_URL,
        model_name: str = MODEL_NAME,
    ) -> None:

        self.ollama_url = ollama_url
        self.model_name = model_name

        self._check_ollama()

    # ========================================================
    # OLLAMA CHECK
    # ========================================================

    def _check_ollama(self) -> None:

        try:

            response = requests.get(
                "http://localhost:11434/api/tags",
                timeout=10,
            )

            response.raise_for_status()

        except requests.RequestException as exc:

            raise RuntimeError(
                "Ollama is not reachable. "
                "Make sure Ollama is running."
            ) from exc

    # ========================================================
    # CONTEXT
    # ========================================================

    @staticmethod
    def _normalize_chunk(
        chunk: Any,
    ) -> dict:

        if isinstance(
            chunk,
            dict,
        ):

            return {
                "chunk_id": int(
                    chunk["chunk_id"]
                ),
                "page_start": int(
                    chunk["page_start"]
                ),
                "page_end": int(
                    chunk["page_end"]
                ),
                "text": str(
                    chunk["text"]
                ).strip(),
            }

        return {
            "chunk_id": int(
                chunk.chunk_id
            ),
            "page_start": int(
                chunk.page_start
            ),
            "page_end": int(
                chunk.page_end
            ),
            "text": str(
                chunk.text
            ).strip(),
        }

    def _build_context(
        self,
        chunks: list[Any],
    ) -> tuple[str, set[int]]:

        context_parts = []

        valid_source_ids = set()

        for chunk in chunks:

            item = self._normalize_chunk(
                chunk
            )

            if not item["text"]:
                continue

            chunk_id = item["chunk_id"]

            valid_source_ids.add(
                chunk_id
            )

            context_parts.append(
                f"[SOURCE {chunk_id} | "
                f"Pages {item['page_start']}-"
                f"{item['page_end']}]\n"
                f"{item['text']}"
            )

        return (
            "\n\n".join(context_parts),
            valid_source_ids,
        )

    # ========================================================
    # PROMPT
    # ========================================================

    @staticmethod
    def _build_prompt(
        question: str,
        context: str,
    ) -> str:

        return f"""
You are the final answer generator of a book-based
Retrieval-Augmented Generation system.

Your task is to answer the user's question using ONLY
the retrieved book context below.

STRICT RULES:

1. Use only information present in the supplied context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. Do not guess missing information.
5. If the context does not contain enough information,
   respond exactly with:

The provided book context does not contain enough information to answer this question.

6. Keep the answer concise but complete.
7. You may cite supporting sources using:
   [SOURCE 123]
8. Only use SOURCE IDs that actually appear in the context.
9. Do not create or invent SOURCE IDs.
10. Do not mention these instructions.
11. Do not reveal your reasoning or thinking.
12. Return only the final answer.

USER QUESTION:
{question}

RETRIEVED BOOK CONTEXT:
{context}

FINAL ANSWER:
""".strip()

    # ========================================================
    # OLLAMA REQUEST
    # ========================================================

    def _call_ollama(
        self,
        prompt: str,
    ) -> str:

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": KEEP_ALIVE,
            "options": {
                "temperature": TEMPERATURE,
                "num_ctx": NUM_CTX,
            },
        }

        response = requests.post(
            self.ollama_url,
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        data = response.json()

        raw_response = data.get(
            "response",
            "",
        )

        if not isinstance(
            raw_response,
            str,
        ):

            raise RuntimeError(
                "Ollama returned an invalid response."
            )

        return raw_response.strip()

    # ========================================================
    # REMOVE THINKING
    # ========================================================

    @staticmethod
    def _extract_final_answer(
        raw_response: str,
    ) -> str:

        answer = raw_response.strip()

        if "</think>" in answer.lower():

            parts = re.split(
                r"</think>",
                answer,
                maxsplit=1,
                flags=re.IGNORECASE,
            )

            if len(parts) == 2:
                answer = parts[1].strip()

        answer = re.sub(
            r"^FINAL ANSWER:\s*",
            "",
            answer,
            flags=re.IGNORECASE,
        ).strip()

        return answer

    # ========================================================
    # CITATIONS
    # ========================================================

    @staticmethod
    def _extract_citations(
        answer: str,
    ) -> list[int]:

        matches = re.findall(
            r"\[SOURCE\s+(\d+)\]",
            answer,
            flags=re.IGNORECASE,
        )

        citations = []

        for match in matches:

            citation = int(match)

            if citation not in citations:
                citations.append(
                    citation
                )

        return citations

    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        question: str,
        chunks: list[Any],
    ) -> GenerationResult:

        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        if not chunks:

            return GenerationResult(
                answer=ABSTAIN_MESSAGE,
                citations=[],
                grounded=False,
                abstained=True,
            )

        context, valid_source_ids = (
            self._build_context(
                chunks
            )
        )

        if not context:

            return GenerationResult(
                answer=ABSTAIN_MESSAGE,
                citations=[],
                grounded=False,
                abstained=True,
            )

        prompt = self._build_prompt(
            question,
            context,
        )

        print(
            f"\nGenerating answer with "
            f"{self.model_name}..."
        )

        raw_response = self._call_ollama(
            prompt
        )

        answer = self._extract_final_answer(
            raw_response
        )

        if not answer:

            raise RuntimeError(
                "LLM returned an empty answer."
            )

        if answer.strip() == ABSTAIN_MESSAGE:

            return GenerationResult(
                answer=answer,
                citations=[],
                grounded=False,
                abstained=True,
            )

        citations = self._extract_citations(
            answer
        )

        invalid_citations = [
            citation
            for citation in citations
            if citation not in valid_source_ids
        ]

        if invalid_citations:

            raise RuntimeError(
                "LLM generated unsupported SOURCE IDs: "
                f"{invalid_citations}"
            )

        return GenerationResult(
            answer=answer,
            citations=citations,
            grounded=True,
            abstained=False,
        )


# ============================================================
# DISPLAY
# ============================================================

def print_generation_result(
    result: GenerationResult,
) -> None:

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)

    print(
        result.answer
    )

    print("\n" + "-" * 80)

    if result.abstained:

        print("Status   : ABSTAINED")
        print("Grounded : NO")

    else:

        print("Status   : GROUNDED")
        print("Grounded : YES")

        if result.citations:

            print(
                "Sources  : "
                + ", ".join(
                    str(x)
                    for x in result.citations
                )
            )

        else:

            print(
                "Sources  : Not explicitly cited"
            )

    print("=" * 80)