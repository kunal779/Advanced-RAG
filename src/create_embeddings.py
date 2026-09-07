import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path("data/chunks.json")
OUTPUT_PATH = Path("data/embeddings.json")

MODEL_NAME = "all-MiniLM-L6-v2"

# Process multiple chunks together
BATCH_SIZE = 32

# Normalize vectors to unit length.
# This is useful for cosine-similarity based retrieval.
NORMALIZE_EMBEDDINGS = True

# Safety checks
VALIDATE_EMBEDDINGS = True


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks() -> list[dict]:
    """
    Load chunks created by chunk_text.py.
    """

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {INPUT_PATH}"
        )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    if not isinstance(chunks, list):
        raise ValueError(
            "Invalid chunks.json format. "
            "Expected a list of chunks."
        )

    if not chunks:
        raise ValueError(
            "chunks.json contains no chunks."
        )

    return chunks


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_model() -> SentenceTransformer:
    """
    Load the local Sentence Transformer model.
    """

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    return model


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(
    model: SentenceTransformer,
    chunks: list[dict]
) -> list:

    """
    Convert every chunk's text into an embedding vector.
    """

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"Creating embeddings for "
        f"{len(texts):,} chunks..."
    )

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        convert_to_numpy=True
    )

    return embeddings


# ============================================================
# VALIDATE EMBEDDINGS
# ============================================================

def validate_embeddings(
    chunks: list[dict],
    embeddings
) -> int:

    """
    Make sure every chunk has exactly one embedding
    and every embedding has the expected dimension.
    """

    expected_dimension = 384

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Validation failed: "
            "number of embeddings does not match "
            "number of chunks."
        )

    actual_dimension = embeddings.shape[1]

    if actual_dimension != expected_dimension:
        raise ValueError(
            f"Validation failed: expected "
            f"{expected_dimension} dimensions, "
            f"got {actual_dimension}."
        )

    print("Validation: PASSED")
    print(
        f"Every chunk has a "
        f"{actual_dimension}-dimensional embedding."
    )

    return actual_dimension


# ============================================================
# PREPARE OUTPUT
# ============================================================

def prepare_output(
    chunks: list[dict],
    embeddings,
    dimension: int
) -> dict:

    """
    Combine chunk information with its embedding.

    This keeps the output compatible with the next
    Qdrant/vector database stage.
    """

    embedded_chunks = []

    for chunk, embedding in zip(
        chunks,
        embeddings
    ):

        embedded_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "word_start": chunk["word_start"],
            "word_end": chunk["word_end"],
            "word_count": chunk["word_count"],
            "character_count": chunk["character_count"],
            "text": chunk["text"],
            "embedding": embedding.tolist()
        })

    return {
        "metadata": {
            "model": MODEL_NAME,
            "dimension": dimension,
            "total_chunks": len(chunks),
            "batch_size": BATCH_SIZE,
            "normalized": NORMALIZE_EMBEDDINGS
        },
        "embeddings": embedded_chunks
    }


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_output(data: dict) -> None:
    """
    Save embeddings to JSON.
    """

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"\nEmbeddings saved to: "
        f"{OUTPUT_PATH}"
    )


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(
    chunks: list[dict],
    embeddings,
    dimension: int
) -> None:

    print("\nEmbedding Statistics")
    print("--------------------")

    print(
        f"Chunks:             "
        f"{len(chunks):,}"
    )

    print(
        f"Embedding dimension: "
        f"{dimension}"
    )

    print(
        f"Total values:        "
        f"{len(chunks) * dimension:,}"
    )

    print(
        f"Normalized:          "
        f"{NORMALIZE_EMBEDDINGS}"
    )

    if len(embeddings) > 0:

        print("\nFirst embedding preview")
        print("-----------------------")

        print(
            embeddings[0][:10]
        )

        print(
            "... "
            "(showing first 10 of "
            f"{dimension} values)"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("Starting embedding pipeline...\n")

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    chunks = load_chunks()

    print(
        f"Loaded {len(chunks):,} chunks."
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    embeddings = create_embeddings(
        model,
        chunks
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if VALIDATE_EMBEDDINGS:

        dimension = validate_embeddings(
            chunks,
            embeddings
        )

    else:

        dimension = embeddings.shape[1]

    # --------------------------------------------------------
    # Prepare output
    # --------------------------------------------------------

    output_data = prepare_output(
        chunks,
        embeddings,
        dimension
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_output(output_data)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print_statistics(
        chunks,
        embeddings,
        dimension
    )

    print("\nEmbedding pipeline completed successfully.")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()