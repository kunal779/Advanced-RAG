import json
import re
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path("data/extracted_text.json")
OUTPUT_PATH = Path("data/chunks.json")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Safety check
# If True, the script verifies that every source word appears
# in the chunk stream at least once.
VALIDATE_WORDS = True


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Normalize whitespace without intentionally removing words.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# BUILD CONTINUOUS WORD STREAM
# ============================================================

def build_word_stream(pages: list[dict]) -> list[dict]:
    """
    Convert all pages into one continuous stream of words.

    Each word keeps its original page number and position.
    """

    word_stream = []

    global_position = 0

    for page in pages:

        page_number = page["page"]
        text = clean_text(page.get("text", ""))

        if not text:
            continue

        words = text.split()

        for word in words:

            global_position += 1

            word_stream.append({
                "word": word,
                "page": page_number,
                "position": global_position
            })

    return word_stream


# ============================================================
# CREATE CHUNKS
# ============================================================

def create_chunks(word_stream: list[dict]) -> list[dict]:
    """
    Create overlapping chunks from the complete document.

    Example:

        Chunk 1 -> words 1-500
        Chunk 2 -> words 451-950
        Chunk 3 -> words 901-1400

    The final chunk is always preserved, even when it contains
    fewer than CHUNK_SIZE words.
    """

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    chunks = []

    step = CHUNK_SIZE - CHUNK_OVERLAP

    start = 0
    chunk_id = 1

    while start < len(word_stream):

        end = min(
            start + CHUNK_SIZE,
            len(word_stream)
        )

        chunk_words = word_stream[start:end]

        if not chunk_words:
            break

        text = " ".join(
            item["word"]
            for item in chunk_words
        )

        chunks.append({
            "chunk_id": chunk_id,

            "page_start": chunk_words[0]["page"],
            "page_end": chunk_words[-1]["page"],

            "word_start": chunk_words[0]["position"],
            "word_end": chunk_words[-1]["position"],

            "word_count": len(chunk_words),
            "character_count": len(text),

            "text": text
        })

        chunk_id += 1

        # Move forward while keeping overlap
        start += step

    return chunks


# ============================================================
# VALIDATION
# ============================================================

def validate_chunks(
    word_stream: list[dict],
    chunks: list[dict]
) -> None:
    """
    Verify that chunking did not accidentally skip any source
    words.

    Because chunks overlap, the chunked word count will be
    greater than the original word count. Therefore we compare
    the original stream against the continuous chunk positions
    instead of comparing total counts.
    """

    if not word_stream:
        raise ValueError("No words were extracted from the document.")

    if not chunks:
        raise ValueError("No chunks were created.")

    original_words = [
        item["word"]
        for item in word_stream
    ]

    # Collect all unique source positions represented
    # by chunks.
    covered_positions = set()

    for chunk in chunks:

        start = chunk["word_start"]
        end = chunk["word_end"]

        for position in range(start, end + 1):
            covered_positions.add(position)

    expected_positions = set(
        range(1, len(original_words) + 1)
    )

    missing_positions = expected_positions - covered_positions

    if missing_positions:
        first_missing = min(missing_positions)

        raise ValueError(
            f"Validation failed. "
            f"Words were skipped around position {first_missing}."
        )

    print("Validation: PASSED")
    print("No source word positions were skipped.")


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(
    word_stream: list[dict],
    chunks: list[dict]
) -> None:

    original_word_count = len(word_stream)

    chunk_word_count = sum(
        chunk["word_count"]
        for chunk in chunks
    )

    duplicated_words = (
        chunk_word_count - original_word_count
    )

    print("\nChunk Statistics")
    print("----------------")

    print(
        f"Original words:       "
        f"{original_word_count:,}"
    )

    print(
        f"Total chunks:         "
        f"{len(chunks):,}"
    )

    print(
        f"Chunk size:           "
        f"{CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap:        "
        f"{CHUNK_OVERLAP}"
    )

    print(
        f"Effective step:       "
        f"{CHUNK_SIZE - CHUNK_OVERLAP}"
    )

    print(
        f"Words including overlap: "
        f"{chunk_word_count:,}"
    )

    print(
        f"Overlap duplication:  "
        f"{duplicated_words:,}"
    )

    if chunks:

        print(
            f"First chunk pages:     "
            f"{chunks[0]['page_start']} "
            f"-> "
            f"{chunks[0]['page_end']}"
        )

        print(
            f"Last chunk pages:      "
            f"{chunks[-1]['page_start']} "
            f"-> "
            f"{chunks[-1]['page_end']}"
        )

        print(
            f"Last chunk word count: "
            f"{chunks[-1]['word_count']}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading extracted text...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    # Current extractor structure:
    #
    # {
    #     "metadata": {...},
    #     "pages": [...]
    # }

    if "pages" not in data:
        raise ValueError(
            "Invalid extracted_text.json format. "
            "Expected a 'pages' field."
        )

    pages = data["pages"]

    print(
        f"Total pages found: "
        f"{len(pages)}"
    )

    # --------------------------------------------------------
    # Build continuous word stream
    # --------------------------------------------------------

    print("Building continuous word stream...")

    word_stream = build_word_stream(pages)

    print(
        f"Total source words: "
        f"{len(word_stream):,}"
    )

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    print("Creating overlapping chunks...")

    chunks = create_chunks(word_stream)

    print(
        f"Total chunks created: "
        f"{len(chunks):,}"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if VALIDATE_WORDS:

        print("Validating chunk coverage...")

        validate_chunks(
            word_stream,
            chunks
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print_statistics(
        word_stream,
        chunks
    )

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    if chunks:

        print("\nFirst chunk preview")
        print("-------------------")
        print(
            chunks[0]["text"][:500]
        )

        print("\nLast chunk preview")
        print("------------------")
        print(
            chunks[-1]["text"][:500]
        )

    print(
        f"\nSaved chunks to: "
        f"{OUTPUT_PATH}"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()