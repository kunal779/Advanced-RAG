import json
import re
from pathlib import Path

import pymupdf


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = Path("data/hands_on_llm.pdf")
OUTPUT_PATH = Path("data/extracted_text.json")

# Keep detailed information about every page
SAVE_BLOCKS = True

# Remove repeated header/footer lines if they appear
# on many pages.
REMOVE_REPEATED_HEADERS_FOOTERS = True

# A line appearing on this many pages is considered
# a possible repeated header/footer.
REPEATED_LINE_THRESHOLD = 0.30


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Clean extracted text without aggressively deleting content.
    """

    if not text:
        return ""

    # Normalize different newline formats
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Join words broken by PDF line wrapping:
    #
    # retrie-
    # val
    #
    # becomes:
    #
    # retrieval
    #
    text = re.sub(
        r"(?<=\w)-\n(?=\w)",
        "",
        text
    )

    # Replace remaining line breaks with spaces
    text = re.sub(r"\s*\n\s*", " ", text)

    # Collapse repeated whitespace
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# ============================================================
# BLOCK EXTRACTION
# ============================================================

def extract_page_blocks(page):
    """
    Extract text blocks from a page.

    Blocks retain their position on the page, which gives us
    more control than simply calling page.get_text().
    """

    blocks = page.get_text("blocks")

    extracted_blocks = []

    for block in blocks:

        if len(block) < 5:
            continue

        x0, y0, x1, y1, text = block[:5]

        cleaned = normalize_text(text)

        if not cleaned:
            continue

        extracted_blocks.append({
            "text": cleaned,
            "x0": round(x0, 2),
            "y0": round(y0, 2),
            "x1": round(x1, 2),
            "y1": round(y1, 2),
        })

    # Reading order:
    # top → bottom
    # left → right
    extracted_blocks.sort(
        key=lambda block: (
            block["y0"],
            block["x0"]
        )
    )

    return extracted_blocks


# ============================================================
# FIND REPEATED LINES
# ============================================================

def find_repeated_lines(page_data):
    """
    Find lines that repeat across many pages.

    These are usually headers, footers, or running titles.
    """

    line_counts = {}

    total_pages = len(page_data)

    for page in page_data:

        seen_on_this_page = set()

        for block in page["blocks"]:

            text = block["text"]

            # Break block into lines
            for line in text.split("\n"):

                line = normalize_text(line)

                if not line:
                    continue

                # Ignore extremely long lines.
                # They are unlikely to be headers/footers.
                if len(line) > 150:
                    continue

                normalized_line = line.lower()

                if normalized_line not in seen_on_this_page:

                    line_counts[normalized_line] = (
                        line_counts.get(normalized_line, 0) + 1
                    )

                    seen_on_this_page.add(normalized_line)

    threshold = max(
        2,
        int(total_pages * REPEATED_LINE_THRESHOLD)
    )

    repeated_lines = {
        line
        for line, count in line_counts.items()
        if count >= threshold
    }

    return repeated_lines


# ============================================================
# REMOVE REPEATED HEADER / FOOTER
# ============================================================

def remove_repeated_lines(blocks, repeated_lines):
    """
    Remove only lines identified as repeated headers/footers.
    """

    cleaned_blocks = []

    for block in blocks:

        text = block["text"]

        lines = text.split("\n")

        kept_lines = []

        for line in lines:

            normalized = normalize_text(line).lower()

            if normalized in repeated_lines:
                continue

            kept_lines.append(line)

        new_text = normalize_text("\n".join(kept_lines))

        if new_text:
            new_block = block.copy()
            new_block["text"] = new_text

            cleaned_blocks.append(new_block)

    return cleaned_blocks


# ============================================================
# EXTRACT PDF
# ============================================================

def extract_pdf():

    print("Opening PDF...")

    doc = pymupdf.open(PDF_PATH)

    pages = []

    for page_number, page in enumerate(doc, start=1):

        blocks = extract_page_blocks(page)

        page_text = " ".join(
            block["text"]
            for block in blocks
        )

        page_text = normalize_text(page_text)

        pages.append({
            "page": page_number,
            "text": page_text,
            "blocks": blocks
        })

        if page_number % 50 == 0:
            print(f"Processed {page_number}/{len(doc)} pages...")

    doc.close()

    return pages


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(pages):

    total_characters = 0
    total_words = 0

    for page in pages:

        text = page["text"]

        total_characters += len(text)
        total_words += len(text.split())

    return {
        "pages": len(pages),
        "characters": total_characters,
        "words": total_words
    }


# ============================================================
# MAIN
# ============================================================

def main():

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    pages = extract_pdf()

    print("\nInitial extraction complete.")

    # --------------------------------------------------------
    # Detect repeated headers / footers
    # --------------------------------------------------------

    repeated_lines = set()

    if REMOVE_REPEATED_HEADERS_FOOTERS:

        print("Detecting repeated headers/footers...")

        repeated_lines = find_repeated_lines(pages)

        print(
            f"Possible repeated lines found: "
            f"{len(repeated_lines)}"
        )

        for page in pages:

            page["blocks"] = remove_repeated_lines(
                page["blocks"],
                repeated_lines
            )

            page["text"] = normalize_text(
                " ".join(
                    block["text"]
                    for block in page["blocks"]
                )
            )

    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    statistics = calculate_statistics(pages)

    print("\nExtraction Statistics")
    print("---------------------")
    print(f"Pages:      {statistics['pages']}")
    print(f"Characters: {statistics['characters']:,}")
    print(f"Words:      {statistics['words']:,}")

    # --------------------------------------------------------
    # Prepare output
    # --------------------------------------------------------

    output_data = {
        "metadata": {
            "source": PDF_PATH.name,
            "total_pages": statistics["pages"],
            "total_characters": statistics["characters"],
            "total_words": statistics["words"],
        },
        "pages": pages
    }

    # Optionally remove block-level data from final JSON
    if not SAVE_BLOCKS:

        for page in output_data["pages"]:
            page.pop("blocks", None)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output_data,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"\nSaved extracted data to: {OUTPUT_PATH}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()