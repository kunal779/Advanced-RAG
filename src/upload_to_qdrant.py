import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


INPUT_PATH = Path("data/embeddings.json")
QDRANT_PATH = Path("data/qdrant")

COLLECTION_NAME = "rag_documents"

BATCH_SIZE = 32
VECTOR_SIZE = 384

VALIDATE_UPLOAD = True


def load_embeddings() -> dict:
    """Load embedded chunks from embeddings.json."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {INPUT_PATH}"
        )

    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if "metadata" not in data:
        raise ValueError(
            "Invalid embeddings.json: metadata section missing."
        )

    if "embeddings" not in data:
        raise ValueError(
            "Invalid embeddings.json: embeddings section missing."
        )

    embeddings = data["embeddings"]

    if not embeddings:
        raise ValueError(
            "embeddings.json contains no embeddings."
        )

    return data


def validate_embeddings(data: dict) -> None:
    """Validate vector dimensions and required fields."""

    metadata = data["metadata"]
    embeddings = data["embeddings"]

    stored_dimension = metadata.get("dimension")

    if stored_dimension != VECTOR_SIZE:
        raise ValueError(
            f"Vector dimension mismatch. "
            f"Expected {VECTOR_SIZE}, got {stored_dimension}."
        )

    for index, item in enumerate(embeddings, start=1):

        required_fields = [
            "chunk_id",
            "text",
            "embedding",
        ]

        for field in required_fields:
            if field not in item:
                raise ValueError(
                    f"Missing '{field}' in embedding item {index}."
                )

        vector = item["embedding"]

        if len(vector) != VECTOR_SIZE:
            raise ValueError(
                f"Invalid vector size at item {index}. "
                f"Expected {VECTOR_SIZE}, got {len(vector)}."
            )

    print("Embedding validation: PASSED")
    print(
        f"Verified {len(embeddings):,} vectors "
        f"× {VECTOR_SIZE} dimensions."
    )


def create_client() -> QdrantClient:
    """Create a local persistent Qdrant client."""

    if not QDRANT_PATH.exists():
        raise FileNotFoundError(
            f"Qdrant database not found: {QDRANT_PATH}"
        )

    print(f"Opening Qdrant database: {QDRANT_PATH}")

    return QdrantClient(
        path=str(QDRANT_PATH)
    )


def build_points(embeddings: list[dict]) -> list[PointStruct]:
    """Convert embedded chunks into Qdrant points."""

    points = []

    for item in embeddings:

        point = PointStruct(
            id=item["chunk_id"],

            vector=item["embedding"],

            payload={
                "chunk_id": item["chunk_id"],
                "page_start": item["page_start"],
                "page_end": item["page_end"],
                "word_start": item["word_start"],
                "word_end": item["word_end"],
                "word_count": item["word_count"],
                "character_count": item["character_count"],
                "text": item["text"],
            },
        )

        points.append(point)

    return points


def upload_points(
    client: QdrantClient,
    points: list[PointStruct],
) -> None:
    """Upload points to Qdrant in batches."""

    total_points = len(points)

    print(
        f"\nUploading {total_points:,} points "
        f"in batches of {BATCH_SIZE}..."
    )

    for start in range(0, total_points, BATCH_SIZE):

        batch = points[start:start + BATCH_SIZE]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
        )

        uploaded = min(
            start + len(batch),
            total_points,
        )

        print(
            f"Uploaded {uploaded:,}/{total_points:,}"
        )


def validate_upload(
    client: QdrantClient,
    expected_count: int,
) -> None:
    """Verify that all points reached Qdrant."""

    info = client.get_collection(
        collection_name=COLLECTION_NAME
    )

    actual_count = info.points_count

    if actual_count != expected_count:
        raise ValueError(
            f"Upload validation failed. "
            f"Expected {expected_count} points, "
            f"but Qdrant contains {actual_count}."
        )

    print("\nUpload validation: PASSED")
    print(
        f"Qdrant contains {actual_count:,} points."
    )


def main():
    print("Starting Qdrant upload pipeline...\n")

    data = load_embeddings()

    embeddings = data["embeddings"]

    print(
        f"Loaded {len(embeddings):,} embedded chunks."
    )

    if VALIDATE_UPLOAD:
        validate_embeddings(data)

    client = create_client()

    print("\nBuilding Qdrant points...")

    points = build_points(embeddings)

    print(
        f"Created {len(points):,} Qdrant points."
    )

    upload_points(client, points)

    if VALIDATE_UPLOAD:
        validate_upload(
            client,
            expected_count=len(points),
        )

    print(
        "\nQdrant upload completed successfully."
    )


if __name__ == "__main__":
    main()