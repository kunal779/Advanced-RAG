from __future__ import annotations

from src.agent import create_agent


def main() -> None:

    print(
        "\n" + "=" * 80
    )

    print(
        "ADVANCED RAG SYSTEM"
    )

    print(
        "=" * 80
    )

    print(
        "\nInitializing system..."
    )

    agent = create_agent()

    print(
        "\nSystem ready."
    )

    print(
        "Type 'exit' or 'quit' to stop."
    )

    while True:

        try:

            question = input(
                "\nYou: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print(
                "\n\nExiting..."
            )

            break

        if question.lower() in {
            "exit",
            "quit",
        }:

            print(
                "\nExiting..."
            )

            break

        if not question:

            print(
                "Please enter a question."
            )

            continue

        try:

            agent.run(
                question
            )

        except Exception as exc:

            print(
                "\nRAG pipeline error:"
            )

            print(
                exc
            )


if __name__ == "__main__":
    main()