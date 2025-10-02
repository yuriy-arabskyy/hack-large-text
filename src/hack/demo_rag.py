"""Demo script for RAG with FAISS retriever and DSPy."""

import os
import dspy
from dotenv import load_dotenv
from hack.rag_agent import create_agent

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")


def main():
    """Run demo queries through the RAG agent."""

    # Configure DSPy with OpenAI
    lm = dspy.LM(model="openai/gpt-4o-mini", api_key=openai_api_key)
    dspy.configure(lm=lm)

    print("Initializing agent with FAISS retriever...")
    agent = create_agent()

    print("\n" + "=" * 80)
    print("RAG Agent Demo - Chess Fundamentals by Capablanca")
    print("=" * 80)

    # Example queries
    queries = [
        "What are the key principles for opening moves in chess?",
        "How should I approach the endgame?",
        "What is the importance of pawn structure?",
    ]

    for i, question in enumerate(queries, 1):
        print(f"\n{'─' * 80}")
        print(f"Query #{i}: {question}")
        print(f"{'─' * 80}")

        try:
            # Run the agent
            result = agent.forward(question)

            print("\nAnswer:")
            print(result["answer"])

            print("\nCitations:")
            print(result["citations"])

        except Exception as e:
            print(f"Error processing query: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
