import json
from pathlib import Path
import dspy
from dotenv import load_dotenv
import os

# --- Setup ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

lm = dspy.LM(
    "openai/gpt-4.1",   # or "openai/gpt-4o-mini" for faster/cheaper
    api_key=openai_api_key,
    temperature=0.0     # make answers deterministic
)
dspy.settings.configure(lm=lm)

# --- Signature ---
class AnswerGeneration(dspy.Signature):
    """Answer a question based only on the given chess context text."""
    context = dspy.InputField(desc="Passage from Capablanca's Chess Fundamentals")
    question = dspy.InputField(desc="A well-formed question about the passage")
    answer = dspy.OutputField(desc="Concise, correct answer derived only from the passage")

# --- Main pipeline ---
def main():
    input_path = Path("golden_qa_with_questions.json")
    output_path = Path("golden_qa_with_answers.json")

    # Load data (context + question)
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    gen_answer = dspy.Predict(AnswerGeneration)

    results = []
    for i, item in enumerate(dataset, 1):
        ctx = item["context"]
        q = item["question"]

        try:
            a = gen_answer(context=ctx, question=q)
            results.append({
                "context": ctx,
                "question": q,
                "answer": a.answer.strip()
            })
            print(f"[{i}] Answer: {a.answer.strip()}")
        except Exception as e:
            print(f"[{i}] Error generating answer: {e}")
            results.append({
                "context": ctx,
                "question": q,
                "answer": ""
            })

    # Save final dataset
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} items with answers → {output_path}")

if __name__ == "__main__":
    main()