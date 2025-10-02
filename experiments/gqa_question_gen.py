import json
from pathlib import Path
import dspy
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


lm = dspy.LM(
    "openai/gpt-4.1",
    api_key=openai_api_key,
)

dspy.settings.configure(lm=lm)

class QuestionGeneration(dspy.Signature):
    """Generate a clear, high-quality, learner-style question from the given chess context text."""
    context = dspy.InputField(desc="A passage from Capablanca's Chess Fundamentals")
    question = dspy.OutputField(desc="A well-formed question that can be answered from the passage")

# --------- Main pipeline ---------
def main():
    input_path = Path("golden_qa_data.json")
    output_path = Path("golden_qa_with_questions.json")

    # Load contexts
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Create DSPy predictor
    gen_question = dspy.Predict(QuestionGeneration)

    results = []
    for i, item in enumerate(dataset, 1):
        ctx = item["context"]

        try:
            q = gen_question(context=ctx)
            results.append({
                "context": ctx,
                "question": q.question.strip()
            })
            print(f"[{i}] Question: {q.question.strip()}")
        except Exception as e:
            print(f"[{i}] Error generating question: {e}")
            results.append({
                "context": ctx,
                "question": ""
            })

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} items with questions → {output_path}")

if __name__ == "__main__":
    main()