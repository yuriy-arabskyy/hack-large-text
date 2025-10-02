import json
from pathlib import Path
import dspy
from dotenv import load_dotenv
import os
from statistics import mean

from hack.rag_agent import create_agent  # your WorkspaceAgent factory
try:
    from gepa import GEPA
except Exception:
    GEPA = None

# --- Load API key ---
load_dotenv()
# Silence HF tokenizers fork warning during DSPy bootstrapping
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Judge signature ---
class JudgeSignature(dspy.Signature):
    """Judge whether the predicted answer correctly matches the gold answer given the question."""
    question = dspy.InputField()
    gold_answer = dspy.InputField()
    pred_answer = dspy.InputField()
    correct = dspy.OutputField(desc="yes or no")

def main():
    # --- Load golden QA dataset ---
    input_path = Path("/Users/jamisonproctor/Documents/dev/hack-large-text/experiments/golden_qa_with_answers.json")
    with open(input_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    # Prepare datasets
    raw = [{"question": item["question"], "answer": item["answer"]} for item in golden_data]
    # Use roughly half for training to maximize signal while keeping holdout
    split = max(1, len(raw) // 2)
    train_raw, val = raw[:split], raw[split:]
    # DSPy teleprompters expect dspy.Example with declared inputs
    train = [
        dspy.Example(question=ex["question"], answer=ex["answer"]).with_inputs("question")
        for ex in train_raw
    ]

    # --- Define LMs explicitly ---
    coach_lm = dspy.LM("openai/gpt-4.1", api_key=openai_api_key, temperature=0.7)         # optimization
    worker_lm = dspy.LM("openai/gpt-4.1-nano", api_key=openai_api_key, temperature=0.0)   # runtime
    judge_lm = dspy.LM("openai/gpt-4.1", api_key=openai_api_key, temperature=0.0)         # judge

    # Configure default LM for DSPy predictors (required by dspy 3.x)
    # This ensures modules like WorkspaceAgent have an active LM when invoked.
    dspy.configure(lm=worker_lm)

    # --- Judge predictor ---
    judge = dspy.Predict(JudgeSignature)
    judge.lm = judge_lm

    def judge_metric(example, prediction, trace=None):
        q = example["question"]
        gold = example["answer"]
        pred = prediction["answer"]
        resp = judge(question=q, gold_answer=gold, pred_answer=pred)
        return 1 if resp.correct.strip().lower().startswith("y") else 0

    # --- Baseline evaluation ---
    print("📊 Evaluating baseline agent with worker LM...")
    agent = create_agent(
        lm=worker_lm,
        faiss_index_path="/Users/jamisonproctor/Documents/dev/hack-large-text/experiments/chess_pdf.faiss",
        workspace_json_path="/Users/jamisonproctor/Documents/dev/hack-large-text/experiments/workspace_with_embeddings.json",
    )
    baseline_scores = [judge_metric(ex, agent(ex["question"])) for ex in val]
    baseline_acc = mean(baseline_scores)
    print(f"Baseline validation accuracy: {baseline_acc:.2%}")

    # Baseline on the short slice we'll use for quick comparison
    val_slice = val[:20] if len(val) > 20 else val
    baseline_slice_acc = mean([judge_metric(ex, agent(ex["question"])) for ex in val_slice])
    print(f"Baseline (quick slice) accuracy: {baseline_slice_acc:.2%}")

    # --- Step 1: Optimize agent with GEPA ---
    print("\n🧪 GEPA search (max iterations)…")
    if GEPA is None:
        raise ImportError(
            "gepa is not installed in this environment. Install with `poetry add gepa[dspy]` and retry."
        )

    # Allow switching the GEPA auto level via env, default to 'medium' (try 'heavy' if available).
    gepa_auto = os.getenv("GEPA_AUTO_LEVEL", "medium")

    gepa = GEPA(
        metric=judge_metric,
        auto=gepa_auto,
        num_threads=8,
        track_stats=False,
        use_merge=False,
        reflection_lm=dspy.LM("openai/gpt-4o-mini", temperature=0.3, api_key=openai_api_key),
    )

    optimized_agent = gepa.compile(agent, trainset=train)

    # Save full program for later reuse
    optimized_agent.save("optimized_workspace_agent", save_program=True)
    print("✅ Optimization complete. Saved optimized agent to optimized_workspace_agent/")

    # Inspect compiled demos to verify optimization bound examples
    print("\n📎 Demo counts per predictor (post-compile):")
    for name, predictor in optimized_agent.named_predictors():
        demos = getattr(predictor, "demos", []) or []
        print(f"  - {name}: {len(demos)} demos")

    # --- Step 2: Reload with worker LM for runtime ---
    print("\n🔄 Switching to gpt-4.1-nano for runtime...")
    optimized_agent = dspy.load("optimized_workspace_agent")
    optimized_agent.set_lm(worker_lm)

    # --- Optimized evaluation ---
    print("📊 Evaluating optimized agent with worker LM...")
    # For quicker feedback, evaluate on a small validation slice, then expand
    optimized_scores = [judge_metric(ex, optimized_agent(ex["question"])) for ex in val_slice]
    optimized_acc = mean(optimized_scores)
    print(f"Optimized (quick slice) accuracy: {optimized_acc:.2%}")

    # (Demo output removed per request)

if __name__ == "__main__":
    main()
