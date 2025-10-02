import json
import random
import re
from pathlib import Path

# -------- CONFIG --------
MIN_CHARS = 150
MAX_CHARS = 1200
SAMPLE_SIZE = 50
SEED = 42
INPUT_FILE = "workspace_with_sections.json"
OUTPUT_FILE = "golden_qa_data.json"

# Regexes to catch move notation / tabular sequences
MOVE_NUMBER_RE = re.compile(r"\d+\.\s*[A-Za-z]")
MANY_SPACES_RE = re.compile(r"\s{10,}")  # lots of whitespace
PIECE_SYMBOLS = {"P", "R", "B", "Q", "K", "Kt"}  # basic piece symbols

def looks_like_moves(text: str) -> bool:
    """Heuristic filter: true if the block looks like a chess move list."""
    # Lots of numbered moves
    if len(MOVE_NUMBER_RE.findall(text)) >= 3:
        return True
    # Big whitespace alignment
    if MANY_SPACES_RE.search(text):
        return True
    # If most tokens are piece codes
    tokens = re.split(r"\W+", text)
    piece_tokens = [t for t in tokens if t in PIECE_SYMBOLS]
    if tokens and (len(piece_tokens) / len(tokens)) > 0.3:
        return True
    return False

# -------- MAIN --------
def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    blocks = data["blocks"]

    candidates = []
    for b in blocks:
        text = b.get("text", "").strip()
        if (
            b.get("type") in ("body", "h2", "h3")
            and MIN_CHARS <= len(text) <= MAX_CHARS
            and not looks_like_moves(text)
        ):
            candidates.append(text)

    print(f"Found {len(candidates)} candidate text sections after filtering.")

    random.seed(SEED)
    sampled = random.sample(candidates, min(SAMPLE_SIZE, len(candidates)))

    dataset = [{"context": ctx} for ctx in sampled]

    out_path = Path(OUTPUT_FILE)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(dataset)} contexts → {out_path}")

if __name__ == "__main__":
    main()