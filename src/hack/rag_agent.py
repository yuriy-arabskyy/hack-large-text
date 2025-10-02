import dspy
import os
from dotenv import load_dotenv
from hack.models.rag_models import QueryUnderstanding, EvidenceSelection, AnswerSynthesis
from hack.retriever import FaissRetriever

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Agent / Module ---
class WorkspaceAgent(dspy.Module):
    def __init__(self, retriever, lm):
        super().__init__()
        self.retriever = retriever
        # Do NOT pass `lm` as a kwarg to Predict (treated as config in dspy 3.x)
        self.understand = dspy.Predict(QueryUnderstanding)
        self.select = dspy.Predict(EvidenceSelection)
        self.synthesize = dspy.Predict(AnswerSynthesis)

        # Assign LM on the predictors explicitly (if provided)
        if lm is not None:
            self.understand.lm = lm
            self.select.lm = lm
            self.synthesize.lm = lm

    def forward(self, question: str):
        # 1. Query → retrieval plan
        q_resp = self.understand(question=question)
        search_terms = q_resp.search_terms
        search_plan = q_resp.search_plan

        # 2. Fetch candidate units
        candidates = []
        candidates += self.retriever.search_text(search_terms)
        candidates += self.retriever.search_tables(search_terms)
        candidates += self.retriever.search_images(search_terms)

        # Prepare candidate strings for the LLM
        candidate_strs = [
            f"[{c['unit_id']}] {c['content']} (p. {c['page']}, section {c.get('section_path','')})"
            for c in candidates
        ]

        # 3. Evidence selection
        sel_resp = self.select(question=question, candidates=candidate_strs)
        selected_snippets = sel_resp.selected

        # 4. Answer synthesis
        ans_resp = self.synthesize(question=question, evidence=selected_snippets)
        return {
            "answer": ans_resp.answer_text,
            "citations": ans_resp.citations_json
        }


# Initialize retriever with paths to FAISS index and workspace JSON
# These paths assume the script is run from the project root
def create_agent(
    faiss_index_path="experiments/chess_pdf.faiss",
    workspace_json_path="experiments/workspace_with_embeddings.json",
    lm=None
):
    retriever = FaissRetriever(
        faiss_index_path=faiss_index_path,
        workspace_json_path=workspace_json_path
    )
    return WorkspaceAgent(retriever, lm=lm)
