import dspy
import os
from dotenv import load_dotenv
from models.rag_models import QueryUnderstanding, EvidenceSelection, AnswerSynthesis
from retriever import MockRetriever

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Agent / Module ---
class WorkspaceAgent(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever  # your DB / index backend
        self.understand = dspy.Predict(QueryUnderstanding)
        self.select = dspy.Predict(EvidenceSelection)
        self.synthesize = dspy.Predict(AnswerSynthesis)

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


retriever = MockRetriever()
agent = WorkspaceAgent(retriever)