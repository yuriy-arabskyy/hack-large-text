import dspy

class QueryUnderstanding(dspy.Signature):
    """Expand user question into retrievable terms and a search plan."""
    question = dspy.InputField(desc="The user’s natural language question")
    search_terms = dspy.OutputField(desc="Keywords or expanded query for retrieval")
    search_plan = dspy.OutputField(desc="Description of which retrieval strategies to use")

class EvidenceSelection(dspy.Signature):
    """Select the most relevant evidence from retrieved units."""
    question = dspy.InputField(desc="The user’s question")
    candidates = dspy.InputField(desc="List of candidate text snippet strings")
    selected = dspy.OutputField(desc="List of the most relevant snippet strings")

class AnswerSynthesis(dspy.Signature):
    """Compose final answer with inline citations and JSON anchors."""
    question = dspy.InputField(desc="The user's question")
    evidence = dspy.InputField(desc="List of selected evidence snippets with page numbers and sections")
    answer_text = dspy.OutputField(desc="Generated natural-language answer with inline citations")
    citations_json = dspy.OutputField(
        desc='List of citation objects in JSON format. Each citation must include: '
        '"page" (integer page number), "section" (section path string), and "quote" (relevant text excerpt). '
        'Example: [{"page": 12, "section": "Opening Strategy", "quote": "Control the center"}]'
    )