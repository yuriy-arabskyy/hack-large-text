"""Streamlit chat interface for RAG agent demo."""

import os
import dspy
import streamlit as st
from dotenv import load_dotenv
from hack.rag_agent import create_agent

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OPENAI_API_KEY not found in .env file")
    st.stop()


@st.cache_resource
def initialize_agent():
    """Initialize the RAG agent with DSPy configuration (cached across reruns)."""
    # Configure DSPy with OpenAI
    # lm = dspy.LM(model="openai/gpt-4o-mini", api_key=openai_api_key)
    lm = dspy.LM(model="openai/gpt-4o", api_key=openai_api_key)
    dspy.configure(lm=lm)

    # Create and return agent
    return create_agent()


def initialize_messages():
    """Initialize message history in session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def display_message(role, content, citations=None):
    """Display a chat message with optional citations."""
    with st.chat_message(role):
        st.markdown(content)
        if citations:
            with st.expander("📚 Citations", expanded=False):
                st.code(citations, language="json")


def main():
    """Run the Streamlit chat interface."""
    st.set_page_config(page_title="RAG Agent Chat", page_icon="🤖", layout="wide")

    st.title("🤖 RAG Agent Chat Demo")
    st.caption("Ask questions about your documents - powered by DSPy and FAISS")

    # Initialize agent (cached) and messages
    agent = initialize_agent()
    initialize_messages()

    # Sidebar with controls
    with st.sidebar:
        st.header("Controls")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        st.header("About")
        st.markdown("""
        This chat interface uses:
        - **DSPy** for agent orchestration
        - **FAISS** for semantic search
        - **OpenAI GPT-4o-mini** for language understanding

        The agent retrieves relevant information from indexed documents
        and provides answers with citations.
        """)

    # Display chat history
    for msg in st.session_state.messages:
        display_message(msg["role"], msg["content"], msg.get("citations"))

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message("user", prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # import pdb; pdb.set_trace()
                    result = agent(question=prompt)
                    answer = result["answer"]
                    citations = result["citations"]

                    # Display answer
                    st.markdown(answer)

                    # Display citations in expander
                    if citations:
                        with st.expander("📚 Citations", expanded=False):
                            st.code(citations, language="json")

                    # Add to message history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "citations": citations}
                    )

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )


if __name__ == "__main__":
    main()
