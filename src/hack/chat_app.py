"""Streamlit chat interface for RAG agent demo."""

import os
import dspy
import streamlit as st
from pathlib import Path
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from hack.rag_agent import create_agent
from hack.pdf_processor import process_pdf
from hack.retriever import FaissRetriever

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OPENAI_API_KEY not found in .env file")
    st.stop()


@st.cache_resource
def initialize_dspy():
    """Initialize DSPy configuration (cached across reruns)."""
    # lm = dspy.LM(model="openai/gpt-4o-mini", api_key=openai_api_key)
    lm = dspy.LM(model="openai/gpt-4o", api_key=openai_api_key)
    dspy.configure(lm=lm)


def create_agent_from_uploaded_pdf(uploaded_file):
    """Process uploaded PDF and create agent with in-memory retriever."""
    # Save uploaded file to temporary location
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp_file.name)

    try:
        # Process PDF
        with st.spinner("Processing PDF... This may take a minute."):
            index, blocks, metadata = process_pdf(tmp_path, doc_id=uploaded_file.name)

        # Create retriever with in-memory data
        retriever = FaissRetriever(faiss_index=index, blocks=blocks)

        # Create agent
        agent = create_agent(retriever=retriever)

        return agent, metadata
    finally:
        # Clean up temporary file
        tmp_path.unlink(missing_ok=True)


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

    # Initialize DSPy (cached)
    initialize_dspy()
    initialize_messages()

    # Initialize session state for PDF processing
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "pdf_metadata" not in st.session_state:
        st.session_state.pdf_metadata = None

    # Sidebar with controls
    with st.sidebar:
        st.header("Document Source")

        # PDF upload
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            help="Upload a PDF to create a new workspace"
        )

        if uploaded_file is not None:
            # Check if this is a new file
            current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if "current_file_id" not in st.session_state or st.session_state.current_file_id != current_file_id:
                # Process new PDF
                try:
                    agent, metadata = create_agent_from_uploaded_pdf(uploaded_file)
                    st.session_state.agent = agent
                    st.session_state.pdf_metadata = metadata
                    st.session_state.current_file_id = current_file_id
                    st.session_state.messages = []  # Clear chat history
                    st.success(f"✅ Processed {metadata['num_pages']} pages, {metadata['num_blocks']} blocks")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    st.session_state.agent = None

        # Option to use default chess PDF
        st.divider()
        if st.button("📚 Use Default Chess PDF", use_container_width=True):
            try:
                st.session_state.agent = create_agent()
                st.session_state.pdf_metadata = {
                    "doc_id": "chess_pdf",
                    "num_pages": 95,
                    "num_blocks": 1941
                }
                st.session_state.current_file_id = "default_chess"
                st.session_state.messages = []
                st.success("✅ Loaded default chess PDF")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading default PDF: {str(e)}")

        st.divider()

        # Show current document info
        if st.session_state.pdf_metadata:
            st.header("Current Document")
            metadata = st.session_state.pdf_metadata
            st.markdown(f"""
            - **Document**: {metadata['doc_id']}
            - **Pages**: {metadata['num_pages']}
            - **Blocks**: {metadata['num_blocks']}
            """)

        st.divider()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        st.header("About")
        st.markdown("""
        This chat interface uses:
        - **DSPy** for agent orchestration
        - **FAISS** for semantic search
        - **OpenAI GPT-4o** for language understanding

        Upload a PDF to create a workspace, then ask questions
        about the document content.
        """)

    # Display chat history
    for msg in st.session_state.messages:
        display_message(msg["role"], msg["content"], msg.get("citations"))

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Check if agent is loaded
        if st.session_state.agent is None:
            st.warning("⚠️ Please upload a PDF or load the default Chess PDF first")
        else:
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            display_message("user", prompt)

            # Get agent response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.agent(question=prompt)
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
