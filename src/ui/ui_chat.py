import streamlit as st
import os
import sys
from dotenv import load_dotenv

# --- Path Setup ---
# Get the absolute path of the current script's directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the path to the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
# Get the path to the models directory
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Add RAG directory to sys.path *temporarily* if needed (usually not required if structure is correct)
# sys.path.insert(0, RAG_DIR)

# --- Load Environment Variables ---
# Load from .env file located in the core directory
dotenv_path = os.path.join(PROJECT_ROOT, "src", "core", ".env")
load_dotenv(dotenv_path=dotenv_path)

# LangChain components (assuming they are installed in the environment)
try:
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains import RetrievalQA
    from langchain_core.vectorstores import VectorStore
except ImportError:
    st.error("Required LangChain libraries not found. Make sure you are running this in the correct Conda environment ('policy_env') where packages were installed.")
    st.stop()


# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Constants ---
VECTORSTORE_PATH = MODELS_DIR

EMBEDDING_MODEL = "models/embedding-001" # Must match the model used for creating the index
LLM_MODEL = "gemini-1.5-flash"         # Or "gemini-pro", etc.
RETRIEVER_K = 4                        # Number of relevant chunks to fetch

# --- Error Checks ---
if not GOOGLE_API_KEY:
    st.error("Error: GOOGLE_API_KEY not found.")
    st.info(f"Please ensure a .env file with the key exists in: {PROJECT_ROOT}")
    st.stop()

# Check if the FAISS directory itself exists
if not os.path.isdir(VECTORSTORE_PATH): # Use isdir to check for the directory
    st.error(f"Error: FAISS index directory not found: {VECTORSTORE_PATH}")
    st.info("Please ensure the 'FAISS' folder exists inside 'models' and contains 'index.faiss' and 'index.pkl'.")
    st.stop()
# You could add more specific checks for index.faiss and index.pkl if needed
# if not os.path.exists(os.path.join(VECTORSTORE_PATH, "index.faiss")):
#     st.error(f"Error: index.faiss not found in {VECTORSTORE_PATH}")
#     st.stop()


# --- Caching Functions for Performance ---

# Cache the loading of the vector store (runs only once unless path changes)
@st.cache_resource(show_spinner="Loading Policy Knowledge Base...")
def load_vector_store(_vectorstore_path: str) -> VectorStore | None:
    """Loads the existing FAISS vector store."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
        vector_store = FAISS.load_local(
            _vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except FileNotFoundError:
        # This error might still occur if files *inside* the directory are missing
        st.error(f"Error loading index files from directory: {_vectorstore_path}. Ensure 'index.faiss' and 'index.pkl' are present.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the vector store: {e}")
        return None

# Cache the initialization of the QA chain (runs only once unless vector_store or llm changes)
@st.cache_resource(show_spinner="Initializing AI Assistant...")
def initialize_chat_chain(_vector_store: VectorStore) -> RetrievalQA | None:
    """Initializes the RetrievalQA chain for chatting."""
    try:
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1,
            convert_system_message_to_human=True
        )
        retriever = _vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': RETRIEVER_K}
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False # Only get the answer
        )
        return qa_chain
    except Exception as e:
        st.error(f"Failed to initialize chat chain: {e}")
        return None

# --- Main Streamlit App ---

st.set_page_config(page_title="Smart Policy Assistant", layout="wide")
st.title("🇪🇺 Smart Policy Assistant Chat")
st.info("Ask questions about EU countries' National Energy and Climate Plans (NECPs).")

# Load resources using cached functions
vector_store = load_vector_store(VECTORSTORE_PATH)
if not vector_store:
    st.stop() # Stop execution if vector store failed to load

qa_chain = initialize_chat_chain(vector_store)
if not qa_chain:
    st.stop() # Stop execution if chain failed to initialize

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input using the chat input widget
if prompt := st.chat_input("Ask about energy policies..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display thinking indicator and get assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        try:
            # Get the answer from the QA chain
            result = qa_chain.invoke({"query": prompt})
            answer = result.get('result', 'Sorry, I encountered an issue finding an answer.')
            message_placeholder.markdown(answer) # Display the actual answer
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            error_message = f"An error occurred: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})