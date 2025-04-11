import os
import sys
from dotenv import load_dotenv

# LangChain components
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStore

# --- Configuration ---
load_dotenv()

# Check for API key
if os.getenv("GOOGLE_API_KEY") is None:
    print("Error: GOOGLE_API_KEY not found in environment variables.")
    print("Please ensure a .env file with the key exists in the script's directory.")
    sys.exit(1) # Exit if key is missing

# --- Constants ---
# <<< IMPORTANT: Make sure this path EXACTLY matches where your index was saved >>>
VECTORSTORE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
# If you saved it elsewhere, update this path, e.g.:
# VECTORSTORE_PATH = "/Users/kian/Capstone Project/02.RAG/FAISS"

# Ensure the path exists before proceeding
if not os.path.exists(VECTORSTORE_PATH):
    print(f"Error: Vector store path not found: {VECTORSTORE_PATH}")
    print("Please ensure the path is correct and the index files exist.")
    sys.exit(1)

EMBEDDING_MODEL = "models/embedding-001" # Must match the model used for creating the index
LLM_MODEL = "gemini-1.5-flash" # Or "gemini-pro", etc. Can be changed for chatting
RETRIEVER_K = 4 # Number of relevant chunks to fetch

# --- Core Functions ---

def load_vector_store() -> VectorStore | None:
    """Loads the existing FAISS vector store."""
    print(f"Loading vector store from: {VECTORSTORE_PATH}...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True # Needed for loading FAISS with newer LangChain
        )
        print("Vector store loaded successfully.")
        return vector_store
    except FileNotFoundError:
        print(f"Error: index.faiss or index.pkl not found in {VECTORSTORE_PATH}.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the vector store: {e}")
        return None

def initialize_chat_chain(vector_store: VectorStore) -> RetrievalQA | None:
    """Initializes the RetrievalQA chain for chatting."""
    print("Initializing chat chain...")
    try:
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL,
                                 temperature=0.1, # Keep low for factual answers
                                 convert_system_message_to_human=True)

        # Create the retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': RETRIEVER_K}
        )

        # Create the RetrievalQA chain - NO source documents returned
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # Standard chain type for this use case
            retriever=retriever,
            return_source_documents=False # Set to False to only get the answer
        )
        print("Chat chain ready.")
        return qa_chain
    except Exception as e:
        print(f"Failed to initialize chat chain: {e}")
        return None

def ask_question(qa_chain: RetrievalQA, question: str) -> str:
    """Asks a question and returns only the answer string."""
    if not qa_chain:
        return "Error: Chat chain is not initialized."
    try:
        # Use invoke for newer LangChain versions
        result = qa_chain.invoke({"query": question})
        # Extract the answer string, handle cases where it might be missing
        answer = result.get('result', 'Sorry, I could not find an answer.')
        return answer
    except Exception as e:
        print(f"\nAn error occurred during query: {e}")
        return "An error occurred while processing your question."

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load the existing vector store
    vector_store = load_vector_store()
    if not vector_store:
        sys.exit(1) # Exit if loading failed

    # 2. Initialize the Q&A chain
    qa_chain = initialize_chat_chain(vector_store)
    if not qa_chain:
        sys.exit(1) # Exit if chain initialization failed

    # 3. Start interactive chat loop
    print("\n--- Smart Policy Assistant Chat ---")
    print("Ask questions about the NECP documents. Type 'quit' or 'exit' to stop.")

    while True:
        try:
            user_question = input("\nAsk: ")
            if user_question.lower().strip() in ['quit', 'exit']:
                print("Exiting chat. Goodbye!")
                break
            if not user_question.strip():
                continue

            # Get and print only the answer
            answer = ask_question(qa_chain, user_question)
            print(f"\nAssistant: {answer}")

        except EOFError: # Handle Ctrl+D exit
             print("\nExiting chat. Goodbye!")
             break
        except KeyboardInterrupt: # Handle Ctrl+C exit
            print("\nExiting chat. Goodbye!")
            break