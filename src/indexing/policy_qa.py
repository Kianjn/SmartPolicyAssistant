import os
import glob
from dotenv import load_dotenv

# LangChain components
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# --- Configuration ---
# Load environment variables (especially GOOGLE_API_KEY)
load_dotenv()

# Check for API key
if os.getenv("GOOGLE_API_KEY") is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file.")

# --- Constants ---
TEXT_FILES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")  # Directory containing .txt files
VECTORSTORE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models") # Directory to save/load the FAISS index
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
# You can switch to other embedding models if desired, e.g. from HuggingFace
# from langchain_community.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_MODEL = "models/embedding-001" # Google's embedding model
LLM_MODEL = "gemini-1.5-flash" # Or "gemini-pro", etc.

# --- Global Variables ---
# We'll initialize these lazily
vector_store = None
qa_chain = None

# --- Core Functions ---

def load_documents(directory_path: str):
    """Loads all .txt documents from the specified directory."""
    print(f"Loading documents from: {directory_path}")
    # Using TextLoader for each file to better handle potential encoding issues
    all_files = glob.glob(os.path.join(directory_path, "*.txt"))
    if not all_files:
        raise FileNotFoundError(f"No .txt files found in the directory: {directory_path}")

    docs = []
    for file_path in all_files:
        try:
            # Attempt to load with default UTF-8, add error handling if needed
            loader = TextLoader(file_path, encoding='utf-8')
            docs.extend(loader.load())
        except Exception as e:
            print(f"Warning: Could not load file {file_path}. Error: {e}")
            # You might want to try other encodings here like 'latin-1' if UTF-8 fails
            # try:
            #     loader = TextLoader(file_path, encoding='latin-1')
            #     docs.extend(loader.load())
            # except Exception as e2:
            #     print(f"Warning: Still could not load file {file_path} with latin-1. Error: {e2}")
    print(f"Loaded {len(docs)} document sections.") # TextLoader might load one doc per file
    return docs

def split_documents(documents):
    """Splits documents into smaller chunks."""
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def get_vector_store(force_recreate=False):
    """Creates or loads the FAISS vector store."""
    global vector_store
    if vector_store is not None and not force_recreate:
        print("Using existing vector store.")
        return vector_store

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(VECTORSTORE_PATH) and not force_recreate:
        print(f"Loading existing vector store from: {VECTORSTORE_PATH}")
        vector_store = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True) # Add allow_dangerous_deserialization for newer LangChain versions
        print("Vector store loaded successfully.")
    else:
        print("Creating new vector store...")
        if not os.path.exists(TEXT_FILES_DIR):
             raise FileNotFoundError(f"Text directory not found: {TEXT_FILES_DIR}. Cannot create vector store.")
        documents = load_documents(TEXT_FILES_DIR)
        if not documents:
             raise ValueError(f"No documents were loaded from {TEXT_FILES_DIR}. Cannot create vector store.")
        chunks = split_documents(documents)
        if not chunks:
             raise ValueError("Document splitting resulted in zero chunks. Cannot create vector store.")

        print("Generating embeddings and building FAISS index (this may take a while)...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        print("FAISS index built.")
        print(f"Saving vector store to: {VECTORSTORE_PATH}")
        vector_store.save_local(VECTORSTORE_PATH)
        print("Vector store saved.")

    return vector_store

def initialize_qa_chain(force_recreate_vs=False):
    """Initializes the RetrievalQA chain."""
    global qa_chain
    if qa_chain is not None and not force_recreate_vs: # Only recreate chain if VS was recreated
        print("Using existing Q&A chain.")
        return qa_chain

    print("Initializing Q&A chain...")
    vs = get_vector_store(force_recreate=force_recreate_vs)

    # Initialize the LLM (Gemini)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL,
                             temperature=0.1, # Lower temperature for more factual answers
                             convert_system_message_to_human=True) # Good practice for some models

    # Create the retriever
    retriever = vs.as_retriever(
        search_type="similarity", # Or "mmr" for max marginal relevance
        search_kwargs={'k': 4} # Retrieve top 4 most relevant chunks
    )

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" puts all context in one prompt; others are "map_reduce", "refine"
        retriever=retriever,
        return_source_documents=True, # Set to True to see which chunks were retrieved
        chain_type_kwargs={"prompt": None} # Use default prompt, or customize here
    )
    print("Q&A chain initialized.")
    return qa_chain

def ask_policy(question: str):
    """
    Asks a question to the initialized Q&A system.

    Args:
        question: The question string.

    Returns:
        A dictionary containing the 'answer' and potentially 'source_documents'.
        Returns None if the Q&A chain is not initialized.
    """
    global qa_chain
    if qa_chain is None:
        print("Error: Q&A chain not initialized. Call initialize_qa_chain() first.")
        return None

    print(f"\nAsking question: {question}")
    try:
        # Use invoke for newer LangChain versions
        result = qa_chain.invoke({"query": question})

        print("\nAnswer:")
        print(result.get('result', 'No answer found.')) # 'result' is the key for the answer

        if result.get('source_documents'):
            print("\nSources:")
            # Limiting source output for clarity
            for i, doc in enumerate(result['source_documents'][:2]): # Show first 2 sources
                 source_name = doc.metadata.get('source', 'Unknown source')
                 print(f"  Source {i+1}: {source_name} (excerpt)")
                 # Print only the first few lines of the source chunk
                 print("  -------")
                 print(f"  {doc.page_content[:250]}...") # Print beginning of the chunk
                 print("  -------\n")

        return {
            "answer": result.get('result'),
            "source_documents": result.get('source_documents')
        }

    except Exception as e:
        print(f"An error occurred during query: {e}")
        # You might want to inspect the specific error, e.g., API rate limits, etc.
        return {"answer": f"Error processing question: {e}", "source_documents": []}


# --- Main Execution ---
if __name__ == "__main__":
    # Initialize the system (loads/creates vector store and sets up QA chain)
    # Set force_recreate_vs=True if you updated your .txt files and want to rebuild the index
    initialize_qa_chain(force_recreate_vs=True)

    # Example Usage:
    response1 = ask_policy("What is Germany's target for renewables in electricity generation by 2030?")
    # response2 = ask_policy("Compare the main climate mitigation measures mentioned by France and Spain.")
    # response3 = ask_policy("What are the key policies for the transport sector in Italy's NECP?")

    # --- Optional: Interactive Loop ---
    print("\n--- Interactive Q&A ---")
    print("Type 'quit' or 'exit' to stop.")
    while True:
        user_question = input("Ask a policy question: ")
        if user_question.lower() in ['quit', 'exit']:
            break
        if not user_question.strip():
            continue
        ask_policy(user_question)